import time
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from .config import TZ

# ================= FORMATTING =================


def _to_minutes(t: str) -> int:
    h, m = map(int, t.split(":"))
    return h * 60 + m


def _fmt_duration_ua(minutes: int) -> str:
    h = minutes // 60
    m = minutes % 60
    parts = []
    if h > 0:
        parts.append(f"{h} год")
    if m > 0:
        parts.append(f"{m} хв")
    if not parts:
        return "0 хв"
    return " ".join(parts)


def total_off_minutes(intervals: List[Tuple[str, str]]) -> int:
    s = 0
    for a, b in intervals:
        s += max(0, _to_minutes(b) - _to_minutes(a))
    return s


def intervals_to_text(intervals: List[Tuple[str, str]]) -> str:
    return ";".join([f"{a}-{b}" for a, b in intervals])


def build_update_prefix(
    old_total_off: Optional[int],
    old_intervals_text: Optional[str],
    new_total_off: int,
    new_intervals_text: str,
) -> str:
    header = "🔄 Графік оновлено для твоєї черги"

    lines = [header, ""]
    delta_line = None

    if old_total_off is not None:
        delta = new_total_off - old_total_off
        if delta > 0:
            delta_line = f"❌ Світла стало МЕНШЕ на {_fmt_duration_ua(delta)}"
        elif delta < 0:
            delta_line = f"✅ Світла стало БІЛЬШЕ на {_fmt_duration_ua(abs(delta))}"
        else:
            delta_line = "➖ Кількість світла не змінилась"

    if old_total_off is None and old_intervals_text is None:
        lines.append("ℹ️ Оновили графік — показую актуальний стан")
        return "\n".join(lines).strip()

    if delta_line:
        lines.append(delta_line)

    return "\n".join(lines).strip()


def is_light_now(intervals: List[Tuple[str, str]], now_dt: datetime) -> bool:
    now_m = now_dt.hour * 60 + now_dt.minute
    for a, b in intervals:
        sa = _to_minutes(a)
        sb = _to_minutes(b)
        if sa <= now_m < sb:
            return False
    return True


def time_to_light(intervals: List[Tuple[str, str]], now_dt: datetime) -> Optional[str]:
    now_m = now_dt.hour * 60 + now_dt.minute
    for a, b in intervals:
        sa = _to_minutes(a)
        sb = _to_minutes(b)
        if sa <= now_m < sb:
            delta = sb - now_m
            h = delta // 60
            m = delta % 60
            parts = []
            if h > 0:
                parts.append(f"{h} год")
            if m > 0:
                parts.append(f"{m} хв")
            return "ввімкнеться через " + " ".join(parts)
    return None


def filter_past_intervals(intervals: List[Tuple[str, str]], now_dt: datetime) -> List[Tuple[str, str]]:
    now_m = now_dt.hour * 60 + now_dt.minute
    out: List[Tuple[str, str]] = []
    for a, b in intervals:
        if _to_minutes(b) <= now_m:
            continue
        out.append((a, b))
    return out


def _current_slot_index(now_dt: datetime) -> int:
    return (now_dt.hour * 60 + now_dt.minute) // 30


def future_changed_only(row_states_new: List[bool], row_states_old_text: Optional[str], now_dt: datetime) -> bool:
    """
    True якщо є зміни у майбутніх слотах (від поточного слоту до кінця).
    Якщо old_text нема — вважаємо що зміни релевантні.
    """
    if not row_states_old_text or len(row_states_old_text) != 48:
        return True

    idx = _current_slot_index(now_dt)
    for i in range(idx, 48):
        old_off = (row_states_old_text[i] == "1")
        if row_states_new[i] != old_off:
            return True
    return False


def slots_to_intervals_from_index(slots: List[bool], start_idx: int = 0) -> List[Tuple[str, str]]:
    """
    slots: 48 bool, True=off
    start_idx: з якого слоту рахувати (майбутнє)
    """
    intervals = []
    start = None

    for i in range(start_idx, 48):
        off = slots[i]
        if off and start is None:
            start = i
        if (not off) and start is not None:
            intervals.append((start, i))
            start = None

    if start is not None:
        intervals.append((start, 48))

    def slot_to_time(s: int) -> str:
        if s == 48:
            return "24:00"
        m = s * 30
        return f"{m // 60:02d}:{m % 60:02d}"

    return [(slot_to_time(a), slot_to_time(b)) for a, b in intervals]


def states_text_to_slots(states_text: str) -> List[bool]:
    """states_text: '1'=off, '0'=on, len=48"""
    return [(ch == "1") for ch in states_text[:48]]


def diff_future_intervals(
    old_states_text: Optional[str],
    new_states_48: List[bool],
    now_dt: datetime
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Повертає (cancelled, added) тільки для майбутніх слотів (від поточного до кінця).
    cancelled: було OFF, стало ON
    added:     було ON, стало OFF
    """
    if not old_states_text or len(old_states_text) != 48:
        # нема з чим порівнювати
        return [], []

    old_slots = states_text_to_slots(old_states_text)
    idx = _current_slot_index(now_dt)

    cancelled_slots = [False] * 48
    added_slots = [False] * 48

    for i in range(idx, 48):
        o = old_slots[i]
        n = new_states_48[i]
        if o and not n:
            cancelled_slots[i] = True
        elif (not o) and n:
            added_slots[i] = True

    cancelled = slots_to_intervals_from_index(cancelled_slots, idx)
    added = slots_to_intervals_from_index(added_slots, idx)
    return cancelled, added


def fmt_interval_lines(intervals: List[Tuple[str, str]], prefix: str) -> str:
    if not intervals:
        return "—"
    return "\n".join([f"{prefix} {a}–{b}" for a, b in intervals])


def format_outages_by_dayparts_today(
    intervals: List[Tuple[str, str]],
    now_has_light: bool,
    now_dt: datetime,
    full_intervals: List[Tuple[str, str]],
) -> str:
    if now_has_light:
        status = "💡 Зараз є світло"
    else:
        eta = time_to_light(full_intervals, now_dt)
        status = f"❌ Зараз немає світла ({eta})" if eta else "❌ Зараз немає світла"

    if not intervals:
        return f"{status}\n\n🎉 Відключень більше не бачу"

    parts = {"🌙 Ніч": [], "☀️ День": [], "🌆 Вечір": []}
    for a, b in intervals:
        start = _to_minutes(a)
        if start < 6 * 60:          # 00:00–05:59
            parts["🌙 Ніч"].append((a, b))
        elif start < 16 * 60:       # 06:00–15:59
            parts["☀️ День"].append((a, b))
        else:                       # 16:00–24:00
            parts["🌆 Вечір"].append((a, b))

    lines = [status, "", "🔌 Без світла:\n"]
    for title in ["🌙 Ніч", "☀️ День", "🌆 Вечір"]:
        if parts[title]:
            lines.append(title)
            for a, b in parts[title]:
                lines.append(f"{a}–{b}")
            lines.append("")
    return "\n".join(lines).strip()


def format_outages_by_dayparts_plain(intervals: List[Tuple[str, str]]) -> str:
    if not intervals:
        return "🎉 Відключень не бачу"

    parts = {"🌙 Ніч": [], "☀️ День": [], "🌆 Вечір": []}
    for a, b in intervals:
        start = _to_minutes(a)
        if start < 6 * 60:
            parts["🌙 Ніч"].append((a, b))
        elif start < 16 * 60:
            parts["☀️ День"].append((a, b))
        else:
            parts["🌆 Вечір"].append((a, b))

    lines = ["🔌 Без світла:\n"]
    for title in ["🌙 Ніч", "☀️ День", "🌆 Вечір"]:
        if parts[title]:
            lines.append(title)
            for a, b in parts[title]:
                lines.append(f"{a}–{b}")
            lines.append("")
    return "\n".join(lines).strip()


def tomorrow_label(now_dt: Optional[datetime] = None) -> str:
    if now_dt is None:
        now_dt = datetime.now(TZ)
    tomorrow = now_dt.date() + timedelta(days=1)
    return tomorrow.strftime("%d.%m")


def day_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def tomorrow_day_str(now_dt: Optional[datetime] = None) -> str:
    if now_dt is None:
        now_dt = datetime.now(TZ)
    return (now_dt.date() + timedelta(days=1)).strftime("%Y-%m-%d")


def _cooldown_left(last_ts: float, cooldown: int) -> int:
    left = int(cooldown - (time.time() - last_ts))
    return max(0, left)
