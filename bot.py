import os
import time
import sqlite3
import logging
import hashlib
import shutil
import asyncio
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import requests
import numpy as np
import cv2

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import Forbidden, BadRequest
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ================= CONFIG =================

API_BASE = "https://api-voe-poweron.inneti.net"
API_TODAY = f"{API_BASE}/api/options?option_key=pw_gpv_image_today"
API_TOMORROW = f"{API_BASE}/api/options?option_key=pw_gpv_image_tomorrow"

DB_PATH = os.getenv("DB_PATH", "users.db")

TZ = ZoneInfo("Europe/Uzhgorod")

# Адміни (user_id). У приватному чаті user_id == chat_id.
ADMIN_IDS = {328587643}

# Анти-спам / анти-трафік
CACHE_TTL_SEC = 60          # не качати картинку частіше ніж раз/хв (на весь процес)
PARSE_TTL_SEC = 60          # не парсити OpenCV частіше ніж раз/хв (на весь процес)
NOW_COOLDOWN_SEC = 30       # не обробляти Now/Tomorrow частіше ніж раз/30с на один чат

# Поріг визначення "темна клітинка" (нема світла) по V каналу HSV (підібраний)
V_THRESHOLD = 185

# ================= QUIET HOURS =================
# Тихі години для ПУШ-розсилок (scheduler jobs). Ручні кнопки/запити працюють як і раніше.
# Інтервал: [start, end), тобто END не включно.
QUIET_HOURS_START = 0   # 00:00
QUIET_HOURS_END   = 6   # 06:00

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("light_bot")

# При бажанні приглушити шумні логи:
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("telegram.ext").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

SESSION = requests.Session()
_retry = Retry(
    total=4,
    connect=4,
    read=4,
    backoff_factor=0.6,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    raise_on_status=False,
)
_adapter = HTTPAdapter(max_retries=_retry, pool_connections=10, pool_maxsize=10)
SESSION.mount("https://", _adapter)
SESSION.mount("http://", _adapter)
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json,image/*,*/*",
    "Accept-Language": "uk-UA,uk;q=0.9,en;q=0.8",
})


# ================= DATABASE =================

def db_connect() -> sqlite3.Connection:
    """
    Єдине місце для підключення до SQLite з корисними PRAGMA:
    - WAL: краще для паралельних читань/записів (scheduler + handlers)
    - busy_timeout: менше шансів на "database is locked"
    """
    con = sqlite3.connect(DB_PATH)
    try:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA busy_timeout=3000")
        con.execute("PRAGMA synchronous=NORMAL")
    except Exception:
        pass
    return con


def _ensure_users_columns(con: sqlite3.Connection):
    """
    Soft-migration: додаємо колонки, якщо їх нема.
    """
    ddls = [
        # було раніше
        "ALTER TABLE users ADD COLUMN last_fingerprint TEXT",
        "ALTER TABLE users ADD COLUMN username TEXT",
        "ALTER TABLE users ADD COLUMN created_at INTEGER",
        "ALTER TABLE users ADD COLUMN last_seen_at INTEGER",
        "ALTER TABLE users ADD COLUMN last_total_off INTEGER",
        "ALTER TABLE users ADD COLUMN last_intervals TEXT",
        "ALTER TABLE users ADD COLUMN last_states TEXT",

        # ✅ NEW: розділена памʼять TODAY (в межах одного календарного дня)
        "ALTER TABLE users ADD COLUMN today_day TEXT",             # YYYY-MM-DD
        "ALTER TABLE users ADD COLUMN today_fingerprint TEXT",
        "ALTER TABLE users ADD COLUMN today_total_off INTEGER",
        "ALTER TABLE users ADD COLUMN today_intervals TEXT",
        "ALTER TABLE users ADD COLUMN today_states TEXT",

        # ✅ NEW: памʼять TOMORROW
        "ALTER TABLE users ADD COLUMN tomorrow_day TEXT",          # YYYY-MM-DD (день, на який графік)
        "ALTER TABLE users ADD COLUMN tomorrow_fingerprint TEXT",
        "ALTER TABLE users ADD COLUMN tomorrow_total_off INTEGER",
        "ALTER TABLE users ADD COLUMN tomorrow_intervals TEXT",
        "ALTER TABLE users ADD COLUMN tomorrow_states TEXT",
    ]
    for ddl in ddls:
        try:
            con.execute(ddl)
        except sqlite3.OperationalError:
            pass


def db_init():
    """
    Створює таблиці, і робить "soft-migration" для старих users.db.
    """
    with db_connect() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS users(
                chat_id INTEGER PRIMARY KEY,
                queue INTEGER NOT NULL,
                subqueue INTEGER NOT NULL,

                last_fingerprint TEXT,
                username TEXT,
                created_at INTEGER,
                last_seen_at INTEGER,
                last_total_off INTEGER,
                last_intervals TEXT,
                last_states TEXT,

                today_day TEXT,
                today_fingerprint TEXT,
                today_total_off INTEGER,
                today_intervals TEXT,
                today_states TEXT,

                tomorrow_day TEXT,
                tomorrow_fingerprint TEXT,
                tomorrow_total_off INTEGER,
                tomorrow_intervals TEXT,
                tomorrow_states TEXT
            )
        """)

        _ensure_users_columns(con)

        con.execute("""
            CREATE TABLE IF NOT EXISTS meta(
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        con.commit()


def db_touch_user(chat_id: int, username: Optional[str]):
    """
    created_at — ставимо тільки якщо ще NULL
    last_seen_at — оновлюємо завжди
    username — оновлюємо (може мінятись / бути None)
    """
    now_ts = int(time.time())
    with db_connect() as con:
        _ensure_users_columns(con)
        con.execute("""
            INSERT INTO users(
                chat_id, queue, subqueue,
                last_fingerprint, username, created_at, last_seen_at,
                last_total_off, last_intervals, last_states,
                today_day, today_fingerprint, today_total_off, today_intervals, today_states,
                tomorrow_day, tomorrow_fingerprint, tomorrow_total_off, tomorrow_intervals, tomorrow_states
            )
            VALUES(?, 1, 1,
                   NULL, ?, ?, ?,
                   NULL, NULL, NULL,
                   NULL, NULL, NULL, NULL, NULL,
                   NULL, NULL, NULL, NULL, NULL)
            ON CONFLICT(chat_id) DO UPDATE SET
                username=excluded.username,
                last_seen_at=excluded.last_seen_at,
                created_at=COALESCE(users.created_at, excluded.created_at)
        """, (chat_id, username, now_ts, now_ts))
        con.commit()


def db_upsert_user(chat_id: int, queue: int, subqueue: int):
    """
    Оновлює чергу/підчергу.
    """
    with db_connect() as con:
        _ensure_users_columns(con)
        con.execute("""
            INSERT INTO users(
                chat_id, queue, subqueue,
                last_fingerprint, last_total_off, last_intervals, last_states,
                today_day, today_fingerprint, today_total_off, today_intervals, today_states,
                tomorrow_day, tomorrow_fingerprint, tomorrow_total_off, tomorrow_intervals, tomorrow_states
            )
            VALUES(?, ?, ?,
                   NULL, NULL, NULL, NULL,
                   NULL, NULL, NULL, NULL, NULL,
                   NULL, NULL, NULL, NULL, NULL)
            ON CONFLICT(chat_id) DO UPDATE
            SET queue=excluded.queue,
                subqueue=excluded.subqueue
        """, (chat_id, queue, subqueue))
        con.commit()


def db_get_users_for_push() -> List[Tuple[int, int, int, Optional[str], Optional[int], Optional[str], Optional[str], Optional[str]]]:
    """
    Для today-changes пушів:
    Повертає:
      (chat_id, queue, subqueue, today_fingerprint, today_total_off, today_intervals, today_states, today_day)
    """
    with db_connect() as con:
        _ensure_users_columns(con)
        cur = con.execute("""
            SELECT chat_id, queue, subqueue,
                   today_fingerprint, today_total_off, today_intervals, today_states, today_day
            FROM users
        """)
        return list(cur.fetchall())


def db_get_users_basic() -> List[Tuple[int, int, int, Optional[str], Optional[str], Optional[str]]]:
    """
    Для tomorrow-розсилок:
      (chat_id, queue, subqueue, tomorrow_fingerprint, tomorrow_states, tomorrow_day)
    """
    with db_connect() as con:
        _ensure_users_columns(con)
        cur = con.execute("""
            SELECT chat_id, queue, subqueue,
                   tomorrow_fingerprint, tomorrow_states, tomorrow_day
            FROM users
        """)
        return list(cur.fetchall())


def db_get_user_queue(chat_id: int) -> Optional[Tuple[int, int]]:
    with db_connect() as con:
        _ensure_users_columns(con)
        cur = con.execute("SELECT queue, subqueue FROM users WHERE chat_id=?", (chat_id,))
        row = cur.fetchone()
        if not row:
            return None
        return int(row[0]), int(row[1])


def db_set_today_memory(chat_id: int, day: str, fp: str, total_off: int, intervals_text: str, states_text: str):
    with db_connect() as con:
        _ensure_users_columns(con)
        con.execute("""
            UPDATE users
            SET today_day=?,
                today_fingerprint=?,
                today_total_off=?,
                today_intervals=?,
                today_states=?
            WHERE chat_id=?
        """, (day, fp, total_off, intervals_text, states_text, chat_id))
        con.commit()


def db_set_tomorrow_memory(chat_id: int, day: str, fp: str, total_off: int, intervals_text: str, states_text: Optional[str]):
    with db_connect() as con:
        _ensure_users_columns(con)
        con.execute("""
            UPDATE users
            SET tomorrow_day=?,
                tomorrow_fingerprint=?,
                tomorrow_total_off=?,
                tomorrow_intervals=?,
                tomorrow_states=?
            WHERE chat_id=?
        """, (day, fp, total_off, intervals_text, states_text, chat_id))
        con.commit()


def db_delete_user(chat_id: int):
    with db_connect() as con:
        con.execute("DELETE FROM users WHERE chat_id=?", (chat_id,))
        con.commit()


def db_meta_get(key: str) -> Optional[str]:
    with db_connect() as con:
        cur = con.execute("SELECT value FROM meta WHERE key=?", (key,))
        row = cur.fetchone()
        return row[0] if row else None


def db_meta_set(key: str, value: str):
    with db_connect() as con:
        con.execute("""
            INSERT INTO meta(key, value) VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """, (key, value))
        con.commit()


def db_get_last_user() -> Optional[Tuple[int, Optional[str], Optional[int], Optional[int]]]:
    with db_connect() as con:
        _ensure_users_columns(con)
        cur = con.execute("""
            SELECT chat_id, username, created_at, last_seen_at
            FROM users
            WHERE created_at IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 1
        """)
        row = cur.fetchone()
        return row if row else None


def touch_from_update(update: Update) -> None:
    try:
        chat_id = update.effective_chat.id if update.effective_chat else None
        username = update.effective_user.username if update.effective_user else None
        if chat_id is not None:
            db_touch_user(chat_id, username)
    except Exception:
        log.exception("touch_from_update failed")


# ================= QUIET HOURS HELPERS =================

def is_quiet_hours(now_dt: Optional[datetime] = None) -> bool:
    if now_dt is None:
        now_dt = datetime.now(TZ)
    return QUIET_HOURS_START <= now_dt.hour < QUIET_HOURS_END


# ================= API / IMAGE =================

def fetch_latest_image_url_from_api(api_url: str) -> Optional[str]:
    r = SESSION.get(api_url, timeout=30)
    r.raise_for_status()
    data = r.json()

    def _normalize(v: Optional[str]) -> Optional[str]:
        if not isinstance(v, str):
            return None
        v = v.strip()
        if not v:
            return None
        return API_BASE + v if v.startswith("/") else v

    if isinstance(data, dict) and isinstance(data.get("hydra:member"), list) and data["hydra:member"]:
        item = data["hydra:member"][0]
        if isinstance(item, dict):
            return _normalize(item.get("value"))

    if isinstance(data, dict):
        return _normalize(data.get("value"))

    return None


def download_image(url: str) -> np.ndarray:
    r = SESSION.get(url, timeout=30)
    r.raise_for_status()
    arr = np.frombuffer(r.content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Не вдалося декодувати зображення")
    return img


# ================= CACHES (per endpoint) =================

_cached_img: Dict[str, Dict[str, Any]] = {
    API_TODAY: {"ts": 0.0, "url": None, "img": None},
    API_TOMORROW: {"ts": 0.0, "url": None, "img": None},
}

_parsed_cache: Dict[str, Dict[str, Any]] = {
    API_TODAY: {"ts": 0.0, "url": None, "states": None, "intervals_by_row": None},
    API_TOMORROW: {"ts": 0.0, "url": None, "states": None, "intervals_by_row": None},
}


def get_image_cached(api_url: str) -> Tuple[Optional[str], Optional[np.ndarray]]:
    now = time.time()
    c = _cached_img.setdefault(api_url, {"ts": 0.0, "url": None, "img": None})

    if c["img"] is not None and c["url"] is not None and (now - c["ts"]) < CACHE_TTL_SEC:
        return c["url"], c["img"]

    url = fetch_latest_image_url_from_api(api_url)
    if not url:
        return None, None

    img = download_image(url)

    c["ts"] = now
    c["url"] = url
    c["img"] = img
    return url, img


# ================= FIXED GRID (ANCHOR + STEP) =================

@dataclass
class FixedGrid:
    cols: int = 24
    rows: int = 12

    # !!! ПІДСТАВ СВОЇ ПРАВИЛЬНІ ЦЕНТРИ !!!
    x0_top: int = 205
    y0_top: int = 450

    x0_bot: int = 205
    y0_bot: int = 1340

    dx: int = 65
    dy: int = 60


GRID = FixedGrid()


def _row_index(queue: int, subqueue: int) -> int:
    return (queue - 1) * 2 + (subqueue - 1)


def _cell_center(half: int, col: int, row: int, grid: FixedGrid) -> Tuple[int, int]:
    if half == 0:
        x0, y0 = grid.x0_top, grid.y0_top
    else:
        x0, y0 = grid.x0_bot, grid.y0_bot
    return x0 + col * grid.dx, y0 + row * grid.dy


def _states_by_threshold(v_vals: List[int], thr: int) -> List[bool]:
    """True = нема світла (темна клітинка)."""
    return [v < thr for v in v_vals]


def read_all_rows_states(img: np.ndarray, grid: FixedGrid) -> List[List[bool]]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    V = hsv[..., 2]  # uint8

    states_all: List[List[bool]] = []

    for row in range(grid.rows):
        vals: List[int] = []

        for c in range(grid.cols):  # 00-12
            cx, cy = _cell_center(0, c, row, grid)
            vals.append(int(V[cy, cx]))

        for c in range(grid.cols):  # 12-24
            cx, cy = _cell_center(1, c, row, grid)
            vals.append(int(V[cy, cx]))

        states_all.append(_states_by_threshold(vals, V_THRESHOLD))

    return states_all


def intervals_from_states(states_48: List[bool]) -> List[Tuple[str, str]]:
    intervals = []
    start = None
    for i, off in enumerate(states_48):
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


def states_to_text(states_48: List[bool]) -> str:
    # "1" = off, "0" = on
    return "".join("1" if x else "0" for x in states_48)


def make_fingerprint(q: int, sq: int, row_states: List[bool]) -> str:
    payload = (q, sq, tuple(row_states))
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def hash_intervals_by_row(intervals_by_row: List[List[Tuple[str, str]]]) -> str:
    payload = "|".join(
        ",".join(f"{a}-{b}" for a, b in row)
        for row in intervals_by_row
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_states_cached(api_url: str) -> Tuple[Optional[str], Optional[List[List[bool]]]]:
    """
    Парсить (або віддає з кешу) матрицю states 12x48.
    """
    now = time.time()
    pc = _parsed_cache.setdefault(api_url, {"ts": 0.0, "url": None, "states": None, "intervals_by_row": None})

    if pc["states"] is not None and pc["url"] is not None and (now - pc["ts"]) < PARSE_TTL_SEC:
        return pc["url"], pc["states"]

    try:
        url, img = get_image_cached(api_url)
        if not url or img is None:
            return None, None

        if pc["states"] is not None and pc["url"] == url and (now - pc["ts"]) < PARSE_TTL_SEC:
            return url, pc["states"]

        states = read_all_rows_states(img, GRID)
        intervals_by_row = [intervals_from_states(r) for r in states]

        pc["ts"] = now
        pc["url"] = url
        pc["states"] = states
        pc["intervals_by_row"] = intervals_by_row
        return url, states

    except requests.exceptions.RequestException as e:
        if pc["states"] is not None and pc["url"] is not None:
            log.warning("Network error, using cached states for %s: %s", api_url, e)
            return pc["url"], pc["states"]

        log.warning("Network error, no cache available for %s: %s", api_url, e)
        return None, None


def get_intervals_by_row_cached(api_url: str) -> Tuple[Optional[str], Optional[List[List[Tuple[str, str]]]]]:
    """
    Кешований доступ до інтервалів для всіх 12 рядків.
    """
    now = time.time()
    pc = _parsed_cache.setdefault(api_url, {"ts": 0.0, "url": None, "states": None, "intervals_by_row": None})

    if pc.get("intervals_by_row") is not None and pc.get("url") is not None and (now - pc["ts"]) < PARSE_TTL_SEC:
        return pc["url"], pc["intervals_by_row"]

    url, states = get_states_cached(api_url)
    if not url or not states:
        return None, None

    if pc.get("intervals_by_row") is None:
        pc["intervals_by_row"] = [intervals_from_states(r) for r in states]

    return pc["url"], pc["intervals_by_row"]


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


# ================= UI (BUTTONS) =================

def main_menu_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("⚙️ Встановити/змінити чергу", callback_data="menu:set")],
        [
            InlineKeyboardButton("🔍 Чи є світло зараз?", callback_data="menu:now"),
            InlineKeyboardButton("📅 Графік на завтра", callback_data="menu:tomorrow"),
        ],
    ])


def queue_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("1", callback_data="set:q:1"),
            InlineKeyboardButton("2", callback_data="set:q:2"),
            InlineKeyboardButton("3", callback_data="set:q:3"),
        ],
        [
            InlineKeyboardButton("4", callback_data="set:q:4"),
            InlineKeyboardButton("5", callback_data="set:q:5"),
            InlineKeyboardButton("6", callback_data="set:q:6"),
        ],
        [InlineKeyboardButton("⬅️ Назад", callback_data="menu:back")],
    ])


def subqueue_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("1", callback_data="set:sq:1"),
            InlineKeyboardButton("2", callback_data="set:sq:2"),
        ],
        [InlineKeyboardButton("⬅️ Назад", callback_data="menu:set")],
    ])


# ================= TELEGRAM HANDLERS =================

_last_now: dict[int, float] = {}
_last_tomorrow: dict[int, float] = {}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    touch_from_update(update)
    await update.message.reply_text("Меню", reply_markup=main_menu_kb())


async def my_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    touch_from_update(update)
    await update.message.reply_text(
        f"user_id: {update.effective_user.id}\n"
        f"chat_id: {update.effective_chat.id}"
    )


async def admin_say(update: Update, context: ContextTypes.DEFAULT_TYPE):
    touch_from_update(update)

    u = update.effective_user
    if not u or u.id not in ADMIN_IDS:
        return

    if len(context.args) < 2:
        await update.message.reply_text("Використання: /say <chat_id> <текст>")
        return

    try:
        chat_id = int(context.args[0])
    except ValueError:
        await update.message.reply_text("chat_id має бути числом. Приклад: /say 123456789 Привіт")
        return

    text = " ".join(context.args[1:]).strip()
    if not text:
        await update.message.reply_text("Текст не може бути порожнім.")
        return

    try:
        await context.bot.send_message(chat_id=chat_id, text=text, reply_markup=main_menu_kb())
        await update.message.reply_text(f"✅ Надіслано в chat_id={chat_id}")
    except Forbidden:
        await update.message.reply_text("❌ Не можу надіслати: користувач заблокував бота або чат недоступний.")
    except Exception:
        log.exception("admin_say failed")
        await update.message.reply_text("⚠️ Помилка надсилання.")


async def admin_last(update: Update, context: ContextTypes.DEFAULT_TYPE):
    touch_from_update(update)

    u = update.effective_user
    if not u or u.id not in ADMIN_IDS:
        return

    last = db_get_last_user()
    if not last:
        await update.message.reply_text("У базі ще немає користувачів.")
        return

    chat_id, username, created_at, last_seen_at = last

    def fmt_ts(ts: Optional[int]) -> str:
        if not ts:
            return "—"
        return datetime.fromtimestamp(ts, TZ).strftime("%Y-%m-%d %H:%M:%S")

    uname = f"@{username}" if username else "—"
    await update.message.reply_text(
        "Останній новий користувач:\n"
        f"chat_id: {chat_id}\n"
        f"username: {uname}\n"
        f"created_at: {fmt_ts(created_at)}\n"
        f"last_seen_at: {fmt_ts(last_seen_at)}"
    )

def build_users_table(rows: list[tuple[int, str, Optional[int]]]) -> str:
    """
    rows: (chat_id, username, last_seen_at)
    Повертає моноширинну "табличку" для <pre>...</pre>
    """
    def fmt_ts(ts: Optional[int]) -> str:
        if not ts:
            return "—"
        return datetime.fromtimestamp(ts, TZ).strftime("%d.%m %H:%M")

    header = [
        f"👥 Users ({len(rows)})",
        "",
        "username        id          last_seen",
        "-------------------------------------",
    ]

    lines = []
    for chat_id, username, last_seen_at in rows:
        uname = username if username and username != "-" else "-"
        # робимо @ якщо є юзернейм
        if uname != "-" and not uname.startswith("@"):
            uname = "@" + uname

        lines.append(f"{uname:<14} {str(chat_id):<11} {fmt_ts(last_seen_at)}")

    return "\n".join(header + lines)


async def admin_users(update: Update, context: ContextTypes.DEFAULT_TYPE):
    touch_from_update(update)

    u = update.effective_user
    if not u or u.id not in ADMIN_IDS:
        return

    with db_connect() as con:
        _ensure_users_columns(con)
        rows = con.execute("""
            SELECT
                chat_id,
                COALESCE(username, '-') AS username,
                last_seen_at
            FROM users
            ORDER BY COALESCE(last_seen_at, 0) DESC
            LIMIT 200
        """).fetchall()

    if not rows:
        await update.message.reply_text("У базі ще немає користувачів.")
        return

    text = build_users_table(rows)

    # Telegram має ліміт на повідомлення — підріжемо якщо треба
    if len(text) > 3900:
        text = text[:3900] + "\n…(обрізано)"

    await update.message.reply_text(f"<pre>{text}</pre>", parse_mode="HTML")


async def on_menu_set(update: Update, context: ContextTypes.DEFAULT_TYPE):
    touch_from_update(update)
    q = update.callback_query
    await q.answer()
    await q.edit_message_text("Обери чергу (1–6):", reply_markup=queue_kb())


async def on_set_queue_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    touch_from_update(update)
    q = update.callback_query
    await q.answer()
    queue = int(q.data.split(":")[-1])
    context.user_data["tmp_queue"] = queue
    await q.edit_message_text(f"Черга {queue}. Тепер обери підчергу (1–2):", reply_markup=subqueue_kb())


async def on_set_subqueue_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    touch_from_update(update)
    q = update.callback_query
    await q.answer()

    queue = context.user_data.get("tmp_queue")
    if not queue:
        await q.edit_message_text("Обери чергу (1–6):", reply_markup=queue_kb())
        return

    subqueue = int(q.data.split(":")[-1])
    chat_id = q.message.chat_id

    db_upsert_user(chat_id, queue, subqueue)

    # ✅ Одразу показати графік (сьогодні) для цієї черги
    try:
        url, states = await asyncio.to_thread(get_states_cached, API_TODAY)

        if url and states:
            row = _row_index(queue, subqueue)
            row_states = states[row]
            full_intervals = intervals_from_states(row_states)

            now_dt = datetime.now(TZ)
            view_intervals = filter_past_intervals(full_intervals, now_dt)
            now_has_light = is_light_now(full_intervals, now_dt)

            text = (
                f"✅ Збережено: черга {queue}/{subqueue}\n\n"
                + format_outages_by_dayparts_today(view_intervals, now_has_light, now_dt, full_intervals)
            )

            # оновлюємо TODAY-памʼять, щоб не було “першого пуша”
            today = day_str(now_dt)
            fp = make_fingerprint(queue, subqueue, row_states)
            db_set_today_memory(
                chat_id,
                today,
                fp,
                total_off_minutes(full_intervals),
                intervals_to_text(full_intervals),
                states_to_text(row_states),
            )

            await q.edit_message_text(text, reply_markup=main_menu_kb())
            return

    except Exception:
        log.exception("Failed to show schedule right after set subqueue")

    # fallback (якщо API недоступний)
    await q.edit_message_text(
        f"✅ Збережено: черга {queue}/{subqueue}\n\nМеню 👇",
        reply_markup=main_menu_kb()
    )


async def on_menu_now(update: Update, context: ContextTypes.DEFAULT_TYPE):
    touch_from_update(update)
    q = update.callback_query
    if not q or not q.message:
        return

    chat_id = q.message.chat_id

    # cooldown (на чат) — ПОПАП і вихід
    t = time.time()
    last = _last_now.get(chat_id, 0.0)
    if t - last < NOW_COOLDOWN_SEC:
        left = _cooldown_left(last, NOW_COOLDOWN_SEC)
        try:
            await q.answer(f"⏳ Зачекай {left}с", show_alert=False)
        except Exception:
            pass
        return
    _last_now[chat_id] = t

    qs = db_get_user_queue(chat_id)
    if not qs:
        try:
            await q.answer("Спочатку обери чергу через ⚙️ Set", show_alert=False)
        except Exception:
            pass
        return
    queue, subqueue = qs

    try:
        # не блокуємо event loop
        url, states = await asyncio.to_thread(get_states_cached, API_TODAY)

        if not url or not states:
            try:
                await q.answer("⚠️ Сервер графіків недоступний", show_alert=False)
            except Exception:
                pass
            return

        row = _row_index(queue, subqueue)
        row_states = states[row]
        full_intervals = intervals_from_states(row_states)

        now_dt = datetime.now(TZ)
        view_intervals = filter_past_intervals(full_intervals, now_dt)
        now_has_light = is_light_now(full_intervals, now_dt)

        text = format_outages_by_dayparts_today(view_intervals, now_has_light, now_dt, full_intervals)

        # якщо нічого не змінилось — показуємо "Актуально" і виходимо
        msg = q.message
        if msg and msg.text == text:
            try:
                await q.answer("✅ Актуально", show_alert=False)
            except Exception:
                pass
            return

        # ✅ тут робимо один "порожній" ack (або можна взагалі не робити)
        try:
            await q.answer()
        except Exception:
            pass

        # оновлюємо TODAY-памʼять
        today = day_str(now_dt)
        fp = make_fingerprint(queue, subqueue, row_states)
        db_set_today_memory(
            chat_id,
            today,
            fp,
            total_off_minutes(full_intervals),
            intervals_to_text(full_intervals),
            states_to_text(row_states),
        )

        await q.edit_message_text(text, reply_markup=main_menu_kb())

    except BadRequest as e:
        log.warning("Telegram BadRequest in on_menu_now: %s", e)
    except Exception:
        log.exception("now failed")
        try:
            await q.answer("⚠️ Сталася помилка. Спробуй ще раз.", show_alert=False)
        except Exception:
            pass


async def on_menu_tomorrow(update: Update, context: ContextTypes.DEFAULT_TYPE):
    touch_from_update(update)
    q = update.callback_query
    if not q or not q.message:
        return

    chat_id = q.message.chat_id

    # cooldown — ПОПАП і вихід
    t = time.time()
    last = _last_tomorrow.get(chat_id, 0.0)
    if t - last < NOW_COOLDOWN_SEC:
        left = _cooldown_left(last, NOW_COOLDOWN_SEC)
        try:
            await q.answer(f"⏳ Зачекай {left}с", show_alert=False)
        except Exception:
            pass
        return
    _last_tomorrow[chat_id] = t

    qs = db_get_user_queue(chat_id)
    if not qs:
        try:
            await q.answer("Спочатку обери чергу через ⚙️ Set", show_alert=False)
        except Exception:
            pass
        return
    queue, subqueue = qs

    try:
        url, intervals_by_row = await asyncio.to_thread(get_intervals_by_row_cached, API_TOMORROW)

        # ✅ якщо графіків нема — завжди ПОПАП і вихід
        if not url or not intervals_by_row:
            label = tomorrow_label()
            try:
                await q.answer(f"🕓 Графіків на завтра ({label}) ще немає", show_alert=False)
            except Exception:
                pass
            return

        row = _row_index(queue, subqueue)
        full_intervals = intervals_by_row[row]

        label = tomorrow_label()
        text = f"📅 Завтра ({label})\n\n" + format_outages_by_dayparts_plain(full_intervals)

        msg = q.message
        if msg and msg.text == text:
            try:
                await q.answer("✅ Актуально", show_alert=False)
            except Exception:
                pass
            return

        # ✅ один ack перед edit
        try:
            await q.answer()
        except Exception:
            pass

        await q.edit_message_text(text, reply_markup=main_menu_kb())

    except BadRequest as e:
        log.warning("Telegram BadRequest in on_menu_tomorrow: %s", e)
    except Exception:
        log.exception("tomorrow failed")
        try:
            await q.answer("⚠️ Сталася помилка. Спробуй ще раз.", show_alert=False)
        except Exception:
            pass


async def on_menu_back(update: Update, context: ContextTypes.DEFAULT_TYPE):
    touch_from_update(update)
    q = update.callback_query
    await q.answer()
    try:
        await q.edit_message_text("Меню 👇", reply_markup=main_menu_kb())
    except Exception:
        await q.edit_message_reply_markup(reply_markup=main_menu_kb())


# ================= SCHEDULER JOBS =================

async def broadcast_tomorrow_poll(app: Application):
    if is_quiet_hours():
        return

    now_dt = datetime.now(TZ)
    tday = tomorrow_day_str(now_dt)
    label = tomorrow_label(now_dt)

    # 1 запит+парс на весь бот
    url, states = get_states_cached(API_TOMORROW)
    if not url or not states:
        return

    intervals_by_row = _parsed_cache.get(API_TOMORROW, {}).get("intervals_by_row")
    if not intervals_by_row:
        intervals_by_row = [intervals_from_states(r) for r in states]

    users = db_get_users_basic()
    if not users:
        return

    sent_first = 0
    sent_update = 0
    removed = 0

    for chat_id, queue, subqueue, old_fp, old_states_txt, old_day in users:
        row = _row_index(queue, subqueue)

        row_states = states[row]
        full_intervals = intervals_by_row[row]
        new_fp = make_fingerprint(queue, subqueue, row_states)

        # 1) якщо для користувача це новий tday — перша публікація
        if old_day != tday:
            msg = f"📅 Завтра ({label})\n\n" + format_outages_by_dayparts_plain(full_intervals)
            try:
                await app.bot.send_message(chat_id, msg, reply_markup=main_menu_kb())
                sent_first += 1
            except Forbidden:
                db_delete_user(chat_id)
                removed += 1
                continue
            except Exception:
                log.exception("Failed sending tomorrow first to chat_id=%s", chat_id)
                continue

            db_set_tomorrow_memory(
                chat_id, tday, new_fp,
                total_off_minutes(full_intervals),
                intervals_to_text(full_intervals),
                states_to_text(row_states),
            )
            continue

        # 2) якщо нема з чим порівнювати — просто ініціалізуємо без пуша
        if not old_fp:
            db_set_tomorrow_memory(
                chat_id, tday, new_fp,
                total_off_minutes(full_intervals),
                intervals_to_text(full_intervals),
                states_to_text(row_states),
            )
            continue

        # 3) якщо fp не змінився — мовчимо
        if old_fp == new_fp:
            continue

        # 4) fp змінився — пушимо оновлення
        msg = "🔄 Графіки на завтра оновились\n\n" + f"📅 Завтра ({label})\n\n" + format_outages_by_dayparts_plain(full_intervals)
        try:
            await app.bot.send_message(chat_id, msg, reply_markup=main_menu_kb())
            sent_update += 1
        except Forbidden:
            db_delete_user(chat_id)
            removed += 1
            continue
        except Exception:
            log.exception("Failed sending tomorrow update to chat_id=%s", chat_id)
            continue

        db_set_tomorrow_memory(
            chat_id, tday, new_fp,
            total_off_minutes(full_intervals),
            intervals_to_text(full_intervals),
            states_to_text(row_states),
        )

    log.info("Tomorrow poll: first=%d update=%d removed=%d total=%d day=%s",
             sent_first, sent_update, removed, len(users), tday)

async def broadcast_today_changes(app: Application):
    """
    Вимога:
    - впродовж дня сигналізуємо зміни в черзі користувача
    - порівнюємо завжди з останнім СЬОГОДНІ
    - НЕ порівнюємо з вчорашнім today (якщо day змінився — просто ініціалізуємо today-базу і НЕ пушимо)
    """
    if is_quiet_hours():
        return

    users = db_get_users_for_push()
    if not users:
        return

    now_dt = datetime.now(TZ)
    today = day_str(now_dt)

    try:
        url, states = get_states_cached(API_TODAY)
        if not url or not states:
            return

        intervals_by_row = _parsed_cache.get(API_TODAY, {}).get("intervals_by_row")
        if not intervals_by_row:
            intervals_by_row = [intervals_from_states(r) for r in states]

        for chat_id, queue, subqueue, last_fp, last_total_off, last_intervals_txt, last_states_txt, last_day in users:
            row = _row_index(queue, subqueue)
            row_states = states[row]
            fp = make_fingerprint(queue, subqueue, row_states)
            full_intervals = intervals_by_row[row]

            # ✅ якщо настав новий день — просто приймаємо поточний графік як базу і НЕ пушимо
            if last_day != today:
                db_set_today_memory(
                    chat_id,
                    today,
                    fp,
                    total_off_minutes(full_intervals),
                    intervals_to_text(full_intervals),
                    states_to_text(row_states),
                )
                continue

            # якщо ще нема з чим порівнювати в межах дня — ініціалізуємо і НЕ пушимо
            if (last_fp is None) or (last_total_off is None) or (last_intervals_txt is None) or (last_states_txt is None):
                db_set_today_memory(
                    chat_id,
                    today,
                    fp,
                    total_off_minutes(full_intervals),
                    intervals_to_text(full_intervals),
                    states_to_text(row_states),
                )
                continue

            if fp == last_fp:
                continue  # жодних змін

            # якщо зміни тільки в минулих слотах — НЕ пушимо, але оновлюємо базу, щоб не зациклювало
            if not future_changed_only(row_states, last_states_txt, now_dt):
                db_set_today_memory(
                    chat_id,
                    today,
                    fp,
                    total_off_minutes(full_intervals),
                    intervals_to_text(full_intervals),
                    states_to_text(row_states),
                )
                continue

            cancelled, added = diff_future_intervals(last_states_txt, row_states, now_dt)

            lines = ["🔄 Графік оновлено для твоєї черги", ""]

            if cancelled:
                lines.append("❎ Скасували відключення:")
                lines.append(fmt_interval_lines(cancelled, "❌"))
                lines.append("")

            if added:
                lines.append("➕ Додали відключення:")
                lines.append(fmt_interval_lines(added, "➕"))
                lines.append("")

            # на випадок, якщо fp змінився, але в майбутньому перетворення “в нуль” (дуже рідко)
            if (not cancelled) and (not added):
                lines.append("➖ Змін у майбутніх годинах не бачу")
                lines.append("")

            prefix = "\n".join(lines).strip()


            view_intervals = filter_past_intervals(full_intervals, now_dt)
            now_has_light = is_light_now(full_intervals, now_dt)

            body = format_outages_by_dayparts_today(view_intervals, now_has_light, now_dt, full_intervals)
            msg = prefix + "\n\n" + body

            try:
                await app.bot.send_message(chat_id, msg, reply_markup=main_menu_kb())

                db_set_today_memory(
                    chat_id,
                    today,
                    fp,
                    total_off_minutes(full_intervals),
                    intervals_to_text(full_intervals),
                    states_to_text(row_states),
                )

            except Forbidden:
                db_delete_user(chat_id)
            except Exception:
                log.exception("Failed sending today change to chat_id=%s", chat_id)

    except Exception:
        log.exception("broadcast(today) failed")


async def post_init(app: Application):
    scheduler = AsyncIOScheduler(timezone=TZ)

    # протягом дня — перевірка today змін (як і було)
    scheduler.add_job(
        broadcast_today_changes,
        "cron",
        minute="*/5",
        args=[app],
        timezone=TZ,
    )

    scheduler.add_job(
        broadcast_tomorrow_poll,
        "cron",
        hour="18-23",
        minute="*/5",
        args=[app],
        timezone=TZ,
    )

    scheduler.start()
    app.bot_data["scheduler"] = scheduler
    log.info(
        "Scheduler started (today: every 5 min, tomorrow: 18-23 every 5 min)"
    )


# ================= ERROR HANDLER =================

async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    log.exception("Unhandled error", exc_info=context.error)


# ================= TRANSFER DB =================

def ensure_seed_db():
    if os.path.exists(DB_PATH):
        return

    seed = os.path.join(os.path.dirname(__file__), "data", "users.db")
    if os.path.exists(seed):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        shutil.copy2(seed, DB_PATH)


# ================= MAIN =================

def main():
    token = os.environ["BOT_TOKEN"]

    if DB_PATH and os.path.dirname(DB_PATH):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    ensure_seed_db()
    db_init()

    app = Application.builder().token(token).post_init(post_init).build()

    app.add_handler(CommandHandler("start", start))

    app.add_handler(CommandHandler("id", my_id))
    app.add_handler(CommandHandler("say", admin_say))
    app.add_handler(CommandHandler("last", admin_last))
    app.add_handler(CommandHandler("users", admin_users))

    app.add_handler(CallbackQueryHandler(on_menu_set, pattern=r"^menu:set$"))
    app.add_handler(CallbackQueryHandler(on_menu_now, pattern=r"^menu:now$"))
    app.add_handler(CallbackQueryHandler(on_menu_tomorrow, pattern=r"^menu:tomorrow$"))
    app.add_handler(CallbackQueryHandler(on_menu_back, pattern=r"^menu:back$"))
    app.add_handler(CallbackQueryHandler(on_set_queue_button, pattern=r"^set:q:\d+$"))
    app.add_handler(CallbackQueryHandler(on_set_subqueue_button, pattern=r"^set:sq:[12]$"))

    app.add_error_handler(on_error)

    log.info("Bot started")
    app.run_polling()


if __name__ == "__main__":
    main()
