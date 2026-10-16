from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram.error import Forbidden
from telegram.ext import Application

from .config import API_TODAY, API_TOMORROW, TZ, is_quiet_hours, log
from .db import (
    db_delete_user,
    db_get_users_basic,
    db_get_users_for_push,
    db_set_today_memory,
    db_set_tomorrow_memory,
)
from .formatting import (
    day_str,
    diff_future_intervals,
    filter_past_intervals,
    fmt_interval_lines,
    format_outages_by_dayparts_plain,
    format_outages_by_dayparts_today,
    future_changed_only,
    intervals_to_text,
    is_light_now,
    tomorrow_day_str,
    tomorrow_label,
    total_off_minutes,
)
from .grid import _parsed_cache, _row_index, get_states_cached, intervals_from_states, make_fingerprint, states_to_text
from .keyboards import main_menu_kb

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
            # якщо відключень немає — мовчимо, але зберігаємо стан
            if not full_intervals:
                db_set_tomorrow_memory(
                    chat_id, tday, new_fp,
                    0,
                    "",
                    states_to_text(row_states),
                )
                continue

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


async def seed_today_baseline(app: Application):
    """
    О 00:00 робимо "тихий baseline" на сьогодні:
    - НЕ шлемо пуші
    - просто записуємо today_* для всіх сконфігурованих юзерів
    """
    now_dt = datetime.now(TZ)
    today = day_str(now_dt)

    users = db_get_users_for_push()  # тут є chat_id, queue, subqueue ...
    if not users:
        return

    try:
        url, states = get_states_cached(API_TODAY)
        if not url or not states:
            log.warning("seed_today_baseline: no states/url")
            return

        intervals_by_row = _parsed_cache.get(API_TODAY, {}).get("intervals_by_row")
        if not intervals_by_row:
            intervals_by_row = [intervals_from_states(r) for r in states]

        updated = 0
        for chat_id, queue, subqueue, last_fp, last_total_off, last_intervals_txt, last_states_txt, last_day in users:
            row = _row_index(queue, subqueue)
            row_states = states[row]
            fp = make_fingerprint(queue, subqueue, row_states)
            full_intervals = intervals_by_row[row]

            db_set_today_memory(
                chat_id,
                today,
                fp,
                total_off_minutes(full_intervals),
                intervals_to_text(full_intervals),
                states_to_text(row_states),
            )
            updated += 1

        log.info("seed_today_baseline: updated=%d day=%s", updated, today)

    except Exception:
        log.exception("seed_today_baseline failed")


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

    # ✅ тихий baseline для today о 00:30 (щоб після 06:00 був з чим порівнювати)
    scheduler.add_job(
        seed_today_baseline,
        "cron",
        hour=0,
        minute=30,
        args=[app],
        timezone=TZ,
    )
