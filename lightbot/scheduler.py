import html
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram.error import Forbidden
from telegram.ext import Application

from .config import API_TODAY, API_TOMORROW, TZ, is_quiet_hours, log
from .db import (
    db_delete_user,
    db_get_all_subscriptions,
    db_set_today_memory,
    db_set_tomorrow_memory,
)
from .formatting import (
    day_str,
    diff_future_intervals,
    filter_past_intervals,
    format_outages_by_dayparts_plain,
    format_outages_by_dayparts_today,
    future_changed_only,
    intervals_to_text,
    is_light_now,
    subscription_short_name,
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

    subs = db_get_all_subscriptions()
    if not subs:
        return

    # групуємо по chat_id, щоб надіслати один пуш на чат, а не по одному на чергу
    pending_first: dict[int, list[tuple[str, tuple]]] = {}
    pending_update: dict[int, list[tuple[str, tuple]]] = {}

    for sub in subs:
        chat_id, queue, subqueue = sub.chat_id, sub.queue, sub.subqueue
        row = _row_index(queue, subqueue)

        row_states = states[row]
        full_intervals = intervals_by_row[row]
        new_fp = make_fingerprint(queue, subqueue, row_states)

        save_args = (
            chat_id, queue, subqueue, tday, new_fp,
            total_off_minutes(full_intervals),
            intervals_to_text(full_intervals),
            states_to_text(row_states),
        )

        # 1) якщо для користувача це новий tday — перша публікація
        if sub.tomorrow_day != tday:
            # якщо відключень немає — мовчимо, але зберігаємо стан
            if not full_intervals:
                db_set_tomorrow_memory(
                    chat_id, queue, subqueue, tday, new_fp,
                    0,
                    "",
                    states_to_text(row_states),
                )
                continue

            name = html.escape(subscription_short_name(queue, subqueue, sub.label))
            block = f"📍 <b>{name}</b>\n" + format_outages_by_dayparts_plain(full_intervals)
            pending_first.setdefault(chat_id, []).append((block, save_args))
            continue

        # 2) якщо нема з чим порівнювати — просто ініціалізуємо без пуша
        if not sub.tomorrow_fingerprint:
            db_set_tomorrow_memory(*save_args)
            continue

        # 3) якщо fp не змінився — мовчимо
        if sub.tomorrow_fingerprint == new_fp:
            continue

        # 4) fp змінився — в чергу на пуш-оновлення
        name = html.escape(subscription_short_name(queue, subqueue, sub.label))
        block = f"📍 <b>{name}</b>\n" + format_outages_by_dayparts_plain(full_intervals)
        pending_update.setdefault(chat_id, []).append((block, save_args))

    sent_first = 0
    sent_update = 0
    removed = 0

    for chat_id, items in pending_first.items():
        msg = f"📅 Завтра ({label})\n\n" + "\n\n".join(block for block, _ in items)
        try:
            await app.bot.send_message(chat_id, msg, reply_markup=main_menu_kb(), parse_mode="HTML")
            sent_first += 1
            for _, save_args in items:
                db_set_tomorrow_memory(*save_args)
        except Forbidden:
            db_delete_user(chat_id)
            removed += 1
        except Exception:
            log.exception("Failed sending tomorrow first to chat_id=%s", chat_id)

    for chat_id, items in pending_update.items():
        msg = "🔄 Графіки на завтра оновились\n\n" + f"📅 Завтра ({label})\n\n" + "\n\n".join(block for block, _ in items)
        try:
            await app.bot.send_message(chat_id, msg, reply_markup=main_menu_kb(), parse_mode="HTML")
            sent_update += 1
            for _, save_args in items:
                db_set_tomorrow_memory(*save_args)
        except Forbidden:
            db_delete_user(chat_id)
            removed += 1
        except Exception:
            log.exception("Failed sending tomorrow update to chat_id=%s", chat_id)

    log.info("Tomorrow poll: first_chats=%d update_chats=%d removed=%d total_subs=%d day=%s",
              sent_first, sent_update, removed, len(subs), tday)


async def broadcast_today_changes(app: Application):
    """
    Вимога:
    - впродовж дня сигналізуємо зміни в черзі користувача
    - порівнюємо завжди з останнім СЬОГОДНІ
    - НЕ порівнюємо з вчорашнім today (якщо day змінився — просто ініціалізуємо today-базу і НЕ пушимо)
    """
    if is_quiet_hours():
        return

    subs = db_get_all_subscriptions()
    if not subs:
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

        # групуємо по chat_id, щоб надіслати один пуш на чат, а не по одному на чергу
        pending: dict[int, list[tuple[str, tuple]]] = {}

        for sub in subs:
            chat_id, queue, subqueue = sub.chat_id, sub.queue, sub.subqueue
            row = _row_index(queue, subqueue)
            row_states = states[row]
            fp = make_fingerprint(queue, subqueue, row_states)
            full_intervals = intervals_by_row[row]

            save_args = (
                chat_id, queue, subqueue, today, fp,
                total_off_minutes(full_intervals),
                intervals_to_text(full_intervals),
                states_to_text(row_states),
            )

            # ✅ якщо настав новий день — просто приймаємо поточний графік як базу і НЕ пушимо
            if sub.today_day != today:
                db_set_today_memory(*save_args)
                continue

            # якщо ще нема з чим порівнювати в межах дня — ініціалізуємо і НЕ пушимо
            if (sub.today_fingerprint is None) or (sub.today_total_off is None) or (sub.today_intervals is None) or (sub.today_states is None):
                db_set_today_memory(*save_args)
                continue

            if fp == sub.today_fingerprint:
                continue  # жодних змін

            # якщо зміни тільки в минулих слотах — НЕ пушимо, але оновлюємо базу, щоб не зациклювало
            if not future_changed_only(row_states, sub.today_states, now_dt):
                db_set_today_memory(*save_args)
                continue

            cancelled, added = diff_future_intervals(sub.today_states, row_states, now_dt)

            name = html.escape(subscription_short_name(queue, subqueue, sub.label))
            lines = [f"📍 <b>{name}</b>"]

            if cancelled:
                lines.append("❎ Скасовано: " + ", ".join(f"{a}–{b}" for a, b in cancelled))

            if added:
                lines.append("➕ Додано: " + ", ".join(f"{a}–{b}" for a, b in added))

            # на випадок, якщо fp змінився, але в майбутньому перетворення “в нуль” (дуже рідко)
            if (not cancelled) and (not added):
                lines.append("➖ Змін у майбутніх годинах не бачу")

            view_intervals = filter_past_intervals(full_intervals, now_dt)
            now_has_light = is_light_now(full_intervals, now_dt)
            lines.append(format_outages_by_dayparts_today(view_intervals, now_has_light, now_dt, full_intervals))

            pending.setdefault(chat_id, []).append(("\n".join(lines), save_args))

        sent = 0
        removed = 0

        for chat_id, items in pending.items():
            msg = "🔄 Графік оновлено\n\n" + "\n\n".join(block for block, _ in items)
            try:
                await app.bot.send_message(chat_id, msg, reply_markup=main_menu_kb(), parse_mode="HTML")
                sent += 1
                for _, save_args in items:
                    db_set_today_memory(*save_args)
            except Forbidden:
                db_delete_user(chat_id)
                removed += 1
            except Exception:
                log.exception("Failed sending today change to chat_id=%s", chat_id)

        log.info("Today push: chats=%d removed=%d subs=%d", sent, removed, len(subs))

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

    subs = db_get_all_subscriptions()
    if not subs:
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
        for sub in subs:
            chat_id, queue, subqueue = sub.chat_id, sub.queue, sub.subqueue
            row = _row_index(queue, subqueue)
            row_states = states[row]
            fp = make_fingerprint(queue, subqueue, row_states)
            full_intervals = intervals_by_row[row]

            db_set_today_memory(
                chat_id, queue, subqueue,
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
