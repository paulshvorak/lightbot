import asyncio
import time
from datetime import datetime

from telegram import Update
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from ..config import API_TODAY, API_TOMORROW, NOW_COOLDOWN_SEC, TZ, log
from ..db import db_get_user_queue, db_is_configured, db_set_today_memory, db_upsert_user, touch_from_update
from ..formatting import (
    _cooldown_left,
    day_str,
    filter_past_intervals,
    format_outages_by_dayparts_plain,
    format_outages_by_dayparts_today,
    intervals_to_text,
    is_light_now,
    tomorrow_label,
    total_off_minutes,
)
from ..grid import (
    _row_index,
    get_intervals_by_row_cached,
    get_states_cached,
    intervals_from_states,
    make_fingerprint,
    states_to_text,
)
from ..keyboards import first_menu_kb, main_menu_kb, queue_kb, subqueue_kb

# ================= TELEGRAM HANDLERS (user-facing) =================

_last_now: dict[int, float] = {}
_last_tomorrow: dict[int, float] = {}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    touch_from_update(update)

    chat_id = update.effective_chat.id
    if db_is_configured(chat_id):
        await update.message.reply_text("Меню", reply_markup=main_menu_kb())
    else:
        await update.message.reply_text(
            "Привіт! Спочатку обери свою чергу 👇",
            reply_markup=first_menu_kb()
        )


async def my_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    touch_from_update(update)
    await update.message.reply_text(
        f"user_id: {update.effective_user.id}\n"
        f"chat_id: {update.effective_chat.id}"
    )


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

    chat_id = q.message.chat_id

    if db_is_configured(chat_id):
        kb = main_menu_kb()
        text = "Меню 👇"
    else:
        kb = first_menu_kb()
        text = "Спочатку обери чергу 👇"

    try:
        await q.edit_message_text(text, reply_markup=kb)
    except Exception:
        await q.edit_message_reply_markup(reply_markup=kb)
