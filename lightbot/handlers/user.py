import asyncio
import html
import time
from datetime import datetime
from typing import Optional

from telegram import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import BadRequest, Forbidden
from telegram.ext import ContextTypes

from ..config import API_TODAY, MAX_SUBSCRIPTIONS, NOW_COOLDOWN_SEC, TZ, log
from ..db import (
    db_add_subscription,
    db_get_subscriptions,
    db_has_subscriptions,
    db_remove_subscription,
    db_set_subscription_label,
    db_set_today_memory,
    touch_from_update,
)
from ..formatting import (
    _cooldown_left,
    day_str,
    filter_past_intervals,
    format_outages_by_dayparts_today,
    intervals_to_text,
    is_light_now,
    subscription_display_name,
    subscription_short_name,
    total_off_minutes,
)
from ..grid import (
    _row_index,
    get_states_cached,
    intervals_from_states,
    make_fingerprint,
    states_to_text,
)
from ..keyboards import after_add_kb, all_queues_kb, first_menu_kb, main_menu_kb, manage_kb

# ================= TELEGRAM HANDLERS (user-facing) =================

_last_now: dict[int, float] = {}
_LAST_NOW_TTL_SEC = 3600  # набагато більше за кулдаун — просто щоб не накопичувалось вічно
MAX_LABEL_LEN = 24


def _prune_last_now(now: float) -> None:
    stale = [chat_id for chat_id, ts in _last_now.items() if now - ts > _LAST_NOW_TTL_SEC]
    for chat_id in stale:
        del _last_now[chat_id]


async def _safe_answer(q: CallbackQuery, text: Optional[str] = None, show_alert: bool = False) -> None:
    """
    Відповідає на callback-запит, ігноруючи нешкідливі помилки Telegram
    (стара кнопка / таймаут) — щоб не засмічувати логи "Unhandled error".
    """
    try:
        await q.answer(text, show_alert=show_alert)
    except BadRequest:
        pass


async def _safe_edit_text(q: CallbackQuery, text: str, reply_markup=None, parse_mode: Optional[str] = None) -> None:
    """
    Редагує повідомлення, ігноруючи BadRequest "Message is not modified"
    (трапляється, коли юзер тисне кнопку, що веде на той самий екран).
    """
    try:
        await q.edit_message_text(text, reply_markup=reply_markup, parse_mode=parse_mode)
    except BadRequest as e:
        if "not modified" not in str(e).lower():
            log.warning("edit_message_text failed: %s", e)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    touch_from_update(update)

    chat_id = update.effective_chat.id
    try:
        if db_has_subscriptions(chat_id):
            await update.message.reply_text("Меню", reply_markup=main_menu_kb())
        else:
            await update.message.reply_text(
                "Привіт! Спочатку обери свою чергу 👇",
                reply_markup=first_menu_kb()
            )
    except Forbidden:
        pass


async def my_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    touch_from_update(update)
    await update.message.reply_text(
        f"user_id: {update.effective_user.id}\n"
        f"chat_id: {update.effective_chat.id}"
    )


async def on_menu_set(update: Update, context: ContextTypes.DEFAULT_TYPE):
    touch_from_update(update)
    context.user_data.pop("awaiting_label", None)
    q = update.callback_query
    await _safe_answer(q)

    chat_id = q.message.chat_id
    subs = db_get_subscriptions(chat_id)
    text = f"Твої черги ({len(subs)}/{MAX_SUBSCRIPTIONS}):" if subs else "Ще немає жодної черги 👇"
    await _safe_edit_text(q, text, reply_markup=manage_kb(subs))


async def on_sub_add(update: Update, context: ContextTypes.DEFAULT_TYPE):
    touch_from_update(update)
    context.user_data.pop("awaiting_label", None)
    q = update.callback_query
    await _safe_answer(q)

    chat_id = q.message.chat_id
    existing = [(s.queue, s.subqueue) for s in db_get_subscriptions(chat_id)]
    limit_reached = len(existing) >= MAX_SUBSCRIPTIONS
    await _safe_edit_text(q, "Обери чергу:", reply_markup=all_queues_kb(existing, limit_reached))


async def on_add_queue(update: Update, context: ContextTypes.DEFAULT_TYPE):
    touch_from_update(update)
    context.user_data.pop("awaiting_label", None)
    q = update.callback_query

    _, _, queue_s, subqueue_s = q.data.split(":")
    queue, subqueue = int(queue_s), int(subqueue_s)
    chat_id = q.message.chat_id

    existing = [(s.queue, s.subqueue) for s in db_get_subscriptions(chat_id)]
    if (queue, subqueue) not in existing and len(existing) >= MAX_SUBSCRIPTIONS:
        await _safe_answer(q, f"Максимум {MAX_SUBSCRIPTIONS} черги. Спочатку видали одну через ⚙️", show_alert=True)
        return

    await _safe_answer(q)
    db_add_subscription(chat_id, queue, subqueue)

    # ✅ Одразу показати графік (сьогодні) для щойно доданої пари
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
                f"✅ Додано: черга {queue}.{subqueue}\n\n"
                + format_outages_by_dayparts_today(view_intervals, now_has_light, now_dt, full_intervals)
            )

            today = day_str(now_dt)
            fp = make_fingerprint(queue, subqueue, row_states)
            db_set_today_memory(
                chat_id, queue, subqueue,
                today,
                fp,
                total_off_minutes(full_intervals),
                intervals_to_text(full_intervals),
                states_to_text(row_states),
            )

            await _safe_edit_text(q, text, reply_markup=after_add_kb())
            return

    except Exception:
        log.exception("Failed to show schedule right after adding subscription")

    # fallback (якщо API недоступний)
    await _safe_edit_text(
        q,
        f"✅ Додано: черга {queue}.{subqueue}",
        reply_markup=after_add_kb()
    )


async def on_sub_delete(update: Update, context: ContextTypes.DEFAULT_TYPE):
    touch_from_update(update)
    context.user_data.pop("awaiting_label", None)
    q = update.callback_query
    await _safe_answer(q)

    _, _, queue_s, subqueue_s = q.data.split(":")
    queue, subqueue = int(queue_s), int(subqueue_s)
    chat_id = q.message.chat_id

    db_remove_subscription(chat_id, queue, subqueue)

    subs = db_get_subscriptions(chat_id)
    text = "Твої черги:" if subs else "Ще немає жодної черги 👇"
    await _safe_edit_text(q, text, reply_markup=manage_kb(subs))


async def on_sub_rename(update: Update, context: ContextTypes.DEFAULT_TYPE):
    touch_from_update(update)
    q = update.callback_query
    await _safe_answer(q)

    _, _, queue_s, subqueue_s = q.data.split(":")
    queue, subqueue = int(queue_s), int(subqueue_s)

    context.user_data["awaiting_label"] = (queue, subqueue)
    await _safe_edit_text(
        q,
        f"Напиши назву для черги {queue}.{subqueue} (до {MAX_LABEL_LEN} символів).\n"
        f"Щоб прибрати назву — напиши \"-\".",
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Скасувати", callback_data="menu:set")]]),
    )


async def on_label_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pending = context.user_data.pop("awaiting_label", None)
    if not pending or not update.message or not update.message.text:
        return

    touch_from_update(update)
    queue, subqueue = pending
    chat_id = update.effective_chat.id
    text = update.message.text.strip()

    label = None if text in ("-", "видалити", "очистити") else text[:MAX_LABEL_LEN]
    db_set_subscription_label(chat_id, queue, subqueue, label)

    name = subscription_display_name(queue, subqueue, label)
    subs = db_get_subscriptions(chat_id)
    await update.message.reply_text(f"✅ Збережено: {name}", reply_markup=manage_kb(subs))


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
        await _safe_answer(q, f"⏳ Зачекай {left}с")
        return
    _last_now[chat_id] = t
    _prune_last_now(t)

    subs = db_get_subscriptions(chat_id)
    if not subs:
        await _safe_answer(q, "Спочатку обери чергу через ⚙️ Set")
        return

    try:
        # не блокуємо event loop
        url, states = await asyncio.to_thread(get_states_cached, API_TODAY)

        if not url or not states:
            await _safe_answer(q, "⚠️ Сервер графіків недоступний")
            return

        now_dt = datetime.now(TZ)
        today = day_str(now_dt)
        blocks = []
        plain_blocks = []

        for sub in subs:
            row = _row_index(sub.queue, sub.subqueue)
            row_states = states[row]
            full_intervals = intervals_from_states(row_states)

            view_intervals = filter_past_intervals(full_intervals, now_dt)
            now_has_light = is_light_now(full_intervals, now_dt)

            block = format_outages_by_dayparts_today(view_intervals, now_has_light, now_dt, full_intervals)
            raw_name = subscription_short_name(sub.queue, sub.subqueue, sub.label)
            blocks.append(f"📍 <b>{html.escape(raw_name)}</b>\n{block}")
            plain_blocks.append(f"📍 {raw_name}\n{block}")

            fp = make_fingerprint(sub.queue, sub.subqueue, row_states)
            db_set_today_memory(
                chat_id, sub.queue, sub.subqueue,
                today,
                fp,
                total_off_minutes(full_intervals),
                intervals_to_text(full_intervals),
                states_to_text(row_states),
            )

        text = "\n\n".join(blocks)
        plain_text = "\n\n".join(plain_blocks)

        # якщо нічого не змінилось — показуємо "Актуально" і виходимо
        # (q.message.text завжди plain, без HTML-тегів, тому порівнюємо з plain_text)
        msg = q.message
        if msg and msg.text == plain_text:
            await _safe_answer(q, "✅ Актуально")
            return

        await _safe_answer(q)
        await _safe_edit_text(q, text, reply_markup=main_menu_kb(), parse_mode="HTML")

    except BadRequest as e:
        log.warning("Telegram BadRequest in on_menu_now: %s", e)
    except Exception:
        log.exception("now failed")
        await _safe_answer(q, "⚠️ Сталася помилка. Спробуй ще раз.")


async def on_menu_back(update: Update, context: ContextTypes.DEFAULT_TYPE):
    touch_from_update(update)
    q = update.callback_query
    await _safe_answer(q)

    chat_id = q.message.chat_id

    if db_has_subscriptions(chat_id):
        kb = main_menu_kb()
        text = "Меню 👇"
    else:
        kb = first_menu_kb()
        text = "Спочатку обери чергу 👇"

    await _safe_edit_text(q, text, reply_markup=kb)
