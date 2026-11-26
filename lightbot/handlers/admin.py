import asyncio
from datetime import datetime
from typing import Optional

from telegram import Update
from telegram.error import Forbidden
from telegram.ext import ContextTypes

from ..config import ADMIN_IDS, TZ, log
from ..db import (
    db_click_summary,
    db_connect,
    db_delete_user,
    db_get_last_user,
    db_referral_top,
    db_referral_totals,
    db_users_stats_by_day,
    touch_from_update,
)
from ..keyboards import main_menu_kb

# ================= TELEGRAM HANDLERS (admin) =================


async def admin_say(update: Update, context: ContextTypes.DEFAULT_TYPE):
    touch_from_update(update)

    u = update.effective_user
    if not u or u.id not in ADMIN_IDS:
        return

    if not update.message or not update.message.text:
        return

    parts = update.message.text.split(maxsplit=2)

    if len(parts) < 2:
        await update.message.reply_text(
            "Використання:\n"
            "/say <chat_id> <текст>\n"
            "або reply на повідомлення командою:\n"
            "/say <chat_id>"
        )
        return

    try:
        chat_id = int(parts[1])
    except ValueError:
        await update.message.reply_text("chat_id має бути числом. Приклад: /say 123456789")
        return

    text = None

    # 1) пріоритет: reply на повідомлення
    if update.message.reply_to_message:
        reply_msg = update.message.reply_to_message
        text = reply_msg.text or reply_msg.caption

    # 2) fallback: текст після chat_id
    if not text and len(parts) >= 3:
        text = parts[2]

    if not text:
        await update.message.reply_text(
            "Немає тексту для надсилання.\n"
            "Варіанти:\n"
            "1) /say <chat_id> <текст>\n"
            "2) reply на повідомлення командою /say <chat_id>"
        )
        return

    try:
        await context.bot.send_message(
            chat_id=chat_id,
            text=text,
            reply_markup=main_menu_kb()
        )
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

    def fmt_ts(ts: Optional[int]) -> str:
        if not ts:
            return "—"
        return datetime.fromtimestamp(ts, TZ).strftime("%Y-%m-%d %H:%M:%S")

    uname = f"@{last.username}" if last.username else "—"
    await update.message.reply_text(
        "Останній новий користувач:\n"
        f"chat_id: {last.chat_id}\n"
        f"username: {uname}\n"
        f"created_at: {fmt_ts(last.created_at)}\n"
        f"last_seen_at: {fmt_ts(last.last_seen_at)}"
    )


BUTTON_LABELS = {
    "menu:now": "🔍 Чи є світло зараз?",
    "menu:set": "⚙️ Встановити/змінити чергу",
}


async def admin_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    touch_from_update(update)

    u = update.effective_user
    if not u or u.id not in ADMIN_IDS:
        return

    total, by_days = db_users_stats_by_day(days=7)

    lines = [
        "📊 Статистика користувачів",
        "",
        f"👥 Всього користувачів: {total}",
        "",
        "📅 Нові по днях (останні 7 днів):",
    ]

    if not by_days:
        lines.append("— немає даних —")
    else:
        for day, cnt in by_days:
            lines.append(f"{day}: {cnt}")

    day_start_ts = int(datetime.now(TZ).replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
    clicks = db_click_summary(day_start_ts)

    lines.append("")
    lines.append("🖱 Кліки по кнопках:")
    if not clicks:
        lines.append("— немає даних —")
    else:
        for button_key, today_cnt, total_cnt in clicks:
            label = BUTTON_LABELS.get(button_key, button_key)
            lines.append(f"{label}: сьогодні {today_cnt}, всього {total_cnt}")

    referral_total, referral_activated = db_referral_totals()
    top_referrers = db_referral_top(limit=5)

    lines.append("")
    lines.append(f"📤 Прийшли через «Поділитись»: {referral_total} (обрали чергу: {referral_activated})")
    if top_referrers:
        lines.append("Топ рефералів:")
        for chat_id, username, total_cnt, activated_cnt in top_referrers:
            uname = f"@{username}" if username else str(chat_id)
            lines.append(f"{uname}: {total_cnt} (активних: {activated_cnt})")

    text = "\n".join(lines)

    await update.message.reply_text(text)


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


async def admin_broadcast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    touch_from_update(update)

    u = update.effective_user
    if not u or u.id not in ADMIN_IDS:
        return

    if not update.message or not update.message.text:
        return

    text = None

    # 1) пріоритет: reply на повідомлення
    if update.message.reply_to_message:
        reply_msg = update.message.reply_to_message
        text = reply_msg.text or reply_msg.caption

    # 2) fallback: текст після /broadcast
    if not text:
        parts = update.message.text.split(maxsplit=1)
        if len(parts) >= 2:
            text = parts[1]

    if not text:
        await update.message.reply_text(
            "Використання:\n"
            "/broadcast <текст>\n"
            "або reply на повідомлення командою:\n"
            "/broadcast"
        )
        return

    with db_connect() as con:
        rows = con.execute("""
            SELECT DISTINCT chat_id
            FROM subscriptions
        """).fetchall()

    if not rows:
        await update.message.reply_text("Немає користувачів для розсилки.")
        return

    total = len(rows)
    sent = 0
    removed = 0
    failed = 0

    await update.message.reply_text(f"📣 Починаю розсилку на {total} користувачів...")

    for (chat_id,) in rows:
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_markup=main_menu_kb()
            )
            sent += 1
        except Forbidden:
            db_delete_user(chat_id)
            removed += 1
        except Exception:
            log.exception("broadcast failed for chat_id=%s", chat_id)
            failed += 1

        await asyncio.sleep(0.05)

    await update.message.reply_text(
        "✅ Розсилка завершена\n"
        f"Надіслано: {sent}\n"
        f"Видалено з бази: {removed}\n"
        f"Помилок: {failed}"
    )
