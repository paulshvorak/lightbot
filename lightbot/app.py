import os

from telegram.ext import Application, CallbackQueryHandler, CommandHandler, MessageHandler, filters

from .config import DB_PATH, log
from .db import db_init
from .errors import on_error
from .handlers import admin, user
from .scheduler import post_init

# ================= MAIN =================


def main():
    token = os.environ["BOT_TOKEN"]

    if DB_PATH and os.path.dirname(DB_PATH):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    db_init()

    app = Application.builder().token(token).post_init(post_init).build()

    app.add_handler(CommandHandler("start", user.start))

    app.add_handler(CommandHandler("id", user.my_id))
    app.add_handler(CommandHandler("say", admin.admin_say))
    app.add_handler(CommandHandler("broadcast", admin.admin_broadcast))
    app.add_handler(CommandHandler("last", admin.admin_last))
    app.add_handler(CommandHandler("users", admin.admin_users))
    app.add_handler(CommandHandler("stats", admin.admin_stats))

    app.add_handler(CallbackQueryHandler(user.on_menu_set, pattern=r"^menu:set$"))
    app.add_handler(CallbackQueryHandler(user.on_menu_now, pattern=r"^menu:now$"))
    app.add_handler(CallbackQueryHandler(user.on_menu_share, pattern=r"^menu:share$"))
    app.add_handler(CallbackQueryHandler(user.on_menu_back, pattern=r"^menu:back$"))
    app.add_handler(CallbackQueryHandler(user.on_add_queue, pattern=r"^set:qs:[1-6]:[12]$"))
    app.add_handler(CallbackQueryHandler(user.on_sub_add, pattern=r"^sub:add$"))
    app.add_handler(CallbackQueryHandler(user.on_sub_delete, pattern=r"^sub:del:[1-6]:[12]$"))
    app.add_handler(CallbackQueryHandler(user.on_sub_rename, pattern=r"^sub:rename:[1-6]:[12]$"))
    app.add_handler(CallbackQueryHandler(user.on_restart, pattern=r"^menu:restart$"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, user.on_label_text))

    # ТИМЧАСОВО: ловить будь-яку необроблену кнопку (застарілі клавіатури
    # з-перед рефакторингу на мульти-черги) — має бути останнім CallbackQueryHandler-ом.
    app.add_handler(CallbackQueryHandler(user.on_unknown_callback))

    app.add_error_handler(on_error)

    log.info("Bot started")
    app.run_polling()
