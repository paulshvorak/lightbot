from telegram.ext import ContextTypes

from .config import log

# ================= ERROR HANDLER =================


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    log.exception("Unhandled error", exc_info=context.error)
