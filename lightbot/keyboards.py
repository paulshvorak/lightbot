from telegram import InlineKeyboardButton, InlineKeyboardMarkup

# ================= UI (BUTTONS) =================


def first_menu_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("⚙️ Встановити чергу", callback_data="menu:set")],
    ])


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
