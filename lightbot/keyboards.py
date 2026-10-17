from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from .formatting import subscription_display_name

# ================= UI (BUTTONS) =================


def first_menu_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("⚙️ Встановити чергу", callback_data="menu:set")],
    ])


def main_menu_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🔍 Чи є світло зараз?", callback_data="menu:now"),
            InlineKeyboardButton("⚙️ Встановити/змінити чергу", callback_data="menu:set"),
        ],
    ])


def all_queues_kb(existing: list[tuple[int, int]], limit_reached: bool = False) -> InlineKeyboardMarkup:
    """
    Сітка всіх 12 варіантів черга/підчерга (1-6 × 1-2) на одному екрані.
    Уже додані пари позначені ✅. Якщо ліміт підписок вичерпано —
    решта пар позначені 🔒 (натискання поверне попап з поясненням).
    """
    existing_set = set(existing)
    rows = []
    for q in range(1, 7):
        row = []
        for sq in (1, 2):
            if (q, sq) in existing_set:
                mark = "✅ "
            elif limit_reached:
                mark = "🔒 "
            else:
                mark = ""
            row.append(InlineKeyboardButton(f"{mark}{q}.{sq}", callback_data=f"set:qs:{q}:{sq}"))
        rows.append(row)
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="menu:set")])
    return InlineKeyboardMarkup(rows)


def manage_kb(subs) -> InlineKeyboardMarkup:
    """
    subs: список Subscription (потрібні .queue, .subqueue, .label)
    """
    rows = []
    for s in subs:
        name = subscription_display_name(s.queue, s.subqueue, s.label)
        rows.append([
            InlineKeyboardButton(f"✏️ {name}", callback_data=f"sub:rename:{s.queue}:{s.subqueue}"),
            InlineKeyboardButton("❌", callback_data=f"sub:del:{s.queue}:{s.subqueue}"),
        ])
    rows.append([InlineKeyboardButton("➕ Додати чергу", callback_data="sub:add")])
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="menu:back")])
    return InlineKeyboardMarkup(rows)


def after_add_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("➕ Додати ще чергу", callback_data="sub:add")],
        [InlineKeyboardButton("✅ Готово", callback_data="menu:back")],
    ])
