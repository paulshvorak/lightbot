import sqlite3
import time
from dataclasses import dataclass
from typing import List, Optional

from telegram import Update

from .config import DB_PATH, log

# ================= DATABASE =================


@dataclass
class UserRecord:
    chat_id: int
    username: Optional[str]
    created_at: Optional[int]
    last_seen_at: Optional[int]


@dataclass
class Subscription:
    chat_id: int
    queue: int
    subqueue: int
    label: Optional[str]
    today_day: Optional[str]
    today_fingerprint: Optional[str]
    today_total_off: Optional[int]
    today_intervals: Optional[str]
    today_states: Optional[str]
    tomorrow_day: Optional[str]
    tomorrow_fingerprint: Optional[str]
    tomorrow_total_off: Optional[int]
    tomorrow_intervals: Optional[str]
    tomorrow_states: Optional[str]


class _AutoCloseConnection(sqlite3.Connection):
    """
    sqlite3.Connection.__exit__ сам по собі лише commit/rollback — з'єднання
    НЕ закривається. Тут додатково закриваємо його, щоб `with db_connect() as con:`
    не лишало відкритий файловий дескриптор після кожного виклику.
    """
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            return super().__exit__(exc_type, exc_val, exc_tb)
        finally:
            self.close()


def db_connect() -> sqlite3.Connection:
    """
    Єдине місце для підключення до SQLite з корисними PRAGMA:
    - WAL: краще для паралельних читань/записів (scheduler + handlers)
    - busy_timeout: менше шансів на "database is locked"
    - row_factory=Row: доступ до колонок за іменем, без tuple-позицій
    """
    con = sqlite3.connect(DB_PATH, factory=_AutoCloseConnection)
    con.row_factory = sqlite3.Row
    try:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA busy_timeout=3000")
        con.execute("PRAGMA synchronous=NORMAL")
    except Exception:
        pass
    return con


def db_init():
    """
    Створює таблиці.
    """
    with db_connect() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS users(
                chat_id INTEGER PRIMARY KEY,
                username TEXT,
                created_at INTEGER,
                last_seen_at INTEGER
            )
        """)

        con.execute("""
            CREATE TABLE IF NOT EXISTS subscriptions(
                chat_id INTEGER NOT NULL,
                queue INTEGER NOT NULL,
                subqueue INTEGER NOT NULL,
                label TEXT,

                today_day TEXT,
                today_fingerprint TEXT,
                today_total_off INTEGER,
                today_intervals TEXT,
                today_states TEXT,

                tomorrow_day TEXT,
                tomorrow_fingerprint TEXT,
                tomorrow_total_off INTEGER,
                tomorrow_intervals TEXT,
                tomorrow_states TEXT,

                PRIMARY KEY (chat_id, queue, subqueue)
            )
        """)

        con.commit()


def _row_to_subscription(row: sqlite3.Row) -> Subscription:
    return Subscription(
        chat_id=row["chat_id"],
        queue=row["queue"],
        subqueue=row["subqueue"],
        label=row["label"],
        today_day=row["today_day"],
        today_fingerprint=row["today_fingerprint"],
        today_total_off=row["today_total_off"],
        today_intervals=row["today_intervals"],
        today_states=row["today_states"],
        tomorrow_day=row["tomorrow_day"],
        tomorrow_fingerprint=row["tomorrow_fingerprint"],
        tomorrow_total_off=row["tomorrow_total_off"],
        tomorrow_intervals=row["tomorrow_intervals"],
        tomorrow_states=row["tomorrow_states"],
    )


def db_touch_user(chat_id: int, username: Optional[str]):
    """
    created_at — ставимо тільки якщо ще NULL
    last_seen_at — оновлюємо завжди
    username — оновлюємо (може мінятись / бути None)
    """
    now_ts = int(time.time())
    with db_connect() as con:
        con.execute("""
            INSERT INTO users(chat_id, username, created_at, last_seen_at)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(chat_id) DO UPDATE SET
                username = COALESCE(excluded.username, users.username),
                last_seen_at=excluded.last_seen_at,
                created_at=COALESCE(users.created_at, excluded.created_at)
        """, (chat_id, username, now_ts, now_ts))
        con.commit()


def db_has_subscriptions(chat_id: int) -> bool:
    with db_connect() as con:
        cur = con.execute(
            "SELECT 1 FROM subscriptions WHERE chat_id=? LIMIT 1",
            (chat_id,)
        )
        return cur.fetchone() is not None


def db_add_subscription(chat_id: int, queue: int, subqueue: int):
    with db_connect() as con:
        con.execute("""
            INSERT OR IGNORE INTO subscriptions(chat_id, queue, subqueue)
            VALUES(?, ?, ?)
        """, (chat_id, queue, subqueue))
        con.commit()


def db_set_subscription_label(chat_id: int, queue: int, subqueue: int, label: Optional[str]) -> bool:
    """Повертає True, якщо підписка існувала і назву дійсно оновлено."""
    with db_connect() as con:
        cur = con.execute(
            "UPDATE subscriptions SET label=? WHERE chat_id=? AND queue=? AND subqueue=?",
            (label, chat_id, queue, subqueue)
        )
        con.commit()
        return cur.rowcount > 0


def db_remove_subscription(chat_id: int, queue: int, subqueue: int):
    with db_connect() as con:
        con.execute(
            "DELETE FROM subscriptions WHERE chat_id=? AND queue=? AND subqueue=?",
            (chat_id, queue, subqueue)
        )
        con.commit()


def db_get_subscriptions(chat_id: int) -> List[Subscription]:
    with db_connect() as con:
        cur = con.execute("""
            SELECT * FROM subscriptions
            WHERE chat_id=?
            ORDER BY queue, subqueue
        """, (chat_id,))
        return [_row_to_subscription(r) for r in cur.fetchall()]


def db_get_all_subscriptions() -> List[Subscription]:
    with db_connect() as con:
        cur = con.execute("SELECT * FROM subscriptions")
        return [_row_to_subscription(r) for r in cur.fetchall()]


def db_set_today_memory(chat_id: int, queue: int, subqueue: int, day: str, fp: str, total_off: int, intervals_text: str, states_text: str):
    with db_connect() as con:
        con.execute("""
            UPDATE subscriptions
            SET today_day=?,
                today_fingerprint=?,
                today_total_off=?,
                today_intervals=?,
                today_states=?
            WHERE chat_id=? AND queue=? AND subqueue=?
        """, (day, fp, total_off, intervals_text, states_text, chat_id, queue, subqueue))
        con.commit()


def db_set_tomorrow_memory(chat_id: int, queue: int, subqueue: int, day: str, fp: str, total_off: int, intervals_text: str, states_text: Optional[str]):
    with db_connect() as con:
        con.execute("""
            UPDATE subscriptions
            SET tomorrow_day=?,
                tomorrow_fingerprint=?,
                tomorrow_total_off=?,
                tomorrow_intervals=?,
                tomorrow_states=?
            WHERE chat_id=? AND queue=? AND subqueue=?
        """, (day, fp, total_off, intervals_text, states_text, chat_id, queue, subqueue))
        con.commit()


def db_delete_user(chat_id: int):
    with db_connect() as con:
        con.execute("DELETE FROM subscriptions WHERE chat_id=?", (chat_id,))
        con.execute("DELETE FROM users WHERE chat_id=?", (chat_id,))
        con.commit()


def db_get_last_user() -> Optional[UserRecord]:
    with db_connect() as con:
        cur = con.execute("""
            SELECT chat_id, username, created_at, last_seen_at
            FROM users
            WHERE created_at IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 1
        """)
        row = cur.fetchone()
        if not row:
            return None
        return UserRecord(
            chat_id=row["chat_id"],
            username=row["username"],
            created_at=row["created_at"],
            last_seen_at=row["last_seen_at"],
        )


def touch_from_update(update: Update) -> None:
    try:
        chat_id = update.effective_chat.id if update.effective_chat else None
        username = update.effective_user.username if update.effective_user else None
        if chat_id is not None:
            db_touch_user(chat_id, username)
    except Exception:
        log.exception("touch_from_update failed")


def db_users_stats_by_day(days: int = 14) -> tuple[int, list[tuple[str, int]]]:
    """
    Повертає:
    - total_users
    - список (day, count) за останні N днів
    """
    with db_connect() as con:
        total = con.execute(
            "SELECT COUNT(*) FROM users WHERE created_at IS NOT NULL"
        ).fetchone()[0]

        rows = con.execute("""
            SELECT
              date(created_at, 'unixepoch', 'localtime') AS day,
              COUNT(*) AS cnt
            FROM users
            WHERE created_at IS NOT NULL
              AND created_at >= strftime('%s','now', ?)
            GROUP BY day
            ORDER BY day DESC
        """, (f"-{days} days",)).fetchall()

    return int(total), [(row["day"], int(row["cnt"])) for row in rows]
