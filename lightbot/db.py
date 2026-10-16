import os
import shutil
import sqlite3
import time
from typing import List, Optional, Tuple

from telegram import Update

from .config import DB_PATH, log

# ================= DATABASE =================


def db_connect() -> sqlite3.Connection:
    """
    Єдине місце для підключення до SQLite з корисними PRAGMA:
    - WAL: краще для паралельних читань/записів (scheduler + handlers)
    - busy_timeout: менше шансів на "database is locked"
    """
    con = sqlite3.connect(DB_PATH)
    try:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA busy_timeout=3000")
        con.execute("PRAGMA synchronous=NORMAL")
    except Exception:
        pass
    return con


def _ensure_users_columns(con: sqlite3.Connection):
    """
    Soft-migration: додаємо колонки, якщо їх нема.
    """
    ddls = [
        # було раніше
        "ALTER TABLE users ADD COLUMN last_fingerprint TEXT",
        "ALTER TABLE users ADD COLUMN username TEXT",
        "ALTER TABLE users ADD COLUMN created_at INTEGER",
        "ALTER TABLE users ADD COLUMN last_seen_at INTEGER",
        "ALTER TABLE users ADD COLUMN last_total_off INTEGER",
        "ALTER TABLE users ADD COLUMN last_intervals TEXT",
        "ALTER TABLE users ADD COLUMN last_states TEXT",

        # ✅ NEW: розділена памʼять TODAY (в межах одного календарного дня)
        "ALTER TABLE users ADD COLUMN today_day TEXT",             # YYYY-MM-DD
        "ALTER TABLE users ADD COLUMN today_fingerprint TEXT",
        "ALTER TABLE users ADD COLUMN today_total_off INTEGER",
        "ALTER TABLE users ADD COLUMN today_intervals TEXT",
        "ALTER TABLE users ADD COLUMN today_states TEXT",

        # ✅ NEW: памʼять TOMORROW
        "ALTER TABLE users ADD COLUMN tomorrow_day TEXT",          # YYYY-MM-DD (день, на який графік)
        "ALTER TABLE users ADD COLUMN tomorrow_fingerprint TEXT",
        "ALTER TABLE users ADD COLUMN tomorrow_total_off INTEGER",
        "ALTER TABLE users ADD COLUMN tomorrow_intervals TEXT",
        "ALTER TABLE users ADD COLUMN tomorrow_states TEXT",

        # ✅ NEW: флаг, що юзер вже обрав чергу
        "ALTER TABLE users ADD COLUMN is_configured INTEGER DEFAULT 0",
    ]
    for ddl in ddls:
        try:
            con.execute(ddl)
        except sqlite3.OperationalError:
            pass


def db_init():
    """
    Створює таблиці, і робить "soft-migration" для старих users.db.
    """
    with db_connect() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS users(
                chat_id INTEGER PRIMARY KEY,
                queue INTEGER NOT NULL,
                subqueue INTEGER NOT NULL,

                last_fingerprint TEXT,
                username TEXT,
                created_at INTEGER,
                last_seen_at INTEGER,
                last_total_off INTEGER,
                last_intervals TEXT,
                last_states TEXT,

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

                is_configured INTEGER DEFAULT 0
            )
        """)

        _ensure_users_columns(con)

        # ✅ МІГРАЦІЯ is_configured ДЛЯ СТАРОЇ БАЗИ:
        # ставимо configured=1 ТІЛЬКИ тим, у кого вже є збережений стан/памʼять
        # (тобто це точно "старі реальні" юзери, а не нові, яких лише touchнули)
        con.execute("""
                    UPDATE users
                    SET is_configured = 1
                    WHERE COALESCE(is_configured, 0) = 0
                      AND (
                            today_fingerprint IS NOT NULL
                         OR today_states IS NOT NULL
                         OR last_fingerprint IS NOT NULL
                         OR last_states IS NOT NULL
                      )
                """)

        con.execute("""
            CREATE TABLE IF NOT EXISTS meta(
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        con.commit()


def db_touch_user(chat_id: int, username: Optional[str]):
    """
    created_at — ставимо тільки якщо ще NULL
    last_seen_at — оновлюємо завжди
    username — оновлюємо (може мінятись / бути None)
    """
    now_ts = int(time.time())
    with db_connect() as con:
        _ensure_users_columns(con)
        con.execute("""
            INSERT INTO users(
                chat_id, queue, subqueue,
                last_fingerprint, username, created_at, last_seen_at,
                last_total_off, last_intervals, last_states,
                today_day, today_fingerprint, today_total_off, today_intervals, today_states,
                tomorrow_day, tomorrow_fingerprint, tomorrow_total_off, tomorrow_intervals, tomorrow_states
            )
            VALUES(?, 1, 1,
                   NULL, ?, ?, ?,
                   NULL, NULL, NULL,
                   NULL, NULL, NULL, NULL, NULL,
                   NULL, NULL, NULL, NULL, NULL)
            ON CONFLICT(chat_id) DO UPDATE SET
                username = COALESCE(excluded.username, users.username),
                last_seen_at=excluded.last_seen_at,
                created_at=COALESCE(users.created_at, excluded.created_at)
        """, (chat_id, username, now_ts, now_ts))
        con.commit()


def db_upsert_user(chat_id: int, queue: int, subqueue: int):
    """
    Оновлює чергу/підчергу + позначає юзера як сконфігурованого.
    """
    with db_connect() as con:
        _ensure_users_columns(con)
        con.execute("""
            INSERT INTO users(
                chat_id, queue, subqueue, is_configured,
                last_fingerprint, last_total_off, last_intervals, last_states,
                today_day, today_fingerprint, today_total_off, today_intervals, today_states,
                tomorrow_day, tomorrow_fingerprint, tomorrow_total_off, tomorrow_intervals, tomorrow_states
            )
            VALUES(?, ?, ?, 1,
                   NULL, NULL, NULL, NULL,
                   NULL, NULL, NULL, NULL, NULL,
                   NULL, NULL, NULL, NULL, NULL)
            ON CONFLICT(chat_id) DO UPDATE
            SET queue=excluded.queue,
                subqueue=excluded.subqueue,
                is_configured=1
        """, (chat_id, queue, subqueue))
        con.commit()


def db_is_configured(chat_id: int) -> bool:
    with db_connect() as con:
        _ensure_users_columns(con)
        cur = con.execute(
            "SELECT COALESCE(is_configured, 0) FROM users WHERE chat_id=?",
            (chat_id,)
        )
        row = cur.fetchone()
        return bool(row and row[0] == 1)


def db_get_users_for_push() -> List[Tuple[int, int, int, Optional[str], Optional[int], Optional[str], Optional[str], Optional[str]]]:
    with db_connect() as con:
        _ensure_users_columns(con)
        cur = con.execute("""
            SELECT chat_id, queue, subqueue,
                   today_fingerprint, today_total_off, today_intervals, today_states, today_day
            FROM users
            WHERE COALESCE(is_configured, 0) = 1
        """)
        return list(cur.fetchall())


def db_get_users_basic() -> List[Tuple[int, int, int, Optional[str], Optional[str], Optional[str]]]:
    with db_connect() as con:
        _ensure_users_columns(con)
        cur = con.execute("""
            SELECT chat_id, queue, subqueue,
                   tomorrow_fingerprint, tomorrow_states, tomorrow_day
            FROM users
            WHERE COALESCE(is_configured, 0) = 1
        """)
        return list(cur.fetchall())


def db_get_user_queue(chat_id: int) -> Optional[Tuple[int, int]]:
    with db_connect() as con:
        _ensure_users_columns(con)
        cur = con.execute("""
            SELECT queue, subqueue
            FROM users
            WHERE chat_id=? AND COALESCE(is_configured, 0) = 1
        """, (chat_id,))
        row = cur.fetchone()
        if not row:
            return None
        return int(row[0]), int(row[1])


def db_set_today_memory(chat_id: int, day: str, fp: str, total_off: int, intervals_text: str, states_text: str):
    with db_connect() as con:
        _ensure_users_columns(con)
        con.execute("""
            UPDATE users
            SET today_day=?,
                today_fingerprint=?,
                today_total_off=?,
                today_intervals=?,
                today_states=?
            WHERE chat_id=?
        """, (day, fp, total_off, intervals_text, states_text, chat_id))
        con.commit()


def db_set_tomorrow_memory(chat_id: int, day: str, fp: str, total_off: int, intervals_text: str, states_text: Optional[str]):
    with db_connect() as con:
        _ensure_users_columns(con)
        con.execute("""
            UPDATE users
            SET tomorrow_day=?,
                tomorrow_fingerprint=?,
                tomorrow_total_off=?,
                tomorrow_intervals=?,
                tomorrow_states=?
            WHERE chat_id=?
        """, (day, fp, total_off, intervals_text, states_text, chat_id))
        con.commit()


def db_delete_user(chat_id: int):
    with db_connect() as con:
        con.execute("DELETE FROM users WHERE chat_id=?", (chat_id,))
        con.commit()


def db_meta_get(key: str) -> Optional[str]:
    with db_connect() as con:
        cur = con.execute("SELECT value FROM meta WHERE key=?", (key,))
        row = cur.fetchone()
        return row[0] if row else None


def db_meta_set(key: str, value: str):
    with db_connect() as con:
        con.execute("""
            INSERT INTO meta(key, value) VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """, (key, value))
        con.commit()


def db_get_last_user() -> Optional[Tuple[int, Optional[str], Optional[int], Optional[int]]]:
    with db_connect() as con:
        _ensure_users_columns(con)
        cur = con.execute("""
            SELECT chat_id, username, created_at, last_seen_at
            FROM users
            WHERE created_at IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 1
        """)
        row = cur.fetchone()
        return row if row else None


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

    return int(total), [(day, int(cnt)) for day, cnt in rows]


# ================= TRANSFER DB =================

def ensure_seed_db():
    if os.path.exists(DB_PATH):
        return

    seed = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "users.db")
    if os.path.exists(seed):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        shutil.copy2(seed, DB_PATH)
