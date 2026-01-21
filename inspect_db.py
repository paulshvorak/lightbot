import sqlite3

con = sqlite3.connect("users.db")
rows = con.execute("""
    SELECT chat_id, username, queue, subqueue, last_fingerprint
    FROM users
""").fetchall()

print(f"Users count: {len(rows)}\n")
for r in rows:
    print(r)