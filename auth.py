"""User authentication and session management using SQLite and PBKDF2."""
import hashlib
import os
import secrets
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.db")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    if salt is None:
        salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
    return key.hex(), salt


def verify_password(password: str, password_hash: str, salt: str) -> bool:
    calc_hash, _ = hash_password(password, salt)
    return secrets.compare_digest(calc_hash, password_hash)


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transcripts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                speaker TEXT NOT NULL DEFAULT 'Signer',
                raw_text TEXT,
                english TEXT NOT NULL,
                hindi TEXT,
                kannada TEXT,
                is_smoothed INTEGER DEFAULT 0,
                timestamp TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_transcripts_user ON transcripts(user_id, timestamp)")

        # Seed default admin 'raghav'
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE username = 'raghav'")
        row_raghav = cur.fetchone()
        now_str = datetime.utcnow().isoformat()
        pwd_hash_admin, salt_admin = hash_password("admin123")
        if not row_raghav:
            cur.execute(
                "INSERT INTO users (username, password_hash, salt, role, created_at) VALUES (?, ?, ?, ?, ?)",
                ("raghav", pwd_hash_admin, salt_admin, "admin", now_str)
            )
        else:
            cur.execute(
                "UPDATE users SET password_hash = ?, salt = ?, role = 'admin' WHERE id = ?",
                (pwd_hash_admin, salt_admin, row_raghav["id"])
            )

        # Seed default normal user 'user'
        cur.execute("SELECT id FROM users WHERE username = 'user'")
        row_user = cur.fetchone()
        pwd_hash_user, salt_user = hash_password("user123")
        if not row_user:
            cur.execute(
                "INSERT INTO users (username, password_hash, salt, role, created_at) VALUES (?, ?, ?, ?, ?)",
                ("user", pwd_hash_user, salt_user, "user", now_str)
            )
        else:
            cur.execute(
                "UPDATE users SET password_hash = ?, salt = ?, role = 'user' WHERE id = ?",
                (pwd_hash_user, salt_user, row_user["id"])
            )

        # Remove legacy accounts so only raghav and user remain
        cur.execute("DELETE FROM users WHERE username IN ('admin', 'user1')")
        conn.commit()


def register_user(username: str, password: str, role: str = "user") -> Dict[str, Any]:
    username = username.strip()
    if len(username) < 3:
        raise ValueError("Username must be at least 3 characters long.")
    if len(password) < 4:
        raise ValueError("Password must be at least 4 characters long.")
    if role not in ("admin", "user"):
        role = "user"

    pwd_hash, salt = hash_password(password)
    now_str = datetime.utcnow().isoformat()
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO users (username, password_hash, salt, role, created_at) VALUES (?, ?, ?, ?, ?)",
                (username, pwd_hash, salt, role, now_str)
            )
            user_id = cur.lastrowid
            conn.commit()
            return {"id": user_id, "username": username, "role": role, "created_at": now_str}
    except sqlite3.IntegrityError:
        raise ValueError(f"Username '{username}' is already taken.")


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    username = username.strip()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, username, password_hash, salt, role, created_at FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        if not row:
            return None
        if verify_password(password, row["password_hash"], row["salt"]):
            return {
                "id": row["id"],
                "username": row["username"],
                "role": row["role"],
                "created_at": row["created_at"]
            }
        return None


def create_session(user_id: int, days_valid: int = 7) -> str:
    token = secrets.token_hex(32)
    now = datetime.utcnow()
    expires = now + timedelta(days=days_valid)
    with get_db() as conn:
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token, user_id, now.isoformat(), expires.isoformat())
        )
        conn.commit()
    return token


def get_user_by_session(token: str) -> Optional[Dict[str, Any]]:
    if not token:
        return None
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT u.id, u.username, u.role, u.created_at, s.expires_at
            FROM sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.token = ?
        """, (token,))
        row = cur.fetchone()
        if not row:
            return None

        # Check expiration
        try:
            exp = datetime.fromisoformat(row["expires_at"])
            if exp < datetime.utcnow():
                delete_session(token)
                return None
        except Exception:
            pass

        return {
            "id": row["id"],
            "username": row["username"],
            "role": row["role"],
            "created_at": row["created_at"]
        }


def delete_session(token: str):
    with get_db() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()


def list_users() -> List[Dict[str, Any]]:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, username, role, created_at FROM users ORDER BY id ASC")
        return [dict(r) for r in cur.fetchall()]


def update_user_role(user_id: int, new_role: str) -> bool:
    if new_role not in ("admin", "user"):
        return False
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT username, role FROM users WHERE id = ?", (user_id,))
        row = cur.fetchone()
        if not row:
            return False
        if row["username"].lower() == "raghav":
            raise ValueError("Admin 'raghav' is a fixed permanent administrator and cannot be demoted.")
        cur.execute("UPDATE users SET role = ? WHERE id = ?", (new_role, user_id))
        conn.commit()
        return cur.rowcount > 0


def delete_user(user_id: int) -> bool:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT username, role FROM users WHERE id = ?", (user_id,))
        row = cur.fetchone()
        if not row:
            return False
        if row["username"].lower() == "raghav":
            raise ValueError("Admin 'raghav' is a permanent administrator and cannot be deleted.")
        if row["role"] == "admin":
            cur.execute("SELECT COUNT(*) as count FROM users WHERE role = 'admin'")
            if cur.fetchone()["count"] <= 1:
                raise ValueError("Cannot delete the last remaining administrator.")
        cur.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        return cur.rowcount > 0


def get_users_count() -> Dict[str, int]:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT role, COUNT(*) as count FROM users GROUP BY role")
        counts = {"total": 0, "admin": 0, "user": 0}
        for r in cur.fetchall():
            counts[r["role"]] = r["count"]
            counts["total"] += r["count"]
        return counts


# Session transcript records
def add_transcript(
    user_id: int,
    speaker: str,
    english: str,
    hindi: str = "",
    kannada: str = "",
    raw_text: str = "",
    is_smoothed: bool = False
) -> Dict[str, Any]:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO transcripts (user_id, speaker, raw_text, english, hindi, kannada, is_smoothed, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, speaker, raw_text, english, hindi, kannada, 1 if is_smoothed else 0, timestamp)
        )
        conn.commit()
        t_id = cur.lastrowid
        return {
            "id": t_id,
            "user_id": user_id,
            "speaker": speaker,
            "raw_text": raw_text,
            "english": english,
            "hindi": hindi,
            "kannada": kannada,
            "is_smoothed": bool(is_smoothed),
            "timestamp": timestamp
        }


def get_transcripts(user_id: int, limit: int = 150) -> List[Dict[str, Any]]:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, user_id, speaker, raw_text, english, hindi, kannada, is_smoothed, timestamp
            FROM transcripts
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit)
        )
        rows = [dict(r) for r in cur.fetchall()]
        return rows


def clear_transcripts(user_id: int) -> bool:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM transcripts WHERE user_id = ?", (user_id,))
        conn.commit()
        return True

