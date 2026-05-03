from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import secrets
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import Request

from app.settings import settings


USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
PASSWORD_MIN_LENGTH = 8
PBKDF2_ITERATIONS = 260_000


@dataclass(frozen=True)
class AuthUser:
    id: str
    username: str
    is_admin: bool = False


def _auth_db_path() -> Path:
    return settings.project_root / settings.auth_db_path if not settings.auth_db_path.is_absolute() else settings.auth_db_path


def _secret_path() -> Path:
    return settings.project_root / settings.auth_secret_path if not settings.auth_secret_path.is_absolute() else settings.auth_secret_path


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode((data + padding).encode("ascii"))


def _session_secret() -> bytes:
    path = _secret_path()
    if path.exists():
        return path.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    secret = secrets.token_bytes(32)
    path.write_bytes(secret)
    return secret


def ensure_auth_db() -> Path:
    path = _auth_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                password_salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                iterations INTEGER NOT NULL,
                is_admin INTEGER NOT NULL DEFAULT 0,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_login_at TEXT
            )
            """
        )
        _ensure_user_column(conn, "is_admin", "INTEGER NOT NULL DEFAULT 0")
        _ensure_user_column(conn, "is_active", "INTEGER NOT NULL DEFAULT 1")
        _ensure_bootstrap_admin(conn)
        _ensure_at_least_one_admin(conn)
        conn.commit()
    return path


def _ensure_user_column(conn: sqlite3.Connection, name: str, definition: str) -> None:
    cols = {str(row[1]) for row in conn.execute("PRAGMA table_info(users)").fetchall()}
    if name not in cols:
        conn.execute(f"ALTER TABLE users ADD COLUMN {name} {definition}")


def _ensure_at_least_one_admin(conn: sqlite3.Connection) -> None:
    user_total = int(conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] or 0)
    if user_total == 0:
        return
    admin_total = int(conn.execute("SELECT COUNT(*) FROM users WHERE is_admin = 1 AND is_active = 1").fetchone()[0] or 0)
    if admin_total > 0:
        return
    row = conn.execute("SELECT id FROM users ORDER BY created_at ASC, username ASC LIMIT 1").fetchone()
    if row:
        conn.execute("UPDATE users SET is_admin = 1, is_active = 1 WHERE id = ?", (str(row[0]),))


def _ensure_bootstrap_admin(conn: sqlite3.Connection) -> None:
    if not settings.bootstrap_admin_username or not settings.bootstrap_admin_password:
        return

    clean_username = validate_username(settings.bootstrap_admin_username)
    raw_password = validate_password(settings.bootstrap_admin_password)
    salt = secrets.token_bytes(16)
    password_hash = _hash_password(raw_password, salt)
    row = conn.execute("SELECT id FROM users WHERE username = ?", (clean_username,)).fetchone()
    if row:
        conn.execute(
            """
            UPDATE users
            SET password_salt = ?, password_hash = ?, iterations = ?, is_admin = 1, is_active = 1
            WHERE id = ?
            """,
            (_b64url_encode(salt), password_hash, PBKDF2_ITERATIONS, str(row[0])),
        )
        return

    conn.execute(
        """
        INSERT INTO users(id, username, password_salt, password_hash, iterations, is_admin, is_active)
        VALUES (?, ?, ?, ?, ?, 1, 1)
        """,
        (secrets.token_urlsafe(18), clean_username, _b64url_encode(salt), password_hash, PBKDF2_ITERATIONS),
    )


def user_count() -> int:
    path = ensure_auth_db()
    with sqlite3.connect(path) as conn:
        row = conn.execute("SELECT COUNT(*) FROM users").fetchone()
    return int(row[0] or 0) if row else 0


def validate_username(username: str) -> str:
    clean = str(username or "").strip().lower()
    if not USERNAME_RE.match(clean):
        raise ValueError("사용자명은 영문, 숫자, 점, 밑줄, 하이픈만 사용해 3~32자로 입력하세요.")
    return clean


def validate_password(password: str) -> str:
    raw = str(password or "")
    if len(raw) < PASSWORD_MIN_LENGTH:
        raise ValueError(f"비밀번호는 {PASSWORD_MIN_LENGTH}자 이상이어야 합니다.")
    return raw


def _hash_password(password: str, salt: bytes, iterations: int = PBKDF2_ITERATIONS) -> str:
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return _b64url_encode(digest)


def create_user(username: str, password: str, *, is_admin: bool | None = None, require_registration_open: bool = True) -> AuthUser:
    existing_count = user_count()
    if require_registration_open and not settings.auth_allow_registration and existing_count > 0:
        raise ValueError("신규 사용자 등록이 비활성화되어 있습니다.")
    clean_username = validate_username(username)
    raw_password = validate_password(password)
    user_id = secrets.token_urlsafe(18)
    salt = secrets.token_bytes(16)
    password_hash = _hash_password(raw_password, salt)
    admin_flag = 1 if (existing_count == 0 if is_admin is None else is_admin) else 0
    path = ensure_auth_db()
    try:
        with sqlite3.connect(path) as conn:
            conn.execute(
                """
                INSERT INTO users(id, username, password_salt, password_hash, iterations, is_admin, is_active)
                VALUES (?, ?, ?, ?, ?, ?, 1)
                """,
                (user_id, clean_username, _b64url_encode(salt), password_hash, PBKDF2_ITERATIONS, admin_flag),
            )
            conn.commit()
    except sqlite3.IntegrityError as exc:
        raise ValueError("이미 사용 중인 사용자명입니다.") from exc
    return AuthUser(id=user_id, username=clean_username, is_admin=bool(admin_flag))


def get_user_by_id(user_id: str) -> AuthUser | None:
    path = ensure_auth_db()
    with sqlite3.connect(path) as conn:
        row = conn.execute("SELECT id, username, is_admin FROM users WHERE id = ? AND is_active = 1", (str(user_id or ""),)).fetchone()
    if not row:
        return None
    return AuthUser(id=str(row[0]), username=str(row[1]), is_admin=bool(row[2]))


def authenticate(username: str, password: str) -> AuthUser | None:
    clean_username = str(username or "").strip().lower()
    path = ensure_auth_db()
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            """
            SELECT id, username, password_salt, password_hash, iterations, is_admin, is_active
            FROM users
            WHERE username = ?
            """,
            (clean_username,),
        ).fetchone()
        if not row:
            return None
        user_id, stored_username, salt_text, stored_hash, iterations, is_admin, is_active = row
        if not int(is_active or 0):
            return None
        candidate = _hash_password(str(password or ""), _b64url_decode(str(salt_text)), int(iterations))
        if not hmac.compare_digest(candidate, str(stored_hash)):
            return None
        conn.execute("UPDATE users SET last_login_at = CURRENT_TIMESTAMP WHERE id = ?", (str(user_id),))
        conn.commit()
    return AuthUser(id=str(user_id), username=str(stored_username), is_admin=bool(is_admin))


def make_session_token(user: AuthUser) -> str:
    now = int(time.time())
    payload: dict[str, Any] = {
        "sub": user.id,
        "name": user.username,
        "admin": user.is_admin,
        "iat": now,
        "exp": now + max(int(settings.auth_session_days), 1) * 86_400,
    }
    payload_bytes = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    payload_part = _b64url_encode(payload_bytes)
    sig = hmac.new(_session_secret(), payload_part.encode("ascii"), hashlib.sha256).digest()
    return f"{payload_part}.{_b64url_encode(sig)}"


def verify_session_token(token: str) -> AuthUser | None:
    try:
        payload_part, sig_part = str(token or "").split(".", 1)
        expected = hmac.new(_session_secret(), payload_part.encode("ascii"), hashlib.sha256).digest()
        if not hmac.compare_digest(expected, _b64url_decode(sig_part)):
            return None
        payload = json.loads(_b64url_decode(payload_part).decode("utf-8"))
        if int(payload.get("exp", 0)) < int(time.time()):
            return None
        return get_user_by_id(str(payload.get("sub", "")))
    except Exception:
        return None


def current_user(request: Request) -> AuthUser | None:
    if not settings.auth_enabled:
        return None
    state_user = getattr(request.state, "user", None)
    if isinstance(state_user, AuthUser):
        return state_user
    token = request.cookies.get(settings.auth_cookie_name, "")
    return verify_session_token(token)


def portfolio_db_for_user(user: AuthUser | str) -> Path:
    user_id = user.id if isinstance(user, AuthUser) else str(user)
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", user_id).strip("._-") or "default"
    root = settings.portfolio_db_root if settings.portfolio_db_root.is_absolute() else settings.project_root / settings.portfolio_db_root
    return root / "users" / safe_id / "portfolio.sqlite"


def list_users() -> list[dict[str, object]]:
    path = ensure_auth_db()
    with sqlite3.connect(path) as conn:
        rows = conn.execute(
            """
            SELECT id, username, is_admin, is_active, created_at, last_login_at
            FROM users
            ORDER BY is_admin DESC, username ASC
            """
        ).fetchall()
    return [
        {
            "id": str(row[0]),
            "username": str(row[1]),
            "is_admin": bool(row[2]),
            "is_active": bool(row[3]),
            "created_at": row[4],
            "last_login_at": row[5],
            "portfolio_db": str(portfolio_db_for_user(str(row[0]))),
        }
        for row in rows
    ]


def _admin_count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM users WHERE is_admin = 1 AND is_active = 1").fetchone()[0] or 0)


def set_user_active(user_id: str, is_active: bool) -> None:
    path = ensure_auth_db()
    with sqlite3.connect(path) as conn:
        row = conn.execute("SELECT is_admin FROM users WHERE id = ?", (str(user_id),)).fetchone()
        if not row:
            raise ValueError("사용자를 찾을 수 없습니다.")
        if not is_active and int(row[0] or 0) and _admin_count(conn) <= 1:
            raise ValueError("마지막 활성 관리자 계정은 비활성화할 수 없습니다.")
        conn.execute("UPDATE users SET is_active = ? WHERE id = ?", (1 if is_active else 0, str(user_id)))
        conn.commit()


def set_user_admin(user_id: str, is_admin: bool) -> None:
    path = ensure_auth_db()
    with sqlite3.connect(path) as conn:
        row = conn.execute("SELECT is_admin FROM users WHERE id = ?", (str(user_id),)).fetchone()
        if not row:
            raise ValueError("사용자를 찾을 수 없습니다.")
        if not is_admin and int(row[0] or 0) and _admin_count(conn) <= 1:
            raise ValueError("마지막 활성 관리자 권한은 해제할 수 없습니다.")
        conn.execute("UPDATE users SET is_admin = ? WHERE id = ?", (1 if is_admin else 0, str(user_id)))
        conn.commit()


def reset_password(user_id: str, password: str) -> None:
    raw_password = validate_password(password)
    salt = secrets.token_bytes(16)
    password_hash = _hash_password(raw_password, salt)
    path = ensure_auth_db()
    with sqlite3.connect(path) as conn:
        cur = conn.execute(
            """
            UPDATE users
            SET password_salt = ?, password_hash = ?, iterations = ?
            WHERE id = ?
            """,
            (_b64url_encode(salt), password_hash, PBKDF2_ITERATIONS, str(user_id)),
        )
        conn.commit()
    if int(cur.rowcount or 0) <= 0:
        raise ValueError("사용자를 찾을 수 없습니다.")
