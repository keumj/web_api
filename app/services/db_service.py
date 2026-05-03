from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from app.settings import settings


def using_remote_app_db() -> bool:
    return bool(settings.app_database_url)


def _local_auth_db_path() -> Path:
    return settings.auth_db_path if settings.auth_db_path.is_absolute() else settings.project_root / settings.auth_db_path


def storage_label() -> str:
    if using_remote_app_db():
        return "turso:" + settings.app_database_url
    return str(_local_auth_db_path())


def _connect_remote():
    try:
        import libsql
    except ImportError as exc:  # pragma: no cover - depends on deployment extras.
        raise RuntimeError(
            "TURSO_DATABASE_URL/KEUMJM_DATABASE_URL is set, but the 'libsql' package is not installed."
        ) from exc
    kwargs: dict[str, str] = {"database": settings.app_database_url}
    if settings.app_database_auth_token:
        kwargs["auth_token"] = settings.app_database_auth_token
    return libsql.connect(**kwargs)


def _connect_local():
    path = _local_auth_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path)


@contextmanager
def app_db_connection() -> Iterator[object]:
    conn = _connect_remote() if using_remote_app_db() else _connect_local()
    try:
        yield conn
    finally:
        close = getattr(conn, "close", None)
        if callable(close):
            close()
