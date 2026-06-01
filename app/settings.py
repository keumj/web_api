from __future__ import annotations

import os
from dataclasses import dataclass
from ipaddress import ip_network
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - startup still works if dotenv is absent.
    load_dotenv = None


if load_dotenv is not None:
    load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str) -> tuple[str, ...]:
    raw = os.getenv(name, "")
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw.strip())


def _env_first(*names: str, default: str = "") -> str:
    for name in names:
        raw = os.getenv(name)
        if raw is not None and raw.strip():
            return raw.strip()
    return default


@dataclass(frozen=True)
class Settings:
    app_name: str = "Keumjm Portfolio Lab"
    host: str = os.getenv("KEUMJM_HOST", "0.0.0.0")
    port: int = int(os.getenv("KEUMJM_PORT", "8515"))
    project_root: Path = Path(__file__).resolve().parents[1]
    access_mode: str = os.getenv("KEUMJM_ACCESS_MODE", "lan").strip().lower()
    allowed_cidrs: tuple[str, ...] = _env_list("KEUMJM_ALLOWED_CIDRS")
    enable_docs: bool = _env_bool("KEUMJM_ENABLE_DOCS", True)
    enable_macro: bool = _env_bool("ENABLE_MACRO", False)
    gzip_minimum_size: int = _env_int("KEUMJM_GZIP_MINIMUM_SIZE", 1024)
    gzip_compresslevel: int = _env_int("KEUMJM_GZIP_COMPRESSLEVEL", 5)
    auth_enabled: bool = _env_bool("KEUMJM_AUTH_ENABLED", True)
    auth_allow_registration: bool = _env_bool("KEUMJM_AUTH_ALLOW_REGISTRATION", True)
    auth_cookie_name: str = os.getenv("KEUMJM_AUTH_COOKIE_NAME", "keumjm_session")
    auth_cookie_secure: bool = _env_bool("KEUMJM_AUTH_COOKIE_SECURE", False)
    auth_session_days: int = _env_int("KEUMJM_AUTH_SESSION_DAYS", 14)
    auth_session_secret: str = _env_first("KEUMJM_AUTH_SECRET", "KEUMJ_AUTH_SECRET")
    auth_db_path: Path = Path(os.getenv("KEUMJM_AUTH_DB_PATH", "data/users/auth.sqlite"))
    auth_secret_path: Path = Path(os.getenv("KEUMJM_AUTH_SECRET_PATH", "data/users/session_secret.key"))
    bootstrap_admin_username: str = os.getenv("KEUMJM_BOOTSTRAP_ADMIN_USERNAME", "").strip()
    bootstrap_admin_password: str = os.getenv("KEUMJM_BOOTSTRAP_ADMIN_PASSWORD", "")
    portfolio_db_root: Path = Path(os.getenv("KEUMJ_PORTFOLIO_DB_DIR", "data/portfolio"))
    use_remote_app_db: bool = _env_bool("KEUMJM_USE_REMOTE_APP_DB", True)
    app_database_url: str = _env_first("KEUMJM_DATABASE_URL", "KEUMJ_DATABASE_URL", "TURSO_DATABASE_URL")
    app_database_auth_token: str = _env_first("KEUMJM_DATABASE_AUTH_TOKEN", "KEUMJ_DATABASE_AUTH_TOKEN", "TURSO_AUTH_TOKEN")
    sp500_database_url: str = _env_first(
        "KEUMJ_SP500_DATABASE_URL",
        "KEUMJ_SP500_TURSO_DATABASE_URL",
        "TURSO_SP500_DATABASE_URL",
    )
    sp500_database_auth_token: str = _env_first(
        "KEUMJ_SP500_DATABASE_AUTH_TOKEN",
        "KEUMJ_SP500_TURSO_AUTH_TOKEN",
        "TURSO_SP500_AUTH_TOKEN",
    )
    sp500_supabase_database_url: str = _env_first("KEUMJ_SP500_SUPABASE_DATABASE_URL")
    sp500_supabase_host: str = os.getenv("KEUMJ_SP500_SUPABASE_HOST", "").strip()
    sp500_supabase_port: int = _env_int("KEUMJ_SP500_SUPABASE_PORT", 6543)
    sp500_supabase_database: str = os.getenv("KEUMJ_SP500_SUPABASE_DATABASE", "postgres").strip()
    sp500_supabase_user: str = os.getenv("KEUMJ_SP500_SUPABASE_USER", "").strip()
    sp500_supabase_password: str = os.getenv("KEUMJ_SP500_SUPABASE_PASSWORD", "")
    sp500_supabase_sslmode: str = os.getenv("KEUMJ_SP500_SUPABASE_SSLMODE", "require").strip() or "require"
    macro_database_url: str = _env_first(
        "KEUMJ_MACRO_DATABASE_URL",
        "KEUMJ_MACRO_TURSO_DATABASE_URL",
        "TURSO_MACRO_DATABASE_URL",
    )
    macro_database_auth_token: str = _env_first(
        "KEUMJ_MACRO_DATABASE_AUTH_TOKEN",
        "KEUMJ_MACRO_TURSO_AUTH_TOKEN",
        "TURSO_MACRO_AUTH_TOKEN",
    )
    macro_supabase_database_url: str = _env_first("KEUMJ_MACRO_SUPABASE_DATABASE_URL")
    macro_supabase_host: str = os.getenv("KEUMJ_MACRO_SUPABASE_HOST", "").strip()
    macro_supabase_port: int = _env_int("KEUMJ_MACRO_SUPABASE_PORT", 6543)
    macro_supabase_database: str = os.getenv("KEUMJ_MACRO_SUPABASE_DATABASE", "postgres").strip()
    macro_supabase_user: str = os.getenv("KEUMJ_MACRO_SUPABASE_USER", "").strip()
    macro_supabase_password: str = os.getenv("KEUMJ_MACRO_SUPABASE_PASSWORD", "")
    macro_supabase_sslmode: str = os.getenv("KEUMJ_MACRO_SUPABASE_SSLMODE", "require").strip() or "require"

    def parsed_allowed_networks(self):
        return tuple(ip_network(cidr, strict=False) for cidr in self.allowed_cidrs)


settings = Settings()
