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


@dataclass(frozen=True)
class Settings:
    app_name: str = "Keumjm Portfolio Lab"
    host: str = os.getenv("KEUMJM_HOST", "0.0.0.0")
    port: int = int(os.getenv("KEUMJM_PORT", "8515"))
    project_root: Path = Path(__file__).resolve().parents[1]
    access_mode: str = os.getenv("KEUMJM_ACCESS_MODE", "lan").strip().lower()
    allowed_cidrs: tuple[str, ...] = _env_list("KEUMJM_ALLOWED_CIDRS")
    enable_docs: bool = _env_bool("KEUMJM_ENABLE_DOCS", True)
    gzip_minimum_size: int = _env_int("KEUMJM_GZIP_MINIMUM_SIZE", 1024)
    gzip_compresslevel: int = _env_int("KEUMJM_GZIP_COMPRESSLEVEL", 5)
    auth_enabled: bool = _env_bool("KEUMJM_AUTH_ENABLED", True)
    auth_allow_registration: bool = _env_bool("KEUMJM_AUTH_ALLOW_REGISTRATION", True)
    auth_cookie_name: str = os.getenv("KEUMJM_AUTH_COOKIE_NAME", "keumjm_session")
    auth_cookie_secure: bool = _env_bool("KEUMJM_AUTH_COOKIE_SECURE", False)
    auth_session_days: int = _env_int("KEUMJM_AUTH_SESSION_DAYS", 14)
    auth_db_path: Path = Path(os.getenv("KEUMJM_AUTH_DB_PATH", "data/users/auth.sqlite"))
    auth_secret_path: Path = Path(os.getenv("KEUMJM_AUTH_SECRET_PATH", "data/users/session_secret.key"))
    portfolio_db_root: Path = Path(os.getenv("KEUMJ_PORTFOLIO_DB_DIR", "data/portfolio"))

    def parsed_allowed_networks(self):
        return tuple(ip_network(cidr, strict=False) for cidr in self.allowed_cidrs)


settings = Settings()
