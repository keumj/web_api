from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import uvicorn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw.strip())


def _ssl_arg(name: str) -> str | None:
    raw = os.getenv(name, "").strip()
    return raw or None


def main() -> None:
    if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    ssl_certfile = _ssl_arg("KEUMJM_SSL_CERTFILE")
    ssl_keyfile = _ssl_arg("KEUMJM_SSL_KEYFILE")
    uvicorn.run(
        "app.main:app",
        host=os.getenv("KEUMJM_HOST", "0.0.0.0"),
        port=_env_int("KEUMJM_PORT", 8515),
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
        timeout_keep_alive=_env_int("KEUMJM_UVICORN_KEEP_ALIVE", 30),
        log_level=os.getenv("KEUMJM_UVICORN_LOG_LEVEL", "warning"),
        access_log=False,
        loop="asyncio",
    )


if __name__ == "__main__":
    main()
