from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import uvicorn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _env_int(default: int, *names: str) -> int:
    for name in names:
        raw = os.getenv(name)
        if raw is not None and raw.strip():
            return int(raw.strip())
    return default


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
        port=_env_int(8515, "PORT", "KEUMJM_PORT"),
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
        timeout_keep_alive=_env_int(30, "KEUMJM_UVICORN_KEEP_ALIVE"),
        log_level=os.getenv("KEUMJM_UVICORN_LOG_LEVEL", "warning"),
        access_log=False,
        loop="asyncio",
    )


if __name__ == "__main__":
    main()
