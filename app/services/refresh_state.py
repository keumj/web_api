from __future__ import annotations

import json
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from app.settings import settings


STATE_RELATIVE_PATH = Path("outputs") / "refresh_state.json"
SCHEDULER_LOG_RELATIVE_PATH = Path("outputs") / "refresh_local_data_scheduler.log"
TRACKED_REFRESH_TARGETS = (
    "data/sp500_shared_db/sp500_shared_prices.sqlite",
    "data/macro_prices.sqlite",
)
STALE_RUNNING_SECONDS = 8 * 60 * 60


def state_path(root: Path | None = None) -> Path:
    return (root or settings.project_root) / STATE_RELATIVE_PATH


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"state_read_error": f"{type(exc).__name__}: {exc}"}
    return data if isinstance(data, dict) else {"state_read_error": "state file is not a JSON object"}


def read_state(root: Path | None = None) -> dict[str, Any]:
    return _read_json(state_path(root))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _format_mtime(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds")
    except Exception:
        return None


def _sqlite_fetchone(path: Path, query: str) -> tuple[Any, ...] | None:
    if not path.exists():
        return None
    try:
        with sqlite3.connect(path) as conn:
            return conn.execute(query).fetchone()
    except Exception:
        return None


def collect_git_status(root: Path | None = None) -> dict[str, Any]:
    root_dir = root or settings.project_root
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--", *TRACKED_REFRESH_TARGETS],
            cwd=str(root_dir),
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=10,
        )
    except Exception as exc:
        return {
            "ok": False,
            "changed": False,
            "lines": [],
            "error": f"{type(exc).__name__}: {exc}",
        }
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    error = ""
    if result.returncode != 0:
        error = (result.stderr or result.stdout or "git status failed").strip()
    return {
        "ok": result.returncode == 0,
        "changed": bool(lines),
        "lines": lines,
        "error": error,
    }


def collect_data_status(root: Path | None = None) -> dict[str, Any]:
    root_dir = root or settings.project_root
    shared_sqlite = root_dir / "data" / "sp500_shared_db" / "sp500_shared_prices.sqlite"
    macro_sqlite = root_dir / "data" / "macro_prices.sqlite"

    prices = _sqlite_fetchone(shared_sqlite, "SELECT MAX(date), COUNT(*), COUNT(DISTINCT symbol) FROM prices")
    quarterly = _sqlite_fetchone(
        shared_sqlite,
        "SELECT MAX(fiscal_date), COUNT(*), COUNT(DISTINCT symbol), MAX(updated_at) FROM fundamentals_quarterly",
    )
    news = _sqlite_fetchone(shared_sqlite, "SELECT MAX(publish_date), COUNT(*), COUNT(DISTINCT ticker) FROM news_articles")
    macro = _sqlite_fetchone(macro_sqlite, "SELECT MAX(date), COUNT(*), COUNT(DISTINCT series_id) FROM macro_series")

    return {
        "shared_sqlite": {
            "path": "data/sp500_shared_db/sp500_shared_prices.sqlite",
            "exists": shared_sqlite.exists(),
            "size_bytes": shared_sqlite.stat().st_size if shared_sqlite.exists() else 0,
            "modified_at": _format_mtime(shared_sqlite),
        },
        "macro_sqlite": {
            "path": "data/macro_prices.sqlite",
            "exists": macro_sqlite.exists(),
            "size_bytes": macro_sqlite.stat().st_size if macro_sqlite.exists() else 0,
            "modified_at": _format_mtime(macro_sqlite),
        },
        "prices": {
            "latest_date": str(prices[0]) if prices and prices[0] else None,
            "rows": int(prices[1] or 0) if prices else None,
            "symbols": int(prices[2] or 0) if prices else None,
        },
        "quarterly": {
            "latest_fiscal_date": str(quarterly[0]) if quarterly and quarterly[0] else None,
            "rows": int(quarterly[1] or 0) if quarterly else None,
            "symbols": int(quarterly[2] or 0) if quarterly else None,
            "updated_at": str(quarterly[3]) if quarterly and quarterly[3] else None,
        },
        "news": {
            "latest_publish_date": str(news[0]) if news and news[0] else None,
            "rows": int(news[1] or 0) if news else None,
            "tickers": int(news[2] or 0) if news else None,
        },
        "macro": {
            "latest_date": str(macro[0]) if macro and macro[0] else None,
            "rows": int(macro[1] or 0) if macro else None,
            "series": int(macro[2] or 0) if macro else None,
        },
    }


def _latest_scheduler_lines(root: Path) -> list[str]:
    log_path = root / SCHEDULER_LOG_RELATIVE_PATH
    if not log_path.exists():
        return []
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []
    matches = [
        line.strip()
        for line in lines
        if "Scheduled refresh" in line or "SQLite auto sync is disabled" in line or "Refresh result:" in line
    ]
    return matches[-5:]


def record_state(event: str, *, source: str = "unknown", exit_code: int | None = None, root: Path | None = None) -> dict[str, Any]:
    root_dir = root or settings.project_root
    path = state_path(root_dir)
    state = read_state(root_dir)
    now = _now()

    state.setdefault("schema_version", 1)
    state["source"] = source
    state["last_checked_at"] = now
    if event == "started":
        state["status"] = "running"
        state["last_started_at"] = now
        state["last_exit_code"] = None
        state["last_error"] = None
    elif event == "finished":
        state["status"] = "success" if int(exit_code or 0) == 0 else "failed"
        state["last_finished_at"] = now
        state["last_exit_code"] = int(exit_code or 0)
        state["last_error"] = None if int(exit_code or 0) == 0 else f"exit_code={int(exit_code or 0)}"
    elif event == "checked":
        if state.get("status") == "checked":
            state["status"] = "unknown"
    else:
        state["status"] = str(event or "checked")

    state["git"] = collect_git_status(root_dir)
    state["data"] = collect_data_status(root_dir)
    state["scheduler_log_tail"] = _latest_scheduler_lines(root_dir)
    _write_json(path, state)
    return state


def _seconds_since(value: str | None) -> float | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return (datetime.now().astimezone() - parsed).total_seconds()


def notice_state(root: Path | None = None) -> dict[str, Any]:
    root_dir = root or settings.project_root
    state = read_state(root_dir)
    exists = state_path(root_dir).exists()
    state["state_file_exists"] = exists
    state["state_file_path"] = str(STATE_RELATIVE_PATH).replace("\\", "/")
    state["git"] = collect_git_status(root_dir)
    state["data"] = collect_data_status(root_dir)
    state["scheduler_log_tail"] = _latest_scheduler_lines(root_dir)

    status = str(state.get("status") or "unknown")
    running_age = _seconds_since(state.get("last_started_at") if status == "running" else None)
    stale_running = running_age is not None and running_age > STALE_RUNNING_SECONDS
    git = state.get("git") if isinstance(state.get("git"), dict) else {}

    if not exists or state.get("state_read_error"):
        severity = "warning"
        headline = "갱신 상태 파일이 아직 없습니다."
    elif stale_running:
        severity = "error"
        headline = "갱신 작업이 종료 기록 없이 오래 실행 중으로 남아 있습니다."
    elif status == "failed":
        severity = "error"
        headline = "마지막 데이터 갱신이 실패했습니다."
    elif git.get("error"):
        severity = "error"
        headline = "SQLite Git 상태 확인이 필요합니다."
    elif git.get("changed"):
        severity = "warning"
        headline = "GitHub에 아직 반영되지 않은 SQLite 변경이 있습니다."
    elif status == "running":
        severity = "warning"
        headline = "데이터 갱신 작업이 실행 중입니다."
    else:
        severity = "success"
        headline = "현재 추적 중인 SQLite 변경은 없습니다."

    state["notice"] = {
        "severity": severity,
        "headline": headline,
        "stale_running": stale_running,
        "running_age_seconds": running_age,
    }
    return state
