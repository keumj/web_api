from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from app.services import refresh_state
from app.settings import settings


STATE_RELATIVE_PATH = Path("outputs") / "turso_refresh_button_state.json"
LOG_RELATIVE_PATH = Path("outputs") / "turso_refresh_button.log"
ALLOWED_TARGETS: dict[str, str] = {
    "prices": "가격/시총",
    "fundamentals": "펀더멘털",
    "news": "뉴스",
    "macro": "거시",
    "sp500": "S&P500 전체",
    "all": "전체",
}

_LOCK = threading.Lock()
_RUNNING_THREAD: threading.Thread | None = None


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _state_path(root: Path | None = None) -> Path:
    return (root or settings.project_root) / STATE_RELATIVE_PATH


def _log_path(root: Path | None = None) -> Path:
    return (root or settings.project_root) / LOG_RELATIVE_PATH


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"state_read_error": f"{type(exc).__name__}: {exc}"}
    return data if isinstance(data, dict) else {"state_read_error": "state file is not a JSON object"}


def _write_state(payload: dict[str, Any], *, root: Path | None = None) -> None:
    path = _state_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(settings.project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _build_command(target: str) -> list[str]:
    script = settings.project_root / "scripts" / "refresh_turso_daily.py"
    return [sys.executable, "-u", str(script), "--mode", "direct", "--target", target]


def _record_refresh_state(event: str, *, target: str, exit_code: int | None = None, root: Path) -> None:
    old_sync_interval = os.environ.get("KEUMJ_TURSO_REPLICA_SYNC_SECONDS")
    os.environ["KEUMJ_TURSO_REPLICA_SYNC_SECONDS"] = "0"
    try:
        refresh_state.record_state(event, source=f"manual-turso:{target}", exit_code=exit_code, root=root)
    finally:
        if old_sync_interval is None:
            os.environ.pop("KEUMJ_TURSO_REPLICA_SYNC_SECONDS", None)
        else:
            os.environ["KEUMJ_TURSO_REPLICA_SYNC_SECONDS"] = old_sync_interval


def status() -> dict[str, Any]:
    state = _read_json(_state_path())
    state.setdefault("status", "idle")
    state.setdefault("target", "")
    state.setdefault("target_label", "")
    state.setdefault("started_at", "")
    state.setdefault("finished_at", "")
    state.setdefault("exit_code", None)
    state["allowed_targets"] = ALLOWED_TARGETS
    state["log_path"] = _display_path(_log_path())
    with _LOCK:
        state["running"] = bool(_RUNNING_THREAD and _RUNNING_THREAD.is_alive())
    return state


def start(target: str) -> dict[str, Any]:
    clean_target = str(target or "").strip().lower()
    if clean_target not in ALLOWED_TARGETS:
        return {"ok": False, "message": f"unknown target: {target}"}

    with _LOCK:
        global _RUNNING_THREAD
        if _RUNNING_THREAD and _RUNNING_THREAD.is_alive():
            return {"ok": False, "message": "turso refresh is already running"}

        thread = threading.Thread(target=_run, args=(clean_target,), name=f"turso-refresh-{clean_target}", daemon=True)
        _RUNNING_THREAD = thread
        thread.start()

    return {"ok": True, "message": "started", "target": clean_target, "target_label": ALLOWED_TARGETS[clean_target]}


def _run(target: str) -> None:
    root = settings.project_root
    log_path = _log_path(root)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = _build_command(target)
    started_at = _now()
    state = {
        "schema_version": 1,
        "status": "running",
        "target": target,
        "target_label": ALLOWED_TARGETS[target],
        "started_at": started_at,
        "finished_at": "",
        "exit_code": None,
        "command": command,
        "log_path": _display_path(log_path),
    }
    _write_state(state, root=root)
    _record_refresh_state("started", target=target, root=root)

    env = os.environ.copy()
    env["KEUMJ_TURSO_EMBEDDED_REPLICA"] = "false"
    try:
        with log_path.open("a", encoding="utf-8", errors="replace") as log:
            log.write("\n" + "=" * 72 + "\n")
            log.write(f"[{started_at}] start target={target} label={ALLOWED_TARGETS[target]}\n")
            log.write("command=" + " ".join(command) + "\n")
            result = subprocess.run(
                command,
                cwd=str(root),
                env=env,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            exit_code = int(result.returncode)
            finished_at = _now()
            log.write(f"[{finished_at}] finish target={target} exit={exit_code}\n")
    except Exception as exc:
        exit_code = 1
        finished_at = _now()
        with log_path.open("a", encoding="utf-8", errors="replace") as log:
            log.write(f"[{finished_at}] failed to launch target={target}: {type(exc).__name__}: {exc}\n")

    state.update(
        {
            "status": "success" if exit_code == 0 else "failed",
            "finished_at": finished_at,
            "exit_code": exit_code,
        }
    )
    _write_state(state, root=root)
    _record_refresh_state("finished", target=target, exit_code=exit_code, root=root)
