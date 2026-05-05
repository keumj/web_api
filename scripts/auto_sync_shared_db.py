from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = Path("data/sp500_shared_db/sp500_shared_prices.sqlite")


def _run(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=ROOT,
        check=check,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )


def _git_output(*args: str) -> str:
    return _run(["git", *args]).stdout.strip()


def _status_lines(*pathspec: str) -> list[str]:
    result = _run(["git", "status", "--porcelain", "--", *pathspec], check=False)
    return [line.rstrip() for line in result.stdout.splitlines() if line.strip()]


def _changed_paths(lines: list[str]) -> list[str]:
    paths: list[str] = []
    for line in lines:
        if len(line) < 4:
            continue
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1].strip()
        paths.append(path.replace("\\", "/"))
    return paths


def _print(msg: str) -> None:
    print(f"[auto-sync] {msg}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Commit and push only the shared SQLite refresh artifact.")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--branch", default="")
    parser.add_argument("--message", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()

    target_abs = ROOT / TARGET
    if not target_abs.exists():
        _print(f"target file is missing: {TARGET.as_posix()}")
        return 1

    try:
        inside = _git_output("rev-parse", "--is-inside-work-tree")
    except subprocess.CalledProcessError as exc:
        _print(f"git repository check failed: {exc.stderr.strip() or exc}")
        return 1
    if inside.lower() != "true":
        _print("current directory is not a git repository")
        return 1

    branch = args.branch or _git_output("branch", "--show-current")
    if not branch:
        _print("could not determine current branch")
        return 1

    staged_status = _status_lines()
    staged_paths = _changed_paths([line for line in staged_status if line[:2] != "??"])
    unrelated_staged = [path for path in staged_paths if path != TARGET.as_posix()]
    if unrelated_staged:
        _print("refusing to auto-sync because other tracked changes are present in the worktree:")
        for path in unrelated_staged:
            _print(f"  - {path}")
        _print("commit or stash those changes first, or run the sync manually.")
        return 2

    target_status = _status_lines(TARGET.as_posix())
    if not target_status:
        _print(f"no changes detected for {TARGET.as_posix()}")
        return 0

    message = args.message or f"Auto-refresh shared market data ({datetime.now().strftime('%Y-%m-%d')})"
    _print(f"target change detected: {TARGET.as_posix()}")
    _print(f"branch={branch} remote={args.remote}")
    _print(f"commit message: {message}")

    if args.dry_run:
        _print("dry run only; skipping git add/commit/push")
        return 0

    try:
        _run(["git", "add", "--", TARGET.as_posix()])
        _run(["git", "commit", "--only", "-m", message, "--", TARGET.as_posix()])
        _print("commit created successfully")
        if not args.no_push:
            _run(["git", "push", args.remote, branch])
            _print("push completed successfully")
        else:
            _print("push skipped by --no-push")
        return 0
    except subprocess.CalledProcessError as exc:
        _print(exc.stderr.strip() or exc.stdout.strip() or str(exc))
        return exc.returncode or 1


if __name__ == "__main__":
    raise SystemExit(main())
