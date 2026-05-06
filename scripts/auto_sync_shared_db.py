from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = [
    Path("data/sp500_shared_db/sp500_shared_prices.sqlite"),
    Path("data/macro_prices.sqlite"),
]


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


def _print(msg: str) -> None:
    print(f"[auto-sync] {msg}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Commit and push only refreshed SQLite artifacts.")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--branch", default="")
    parser.add_argument("--message", default="")
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="SQLite file to sync. May be repeated. Defaults to shared and macro SQLite files.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()

    targets = [Path(item) for item in args.target] if args.target else DEFAULT_TARGETS
    existing_targets = [target for target in targets if (ROOT / target).exists()]
    if not existing_targets:
        _print("no target SQLite files exist:")
        for target in targets:
            _print(f"  - {target.as_posix()}")
        return 1
    target_pathspecs = [target.as_posix() for target in existing_targets]

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

    target_status = _status_lines(*target_pathspecs)
    ignored_status = _run(["git", "status", "--porcelain", "--ignored", "--", *target_pathspecs], check=False)
    ignored_lines = [line.rstrip() for line in ignored_status.stdout.splitlines() if line.startswith("!! ")]
    if not target_status and not ignored_lines:
        _print("no changes detected for target SQLite files")
        return 0

    message = args.message or f"Auto-refresh SQLite data ({datetime.now().strftime('%Y-%m-%d')})"
    _print("target SQLite changes detected:")
    for target in target_pathspecs:
        _print(f"  - {target}")
    _print(f"branch={branch} remote={args.remote}")
    _print(f"commit message: {message}")

    if args.dry_run:
        _print("dry run only; skipping git add/commit/push")
        return 0

    try:
        _run(["git", "add", "--force", "--", *target_pathspecs])
        _run(["git", "commit", "--only", "-m", message, "--", *target_pathspecs])
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
