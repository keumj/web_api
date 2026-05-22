from __future__ import annotations

import sys

from refresh_turso_daily import main as refresh_main


DEFAULT_ARGS = [
    "--mode",
    "upload-local",
    "--target",
    "all",
    "--overlap-days",
    "7",
    "--news-backfill-days",
    "14",
    "--news-max-items",
    "8",
    "--news-timeout",
    "5",
]


def main(argv: list[str] | None = None) -> int:
    # Keep Render/manual defaults even when a caller overrides only --target.
    # argparse uses the last repeated option, so caller-provided values win.
    return refresh_main([*DEFAULT_ARGS, *(argv or [])])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or None))
