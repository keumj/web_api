from __future__ import annotations

import sys

from refresh_turso_daily import main as refresh_main


DEFAULT_ARGS = [
    "--mode",
    "direct",
    "--target",
    "all",
    "--overlap-days",
    "7",
    "--news-backfill-days",
    "32",
    "--news-max-items",
    "30",
    "--news-timeout",
    "8",
    "--news-retention-days",
    "60",
]


def main(argv: list[str] | None = None) -> int:
    # Keep Render/manual defaults even when a caller overrides only --target.
    # argparse uses the last repeated option, so caller-provided values win.
    return refresh_main([*DEFAULT_ARGS, *(argv or [])])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or None))
