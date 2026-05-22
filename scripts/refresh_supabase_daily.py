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
    "14",
    "--news-max-items",
    "8",
    "--news-timeout",
    "8",
]


def main(argv: list[str] | None = None) -> int:
    return refresh_main(DEFAULT_ARGS if argv is None else argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or None))
