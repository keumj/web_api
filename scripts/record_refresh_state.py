from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services import refresh_state


def main() -> int:
    parser = argparse.ArgumentParser(description="Record local data refresh state.")
    parser.add_argument("event", choices=("started", "finished", "checked"))
    parser.add_argument("--source", default="manual")
    parser.add_argument("--exit-code", type=int, default=0)
    args = parser.parse_args()

    state = refresh_state.record_state(args.event, source=args.source, exit_code=args.exit_code)
    notice = refresh_state.notice_state()
    print(
        "refresh_state "
        f"event={args.event} "
        f"status={state.get('status')} "
        f"severity={notice.get('notice', {}).get('severity')} "
        f"headline={notice.get('notice', {}).get('headline')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
