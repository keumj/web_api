from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def app_root_dir() -> Path:
    if getattr(sys, "frozen", False):
        exe_root = Path(sys.executable).resolve().parent
        if _has_runtime_data(exe_root):
            return exe_root
        internal_root = exe_root / "_internal"
        if _has_runtime_data(internal_root):
            return internal_root
        return exe_root
    return Path(__file__).resolve().parents[1]


def _has_runtime_data(root: Path) -> bool:
    data_dir = root / "data"
    return (
        (data_dir / "sp500_components_full.csv").is_file()
        and (data_dir / "sp500_shared_db" / "sp500_shared_prices.sqlite").is_file()
    )


def _activate_app_root() -> Path:
    root = app_root_dir()
    os.chdir(root)
    return root


def _run_gui(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Launch the Keumj unified stock GUI.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8512)
    parser.add_argument("--ca-bundle", default="")
    parser.add_argument("--insecure-ssl", action="store_true")
    parser.add_argument("--require-live-data", action="store_true")
    parser.add_argument("legacy_args", nargs="*")
    args = parser.parse_args(argv)

    host = str(args.host)
    port = int(args.port)
    ca_bundle = str(args.ca_bundle).strip()
    require_live_data = bool(args.require_live_data)

    if args.legacy_args:
        if len(args.legacy_args) >= 1 and str(args.legacy_args[0]).strip():
            port = int(args.legacy_args[0])
        if len(args.legacy_args) >= 2 and str(args.legacy_args[1]).strip():
            host = str(args.legacy_args[1]).strip()
        if len(args.legacy_args) >= 3 and str(args.legacy_args[2]).strip():
            ca_bundle = str(args.legacy_args[2]).strip()
        if len(args.legacy_args) >= 4 and str(args.legacy_args[3]).strip().lower() in {"1", "true", "yes", "on"}:
            require_live_data = True

    from pipeline_common.security import configure_ssl
    from pipeline_stock.web_gui import launch_web_gui

    configure_ssl(
        insecure_ssl=bool(args.insecure_ssl),
        ca_bundle=ca_bundle or None,
    )
    if require_live_data:
        os.environ["KEUMJ_REQUIRE_LIVE_DATA"] = "1"

    print(f"Working directory: {Path.cwd()}", flush=True)
    print(f"Starting Equity Analysis Lab | S&P500 at http://{host}:{port}", flush=True)
    launch_web_gui(host=host, port=port)
    return 0


def _run_refresh(argv: list[str]) -> int:
    from pipeline_common.refresh_sp500_shared_prices import main as refresh_main

    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0], *argv]
        return int(refresh_main())
    finally:
        sys.argv = old_argv


def _print_help() -> None:
    print(
        "\n".join(
            [
                "KeumjStockLab",
                "",
                "Commands:",
                "  gui [--host localhost] [--port 8512]    Launch the unified stock GUI.",
                "  refresh [refresh options]                Refresh S&P 500 shared data.",
                "",
                "Examples:",
                "  KeumjStockLab.exe gui",
                "  KeumjStockLab.exe refresh --provider yfinance",
            ]
        )
    )


def main(argv: list[str] | None = None) -> int:
    _activate_app_root()
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        return _run_gui([])

    command = args[0].strip().lower()
    if command in {"-h", "--help", "help"}:
        _print_help()
        return 0
    if command in {"gui", "run", "web"}:
        return _run_gui(args[1:])
    if command in {"refresh", "refresh-stock", "refresh_stock_data"}:
        return _run_refresh(args[1:])
    if command.startswith("-"):
        return _run_gui(args)

    print(f"Unknown command: {args[0]}", file=sys.stderr)
    _print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
