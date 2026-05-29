from __future__ import annotations

import argparse
import os
import signal
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .notebook_data import load_sp500_components
from .security import configure_ssl, security_hint
from .shared_sp500_prices_sql import shared_prices_sqlite_path, upsert_shared_quarterly_fundamentals

try:
    import yfinance as yf

    try:
        from yfinance import set_tz_cache_location
    except Exception:  # pragma: no cover - optional helper
        set_tz_cache_location = None
except Exception:  # pragma: no cover - dependency optional at runtime
    yf = None
    set_tz_cache_location = None

try:
    from curl_cffi import requests as curl_requests
except Exception:  # pragma: no cover - optional dependency
    curl_requests = None

import requests


_YF_CACHE_DIR = Path(os.getenv("KEUMJ_YFINANCE_CACHE_DIR", "data/.yfinance_cache"))
_CANCEL_REQUESTED = False


def _log(message: str) -> None:
    print(f"[refresh-quarterly-fundamentals] {message}", flush=True)


def _handle_sigint(_signum: int, _frame: object) -> None:
    global _CANCEL_REQUESTED
    _CANCEL_REQUESTED = True
    raise KeyboardInterrupt


def _raise_if_cancelled() -> None:
    if _CANCEL_REQUESTED:
        raise KeyboardInterrupt


def _interruptible_sleep(seconds: float) -> None:
    remaining = max(float(seconds), 0.0)
    while remaining > 0:
        _raise_if_cancelled()
        chunk = min(remaining, 0.2)
        time.sleep(chunk)
        remaining -= chunk
    _raise_if_cancelled()


def _normalize_symbol(value: object) -> str:
    return str(value or "").strip().upper()


def _provider_symbol(symbol: str) -> str:
    return _normalize_symbol(symbol).replace(".", "-")


def _build_yfinance_session(*, ca_bundle: str | None = None, insecure_ssl: bool = False):
    if curl_requests is not None:
        session = curl_requests.Session()
    else:
        session = requests.Session()
        session.trust_env = True
    if ca_bundle:
        session.verify = ca_bundle
    elif insecure_ssl:
        session.verify = False
    return session


def _to_number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        out = float(value)
        return out if np.isfinite(out) else None
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"none", "nan", "na", "n/a", "-"}:
        return None
    try:
        out = float(text)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def _normalize_statement_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    try:
        out.columns = [pd.to_datetime(col) for col in out.columns]
        out = out.sort_index(axis=1, ascending=False)
    except Exception:
        return pd.DataFrame()
    return out


def _call_statement_method(
    ticker_obj: object,
    *method_names: str,
    freq: str = "quarterly",
) -> pd.DataFrame:
    for method_name in method_names:
        fn = getattr(ticker_obj, method_name, None)
        if fn is None or not callable(fn):
            continue
        try:
            raw = fn(freq=freq)
        except TypeError:
            try:
                raw = fn()
            except Exception:
                continue
        except Exception:
            continue
        frame = _normalize_statement_df(raw)
        if not frame.empty and (freq == "quarterly" or _statement_looks_quarterly(frame)):
            return frame
    return pd.DataFrame()


def _statement_looks_quarterly(df: pd.DataFrame) -> bool:
    if df is None or df.empty or len(df.columns) < 2:
        return False
    try:
        first = pd.to_datetime(df.columns[0])
        second = pd.to_datetime(df.columns[1])
    except Exception:
        return False
    return abs((first - second).days) <= 120


def _load_statement(ticker_obj: object, *attr_names: str) -> pd.DataFrame:
    for attr_name in attr_names:
        try:
            raw = getattr(ticker_obj, attr_name, pd.DataFrame())
        except Exception:
            continue
        frame = _normalize_statement_df(raw)
        if not frame.empty and (attr_name.startswith("quarterly_") or _statement_looks_quarterly(frame)):
            return frame
    return pd.DataFrame()


def _load_sec_statement_frames_local(
    *,
    ticker: str,
    ca_bundle: str | None = None,
    insecure_ssl: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        from pipeline_stock.web_gui import _fetch_sec_statement_frames, _request_verify_value
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    try:
        verify = _request_verify_value(ca_bundle_path=ca_bundle, insecure_ssl=insecure_ssl)
        income, balance, cashflow, _ = _fetch_sec_statement_frames(ticker=ticker, verify=verify)
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    return _normalize_statement_df(income), _normalize_statement_df(balance), _normalize_statement_df(cashflow)


def _pick_column_value(df: pd.DataFrame, candidates: list[str], column: pd.Timestamp) -> float | None:
    if df is None or df.empty or column not in df.columns:
        return None
    for candidate in candidates:
        if candidate in df.index:
            return _to_number(df.loc[candidate, column])
    return None


def _extract_quarterly_rows(
    symbol: str,
    *,
    limit_per_symbol: int,
    ca_bundle: str | None = None,
    insecure_ssl: bool = False,
) -> list[dict[str, object]]:
    if yf is None:
        raise RuntimeError("yfinance is not installed")
    provider_symbol = _provider_symbol(symbol)
    ticker_obj = yf.Ticker(provider_symbol, session=_build_yfinance_session(ca_bundle=ca_bundle, insecure_ssl=insecure_ssl))
    income = _load_statement(ticker_obj, "quarterly_income_stmt", "quarterly_incomestmt", "quarterly_financials", "income_stmt", "incomestmt", "financials")
    if income.empty:
        income = _call_statement_method(ticker_obj, "get_income_stmt", "get_incomestmt", "get_financials", freq="quarterly")
    balance = _load_statement(ticker_obj, "quarterly_balance_sheet", "quarterly_balancesheet", "balance_sheet", "balancesheet")
    if balance.empty:
        balance = _call_statement_method(ticker_obj, "get_balance_sheet", "get_balancesheet", freq="quarterly")
    cashflow = _load_statement(ticker_obj, "quarterly_cash_flow", "quarterly_cashflow", "cash_flow", "cashflow")
    if cashflow.empty:
        cashflow = _call_statement_method(ticker_obj, "get_cash_flow", "get_cashflow", freq="quarterly")
    source_prefix = "yfinance"
    if income.empty and balance.empty and cashflow.empty:
        income, balance, cashflow = _load_sec_statement_frames_local(
            ticker=symbol,
            ca_bundle=ca_bundle,
            insecure_ssl=insecure_ssl,
        )
        source_prefix = "sec_quarterized" if not (income.empty and balance.empty and cashflow.empty) else "yfinance"
    if income.empty and balance.empty and cashflow.empty:
        return []
    all_dates: list[pd.Timestamp] = []
    for frame in [income, balance, cashflow]:
        if frame is None or frame.empty:
            continue
        all_dates.extend([pd.Timestamp(col).normalize() for col in frame.columns[: max(int(limit_per_symbol), 1)]])
    if not all_dates:
        return []
    fiscal_dates = sorted({dt for dt in all_dates if pd.notna(dt)}, reverse=True)[: max(int(limit_per_symbol), 1)]
    rows: list[dict[str, object]] = []
    for fiscal_date in fiscal_dates:
        rows.append(
            {
                "symbol": _normalize_symbol(symbol),
                "fiscal_date": fiscal_date.strftime("%Y-%m-%d"),
                "filing_date": None,
                "period_type": "quarterly",
                "net_income": _pick_column_value(income, ["Net Income"], fiscal_date),
                "diluted_eps": _pick_column_value(income, ["Diluted EPS", "Basic EPS"], fiscal_date),
                "stockholders_equity": _pick_column_value(
                    balance,
                    ["Stockholders Equity", "Total Stockholder Equity"],
                    fiscal_date,
                ),
                "total_assets": _pick_column_value(balance, ["Total Assets"], fiscal_date),
                "total_debt": _pick_column_value(balance, ["Total Debt", "Long Term Debt"], fiscal_date),
                "current_assets": _pick_column_value(balance, ["Current Assets", "Total Current Assets"], fiscal_date),
                "current_liabilities": _pick_column_value(
                    balance,
                    ["Current Liabilities", "Total Current Liabilities", "Current Debt And Capital Lease Obligation"],
                    fiscal_date,
                ),
                "operating_cash_flow": _pick_column_value(
                    cashflow,
                    ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"],
                    fiscal_date,
                ),
                "free_cash_flow": _pick_column_value(cashflow, ["Free Cash Flow"], fiscal_date),
                "source": f"{source_prefix}:{provider_symbol}",
            }
        )
    return rows


def _resolve_symbols(max_symbols: int, explicit_symbols: str) -> list[str]:
    if str(explicit_symbols).strip():
        return [_normalize_symbol(item) for item in str(explicit_symbols).split(",") if _normalize_symbol(item)]
    requested = int(max_symbols)
    load_count = requested if requested > 0 else 10_000
    frame, _ = load_sp500_components(max_symbols=load_count)
    return [_normalize_symbol(symbol) for symbol in frame["Symbol"].astype(str).tolist() if _normalize_symbol(symbol)]


def _load_latest_fiscal_dates(db_path: Path, symbols: list[str]) -> dict[str, str]:
    normalized = [_normalize_symbol(symbol) for symbol in symbols if _normalize_symbol(symbol)]
    if not normalized or not db_path.exists() or not db_path.is_file():
        return {}
    placeholders = ",".join("?" for _ in normalized)
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT symbol, MAX(fiscal_date)
            FROM fundamentals_quarterly
            WHERE symbol IN ({placeholders})
            GROUP BY symbol
            """,
            normalized,
        ).fetchall()
    return {str(symbol).upper(): str(fiscal_date) for symbol, fiscal_date in rows if symbol and fiscal_date}


def _filter_rows_newer_than_db(
    rows: list[dict[str, object]],
    latest_db_fiscal_date: str | None,
) -> tuple[list[dict[str, object]], str | None]:
    if not rows:
        return [], None
    latest_source_fiscal_date = max(str(row["fiscal_date"]) for row in rows if row.get("fiscal_date"))
    if not latest_db_fiscal_date:
        return list(rows), latest_source_fiscal_date
    latest_db_ts = pd.Timestamp(latest_db_fiscal_date).normalize()
    new_rows = [
        row
        for row in rows
        if row.get("fiscal_date") and pd.Timestamp(str(row["fiscal_date"])).normalize() > latest_db_ts
    ]
    return new_rows, latest_source_fiscal_date


def _has_recent_quarterly_data(
    latest_fiscal_date: str | None,
    *,
    refresh_after_days: int,
    as_of_date: str | pd.Timestamp | None = None,
) -> bool:
    if not latest_fiscal_date:
        return False
    latest_ts = pd.Timestamp(latest_fiscal_date).normalize()
    current_ts = pd.Timestamp(as_of_date).normalize() if as_of_date is not None else pd.Timestamp.today().normalize()
    age_days = int((current_ts - latest_ts).days)
    return age_days >= 0 and age_days <= max(int(refresh_after_days), 1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill the latest quarterly fundamentals into the shared S&P 500 SQLite DB.")
    parser.add_argument("--db-path", default="", help="Optional shared SQLite path override")
    parser.add_argument("--symbols", default="", help="Comma-separated symbol list override")
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=0,
        help="Maximum S&P 500 symbols to refresh. Use 0 to process the full component file.",
    )
    parser.add_argument("--limit-per-symbol", type=int, default=4, help="Quarter rows to store per symbol")
    parser.add_argument("--pause-seconds", type=float, default=0.15, help="Delay between symbol requests")
    parser.add_argument("--insecure-ssl", action="store_true", help="Disable TLS verification for temporary testing")
    parser.add_argument("--ca-bundle", default="", help="Custom CA bundle path")
    parser.add_argument(
        "--refresh-after-days",
        type=int,
        default=120,
        help="Skip symbols whose latest fiscal_date is within this many days",
    )
    parser.add_argument("--force-refresh", action="store_true", help="Refresh even if a recent quarter already exists")
    return parser.parse_args()


def main() -> int:
    global _CANCEL_REQUESTED
    args = _parse_args()
    if yf is None:
        raise SystemExit("yfinance is not installed. Install yfinance to refresh quarterly fundamentals.")
    _CANCEL_REQUESTED = False
    previous_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _handle_sigint)
    insecure_ssl = bool(getattr(args, "insecure_ssl", False))
    ca_bundle = str(getattr(args, "ca_bundle", "")).strip() or None
    try:
        configure_ssl(insecure_ssl=insecure_ssl, ca_bundle=ca_bundle)
        if set_tz_cache_location is not None:
            _YF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            try:
                set_tz_cache_location(str(_YF_CACHE_DIR))
            except Exception:
                pass
        db_path = Path(args.db_path) if str(args.db_path).strip() else shared_prices_sqlite_path()
        symbols = _resolve_symbols(args.max_symbols, args.symbols)
        total_symbols = len(symbols)
        stored_rows = 0
        success_count = 0
        failure_count = 0
        no_data_count = 0
        skipped_count = 0
        latest_fiscal_dates = _load_latest_fiscal_dates(db_path, symbols)
        _log(f"Starting quarterly fundamentals refresh for {total_symbols} symbols -> db={db_path}")
        for index, symbol in enumerate(symbols, start=1):
            _raise_if_cancelled()
            latest_fiscal_date = latest_fiscal_dates.get(symbol)
            try:
                rows = _extract_quarterly_rows(
                    symbol,
                    limit_per_symbol=args.limit_per_symbol,
                    ca_bundle=ca_bundle,
                    insecure_ssl=insecure_ssl,
                )
                if rows:
                    rows_to_store = rows
                    latest_source_fiscal_date = max(str(row["fiscal_date"]) for row in rows if row.get("fiscal_date"))
                    if not args.force_refresh:
                        rows_to_store, latest_source_fiscal_date = _filter_rows_newer_than_db(rows, latest_fiscal_date)
                    if not rows_to_store:
                        skipped_count += 1
                        _log(
                            f"{index}/{total_symbols} {symbol}: up-to-date "
                            f"(db_latest={latest_fiscal_date}, source_latest={latest_source_fiscal_date})"
                        )
                        continue
                    stored_rows += upsert_shared_quarterly_fundamentals(
                        pd.DataFrame(rows_to_store),
                        db_path=db_path,
                        source=rows_to_store[0]["source"],
                    )
                    success_count += 1
                else:
                    no_data_count += 1
                    _log(f"{index}/{total_symbols} {symbol}: no quarterly statement rows")
            except Exception as exc:
                failure_count += 1
                _log(f"{index}/{total_symbols} {symbol}: failed ({type(exc).__name__}: {exc})")
            else:
                if rows:
                    _log(
                        f"{index}/{total_symbols} {symbol}: stored "
                        f"{len(rows_to_store)} quarterly rows (db_latest={latest_fiscal_date}, source_latest={latest_source_fiscal_date})"
                    )
            if index < total_symbols and float(args.pause_seconds) > 0:
                _interruptible_sleep(float(args.pause_seconds))
        _log(
            "Completed quarterly fundamentals refresh "
            f"(symbols={total_symbols}, succeeded={success_count}, skipped={skipped_count}, no_data={no_data_count}, failed={failure_count}, rows_changed={stored_rows})"
        )
        if success_count == 0 and skipped_count < total_symbols and (failure_count > 0 or no_data_count > 0):
            hint = security_hint(RuntimeError("yfinance download failed"), output_dir=db_path.parent)
            if hint:
                print(hint, file=sys.stderr)
            print(
                "Quarterly fundamentals refresh finished without loading any new statement rows.",
                file=sys.stderr,
            )
            return 2
        return 0
    except KeyboardInterrupt:
        _log("Cancelled by user (Ctrl+C).")
        return 130
    finally:
        signal.signal(signal.SIGINT, previous_handler)


if __name__ == "__main__":
    raise SystemExit(main())
