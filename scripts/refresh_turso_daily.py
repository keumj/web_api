from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline_common.refresh_sp500_shared_prices import (  # noqa: E402
    DEFAULT_COMPONENTS_CSV,
    DEFAULT_START_DATE,
    _download_symbol_frames,
    _read_symbol_list,
)
from pipeline_common.shared_sp500_prices_sql import (  # noqa: E402
    _connect_shared_prices_read_db,
    shared_prices_database_url,
    shared_prices_sqlite_path,
)
from pipeline_macro.macro_data_store import (  # noqa: E402
    ALL_SPECS,
    MacroSeriesSpec,
    _connect_macro_read_db,
    macro_database_url,
    macro_db_path,
    normalize_series,
)
from pipeline_macro.refresh_macro_prices import _load_series  # noqa: E402

try:
    from fredapi import Fred
except Exception:  # pragma: no cover - optional deployment dependency
    Fred = None


@dataclass(frozen=True)
class JobCounts:
    price_rows: int = 0
    macro_rows: int = 0


def _log(message: str) -> None:
    print(f"[refresh-turso-daily] {message}", flush=True)


def _env_first(*names: str) -> str:
    for name in names:
        value = str(os.getenv(name, "")).strip()
        if value:
            return value
    return ""


def _fred_client():
    key = _env_first("FRED_API_KEY")
    if Fred is None or not key:
        return None
    return Fred(api_key=key)


def _ensure_remote_prices_schema(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prices (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL DEFAULT 0,
            adj_close REAL,
            dividends REAL NOT NULL DEFAULT 0,
            stock_splits REAL NOT NULL DEFAULT 0,
            market_cap REAL,
            PRIMARY KEY (symbol, date)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prices_symbol_date ON prices(symbol, date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS refresh_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_name TEXT NOT NULL,
            mode TEXT NOT NULL,
            status TEXT NOT NULL,
            started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            finished_at TEXT,
            price_rows INTEGER NOT NULL DEFAULT 0,
            macro_rows INTEGER NOT NULL DEFAULT 0,
            message TEXT NOT NULL DEFAULT ''
        )
        """
    )
    conn.commit()


def _ensure_remote_macro_schema(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS macro_series (
            series_id TEXT NOT NULL,
            date TEXT NOT NULL,
            value REAL NOT NULL,
            dataset TEXT NOT NULL,
            frequency TEXT NOT NULL,
            source TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (series_id, date)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_macro_series_date ON macro_series(date)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS macro_metadata (
            series_id TEXT PRIMARY KEY,
            dataset TEXT NOT NULL,
            frequency TEXT NOT NULL,
            source TEXT NOT NULL,
            min_date TEXT,
            max_date TEXT,
            row_count INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS refresh_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_name TEXT NOT NULL,
            mode TEXT NOT NULL,
            status TEXT NOT NULL,
            started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            finished_at TEXT,
            price_rows INTEGER NOT NULL DEFAULT 0,
            macro_rows INTEGER NOT NULL DEFAULT 0,
            message TEXT NOT NULL DEFAULT ''
        )
        """
    )
    conn.commit()


def _table_max_date(conn, table: str, *, where_sql: str = "", params: tuple[object, ...] = ()) -> str | None:
    query = f"SELECT MAX(date) FROM {table}"
    if where_sql:
        query += f" WHERE {where_sql}"
    row = conn.execute(query, params).fetchone()
    value = row[0] if row else None
    return str(value) if value else None


def _overlap_fetch_start_date(default_start_date: str, max_date: str | None, overlap_days: int) -> str:
    if not max_date:
        return pd.Timestamp(default_start_date).normalize().strftime("%Y-%m-%d")
    try:
        overlapped = pd.Timestamp(max_date).normalize() - pd.Timedelta(days=max(int(overlap_days), 0))
    except Exception:
        return pd.Timestamp(default_start_date).normalize().strftime("%Y-%m-%d")
    return max(pd.Timestamp(default_start_date).normalize(), overlapped).strftime("%Y-%m-%d")


def _price_rows_from_frames(frames: dict[str, pd.DataFrame], *, only_after: str | None) -> list[tuple[object, ...]]:
    cutoff = pd.Timestamp(only_after).normalize() if only_after else None
    rows: list[tuple[object, ...]] = []
    for symbol, frame in frames.items():
        if frame is None or frame.empty:
            continue
        clean = frame.copy()
        clean.index = pd.to_datetime(clean.index, errors="coerce").normalize()
        clean = clean[~clean.index.isna()].sort_index()
        if cutoff is not None:
            clean = clean[clean.index > cutoff]
        required = ["open", "high", "low", "close"]
        if clean.empty or any(col not in clean.columns for col in required):
            continue
        if "volume" not in clean.columns:
            clean["volume"] = 0.0
        if "adj_close" not in clean.columns:
            clean["adj_close"] = clean["close"]
        if "dividends" not in clean.columns:
            clean["dividends"] = 0.0
        if "stock_splits" not in clean.columns:
            clean["stock_splits"] = 0.0
        for idx, row in clean.iterrows():
            values = [row.get(col) for col in required]
            if any(pd.isna(value) for value in values):
                continue
            rows.append(
                (
                    str(symbol).strip().upper(),
                    pd.Timestamp(idx).strftime("%Y-%m-%d"),
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    float(row.get("volume", 0.0) or 0.0),
                    float(row["adj_close"]) if pd.notna(row.get("adj_close")) else float(row["close"]),
                    float(row.get("dividends", 0.0) or 0.0),
                    float(row.get("stock_splits", 0.0) or 0.0),
                )
            )
    return rows


def _insert_price_rows(conn, rows: list[tuple[object, ...]], *, chunk_size: int = 1000) -> int:
    if not rows:
        return 0
    sql = """
        INSERT OR IGNORE INTO prices
        (symbol, date, open, high, low, close, volume, adj_close, dividends, stock_splits)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    for i in range(0, len(rows), max(int(chunk_size), 1)):
        conn.executemany(sql, rows[i : i + max(int(chunk_size), 1)])
    conn.commit()
    return len(rows)


def refresh_prices_direct(
    *,
    symbols_csv: Path,
    start_date: str,
    chunk_size: int,
    provider: str,
    pause_seconds: float,
    overlap_days: int,
) -> int:
    if not shared_prices_database_url():
        raise RuntimeError("KEUMJ_SP500_DATABASE_URL must be set for direct Turso refresh.")
    with _connect_shared_prices_read_db() as conn:
        _ensure_remote_prices_schema(conn)
        old_max_date = _table_max_date(conn, "prices")
        fetch_start = _overlap_fetch_start_date(start_date, old_max_date, overlap_days)
        end_date = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        symbols = _read_symbol_list(symbols_csv, Path("__remote_turso__.sqlite"))
        _log(
            f"S&P500 direct refresh: symbols={len(symbols)} old_max={old_max_date or '-'} "
            f"fetch_start={fetch_start} end_exclusive={end_date}"
        )
        if pd.Timestamp(fetch_start).normalize() >= pd.Timestamp(end_date).normalize():
            _log("S&P500 direct refresh skipped: no new date window.")
            return 0
        frames, missing = _download_symbol_frames(
            symbols,
            start_date=fetch_start,
            end_date=end_date,
            chunk_size=chunk_size,
            provider=provider,
            pause_seconds=pause_seconds,
        )
        if missing:
            _log(f"S&P500 missing symbols: {len(missing)}")
        rows = _price_rows_from_frames(
            frames,
            only_after=(pd.Timestamp(fetch_start).normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        )
        attempted = _insert_price_rows(conn, rows)
        _log(f"S&P500 direct refresh wrote candidate rows={attempted}")
        return attempted


def _macro_series_rows(spec: MacroSeriesSpec, series: pd.Series, source: str) -> list[tuple[object, ...]]:
    clean = normalize_series(series, series_id=spec.series_id)
    return [
        (
            spec.series_id,
            pd.Timestamp(idx).strftime("%Y-%m-%d"),
            float(value),
            spec.dataset,
            spec.frequency,
            source,
        )
        for idx, value in clean.items()
    ]


def _upsert_macro_rows(conn, spec: MacroSeriesSpec, rows: list[tuple[object, ...]], source: str) -> int:
    if not rows:
        return 0
    conn.executemany(
        """
        INSERT INTO macro_series (series_id, date, value, dataset, frequency, source, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(series_id, date) DO UPDATE SET
            value=excluded.value,
            dataset=excluded.dataset,
            frequency=excluded.frequency,
            source=excluded.source,
            updated_at=CURRENT_TIMESTAMP
        """,
        rows,
    )
    conn.execute(
        """
        INSERT INTO macro_metadata (series_id, dataset, frequency, source, min_date, max_date, row_count, updated_at)
        VALUES (
            ?, ?, ?, ?,
            (SELECT MIN(date) FROM macro_series WHERE series_id = ?),
            (SELECT MAX(date) FROM macro_series WHERE series_id = ?),
            (SELECT COUNT(*) FROM macro_series WHERE series_id = ?),
            CURRENT_TIMESTAMP
        )
        ON CONFLICT(series_id) DO UPDATE SET
            dataset=excluded.dataset,
            frequency=excluded.frequency,
            source=excluded.source,
            min_date=excluded.min_date,
            max_date=excluded.max_date,
            row_count=excluded.row_count,
            updated_at=CURRENT_TIMESTAMP
        """,
        (spec.series_id, spec.dataset, spec.frequency, source, spec.series_id, spec.series_id, spec.series_id),
    )
    conn.commit()
    return len(rows)


def refresh_macro_direct(*, default_years: int) -> int:
    if not macro_database_url():
        raise RuntimeError("KEUMJ_MACRO_DATABASE_URL must be set for direct Turso refresh.")
    fred = _fred_client()
    total = 0
    with _connect_macro_read_db() as conn:
        _ensure_remote_macro_schema(conn)
        for spec in ALL_SPECS:
            old_max = _table_max_date(conn, "macro_series", where_sql="series_id = ?", params=(spec.series_id,))
            if old_max:
                start = pd.Timestamp(old_max).normalize() + pd.Timedelta(days=1)
            else:
                start = (pd.Timestamp.today().normalize() - pd.DateOffset(years=int(default_years))).normalize()
            series, source = _load_series(spec, fred=fred, start=start)
            if series is None or series.empty or source is None:
                _log(f"macro {spec.series_id}: no new data")
                continue
            if old_max:
                series = series[series.index > pd.Timestamp(old_max).normalize()]
            rows = _macro_series_rows(spec, series, source)
            count = _upsert_macro_rows(conn, spec, rows, source)
            total += count
            _log(f"macro {spec.series_id}: candidate rows={count} source={source}")
    _log(f"macro direct refresh wrote candidate rows={total}")
    return total


def upload_local_prices_to_turso(*, local_db: Path, overlap_days: int) -> int:
    if not shared_prices_database_url():
        raise RuntimeError("KEUMJ_SP500_DATABASE_URL must be set for local upload.")
    if not local_db.is_file():
        raise FileNotFoundError(f"Local S&P500 SQLite DB not found: {local_db}")
    with _connect_shared_prices_read_db() as remote, sqlite3.connect(local_db) as local:
        _ensure_remote_prices_schema(remote)
        old_max = _table_max_date(remote, "prices")
        query = """
            SELECT symbol, date, open, high, low, close, volume, adj_close, dividends, stock_splits
            FROM prices
        """
        params: list[object] = []
        if old_max:
            cutoff = _overlap_fetch_start_date("1900-01-01", old_max, overlap_days)
            query += " WHERE date >= ?"
            params.append(cutoff)
        query += " ORDER BY date, symbol"
        rows = [tuple(row) for row in local.execute(query, params).fetchall()]
        attempted = _insert_price_rows(remote, rows)
        _log(f"local S&P500 upload old_max={old_max or '-'} candidate rows={attempted}")
        return attempted


def upload_local_macro_to_turso(*, local_db: Path) -> int:
    if not macro_database_url():
        raise RuntimeError("KEUMJ_MACRO_DATABASE_URL must be set for local upload.")
    if not local_db.is_file():
        raise FileNotFoundError(f"Local macro SQLite DB not found: {local_db}")
    total = 0
    with _connect_macro_read_db() as remote, sqlite3.connect(local_db) as local:
        _ensure_remote_macro_schema(remote)
        for spec in ALL_SPECS:
            old_max = _table_max_date(remote, "macro_series", where_sql="series_id = ?", params=(spec.series_id,))
            query = """
                SELECT series_id, date, value, dataset, frequency, source
                FROM macro_series
                WHERE series_id = ?
            """
            params: list[object] = [spec.series_id]
            if old_max:
                query += " AND date > ?"
                params.append(old_max)
            query += " ORDER BY date"
            rows = [tuple(row) for row in local.execute(query, params).fetchall()]
            source = str(rows[-1][5]) if rows else "local_sqlite"
            total += _upsert_macro_rows(remote, spec, rows, source)
        _log(f"local macro upload candidate rows={total}")
        return total


def _record_run(status: str, mode: str, counts: JobCounts, message: str) -> None:
    targets = []
    if shared_prices_database_url():
        targets.append(("sp500", _connect_shared_prices_read_db, _ensure_remote_prices_schema))
    if macro_database_url():
        targets.append(("macro", _connect_macro_read_db, _ensure_remote_macro_schema))
    for name, connect, ensure in targets:
        try:
            with connect() as conn:
                ensure(conn)
                conn.execute(
                    """
                    INSERT INTO refresh_runs(job_name, mode, status, finished_at, price_rows, macro_rows, message)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?)
                    """,
                    (name, mode, status, int(counts.price_rows), int(counts.macro_rows), message[:1000]),
                )
                conn.commit()
        except Exception as exc:
            _log(f"failed to record refresh run in {name}: {type(exc).__name__}: {exc}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Incrementally refresh Turso calculation DBs.")
    parser.add_argument("--mode", choices=["direct", "upload-local"], default="direct")
    parser.add_argument("--target", choices=["all", "sp500", "macro"], default="all")
    parser.add_argument("--symbols-csv", default=str(DEFAULT_COMPONENTS_CSV))
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--provider", choices=["auto", "yfinance", "fdr"], default="auto")
    parser.add_argument("--pause-seconds", type=float, default=0.0)
    parser.add_argument("--overlap-days", type=int, default=3, help="Re-read recent days to fill partial failed runs.")
    parser.add_argument("--macro-years", type=int, default=5)
    parser.add_argument("--local-sp500-db", default=str(shared_prices_sqlite_path()))
    parser.add_argument("--local-macro-db", default=str(macro_db_path()))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        price_rows = 0
        macro_rows = 0
        if args.mode == "direct":
            if args.target in {"all", "sp500"}:
                price_rows = refresh_prices_direct(
                    symbols_csv=Path(args.symbols_csv),
                    start_date=str(args.start_date),
                    chunk_size=int(args.chunk_size),
                    provider=str(args.provider),
                    pause_seconds=float(args.pause_seconds),
                    overlap_days=int(args.overlap_days),
                )
            if args.target in {"all", "macro"}:
                macro_rows = refresh_macro_direct(default_years=int(args.macro_years))
        else:
            if args.target in {"all", "sp500"}:
                price_rows = upload_local_prices_to_turso(
                    local_db=Path(args.local_sp500_db),
                    overlap_days=int(args.overlap_days),
                )
            if args.target in {"all", "macro"}:
                macro_rows = upload_local_macro_to_turso(local_db=Path(args.local_macro_db))
        counts = JobCounts(price_rows=price_rows, macro_rows=macro_rows)
        _record_run("ok", str(args.mode), counts, "completed")
        _log(f"SUMMARY status=ok mode={args.mode} price_rows={price_rows} macro_rows={macro_rows}")
        return 0
    except Exception as exc:
        counts = JobCounts()
        _record_run("error", str(args.mode), counts, f"{type(exc).__name__}: {exc}")
        _log(f"SUMMARY status=error mode={args.mode} error_type={type(exc).__name__}")
        print(f"Refresh failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
