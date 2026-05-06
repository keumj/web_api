from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

try:
    from fredapi import Fred
except Exception:  # pragma: no cover - optional dependency
    Fred = None

from .macro_data_store import (
    ALL_SPECS,
    FRED_SPECS,
    LOCAL_ONLY_SPECS,
    MacroSeriesSpec,
    ensure_macro_schema,
    macro_db_path,
    normalize_series,
    read_local_series,
)


@dataclass(frozen=True)
class MacroRefreshResult:
    db_path: Path
    min_date: str
    max_date: str
    inserted_or_replaced_rows: int
    total_rows: int
    series_count: int
    size_bytes: int


def _log(message: str) -> None:
    print(f"[refresh-macro] {message}", flush=True)


def _start_date(years: int) -> pd.Timestamp:
    return (pd.Timestamp.today().normalize() - pd.DateOffset(years=int(years))).normalize()


def _fetch_fred_series(fred: Fred | None, spec: MacroSeriesSpec, start: pd.Timestamp) -> tuple[pd.Series | None, str | None]:
    if fred is None or not spec.fred_id:
        return None, None
    try:
        raw = fred.get_series(spec.fred_id, observation_start=start.strftime("%Y-%m-%d"))
    except Exception as exc:
        _log(f"FRED failed for {spec.series_id} ({spec.fred_id}): {type(exc).__name__}: {exc}")
        return None, None
    series = normalize_series(pd.Series(raw), series_id=spec.series_id, start_date=start)
    if series.empty:
        return None, None
    return series, f"fred:{spec.fred_id}"


def _load_series(spec: MacroSeriesSpec, *, fred: Fred | None, start: pd.Timestamp) -> tuple[pd.Series | None, str | None]:
    if spec.local_csv:
        local = read_local_series(spec.local_csv, series_id=spec.local_name or spec.series_id, start_date=start)
        if local is not None and not local.empty:
            local = local.rename(spec.series_id)
            if spec.fred_id is None:
                return local, f"local_csv:{spec.local_csv}"
            fred_series, fred_source = _fetch_fred_series(fred, spec, start)
            if fred_series is None or fred_series.empty:
                return local, f"local_csv:{spec.local_csv}"
            merged = pd.concat([local, fred_series]).sort_index()
            merged = merged[~merged.index.duplicated(keep="last")].rename(spec.series_id)
            return merged, f"local_csv:{spec.local_csv}+{fred_source}"
    return _fetch_fred_series(fred, spec, start)


def _upsert_series(conn: sqlite3.Connection, spec: MacroSeriesSpec, series: pd.Series, source: str) -> int:
    clean = normalize_series(series, series_id=spec.series_id)
    if clean.empty:
        return 0
    rows = [
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
        VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(series_id) DO UPDATE SET
            dataset=excluded.dataset,
            frequency=excluded.frequency,
            source=excluded.source,
            min_date=excluded.min_date,
            max_date=excluded.max_date,
            row_count=excluded.row_count,
            updated_at=CURRENT_TIMESTAMP
        """,
        (
            spec.series_id,
            spec.dataset,
            spec.frequency,
            source,
            pd.Timestamp(clean.index.min()).strftime("%Y-%m-%d"),
            pd.Timestamp(clean.index.max()).strftime("%Y-%m-%d"),
            int(len(clean)),
        ),
    )
    return len(rows)


def refresh_macro_prices(
    *,
    db_path: str | Path | None = None,
    years: int = 5,
    require_fred: bool = False,
) -> MacroRefreshResult:
    start = _start_date(years)
    target = macro_db_path(db_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fred_key = str(os.getenv("FRED_API_KEY", "")).strip()
    fred = Fred(api_key=fred_key) if Fred is not None and fred_key else None
    if require_fred and fred is None:
        raise RuntimeError("fredapi is not installed; cannot refresh required FRED macro series")

    changed = 0
    missing: list[str] = []
    with sqlite3.connect(target) as conn:
        ensure_macro_schema(conn)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("DELETE FROM macro_series WHERE date < ?", (start.strftime("%Y-%m-%d"),))

        for spec in ALL_SPECS:
            series, source = _load_series(spec, fred=fred, start=start)
            if series is None or series.empty or source is None:
                missing.append(spec.series_id)
                _log(f"missing {spec.series_id}")
                continue
            count = _upsert_series(conn, spec, series, source)
            changed += count
            _log(f"{spec.series_id}: rows={count} source={source}")

        for spec in ALL_SPECS:
            conn.execute(
                """
                UPDATE macro_metadata
                SET min_date=(SELECT MIN(date) FROM macro_series WHERE series_id=?),
                    max_date=(SELECT MAX(date) FROM macro_series WHERE series_id=?),
                    row_count=(SELECT COUNT(*) FROM macro_series WHERE series_id=?),
                    updated_at=CURRENT_TIMESTAMP
                WHERE series_id=?
                """,
                (spec.series_id, spec.series_id, spec.series_id, spec.series_id),
            )

        row = conn.execute("SELECT MIN(date), MAX(date), COUNT(*), COUNT(DISTINCT series_id) FROM macro_series").fetchone()
        conn.commit()
        conn.execute("VACUUM")

    total_rows = int(row[2] or 0) if row else 0
    series_count = int(row[3] or 0) if row else 0
    min_date = str(row[0] or "-") if row else "-"
    max_date = str(row[1] or "-") if row else "-"
    if require_fred:
        required = {spec.series_id for spec in FRED_SPECS}
        missing_required = sorted(required.intersection(missing))
        if missing_required:
            raise RuntimeError(f"Required FRED series missing: {', '.join(missing_required)}")
    if missing:
        _log(f"missing series skipped: {', '.join(missing)}")
    return MacroRefreshResult(
        db_path=target,
        min_date=min_date,
        max_date=max_date,
        inserted_or_replaced_rows=changed,
        total_rows=total_rows,
        series_count=series_count,
        size_bytes=target.stat().st_size if target.exists() else 0,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh five years of macro market/FRED data into data/macro_prices.sqlite.")
    parser.add_argument("--db-path", default="", help="Output SQLite path. Default: data/macro_prices.sqlite")
    parser.add_argument("--years", type=int, default=5, help="History window to keep. Default: 5 years")
    parser.add_argument("--require-fred", action="store_true", help="Fail if any required FRED series cannot be loaded.")
    args = parser.parse_args(argv)
    try:
        result = refresh_macro_prices(
            db_path=args.db_path or None,
            years=args.years,
            require_fred=bool(args.require_fred),
        )
    except Exception as exc:
        print(f"Refresh failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    _log(
        "done "
        f"db={result.db_path} min_date={result.min_date} max_date={result.max_date} "
        f"rows={result.total_rows} series={result.series_count} changed={result.inserted_or_replaced_rows} "
        f"size_bytes={result.size_bytes}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
