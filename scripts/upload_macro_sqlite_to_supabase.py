from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline_macro.macro_data_store import _connect_macro_read_db, macro_db_path, macro_storage_label


def _chunks(rows: list[tuple[object, ...]], size: int) -> Iterable[list[tuple[object, ...]]]:
    for start in range(0, len(rows), max(int(size), 1)):
        yield rows[start : start + size]


def _ensure_postgres_macro_schema(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS macro_series (
            series_id TEXT NOT NULL,
            date TEXT NOT NULL,
            value DOUBLE PRECISION NOT NULL,
            dataset TEXT NOT NULL,
            frequency TEXT NOT NULL,
            source TEXT NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
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
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def _read_local_rows(local, query: str) -> list[tuple[object, ...]]:
    return [tuple(row) for row in local.execute(query).fetchall()]


def upload_macro_sqlite_to_supabase(*, local_db: Path, batch_size: int = 1000) -> tuple[int, int]:
    if not local_db.is_file():
        raise FileNotFoundError(f"Local macro SQLite DB not found: {local_db}")

    with sqlite3.connect(local_db) as local, _connect_macro_read_db() as remote:
        _ensure_postgres_macro_schema(remote)
        series_rows = _read_local_rows(
            local,
            """
            SELECT series_id, date, value, dataset, frequency, source, updated_at
            FROM macro_series
            ORDER BY series_id, date
            """,
        )
        metadata_rows = _read_local_rows(
            local,
            """
            SELECT series_id, dataset, frequency, source, min_date, max_date, row_count, updated_at
            FROM macro_metadata
            ORDER BY series_id
            """,
        )

        for batch in _chunks(series_rows, batch_size):
            remote.executemany(
                """
                INSERT INTO macro_series (series_id, date, value, dataset, frequency, source, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(series_id, date) DO UPDATE SET
                    value = excluded.value,
                    dataset = excluded.dataset,
                    frequency = excluded.frequency,
                    source = excluded.source,
                    updated_at = excluded.updated_at
                """,
                batch,
            )

        for batch in _chunks(metadata_rows, batch_size):
            remote.executemany(
                """
                INSERT INTO macro_metadata (
                    series_id, dataset, frequency, source, min_date, max_date, row_count, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(series_id) DO UPDATE SET
                    dataset = excluded.dataset,
                    frequency = excluded.frequency,
                    source = excluded.source,
                    min_date = excluded.min_date,
                    max_date = excluded.max_date,
                    row_count = excluded.row_count,
                    updated_at = excluded.updated_at
                """,
                batch,
            )
        remote.commit()
        return len(series_rows), len(metadata_rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Upload data/macro_prices.sqlite to the configured Supabase macro DB.")
    parser.add_argument("--local-db", default=str(macro_db_path()), help="Local macro SQLite path.")
    parser.add_argument("--batch-size", type=int, default=1000, help="Rows per INSERT batch.")
    args = parser.parse_args(argv)
    try:
        series_count, metadata_count = upload_macro_sqlite_to_supabase(
            local_db=Path(args.local_db),
            batch_size=int(args.batch_size),
        )
    except Exception as exc:
        print(f"Upload failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Uploaded macro SQLite to {macro_storage_label()}: macro_series={series_count}, macro_metadata={metadata_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
