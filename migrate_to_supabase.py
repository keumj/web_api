from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline_macro.macro_data_store import macro_db_path, macro_storage_label
from scripts.upload_macro_sqlite_to_supabase import upload_macro_sqlite_to_supabase


def migrate_to_supabase() -> int:
    local_db = macro_db_path()
    if not local_db.is_absolute():
        local_db = ROOT / local_db
    if not local_db.is_file():
        print(f"Error: local macro SQLite DB not found: {local_db}", file=sys.stderr)
        return 1

    print(f"Uploading macro SQLite: {local_db}")
    print(f"Target: {macro_storage_label()}")
    try:
        series_count, metadata_count = upload_macro_sqlite_to_supabase(local_db=local_db)
    except Exception as exc:
        print(f"Upload failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(
        "Upload complete: "
        f"macro_series={series_count}, macro_metadata={metadata_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(migrate_to_supabase())
