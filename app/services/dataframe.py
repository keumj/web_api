from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def frame_records(frame: pd.DataFrame | None, *, max_rows: int = 50) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    clean = frame.head(max_rows).copy()
    clean = clean.where(pd.notna(clean), None)
    return [{str(key): _json_value(value) for key, value in row.items()} for row in clean.to_dict(orient="records")]


def frame_html(frame: pd.DataFrame | None, *, max_rows: int = 50) -> str:
    if frame is None or frame.empty:
        return "<p class='service-muted'>데이터가 없습니다.</p>"
    show = frame.head(max_rows).copy()
    table = show.to_html(index=False, border=0, classes="service-table")
    return f"<div class='service-table-wrap'>{table}</div>"
