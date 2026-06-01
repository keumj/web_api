from __future__ import annotations

import base64
import gzip
import json
from dataclasses import dataclass
from typing import Any

from bs4 import BeautifulSoup

from app.services import db_service


@dataclass(frozen=True)
class LatestResult:
    html: str
    updated_at: str | None
    metadata: dict[str, object]


def _ensure_table() -> None:
    with db_service.app_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_latest_results (
                user_id TEXT NOT NULL,
                module TEXT NOT NULL,
                page TEXT NOT NULL,
                payload_encoding TEXT NOT NULL,
                payload_text TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(user_id, module, page)
            )
            """
        )
        conn.commit()


def _encode_html(html: str) -> tuple[str, str]:
    compressed = gzip.compress(str(html or "").encode("utf-8"))
    return "gzip+base64", base64.b64encode(compressed).decode("ascii")


def _decode_html(encoding: str, payload: str) -> str:
    if encoding == "gzip+base64":
        return gzip.decompress(base64.b64decode(str(payload or "").encode("ascii"))).decode("utf-8")
    if encoding == "plain":
        return str(payload or "")
    raise ValueError(f"Unsupported latest result encoding: {encoding}")


def save_latest_result(
    user_id: str | None,
    *,
    module: str,
    page: str,
    html: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    clean_user_id = str(user_id or "").strip()
    if not clean_user_id or not str(html or "").strip():
        return
    _ensure_table()
    encoding, payload = _encode_html(html)
    metadata_json = json.dumps(metadata or {}, ensure_ascii=False, separators=(",", ":"))
    with db_service.app_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO user_latest_results(
                user_id, module, page, payload_encoding, payload_text, metadata_json, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id, module, page)
            DO UPDATE SET
                payload_encoding = excluded.payload_encoding,
                payload_text = excluded.payload_text,
                metadata_json = excluded.metadata_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (clean_user_id, str(module), str(page), encoding, payload, metadata_json),
        )
        conn.commit()


def load_latest_result(user_id: str | None, *, module: str, page: str) -> LatestResult | None:
    clean_user_id = str(user_id or "").strip()
    if not clean_user_id:
        return None
    _ensure_table()
    with db_service.app_db_connection() as conn:
        row = conn.execute(
            """
            SELECT payload_encoding, payload_text, metadata_json, updated_at
            FROM user_latest_results
            WHERE user_id = ? AND module = ? AND page = ?
            """,
            (clean_user_id, str(module), str(page)),
        ).fetchone()
    if not row:
        return None
    metadata: dict[str, object] = {}
    try:
        decoded = json.loads(str(row[2] or "{}"))
        if isinstance(decoded, dict):
            metadata = decoded
    except Exception:
        metadata = {}
    return LatestResult(
        html=_decode_html(str(row[0]), str(row[1])),
        updated_at=str(row[3] or ""),
        metadata=metadata,
    )


def with_loaded_notice(html: str, latest: LatestResult, *, label: str) -> str:
    stamp = str(latest.updated_at or "").strip()
    message = f"저장된 최근 {label} 결과를 불러왔습니다."
    if stamp:
        message += f" 마지막 저장: {stamp}"
    soup = BeautifulSoup(html, "html.parser")
    container = soup.select_one(".wrap") or soup.select_one("main") or soup.body
    if container is None:
        return html
    if container.find(attrs={"data-latest-result-notice": "1"}) is not None:
        return str(soup)
    notice = soup.new_tag("div")
    notice["class"] = "notice ok"
    notice["data-latest-result-notice"] = "1"
    notice.string = message
    container.insert(0, notice)
    return str(soup)
