from __future__ import annotations

import json
from dataclasses import dataclass
from io import StringIO
from typing import Any

import pandas as pd

from app.services import db_service
from pipeline_portfolio.analysis import OptimizationResult, PortfolioDashboard, VirtualTradeResult


SNAPSHOT_TYPE_DASHBOARD = "dashboard"
SNAPSHOT_TYPE_OPTIMIZATION = "optimization"
SNAPSHOT_TYPE_VIRTUAL_TRADE = "virtual_trade"

_DASHBOARD_FRAME_FIELDS = [
    "trades",
    "positions",
    "holdings_performance",
    "portfolio_summary",
    "attribution",
    "stock_attribution",
    "style_attribution",
    "risk_summary",
    "relative_risk_summary",
    "risk_contribution",
    "active_risk_contribution",
    "factor_risk",
    "style_exposure",
    "scoring",
]
_DASHBOARD_OPTIONAL_FRAME_FIELDS = [
    "best_scoring_stocks",
    "worst_scoring_stocks",
    "top_recommendations",
]
_DASHBOARD_CHART_FIELDS = [
    "cumulative_chart",
    "weekly_return_chart",
    "sector_contribution_chart",
    "style_exposure_chart",
    "sector_allocation_chart",
    "benchmark_sector_allocation_chart",
    "risk_contribution_chart",
    "active_risk_contribution_chart",
    "integrated_score_chart",
]


@dataclass(frozen=True)
class SnapshotEnvelope:
    payload: object
    updated_at: str | None
    range_info: dict[str, object] | None = None
    extra: dict[str, object] | None = None


def _ensure_snapshot_table() -> None:
    with db_service.app_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS portfolio_latest_snapshots (
                user_id TEXT NOT NULL,
                snapshot_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(user_id, snapshot_type)
            )
            """
        )
        conn.commit()


def _save_snapshot(user_id: str, snapshot_type: str, payload: dict[str, object]) -> None:
    if not str(user_id or "").strip():
        return
    _ensure_snapshot_table()
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    with db_service.app_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO portfolio_latest_snapshots(user_id, snapshot_type, payload_json, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id, snapshot_type)
            DO UPDATE SET payload_json = excluded.payload_json, updated_at = CURRENT_TIMESTAMP
            """,
            (str(user_id), str(snapshot_type), payload_json),
        )
        conn.commit()


def _load_snapshot(user_id: str, snapshot_type: str) -> tuple[dict[str, object] | None, str | None]:
    if not str(user_id or "").strip():
        return None, None
    _ensure_snapshot_table()
    with db_service.app_db_connection() as conn:
        row = conn.execute(
            """
            SELECT payload_json, updated_at
            FROM portfolio_latest_snapshots
            WHERE user_id = ? AND snapshot_type = ?
            """,
            (str(user_id), str(snapshot_type)),
        ).fetchone()
    if not row:
        return None, None
    try:
        return json.loads(str(row[0])), str(row[1] or "")
    except Exception:
        return None, str(row[1] or "")


def _serialize_frame(frame: pd.DataFrame | None) -> dict[str, object] | None:
    if frame is None:
        return None
    return json.loads(frame.to_json(orient="split", date_format="iso"))


def _deserialize_frame(payload: object) -> pd.DataFrame:
    if payload is None:
        return pd.DataFrame()
    return pd.read_json(StringIO(json.dumps(payload)), orient="split")


def _deserialize_optional_frame(payload: object) -> pd.DataFrame | None:
    if payload is None:
        return None
    return _deserialize_frame(payload)


def save_dashboard_snapshot(user_id: str, dashboard: PortfolioDashboard, *, range_info: dict[str, object]) -> None:
    frames = {name: _serialize_frame(getattr(dashboard, name)) for name in _DASHBOARD_FRAME_FIELDS}
    optional_frames = {name: _serialize_frame(getattr(dashboard, name)) for name in _DASHBOARD_OPTIONAL_FRAME_FIELDS}
    charts = {name: getattr(dashboard, name) for name in _DASHBOARD_CHART_FIELDS}
    payload = {
        "range": range_info,
        "dashboard": {
            "as_of_date": dashboard.as_of_date,
            "diagnostics": dashboard.diagnostics,
            "scoring_commentary": dashboard.scoring_commentary,
            "frames": frames,
            "optional_frames": optional_frames,
            "charts": charts,
        },
    }
    _save_snapshot(user_id, SNAPSHOT_TYPE_DASHBOARD, payload)


def load_dashboard_snapshot(user_id: str) -> SnapshotEnvelope | None:
    payload, updated_at = _load_snapshot(user_id, SNAPSHOT_TYPE_DASHBOARD)
    if not payload:
        return None
    body = payload.get("dashboard", {}) if isinstance(payload, dict) else {}
    frames = body.get("frames", {}) if isinstance(body, dict) else {}
    optional_frames = body.get("optional_frames", {}) if isinstance(body, dict) else {}
    charts = body.get("charts", {}) if isinstance(body, dict) else {}
    dashboard = PortfolioDashboard(
        as_of_date=body.get("as_of_date"),
        trades=_deserialize_frame(frames.get("trades")),
        positions=_deserialize_frame(frames.get("positions")),
        holdings_performance=_deserialize_frame(frames.get("holdings_performance")),
        portfolio_summary=_deserialize_frame(frames.get("portfolio_summary")),
        attribution=_deserialize_frame(frames.get("attribution")),
        stock_attribution=_deserialize_frame(frames.get("stock_attribution")),
        style_attribution=_deserialize_frame(frames.get("style_attribution")),
        risk_summary=_deserialize_frame(frames.get("risk_summary")),
        relative_risk_summary=_deserialize_frame(frames.get("relative_risk_summary")),
        risk_contribution=_deserialize_frame(frames.get("risk_contribution")),
        active_risk_contribution=_deserialize_frame(frames.get("active_risk_contribution")),
        factor_risk=_deserialize_frame(frames.get("factor_risk")),
        style_exposure=_deserialize_frame(frames.get("style_exposure")),
        scoring=_deserialize_frame(frames.get("scoring")),
        diagnostics={str(k): str(v) for k, v in dict(body.get("diagnostics", {})).items()},
        cumulative_chart=charts.get("cumulative_chart"),
        weekly_return_chart=charts.get("weekly_return_chart"),
        sector_contribution_chart=charts.get("sector_contribution_chart"),
        style_exposure_chart=charts.get("style_exposure_chart"),
        sector_allocation_chart=charts.get("sector_allocation_chart"),
        benchmark_sector_allocation_chart=charts.get("benchmark_sector_allocation_chart"),
        risk_contribution_chart=charts.get("risk_contribution_chart"),
        active_risk_contribution_chart=charts.get("active_risk_contribution_chart"),
        integrated_score_chart=charts.get("integrated_score_chart"),
        best_scoring_stocks=_deserialize_optional_frame(optional_frames.get("best_scoring_stocks")),
        worst_scoring_stocks=_deserialize_optional_frame(optional_frames.get("worst_scoring_stocks")),
        top_recommendations=_deserialize_optional_frame(optional_frames.get("top_recommendations")),
        scoring_commentary=body.get("scoring_commentary"),
    )
    return SnapshotEnvelope(payload=dashboard, updated_at=updated_at, range_info=dict(payload.get("range", {})))


def save_optimization_snapshot(
    user_id: str,
    optimization: OptimizationResult,
    *,
    range_info: dict[str, object],
    optimization_params: dict[str, float | int],
) -> None:
    payload = {
        "range": range_info,
        "optimization_params": optimization_params,
        "optimization": {
            "replication": _serialize_frame(optimization.replication),
            "aggressive": _serialize_frame(optimization.aggressive),
            "defensive": _serialize_frame(optimization.defensive),
            "diagnostics": _serialize_frame(optimization.diagnostics),
            "impact_summary": _serialize_frame(optimization.impact_summary),
            "replication_chart": optimization.replication_chart,
            "aggressive_chart": optimization.aggressive_chart,
            "defensive_chart": optimization.defensive_chart,
        },
    }
    _save_snapshot(user_id, SNAPSHOT_TYPE_OPTIMIZATION, payload)


def load_optimization_snapshot(user_id: str) -> SnapshotEnvelope | None:
    payload, updated_at = _load_snapshot(user_id, SNAPSHOT_TYPE_OPTIMIZATION)
    if not payload:
        return None
    body = payload.get("optimization", {}) if isinstance(payload, dict) else {}
    optimization = OptimizationResult(
        replication=_deserialize_frame(body.get("replication")),
        aggressive=_deserialize_frame(body.get("aggressive")),
        defensive=_deserialize_frame(body.get("defensive")),
        diagnostics=_deserialize_frame(body.get("diagnostics")),
        replication_chart=body.get("replication_chart"),
        aggressive_chart=body.get("aggressive_chart"),
        defensive_chart=body.get("defensive_chart"),
        impact_summary=_deserialize_optional_frame(body.get("impact_summary")),
    )
    return SnapshotEnvelope(
        payload=optimization,
        updated_at=updated_at,
        range_info=dict(payload.get("range", {})),
        extra={"optimization_params": dict(payload.get("optimization_params", {}))},
    )


def save_virtual_trade_snapshot(user_id: str, result: VirtualTradeResult, *, range_info: dict[str, object]) -> None:
    payload = {
        "range": range_info,
        "virtual_trade": {
            "input_summary": _serialize_frame(result.input_summary),
            "before_summary": _serialize_frame(result.before_summary),
            "after_summary": _serialize_frame(result.after_summary),
            "position_changes": _serialize_frame(result.position_changes),
            "risk_changes": _serialize_frame(result.risk_changes),
            "diagnostics": result.diagnostics,
        },
    }
    _save_snapshot(user_id, SNAPSHOT_TYPE_VIRTUAL_TRADE, payload)


def load_virtual_trade_snapshot(user_id: str) -> SnapshotEnvelope | None:
    payload, updated_at = _load_snapshot(user_id, SNAPSHOT_TYPE_VIRTUAL_TRADE)
    if not payload:
        return None
    body = payload.get("virtual_trade", {}) if isinstance(payload, dict) else {}
    result = VirtualTradeResult(
        input_summary=_deserialize_frame(body.get("input_summary")),
        before_summary=_deserialize_frame(body.get("before_summary")),
        after_summary=_deserialize_frame(body.get("after_summary")),
        position_changes=_deserialize_frame(body.get("position_changes")),
        risk_changes=_deserialize_frame(body.get("risk_changes")),
        diagnostics={str(k): str(v) for k, v in dict(body.get("diagnostics", {})).items()},
    )
    return SnapshotEnvelope(payload=result, updated_at=updated_at, range_info=dict(payload.get("range", {})))
