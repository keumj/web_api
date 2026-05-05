from __future__ import annotations

import base64
import html
import io
import json
import os
import re
import sqlite3
import subprocess
import sys
import threading
import traceback
import webbrowser

import requests
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import clone

from pipeline_common.notebook_data import fetch_sp500_close_prices
from pipeline_common.security import configure_ssl, ensure_writable_dir, security_hint
from pipeline_common.shared_sp500_prices_sql import load_shared_market_caps_for_symbols
from . import technical_analysis as ta_web_gui

from .forecast import (
    StockForecastResult,
    _build_direction_model_specs,
    _build_model_specs,
    _build_yfinance_session,
    _build_supervised_dataset,
    classify_regime_from_feature_row,
    _fetch_close_prices_with_source,
    _inverse_error_weights,
    _normalize_close_prices,
    load_price_data_csv,
    run_ticker_stock_forecast_pipeline,
)

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - optional runtime dependency
    yf = None

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

_APP_TITLE = "Stock Analysis Lab | S&P 500"

@dataclass
class _RunContext:
    result: StockForecastResult
    saved_dir: str | None
    source_table: pd.DataFrame


@dataclass
class _FinancialContext:
    ticker: str
    company_name: str | None
    currency: str | None
    metrics: dict[str, object]
    summary_table: pd.DataFrame
    income_table: pd.DataFrame
    balance_table: pd.DataFrame
    cashflow_table: pd.DataFrame
    saved_dir: str | None
    source_table: pd.DataFrame
    provider_status_table: pd.DataFrame


@dataclass
class _ReturnsContext:
    ticker: str
    sector: str
    latest_market_date: str
    ticker_latest_date: str
    price_source: str
    market_cap_source: str
    period_returns: dict[str, float | None]
    sector_average_returns: dict[str, float | None]
    market_average_returns: dict[str, float | None]
    sector_rank_ytd: int | None
    sector_count: int
    market_rank_ytd: int | None
    market_count: int
    summary_table: pd.DataFrame
    source_table: pd.DataFrame
    daily_returns_table: pd.DataFrame
    sector_daily_returns_table: pd.DataFrame
    sector_top_table: pd.DataFrame
    sector_bottom_table: pd.DataFrame
    market_top_table: pd.DataFrame
    market_bottom_table: pd.DataFrame
    relative_chart_base64: str
    daily_chart_base64: str


@dataclass
class _RiskContext:
    ticker: str
    sector: str
    latest_market_date: str
    ticker_latest_date: str
    price_source: str
    market_cap_source: str
    ticker_vol_20d: float | None
    ticker_vol_60d: float | None
    ticker_vol_252d: float | None
    ticker_max_drawdown_1y: float | None
    ticker_max_drawdown_3y: float | None
    beta_sector_1y: float | None
    beta_market_1y: float | None
    var_95_1d: float | None
    cvar_95_1d: float | None
    sector_vol_rank_1y: int | None
    sector_count: int
    market_vol_rank_1y: int | None
    market_count: int
    summary_table: pd.DataFrame
    source_table: pd.DataFrame
    recent_shock_table: pd.DataFrame
    sector_high_vol_table: pd.DataFrame
    sector_low_vol_table: pd.DataFrame
    market_high_vol_table: pd.DataFrame
    market_low_vol_table: pd.DataFrame
    drawdown_chart_base64: str
    volatility_chart_base64: str
    commentary: str


@dataclass
class _DecisionContext:
    ticker: str
    sector: str
    latest_market_date: str
    recommendation: str
    confidence_label: str
    total_score: float
    bullish_reasons: list[str]
    bearish_reasons: list[str]
    watch_items: list[str]
    final_commentary: str
    score_table: pd.DataFrame
    signal_table: pd.DataFrame
    source_table: pd.DataFrame
    score_chart_base64: str
    trend_chart_base64: str


@dataclass
class _FactorContext:
    ticker: str
    sector: str
    latest_market_date: str
    ticker_latest_date: str
    price_source: str
    market_cap_source: str
    beta_sector_60d: float | None
    beta_market_60d: float | None
    corr_sector_60d: float | None
    corr_market_60d: float | None
    residual_market_20d: float | None
    residual_sector_20d: float | None
    regime_trend: str
    regime_volatility: str
    regime_beta: str
    regime_overall: str
    summary_table: pd.DataFrame
    source_table: pd.DataFrame
    interpretation_table: pd.DataFrame
    recent_factor_table: pd.DataFrame
    beta_chart_base64: str
    residual_chart_base64: str
    commentary: str


@dataclass
class _WalkForwardContext:
    ticker: str
    price_source: str
    input_mode: str
    horizon_days: int
    evaluation_splits: int
    min_train_rows: int
    step_size: int
    max_splits: int
    train_start_date: str
    latest_as_of_date: str
    latest_realized_date: str
    direction_hit_rate: float | None
    mae_return: float | None
    rmse_return: float | None
    bias_return: float | None
    skill_vs_naive: float | None
    return_correlation: float | None
    classification_hit_rate: float | None
    trade_coverage_rate: float | None
    trade_hit_rate: float | None
    summary_table: pd.DataFrame
    source_table: pd.DataFrame
    interpretation_table: pd.DataFrame
    split_table: pd.DataFrame
    model_table: pd.DataFrame
    threshold_table: pd.DataFrame
    regime_table: pd.DataFrame
    forecast_chart_base64: str
    diagnostics_chart_base64: str
    commentary: str
    saved_dir: str | None


_RETURN_PERIOD_LABELS = ("3Y", "1Y", "6M", "1M", "YTD", "MTD", "WTD", "20D", "60D")


def _format_pct(value: object, ndigits: int = 2) -> str:
    if value is None:
        return "-"
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(numeric):
        return "-"
    return f"{numeric * 100.0:,.{ndigits}f}%"


def _sp500_components_candidates() -> list[Path]:
    candidates = [
        os.getenv("SP500_COMPONENTS_CSV_PATH", "").strip(),
        "data/sp500_components_full.csv",
        "data/sp500_components.csv",
    ]
    return [Path(item) for item in candidates if str(item).strip()]


def _load_sp500_components_full() -> tuple[pd.DataFrame, str]:
    for path in _sp500_components_candidates():
        if not path.exists() or not path.is_file():
            continue
        try:
            raw = pd.read_csv(path)
        except Exception:
            continue
        if raw.empty:
            continue
        cols = {str(c).strip().lower(): c for c in raw.columns}
        symbol_col = cols.get("symbol")
        sector_col = cols.get("sector")
        if symbol_col is None or sector_col is None:
            continue
        out = raw[[symbol_col, sector_col]].copy()
        out.columns = ["Symbol", "Sector"]
        out["Symbol"] = out["Symbol"].astype(str).str.strip().str.upper()
        out["Sector"] = out["Sector"].astype(str).str.strip()
        out = out[(out["Symbol"] != "") & (out["Sector"] != "")]
        out = out.drop_duplicates(subset=["Symbol"], keep="last").reset_index(drop=True)
        if not out.empty:
            return out, f"local_csv:{path.as_posix()}"
    raise FileNotFoundError("S&P 500 components CSV not found.")


def _clean_close_series(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return clean.astype(float)
    out = clean.astype(float).sort_index()
    out.index = pd.to_datetime(out.index)
    return out[~out.index.duplicated(keep="last")]


def _clean_market_cap_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].sort_index()
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out[~out.index.duplicated(keep="last")]
    return out.dropna(how="all")


def _last_observation(
    series: pd.Series,
    *,
    on_or_before: pd.Timestamp | None = None,
    strict_before: pd.Timestamp | None = None,
) -> tuple[pd.Timestamp, float] | None:
    clean = _clean_close_series(series)
    if clean.empty:
        return None
    if strict_before is not None:
        clean = clean[clean.index < pd.Timestamp(strict_before)]
    elif on_or_before is not None:
        clean = clean[clean.index <= pd.Timestamp(on_or_before)]
    if clean.empty:
        return None
    return pd.Timestamp(clean.index[-1]).normalize(), float(clean.iloc[-1])


def _compute_period_returns(series: pd.Series) -> tuple[dict[str, float | None], pd.Timestamp | None]:
    latest = _last_observation(series)
    if latest is None:
        return {label: None for label in _RETURN_PERIOD_LABELS}, None

    latest_date, latest_close = latest
    periods: dict[str, tuple[pd.Timestamp | int, str]] = {
        "3Y": (latest_date - pd.DateOffset(years=3), "on_or_before"),
        "1Y": (latest_date - pd.DateOffset(years=1), "on_or_before"),
        "6M": (latest_date - pd.DateOffset(months=6), "on_or_before"),
        "1M": (latest_date - pd.DateOffset(months=1), "on_or_before"),
        "YTD": (latest_date.to_period("Y").start_time.normalize(), "strict_before"),
        "MTD": (latest_date.to_period("M").start_time.normalize(), "strict_before"),
        "WTD": (latest_date - pd.Timedelta(days=latest_date.dayofweek), "strict_before"),
        "20D": (20, "trading_days"),
        "60D": (60, "trading_days"),
    }

    out: dict[str, float | None] = {}
    clean = _clean_close_series(series)
    for label, (anchor, mode) in periods.items():
        if mode == "trading_days":
            window = int(anchor)
            if len(clean) <= window:
                out[label] = None
                continue
            base = (pd.Timestamp(clean.index[-(window + 1)]).normalize(), float(clean.iloc[-(window + 1)]))
        elif mode == "strict_before":
            base = _last_observation(series, strict_before=pd.Timestamp(anchor))
        else:
            base = _last_observation(series, on_or_before=pd.Timestamp(anchor))
        if base is None or base[1] == 0:
            out[label] = None
            continue
        out[label] = (latest_close / float(base[1])) - 1.0
    return out, latest_date


def _cumulative_return_pct_from_daily_returns(daily_returns: pd.Series) -> pd.Series:
    clean = pd.to_numeric(daily_returns, errors="coerce").dropna().sort_index()
    if clean.empty:
        return pd.Series(dtype=float)
    gross = (1.0 + clean).cumprod()
    rebased = (gross / float(gross.iloc[0])) - 1.0
    return (rebased * 100.0).rename(clean.name or "Cumulative Return")


def _base100_index_from_daily_returns(daily_returns: pd.Series) -> pd.Series:
    clean = pd.to_numeric(daily_returns, errors="coerce").dropna().sort_index()
    if clean.empty:
        return pd.Series(dtype=float)
    gross = (1.0 + clean).cumprod()
    return ((gross / float(gross.iloc[0])) * 100.0).rename(clean.name or "Base 100 Index")


def _apply_year_month_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    ax.tick_params(axis="x", labelrotation=30)


def _compute_period_returns_from_daily_returns(daily_returns: pd.Series) -> tuple[dict[str, float | None], pd.Timestamp | None]:
    clean = pd.to_numeric(daily_returns, errors="coerce").dropna().sort_index()
    if clean.empty:
        return {label: None for label in _RETURN_PERIOD_LABELS}, None

    latest_date = pd.Timestamp(clean.index.max()).normalize()
    anchors: dict[str, tuple[pd.Timestamp | int, str]] = {
        "3Y": (latest_date - pd.DateOffset(years=3), "after"),
        "1Y": (latest_date - pd.DateOffset(years=1), "after"),
        "6M": (latest_date - pd.DateOffset(months=6), "after"),
        "1M": (latest_date - pd.DateOffset(months=1), "after"),
        "YTD": (latest_date.to_period("Y").start_time.normalize(), "on_or_after"),
        "MTD": (latest_date.to_period("M").start_time.normalize(), "on_or_after"),
        "WTD": (latest_date - pd.Timedelta(days=latest_date.dayofweek), "on_or_after"),
        "20D": (20, "tail"),
        "60D": (60, "tail"),
    }
    out: dict[str, float | None] = {}
    for label, (anchor, mode) in anchors.items():
        if mode == "tail":
            window = clean.tail(int(anchor))
        elif mode == "on_or_after":
            window = clean[clean.index >= pd.Timestamp(anchor)]
        else:
            window = clean[clean.index > pd.Timestamp(anchor)]
        out[label] = float((1.0 + window).prod() - 1.0) if not window.empty else None
    return out, latest_date


def _clean_return_series(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return clean.astype(float)
    out = clean.astype(float).sort_index()
    out.index = pd.to_datetime(out.index)
    return out[~out.index.duplicated(keep="last")]


def _annualized_volatility(daily_returns: pd.Series, window: int | None = None) -> float | None:
    clean = _clean_return_series(daily_returns)
    if window is not None:
        clean = clean.tail(window)
    if len(clean) < 2:
        return None
    return float(clean.std(ddof=0) * np.sqrt(252.0))


def _downside_volatility(daily_returns: pd.Series, window: int | None = None) -> float | None:
    clean = _clean_return_series(daily_returns)
    if window is not None:
        clean = clean.tail(window)
    clean = clean[clean < 0.0]
    if len(clean) < 2:
        return None
    return float(clean.std(ddof=0) * np.sqrt(252.0))


def _max_drawdown_from_daily_returns(daily_returns: pd.Series, window: int | None = None) -> float | None:
    clean = _clean_return_series(daily_returns)
    if window is not None:
        clean = clean.tail(window)
    if clean.empty:
        return None
    wealth = (1.0 + clean).cumprod()
    drawdown = (wealth / wealth.cummax()) - 1.0
    return float(drawdown.min()) if not drawdown.empty else None


def _beta_and_correlation(
    asset_daily: pd.Series,
    benchmark_daily: pd.Series,
    *,
    window: int = 252,
) -> tuple[float | None, float | None]:
    asset = _clean_return_series(asset_daily).tail(window)
    benchmark = _clean_return_series(benchmark_daily).tail(window)
    joined = pd.concat([asset.rename("asset"), benchmark.rename("benchmark")], axis=1).dropna()
    if len(joined) < 2:
        return None, None
    benchmark_var = float(joined["benchmark"].var(ddof=0))
    corr = float(joined["asset"].corr(joined["benchmark"])) if len(joined) >= 2 else None
    if benchmark_var <= 0.0 or not np.isfinite(benchmark_var):
        return None, corr
    cov = float(np.cov(joined["asset"], joined["benchmark"], ddof=0)[0, 1])
    return cov / benchmark_var, corr


def _historical_var_cvar(
    daily_returns: pd.Series,
    *,
    confidence: float = 0.95,
    window: int = 252,
) -> tuple[float | None, float | None]:
    clean = _clean_return_series(daily_returns).tail(window)
    if clean.empty:
        return None, None
    quantile = float(clean.quantile(1.0 - confidence))
    tail = clean[clean <= quantile]
    cvar = float(tail.mean()) if not tail.empty else quantile
    return max(0.0, -quantile), max(0.0, -cvar)


def _drawdown_curve_pct(daily_returns: pd.Series, window: int | None = None) -> pd.Series:
    clean = _clean_return_series(daily_returns)
    if window is not None:
        clean = clean.tail(window)
    if clean.empty:
        return pd.Series(dtype=float)
    wealth = (1.0 + clean).cumprod()
    drawdown = ((wealth / wealth.cummax()) - 1.0) * 100.0
    return drawdown.rename(clean.name or "Drawdown")


def _rolling_volatility_pct(daily_returns: pd.Series, window: int = 20, lookback: int = 252) -> pd.Series:
    clean = _clean_return_series(daily_returns).tail(lookback + window + 5)
    if len(clean) < window:
        return pd.Series(dtype=float)
    rolling = clean.rolling(window).std(ddof=0) * np.sqrt(252.0) * 100.0
    return rolling.dropna().rename(clean.name or "Rolling Volatility")


def _rolling_beta_series(asset_daily: pd.Series, benchmark_daily: pd.Series, *, window: int = 60) -> pd.Series:
    asset = _clean_return_series(asset_daily).rename("asset")
    benchmark = _clean_return_series(benchmark_daily).rename("benchmark")
    joined = pd.concat([asset, benchmark], axis=1).dropna()
    if len(joined) < window:
        return pd.Series(dtype=float)
    cov = joined["asset"].rolling(window).cov(joined["benchmark"])
    var = joined["benchmark"].rolling(window).var()
    beta = cov / var.replace(0.0, np.nan)
    return beta.dropna().rename("Rolling Beta")


def _rolling_correlation_series(asset_daily: pd.Series, benchmark_daily: pd.Series, *, window: int = 60) -> pd.Series:
    asset = _clean_return_series(asset_daily).rename("asset")
    benchmark = _clean_return_series(benchmark_daily).rename("benchmark")
    joined = pd.concat([asset, benchmark], axis=1).dropna()
    if len(joined) < window:
        return pd.Series(dtype=float)
    corr = joined["asset"].rolling(window).corr(joined["benchmark"])
    return corr.dropna().rename("Rolling Correlation")


def _drawdown_comparison_chart(
    ticker: str,
    ticker_daily: pd.Series,
    sector_daily: pd.Series,
    market_daily: pd.Series,
) -> str:
    ticker_curve = _drawdown_curve_pct(ticker_daily, 252).rename(ticker)
    sector_curve = _drawdown_curve_pct(sector_daily, 252).rename("Sector Cap-Weighted")
    market_curve = _drawdown_curve_pct(market_daily, 252).rename("S&P 500 Cap-Weighted")
    chart_df = pd.concat([ticker_curve, sector_curve, market_curve], axis=1).dropna(how="all")

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    if not chart_df.empty:
        for name, color in [(ticker, "#0f4c81"), ("Sector Cap-Weighted", "#2e7d32"), ("S&P 500 Cap-Weighted", "#b26a00")]:
            if name in chart_df.columns:
                ax.plot(chart_df.index, chart_df[name], linewidth=2.0, label=name, color=color)
    _apply_year_month_axis(ax)
    ax.axhline(0.0, color="#8a98a8", linewidth=1.0)
    ax.set_title("1Y Drawdown Comparison")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return _render_chart_base64(fig)


def _rolling_volatility_chart(
    ticker: str,
    ticker_daily: pd.Series,
    sector_daily: pd.Series,
    market_daily: pd.Series,
) -> str:
    ticker_curve = _rolling_volatility_pct(ticker_daily, 20, 252).rename(ticker)
    sector_curve = _rolling_volatility_pct(sector_daily, 20, 252).rename("Sector Cap-Weighted")
    market_curve = _rolling_volatility_pct(market_daily, 20, 252).rename("S&P 500 Cap-Weighted")
    chart_df = pd.concat([ticker_curve, sector_curve, market_curve], axis=1).dropna(how="all")

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    if not chart_df.empty:
        for name, color in [(ticker, "#0f4c81"), ("Sector Cap-Weighted", "#2e7d32"), ("S&P 500 Cap-Weighted", "#b26a00")]:
            if name in chart_df.columns:
                ax.plot(chart_df.index, chart_df[name], linewidth=2.0, label=name, color=color)
    _apply_year_month_axis(ax)
    ax.set_title("Rolling 20-Day Annualized Volatility")
    ax.set_ylabel("Volatility (%)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return _render_chart_base64(fig)


def _risk_rank_table_display(df: pd.DataFrame, *, rank_col: str = "Rank") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Rank", "Symbol", "Sector", "As Of", "1Y Ann Vol", "1Y Max DD"])
    out = df.copy()
    if rank_col != "Rank" and rank_col in out.columns:
        out["Rank"] = out[rank_col]
    out["As Of"] = pd.to_datetime(out["latest_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["1Y Ann Vol"] = out["ann_vol_1y"].map(_format_pct)
    out["1Y Max DD"] = out["max_drawdown_1y"].map(_format_pct)
    return out[["Rank", "Symbol", "Sector", "As Of", "1Y Ann Vol", "1Y Max DD"]].reset_index(drop=True)


def _recent_risk_shock_table(
    ticker: str,
    ticker_daily: pd.Series,
    sector_daily: pd.Series,
    market_daily: pd.Series,
) -> pd.DataFrame:
    recent = pd.concat(
        [
            _clean_return_series(ticker_daily).rename(f"{ticker} Daily Return"),
            _clean_return_series(sector_daily).rename("Sector Cap-Weighted Return"),
            _clean_return_series(market_daily).rename("S&P 500 Cap-Weighted Return"),
        ],
        axis=1,
    ).dropna(how="all").tail(20)
    if recent.empty:
        return pd.DataFrame(
            columns=[
                "Date",
                f"{ticker} Daily Return",
                "Sector Cap-Weighted Return",
                "S&P 500 Cap-Weighted Return",
                "Vs Sector",
                "Vs S&P 500",
            ]
        )
    recent["Vs Sector"] = recent[f"{ticker} Daily Return"] - recent["Sector Cap-Weighted Return"]
    recent["Vs S&P 500"] = recent[f"{ticker} Daily Return"] - recent["S&P 500 Cap-Weighted Return"]
    display = recent.reset_index()
    first_col = display.columns[0]
    if first_col != "Date":
        display = display.rename(columns={first_col: "Date"})
    display["Date"] = pd.to_datetime(display["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in [
        f"{ticker} Daily Return",
        "Sector Cap-Weighted Return",
        "S&P 500 Cap-Weighted Return",
        "Vs Sector",
        "Vs S&P 500",
    ]:
        display[col] = display[col].map(_format_pct)
    return display


def _risk_commentary(
    *,
    ticker: str,
    sector: str,
    ticker_vol_252d: float | None,
    sector_vol_252d: float | None,
    market_vol_252d: float | None,
    ticker_max_drawdown_1y: float | None,
    beta_market_1y: float | None,
    var_95_1d: float | None,
) -> str:
    parts: list[str] = []
    risk_bias = "neutral"
    if ticker_vol_252d is not None and sector_vol_252d is not None and market_vol_252d is not None:
        if ticker_vol_252d > sector_vol_252d * 1.15 and ticker_vol_252d > market_vol_252d * 1.15:
            parts.append(
                f"{ticker}는 현재 1년 실현변동성 기준으로 {sector} 섹터와 시가총액가중 S&P 500보다 변동성이 높은 편입니다."
            )
            risk_bias = "high"
        elif ticker_vol_252d < sector_vol_252d * 0.9 and ticker_vol_252d < market_vol_252d * 0.9:
            parts.append(
                f"{ticker}는 현재 1년 실현변동성 기준으로 {sector} 섹터와 시가총액가중 S&P 500보다 방어적인 움직임을 보이고 있습니다."
            )
            risk_bias = "low"
        else:
            parts.append(
                f"{ticker}의 현재 리스크 강도는 {sector} 섹터와 시가총액가중 S&P 500에 대체로 근접한 수준입니다."
            )
    if ticker_max_drawdown_1y is not None:
        if ticker_max_drawdown_1y <= -0.25:
            parts.append("최근 1년 최대낙폭이 깊은 편이라 반등이 나오더라도 가격 경로의 흔들림을 크게 감수해야 하는 종목에 가깝습니다.")
            risk_bias = "high"
        elif ticker_max_drawdown_1y <= -0.12:
            parts.append("최근 1년 최대낙폭은 눈에 띄지만, 대형주 순환 구간에서 충분히 관찰 가능한 범위 안에 있습니다.")
        else:
            parts.append("최근 1년 최대낙폭은 비교적 잘 통제된 편입니다.")
    if beta_market_1y is not None:
        if beta_market_1y >= 1.2:
            parts.append("S&P 500 대비 베타가 1보다 높아 시장이 흔들릴 때 주가 변동이 더 크게 확대될 가능성이 있습니다.")
            if risk_bias != "low":
                risk_bias = "high"
        elif beta_market_1y <= 0.8:
            parts.append("S&P 500 대비 베타가 낮아 최근에는 시장 전체보다 민감도가 낮은 흐름으로 해석할 수 있습니다.")
            if risk_bias == "neutral":
                risk_bias = "low"
    if var_95_1d is not None:
        parts.append(f"95% 기준 1일 역사적 VaR는 약 {_format_pct(var_95_1d)}로, 하루 단위 손실 허용 범위를 점검할 때 참고하기 좋습니다.")

    if risk_bias == "high":
        parts.append("종합하면 현재는 기대수익을 보더라도 추격 진입보다는 분할 접근과 보수적인 손절 관리에 더 무게를 두는 편이 적절합니다.")
    elif risk_bias == "low":
        parts.append("종합하면 현재는 상대적으로 방어적인 구간에 가까워 급격한 변동성 부담보다는 눌림목 대응이나 분할 진입 검토가 쉬운 편입니다.")
    else:
        parts.append("종합하면 현재는 공격적 매수나 과도한 방어 중 한쪽으로 치우치기보다, 중립적인 포지션 관리와 가격 확인이 더 어울리는 구간입니다.")

    return " ".join(parts)


def _clip_score(value: float, *, low: float = -2.0, high: float = 2.0) -> float:
    return float(max(low, min(high, value)))


def _format_score(value: float) -> str:
    return f"{value:+.2f}"


def _decision_label(total_score: float) -> tuple[str, str]:
    abs_score = abs(total_score)
    if total_score >= 4.5:
        label = "매수 우위"
    elif total_score >= 2.0:
        label = "약한 매수 우위"
    elif total_score <= -4.5:
        label = "매도/차익실현 우위"
    elif total_score <= -2.0:
        label = "약한 매도 우위"
    else:
        label = "중립 / 관망"

    if abs_score >= 5.5:
        confidence = "높음"
    elif abs_score >= 3.0:
        confidence = "보통"
    else:
        confidence = "낮음"
    return label, confidence


def _decision_score_chart(score_rows: list[dict[str, object]]) -> str:
    labels = [str(row["Category"]) for row in score_rows]
    values = [float(row["Score"]) for row in score_rows]
    colors = ["#2e7d32" if value >= 0 else "#c62828" for value in values]

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.barh(labels, values, color=colors, alpha=0.85)
    ax.axvline(0.0, color="#8a98a8", linewidth=1.0)
    ax.set_xlim(-2.2, 2.2)
    ax.set_title("Decision Score Breakdown")
    ax.set_xlabel("Score")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return _render_chart_base64(fig)


def _decision_trend_chart(ticker: str, close: pd.Series) -> str:
    clean = _clean_close_series(close).tail(180)
    ma20 = clean.rolling(20).mean()
    ma60 = clean.rolling(60).mean()
    ma120 = clean.rolling(120).mean()
    bb_mid, bb_upper, bb_lower = ta_web_gui._calc_bollinger(clean, 20, 2.0)

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    if not clean.empty:
        ax.plot(clean.index, clean.values, color="#0f4c81", linewidth=1.8, label="Close")
        ax.plot(ma20.index, ma20.values, color="#2e7d32", linewidth=1.2, label="MA20")
        ax.plot(ma60.index, ma60.values, color="#b26a00", linewidth=1.2, label="MA60")
        ax.plot(ma120.index, ma120.values, color="#8e24aa", linewidth=1.2, label="MA120")
        ax.plot(bb_upper.index, bb_upper.values, color="#78909c", linewidth=1.0, linestyle="--", label="BB Upper")
        ax.plot(bb_lower.index, bb_lower.values, color="#78909c", linewidth=1.0, linestyle="--", label="BB Lower")
    _apply_year_month_axis(ax)
    ax.set_title(f"{ticker} Trend and Bollinger Context")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", ncol=3, fontsize=8)
    fig.tight_layout()
    return _render_chart_base64(fig)


def _html_reason_list(items: list[str]) -> str:
    if not items:
        return "<p>-</p>"
    rendered = "".join(f"<li>{html.escape(item)}</li>" for item in items)
    return f"<ul>{rendered}</ul>"


def _run_decision_once(
    form: dict[str, str],
    *,
    returns_ctx: _ReturnsContext,
    risk_ctx: _RiskContext,
    fin_ctx: _FinancialContext | None = None,
) -> _DecisionContext:
    ticker = str(form.get("ticker", "")).strip().upper()
    if not ticker:
        raise ValueError("Provide an S&P 500 ticker for decision analysis.")
    if returns_ctx.ticker != ticker or risk_ctx.ticker != ticker:
        raise ValueError("Decision analysis contexts do not match the requested ticker.")

    close, close_source = _load_common_price_history(
        ticker,
        start_date=(pd.Timestamp.today().normalize() - pd.DateOffset(years=2)).strftime("%Y-%m-%d"),
    )
    if close.empty:
        raise ValueError(f"Shared price history for {ticker} is unavailable.")

    close = _clean_close_series(close)
    ma20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else np.nan
    ma60 = float(close.rolling(60).mean().iloc[-1]) if len(close) >= 60 else np.nan
    ma120 = float(close.rolling(120).mean().iloc[-1]) if len(close) >= 120 else np.nan
    last_close = float(close.iloc[-1])
    rsi14 = float(ta_web_gui._calc_rsi(close, 14).iloc[-1]) if len(close) >= 14 else np.nan
    macd, signal, hist = ta_web_gui._calc_macd(close)
    macd_value = float(macd.iloc[-1]) if not macd.empty else np.nan
    signal_value = float(signal.iloc[-1]) if not signal.empty else np.nan
    hist_value = float(hist.iloc[-1]) if not hist.empty else np.nan
    bb_mid, bb_upper, bb_lower = ta_web_gui._calc_bollinger(close, 20, 2.0)
    bb_upper_value = float(bb_upper.iloc[-1]) if not bb_upper.empty else np.nan
    bb_lower_value = float(bb_lower.iloc[-1]) if not bb_lower.empty else np.nan
    band_width = bb_upper_value - bb_lower_value if np.isfinite(bb_upper_value) and np.isfinite(bb_lower_value) else np.nan
    band_pos = ((last_close - bb_lower_value) / band_width) if np.isfinite(band_width) and band_width > 0 else np.nan

    bullish: list[str] = []
    bearish: list[str] = []
    watch_items: list[str] = []
    score_rows: list[dict[str, object]] = []

    trend_score = 0.0
    trend_details: list[str] = []
    if np.isfinite(ma20):
        if last_close >= ma20:
            trend_score += 0.5
            bullish.append(f"종가가 MA20 위에 있어 단기 추세는 아직 살아 있습니다. 현재가 {last_close:,.2f}, MA20 {ma20:,.2f}입니다.")
            trend_details.append("price>MA20")
        else:
            trend_score -= 0.5
            bearish.append(f"종가가 MA20 아래로 내려와 단기 추세는 약해진 상태입니다. 현재가 {last_close:,.2f}, MA20 {ma20:,.2f}입니다.")
            trend_details.append("price<MA20")
    if np.isfinite(ma60):
        if last_close >= ma60:
            trend_score += 0.5
            bullish.append(f"종가가 MA60 위에 있어 중기 추세 훼손은 아직 제한적입니다. MA60은 {ma60:,.2f}입니다.")
            trend_details.append("price>MA60")
        else:
            trend_score -= 0.5
            bearish.append(f"종가가 MA60 아래에 있어 중기 기준으로는 방어력이 약한 편입니다. MA60은 {ma60:,.2f}입니다.")
            trend_details.append("price<MA60")
    if np.isfinite(ma120):
        if last_close >= ma120:
            trend_score += 0.5
            bullish.append(f"장기 기준선인 MA120 위에서 거래되고 있어 큰 추세 자체는 아직 우상향 해석이 가능합니다. MA120은 {ma120:,.2f}입니다.")
            trend_details.append("price>MA120")
        else:
            trend_score -= 0.75
            bearish.append(f"장기 기준선인 MA120 아래에 있어 추세형 매수 관점에서는 신중할 필요가 있습니다. MA120은 {ma120:,.2f}입니다.")
            trend_details.append("price<MA120")
    if np.isfinite(ma20) and np.isfinite(ma60) and np.isfinite(ma120):
        if ma20 > ma60 > ma120:
            trend_score += 0.5
            bullish.append("MA20 > MA60 > MA120 정배열이라 추세 추종 관점에서는 매수 논리가 비교적 선명합니다.")
            trend_details.append("bull_alignment")
        elif ma20 < ma60 < ma120:
            trend_score -= 0.75
            bearish.append("MA20 < MA60 < MA120 역배열이라 반등이 나와도 구조적으로는 약세 쪽 해석이 우선입니다.")
            trend_details.append("bear_alignment")
    trend_score = _clip_score(trend_score)
    score_rows.append({"Category": "Trend", "Score": trend_score, "Detail": ", ".join(trend_details) or "-"})

    momentum_score = 0.0
    momentum_details: list[str] = []
    if np.isfinite(rsi14):
        if rsi14 <= 35:
            momentum_score += 0.9
            bullish.append(f"RSI(14) {rsi14:,.1f}로 과매도권에 가까워 단기 반등 매수 논리가 살아 있습니다.")
            momentum_details.append("rsi_low")
        elif rsi14 <= 45:
            momentum_score += 0.3
            bullish.append(f"RSI(14) {rsi14:,.1f}로 과열 부담은 크지 않아 눌림목 매수 검토가 가능합니다.")
            momentum_details.append("rsi_near_low")
        elif rsi14 >= 70:
            momentum_score -= 0.9
            bearish.append(f"RSI(14) {rsi14:,.1f}로 과매수권에 가까워 신규 추격매수는 부담이 큽니다.")
            momentum_details.append("rsi_high")
        elif rsi14 >= 60:
            momentum_score -= 0.3
            bearish.append(f"RSI(14) {rsi14:,.1f}로 단기 열기가 높아져 매수보다 차익실현 논리가 조금 더 강합니다.")
            momentum_details.append("rsi_near_high")
    if np.isfinite(macd_value) and np.isfinite(signal_value) and np.isfinite(hist_value):
        if macd_value > signal_value and hist_value > 0:
            momentum_score += 0.8
            bullish.append(f"MACD가 시그널선 위에 있고 히스토그램도 플러스라 모멘텀은 아직 상승 쪽에 가깝습니다. MACD {macd_value:,.3f}, Signal {signal_value:,.3f}입니다.")
            momentum_details.append("macd_bull")
        elif macd_value < signal_value and hist_value < 0:
            momentum_score -= 0.8
            bearish.append(f"MACD가 시그널선 아래이고 히스토그램도 마이너스라 모멘텀은 약세 쪽입니다. MACD {macd_value:,.3f}, Signal {signal_value:,.3f}입니다.")
            momentum_details.append("macd_bear")
    if np.isfinite(band_pos):
        if band_pos <= 0.2:
            momentum_score += 0.6
            bullish.append("주가가 볼린저 밴드 하단에 가까워 기술적 반등 관점의 매수 근거가 생깁니다.")
            momentum_details.append("bb_low")
        elif band_pos >= 0.8:
            momentum_score -= 0.6
            bearish.append("주가가 볼린저 밴드 상단에 가까워 단기 과열과 되돌림 가능성을 함께 봐야 합니다.")
            momentum_details.append("bb_high")
    momentum_score = _clip_score(momentum_score)
    score_rows.append({"Category": "Momentum", "Score": momentum_score, "Detail": ", ".join(momentum_details) or "-"})

    relative_score = 0.0
    relative_details: list[str] = []
    comparisons = [
        ("1M", 0.5),
        ("6M", 0.7),
        ("YTD", 0.8),
    ]
    for label, weight in comparisons:
        ticker_return = returns_ctx.period_returns.get(label)
        sector_return = returns_ctx.sector_average_returns.get(label)
        market_return = returns_ctx.market_average_returns.get(label)
        if ticker_return is None or sector_return is None or market_return is None:
            continue
        if ticker_return > sector_return and ticker_return > market_return:
            relative_score += weight
            bullish.append(f"{label} 수익률이 섹터와 S&P500을 모두 상회해 상대강도는 우호적입니다. 종목 {_format_pct(ticker_return)}, 섹터 {_format_pct(sector_return)}, S&P500 {_format_pct(market_return)}입니다.")
            relative_details.append(f"{label}_outperform")
        elif ticker_return < sector_return and ticker_return < market_return:
            relative_score -= weight
            bearish.append(f"{label} 수익률이 섹터와 S&P500을 모두 밑돌아 상대강도는 약합니다. 종목 {_format_pct(ticker_return)}, 섹터 {_format_pct(sector_return)}, S&P500 {_format_pct(market_return)}입니다.")
            relative_details.append(f"{label}_underperform")
    if returns_ctx.sector_rank_ytd is not None and returns_ctx.sector_count > 0:
        sector_pct = returns_ctx.sector_rank_ytd / max(1, returns_ctx.sector_count)
        if sector_pct <= 0.25:
            relative_score += 0.4
            bullish.append(f"섹터 내 YTD 순위가 상위권({returns_ctx.sector_rank_ytd}/{returns_ctx.sector_count})이라 강한 종목군에 속합니다.")
            relative_details.append("sector_top_quartile")
        elif sector_pct >= 0.75:
            relative_score -= 0.4
            bearish.append(f"섹터 내 YTD 순위가 하위권({returns_ctx.sector_rank_ytd}/{returns_ctx.sector_count})이라 수급 주도권이 약합니다.")
            relative_details.append("sector_bottom_quartile")
    if returns_ctx.market_rank_ytd is not None and returns_ctx.market_count > 0:
        market_pct = returns_ctx.market_rank_ytd / max(1, returns_ctx.market_count)
        if market_pct <= 0.10:
            relative_score += 0.3
            bullish.append(f"S&P500 전체에서도 YTD 상위권({returns_ctx.market_rank_ytd}/{returns_ctx.market_count})이라 시장 주도주 성격이 있습니다.")
            relative_details.append("market_top_decile")
        elif market_pct >= 0.90:
            relative_score -= 0.3
            bearish.append(f"S&P500 전체에서도 YTD 하위권({returns_ctx.market_rank_ytd}/{returns_ctx.market_count})이라 약세 흐름이 뚜렷합니다.")
            relative_details.append("market_bottom_decile")
    relative_score = _clip_score(relative_score)
    score_rows.append({"Category": "Relative Strength", "Score": relative_score, "Detail": ", ".join(relative_details) or "-"})

    risk_score = 0.0
    risk_details: list[str] = []
    if risk_ctx.sector_vol_rank_1y is not None and risk_ctx.sector_count > 0:
        sector_risk_pct = risk_ctx.sector_vol_rank_1y / max(1, risk_ctx.sector_count)
        if sector_risk_pct <= 0.25:
            risk_score -= 0.6
            bearish.append(f"섹터 내 1년 변동성 순위가 상위권({risk_ctx.sector_vol_rank_1y}/{risk_ctx.sector_count})이라 흔들림이 큰 편입니다.")
            risk_details.append("sector_high_vol")
        elif sector_risk_pct >= 0.75:
            risk_score += 0.4
            bullish.append(f"섹터 내 1년 변동성 순위가 낮은 편({risk_ctx.sector_vol_rank_1y}/{risk_ctx.sector_count})이라 방어력이 상대적으로 낫습니다.")
            risk_details.append("sector_low_vol")
    if risk_ctx.ticker_max_drawdown_1y is not None:
        if risk_ctx.ticker_max_drawdown_1y <= -0.25:
            risk_score -= 0.9
            bearish.append(f"최근 1년 최대낙폭이 {_format_pct(risk_ctx.ticker_max_drawdown_1y)}로 깊어, 손실 회복에 시간이 더 필요할 수 있습니다.")
            risk_details.append("deep_drawdown")
        elif risk_ctx.ticker_max_drawdown_1y >= -0.12:
            risk_score += 0.6
            bullish.append(f"최근 1년 최대낙폭이 {_format_pct(risk_ctx.ticker_max_drawdown_1y)} 수준으로 비교적 잘 통제되고 있습니다.")
            risk_details.append("contained_drawdown")
    if risk_ctx.beta_market_1y is not None:
        if risk_ctx.beta_market_1y >= 1.2:
            risk_score -= 0.4
            bearish.append(f"S&P500 대비 베타가 {risk_ctx.beta_market_1y:,.2f}로 높아 시장 급락 시 충격이 더 커질 수 있습니다.")
            risk_details.append("high_beta")
        elif risk_ctx.beta_market_1y <= 0.8:
            risk_score += 0.4
            bullish.append(f"S&P500 대비 베타가 {risk_ctx.beta_market_1y:,.2f}로 낮아 포지션 관리가 상대적으로 수월한 편입니다.")
            risk_details.append("low_beta")
    if risk_ctx.var_95_1d is not None:
        if risk_ctx.var_95_1d >= 0.035:
            risk_score -= 0.3
            bearish.append(f"95% 1일 VaR가 {_format_pct(risk_ctx.var_95_1d)}로 높아 단기 손실 허용 범위를 넉넉히 잡아야 합니다.")
            risk_details.append("high_var")
        elif risk_ctx.var_95_1d <= 0.02:
            risk_score += 0.2
            bullish.append(f"95% 1일 VaR가 {_format_pct(risk_ctx.var_95_1d)} 수준이라 일간 리스크 부담은 과도하지 않습니다.")
            risk_details.append("moderate_var")
    risk_score = _clip_score(risk_score)
    score_rows.append({"Category": "Risk", "Score": risk_score, "Detail": ", ".join(risk_details) or "-"})

    valuation_score = 0.0
    valuation_details: list[str] = []
    if fin_ctx is not None and fin_ctx.ticker.strip().upper() == ticker:
        per = _lookup_number(fin_ctx.metrics, "PER (Trailing)")
        pbr = _lookup_number(fin_ctx.metrics, "PBR")
        roe = _lookup_number(fin_ctx.metrics, "ROE")
        if roe is not None and roe > 1.0:
            roe = float(roe) / 100.0
        if roe is not None:
            if roe >= 0.15:
                valuation_score += 0.7
                bullish.append(f"ROE가 {_format_pct(roe)}로 높아 자본효율은 매수 논리를 보강합니다.")
                valuation_details.append("high_roe")
            elif roe <= 0.08:
                valuation_score -= 0.7
                bearish.append(f"ROE가 {_format_pct(roe)} 수준이라 수익성 측면의 매력은 약합니다.")
                valuation_details.append("low_roe")
        if per is not None:
            if per <= 20:
                valuation_score += 0.4
                bullish.append(f"PER이 {per:,.2f}배 수준이라 고평가 부담은 상대적으로 크지 않습니다.")
                valuation_details.append("fair_per")
            elif per >= 30:
                valuation_score -= 0.5
                bearish.append(f"PER이 {per:,.2f}배로 높아 실적이 조금만 흔들려도 밸류에이션 압박을 받을 수 있습니다.")
                valuation_details.append("high_per")
        if pbr is not None:
            if pbr <= 3:
                valuation_score += 0.2
                bullish.append(f"PBR이 {pbr:,.2f}배 수준이라 자산 대비 과열 부담은 제한적입니다.")
                valuation_details.append("fair_pbr")
            elif pbr >= 6:
                valuation_score -= 0.3
                bearish.append(f"PBR이 {pbr:,.2f}배로 높아 기대치가 이미 가격에 많이 반영되었을 수 있습니다.")
                valuation_details.append("high_pbr")
        if not valuation_details:
            watch_items.append("재무 스냅샷은 존재하지만 PER/PBR/ROE 중 일부가 비어 있어 밸류에이션 판단은 제한적입니다.")
    else:
        watch_items.append("금융 스냅샷이 아직 없어서 밸류에이션 점수는 중립 처리했습니다.")
    valuation_score = _clip_score(valuation_score, low=-1.5, high=1.5)
    score_rows.append({"Category": "Valuation", "Score": valuation_score, "Detail": ", ".join(valuation_details) or "neutral"})

    total_score = float(sum(float(row["Score"]) for row in score_rows))
    recommendation, confidence = _decision_label(total_score)

    if recommendation in {"매수 우위", "약한 매수 우위"}:
        final_commentary = (
            f"{ticker}에 대한 현재 종합 판단은 '{recommendation}'입니다. "
            f"추세와 상대강도에서 받쳐주는 요소가 남아 있고, {'재무지표까지 무난하게 따라오고 있어' if valuation_score > 0 else '재무지표는 추가 확인이 필요하지만'} "
            "신규 진입을 보더라도 분할 접근 관점이 유효합니다. "
            "다만 변동성 구간이 살아 있으면 눌림 확인 후 진입하는 편이 더 안전합니다."
        )
    elif recommendation in {"매도/차익실현 우위", "약한 매도 우위"}:
        final_commentary = (
            f"{ticker}에 대한 현재 종합 판단은 '{recommendation}'입니다. "
            "모멘텀이나 상대강도가 충분히 살아나지 못한 상태에서 리스크 지표까지 부담을 주고 있어, "
            "신규 매수보다는 비중 축소나 차익실현, 혹은 추세 재확인 이후 재진입을 고려하는 쪽이 합리적입니다."
        )
    else:
        final_commentary = (
            f"{ticker}에 대한 현재 종합 판단은 '{recommendation}'입니다. "
            "사는 쪽과 파는 쪽 논리가 모두 존재해 한쪽으로 강하게 기울기 어렵습니다. "
            "지금은 성급한 추격보다 다음 실적, 추세 재정렬, 상대강도 회복 여부를 확인하면서 포지션을 유연하게 관리하는 편이 좋습니다."
        )

    score_table = pd.DataFrame(
        [
            {
                "Category": str(row["Category"]),
                "Score": _format_score(float(row["Score"])),
                "Detail": str(row["Detail"]),
            }
            for row in score_rows
        ]
    )
    signal_table = pd.DataFrame(
        [
            {"Metric": "Close", "Value": f"{last_close:,.2f}"},
            {"Metric": "MA20", "Value": f"{ma20:,.2f}" if np.isfinite(ma20) else "-"},
            {"Metric": "MA60", "Value": f"{ma60:,.2f}" if np.isfinite(ma60) else "-"},
            {"Metric": "MA120", "Value": f"{ma120:,.2f}" if np.isfinite(ma120) else "-"},
            {"Metric": "RSI(14)", "Value": f"{rsi14:,.2f}" if np.isfinite(rsi14) else "-"},
            {"Metric": "MACD", "Value": f"{macd_value:,.4f}" if np.isfinite(macd_value) else "-"},
            {"Metric": "MACD Signal", "Value": f"{signal_value:,.4f}" if np.isfinite(signal_value) else "-"},
            {"Metric": "MACD Hist", "Value": f"{hist_value:,.4f}" if np.isfinite(hist_value) else "-"},
            {"Metric": "Bollinger Upper", "Value": f"{bb_upper_value:,.2f}" if np.isfinite(bb_upper_value) else "-"},
            {"Metric": "Bollinger Lower", "Value": f"{bb_lower_value:,.2f}" if np.isfinite(bb_lower_value) else "-"},
            {"Metric": "Sector YTD Rank", "Value": "-" if returns_ctx.sector_rank_ytd is None else f"{returns_ctx.sector_rank_ytd:,d} / {returns_ctx.sector_count:,d}"},
            {"Metric": "S&P 500 YTD Rank", "Value": "-" if returns_ctx.market_rank_ytd is None else f"{returns_ctx.market_rank_ytd:,d} / {returns_ctx.market_count:,d}"},
            {"Metric": "1Y Ann Vol", "Value": _format_pct(risk_ctx.ticker_vol_252d)},
            {"Metric": "1Y Max Drawdown", "Value": _format_pct(risk_ctx.ticker_max_drawdown_1y)},
            {"Metric": "Beta vs S&P 500", "Value": f"{risk_ctx.beta_market_1y:,.2f}" if risk_ctx.beta_market_1y is not None else "-"},
        ]
    )
    source_table = _metadata_table(
        [
            ("ticker", ticker),
            ("sector", returns_ctx.sector),
            ("close_source", close_source or "-"),
            ("returns_price_source", returns_ctx.price_source),
            ("risk_price_source", risk_ctx.price_source),
            ("market_cap_source", returns_ctx.market_cap_source),
            ("financial_snapshot_used", "yes" if fin_ctx is not None and fin_ctx.ticker.strip().upper() == ticker else "no"),
            ("latest_market_date", returns_ctx.latest_market_date),
            ("decision_score", f"{total_score:+.2f}"),
            ("recommendation", recommendation),
            ("confidence", confidence),
        ]
    )

    return _DecisionContext(
        ticker=ticker,
        sector=returns_ctx.sector,
        latest_market_date=returns_ctx.latest_market_date,
        recommendation=recommendation,
        confidence_label=confidence,
        total_score=total_score,
        bullish_reasons=bullish[:8],
        bearish_reasons=bearish[:8],
        watch_items=watch_items[:5],
        final_commentary=final_commentary,
        score_table=score_table,
        signal_table=signal_table,
        source_table=source_table,
        score_chart_base64=_decision_score_chart(score_rows),
        trend_chart_base64=_decision_trend_chart(ticker, close),
    )

def _weighted_daily_returns(
    price_df: pd.DataFrame,
    market_cap_df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    prices = price_df.copy()
    prices.index = pd.to_datetime(prices.index, errors="coerce")
    prices = prices[~prices.index.isna()].sort_index()
    prices = prices.apply(pd.to_numeric, errors="coerce")
    market_caps = _clean_market_cap_frame(market_cap_df).reindex(prices.index).ffill()
    daily_returns = prices.pct_change(fill_method=None)
    lagged_caps = market_caps.shift(1)
    weights = lagged_caps.where(daily_returns.notna())
    total_weight = weights.sum(axis=1, skipna=True)
    weighted_return = (daily_returns * weights).sum(axis=1, skipna=True) / total_weight.replace(0.0, np.nan)
    member_count = weights.notna().sum(axis=1)
    weighted_return = pd.to_numeric(weighted_return, errors="coerce").dropna()
    member_count = member_count.reindex(weighted_return.index).fillna(0).astype(int)
    return weighted_return, member_count


def _relative_returns_chart(
    ticker: str,
    ticker_daily: pd.Series,
    sector_daily: pd.Series,
    market_daily: pd.Series,
) -> str:
    ticker_curve = _base100_index_from_daily_returns(ticker_daily).rename(ticker)
    sector_curve = _base100_index_from_daily_returns(sector_daily).rename("Sector Cap-Weighted")
    market_curve = _base100_index_from_daily_returns(market_daily).rename("S&P 500 Cap-Weighted")
    chart_df = pd.concat([ticker_curve, sector_curve, market_curve], axis=1).dropna(how="all")

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    if not chart_df.empty:
        for name, color in [(ticker, "#0f4c81"), ("Sector Cap-Weighted", "#2e7d32"), ("S&P 500 Cap-Weighted", "#b26a00")]:
            if name in chart_df.columns:
                ax.plot(chart_df.index, chart_df[name], linewidth=2.0, label=name, color=color)
    _apply_year_month_axis(ax)
    ax.set_title("YTD Base 100 Index")
    ax.set_ylabel("Index (Start=100)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return _render_chart_base64(fig)


def _daily_return_comparison_chart(
    ticker: str,
    ticker_daily_table: pd.DataFrame,
    sector_daily_table: pd.DataFrame,
) -> str:
    ticker_df = ticker_daily_table.copy()
    sector_df = sector_daily_table.copy()
    ticker_df["Date"] = pd.to_datetime(ticker_df["Date"], errors="coerce")
    sector_df["Date"] = pd.to_datetime(sector_df["Date"], errors="coerce")
    chart_df = pd.merge(
        ticker_df[["Date", "Daily Return"]],
        sector_df[["Date", "Sector Cap-Weighted Return"]],
        on="Date",
        how="outer",
    ).sort_values("Date").tail(10)

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    if not chart_df.empty:
        ax.bar(chart_df["Date"], chart_df["Daily Return"] * 100.0, width=0.8, color="#0f4c81", alpha=0.75, label=ticker)
        ax.plot(chart_df["Date"], chart_df["Sector Cap-Weighted Return"] * 100.0, color="#2e7d32", linewidth=2.0, marker="o", label="Sector Cap-Weighted")
    _apply_year_month_axis(ax)
    ax.axhline(0.0, color="#8a98a8", linewidth=1.0)
    ax.set_title("Daily Return vs Sector Cap-Weighted Return (Last 10 Business Days)")
    ax.set_ylabel("Return (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return _render_chart_base64(fig)


def _rank_table_display(df: pd.DataFrame, *, rank_col: str = "Rank") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Rank", "Symbol", "Sector", "As Of", "YTD Return"])
    out = df.copy()
    if rank_col != "Rank" and rank_col in out.columns:
        out["Rank"] = out[rank_col]
    out["As Of"] = pd.to_datetime(out["latest_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["YTD Return"] = out["YTD"].map(_format_pct)
    return out[["Rank", "Symbol", "Sector", "As Of", "YTD Return"]].reset_index(drop=True)


def _run_returns_once(form: dict[str, str]) -> _ReturnsContext:
    raw_ticker = str(form.get("ticker", "")).strip().upper()
    if not raw_ticker:
        raise ValueError("Provide an S&P 500 ticker for return analysis.")

    components, components_source = _load_sp500_components_full()
    components = components.drop_duplicates(subset=["Symbol"], keep="last").reset_index(drop=True)
    if raw_ticker not in set(components["Symbol"]):
        raise ValueError(f"{raw_ticker} is not available in the shared S&P 500 universe.")

    sector = str(components.loc[components["Symbol"] == raw_ticker, "Sector"].iloc[0])
    all_symbols = components["Symbol"].tolist()
    start_date = (pd.Timestamp.today().normalize() - pd.DateOffset(years=4, months=1)).strftime("%Y-%m-%d")
    prices, price_source = fetch_sp500_close_prices(all_symbols, start_date=start_date)
    if prices.empty or raw_ticker not in prices.columns:
        raise ValueError(f"Shared price data for {raw_ticker} is unavailable.")
    market_caps, market_cap_source = load_shared_market_caps_for_symbols(all_symbols, start_date=start_date)
    if market_caps is None or market_caps.empty or raw_ticker not in market_caps.columns:
        raise ValueError(f"Shared market cap data for {raw_ticker} is unavailable.")

    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index, errors="coerce")
    prices = prices[~prices.index.isna()].sort_index()
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.loc[:, [symbol for symbol in all_symbols if symbol in prices.columns]]
    prices = prices.dropna(how="all")
    if prices.empty:
        raise ValueError("Shared price data did not return usable rows.")
    market_caps = _clean_market_cap_frame(market_caps)
    market_caps = market_caps.loc[:, [symbol for symbol in all_symbols if symbol in market_caps.columns]]
    market_caps = market_caps.dropna(how="all")
    if market_caps.empty:
        raise ValueError("Shared market cap data did not return usable rows.")

    market_latest_date = pd.Timestamp(prices.index.max()).normalize()
    ticker_series = _clean_close_series(prices[raw_ticker])
    if ticker_series.empty:
        raise ValueError(f"Shared price data for {raw_ticker} is empty.")

    period_returns, ticker_latest_date = _compute_period_returns(ticker_series)
    if ticker_latest_date is None:
        raise ValueError(f"Could not compute period returns for {raw_ticker}.")

    daily_returns = prices.pct_change(fill_method=None)
    ticker_daily = pd.to_numeric(daily_returns[raw_ticker], errors="coerce").dropna()

    sector_symbols = components.loc[components["Sector"] == sector, "Symbol"].tolist()
    sector_weight_symbols = [symbol for symbol in sector_symbols if symbol in prices.columns and symbol in market_caps.columns]
    market_weight_symbols = [symbol for symbol in all_symbols if symbol in prices.columns and symbol in market_caps.columns]
    if not sector_weight_symbols:
        raise ValueError(f"Shared market cap data for sector {sector} is unavailable.")
    if not market_weight_symbols:
        raise ValueError("Shared market cap data for the S&P 500 universe is unavailable.")
    sector_weighted_daily, sector_daily_counts = _weighted_daily_returns(prices[sector_weight_symbols], market_caps[sector_weight_symbols])
    market_weighted_daily, market_daily_counts = _weighted_daily_returns(prices[market_weight_symbols], market_caps[market_weight_symbols])
    sector_daily = sector_weighted_daily
    market_daily = market_weighted_daily
    ytd_start = ticker_latest_date.to_period("Y").start_time.normalize()
    ticker_ytd_daily = ticker_daily[ticker_daily.index >= ytd_start]
    sector_ytd_daily = sector_daily[sector_daily.index >= ytd_start]
    market_ytd_daily = market_daily[market_daily.index >= ytd_start]

    daily_returns_numeric = pd.DataFrame(
        {
            "Date": ticker_daily.tail(10).index.strftime("%Y-%m-%d"),
            "Close": ticker_series.reindex(ticker_daily.tail(10).index).map(lambda v: "-" if pd.isna(v) else f"{float(v):,.2f}"),
            "Daily Return": ticker_daily.tail(10).values,
        }
    )
    daily_returns_table = pd.DataFrame(
        {
            "Date": daily_returns_numeric["Date"],
            f"{raw_ticker} Close": daily_returns_numeric["Close"],
            f"{raw_ticker} Daily Return": daily_returns_numeric["Daily Return"].map(_format_pct),
        }
    )

    sector_daily_window = sector_daily.tail(10)
    sector_daily_numeric = pd.DataFrame(
        {
            "Date": sector_daily_window.index.strftime("%Y-%m-%d"),
            "Sector Cap-Weighted Return": sector_daily_window.values,
            "Members Used": sector_daily_counts.reindex(sector_daily_window.index).fillna(0).astype(int).values,
        }
    )
    sector_daily_returns_table = pd.DataFrame(
        {
            "Date": sector_daily_numeric["Date"],
            f"{sector} Sector Cap-Weighted Return": sector_daily_numeric["Sector Cap-Weighted Return"].map(_format_pct),
            "Members Used": sector_daily_numeric["Members Used"],
        }
    )

    return_rows: list[dict[str, object]] = []
    for symbol in all_symbols:
        if symbol not in prices.columns:
            continue
        symbol_series = _clean_close_series(prices[symbol])
        if symbol_series.empty:
            continue
        symbol_returns, latest_date = _compute_period_returns(symbol_series)
        row: dict[str, object] = {
            "Symbol": symbol,
            "Sector": str(components.loc[components["Symbol"] == symbol, "Sector"].iloc[0]),
            "latest_date": latest_date,
        }
        row.update(symbol_returns)
        return_rows.append(row)

    returns_df = pd.DataFrame(return_rows)
    if returns_df.empty:
        raise ValueError("Could not build return ranking tables from shared data.")

    for label in _RETURN_PERIOD_LABELS:
        if label not in returns_df.columns:
            returns_df[label] = np.nan

    returns_df["Rank"] = (
        returns_df["YTD"]
        .rank(method="min", ascending=False)
        .where(returns_df["YTD"].notna())
        .astype("Int64")
    )
    returns_df = returns_df.sort_values(["YTD", "Symbol"], ascending=[False, True], na_position="last").reset_index(drop=True)

    sector_df = returns_df[returns_df["Sector"] == sector].copy()
    sector_df["Sector Rank"] = (
        sector_df["YTD"]
        .rank(method="min", ascending=False)
        .where(sector_df["YTD"].notna())
        .astype("Int64")
    )
    ticker_row = returns_df.loc[returns_df["Symbol"] == raw_ticker]
    sector_ticker_row = sector_df.loc[sector_df["Symbol"] == raw_ticker]

    sector_average_returns, _ = _compute_period_returns_from_daily_returns(sector_daily)
    market_average_returns, _ = _compute_period_returns_from_daily_returns(market_daily)

    summary_rows: list[dict[str, object]] = []
    for label in _RETURN_PERIOD_LABELS:
        ticker_value = period_returns.get(label)
        sector_value = sector_average_returns.get(label)
        market_value = market_average_returns.get(label)
        summary_rows.append(
            {
                "Period": label,
                "Ticker": _format_pct(ticker_value),
                "Sector Cap-Weighted": _format_pct(sector_value),
                "S&P 500 Cap-Weighted": _format_pct(market_value),
                "Vs Sector": _format_pct(None if ticker_value is None or sector_value is None else ticker_value - sector_value),
                "Vs S&P 500": _format_pct(None if ticker_value is None or market_value is None else ticker_value - market_value),
            }
        )
    summary_table = pd.DataFrame(summary_rows)

    sector_top_table = _rank_table_display(sector_df.dropna(subset=["YTD"]).head(10), rank_col="Sector Rank")
    sector_bottom_table = _rank_table_display(
        sector_df.dropna(subset=["YTD"]).sort_values(["YTD", "Symbol"], ascending=[True, True]).head(10),
        rank_col="Sector Rank",
    )
    market_top_table = _rank_table_display(returns_df.dropna(subset=["YTD"]).head(10))
    market_bottom_table = _rank_table_display(returns_df.dropna(subset=["YTD"]).sort_values(["YTD", "Symbol"], ascending=[True, True]).head(10))

    source_table = _metadata_table(
        [
            ("ticker", raw_ticker),
            ("sector", sector),
            ("components_source", components_source),
            ("price_source", price_source),
            ("market_cap_source", market_cap_source),
            ("weighting_method", "cap_weighted"),
            ("shared_universe_size", len(all_symbols)),
            ("sector_member_count", len(sector_symbols)),
            ("sector_weight_member_count", len(sector_weight_symbols)),
            ("market_weight_member_count", len(market_weight_symbols)),
            ("latest_market_date", market_latest_date.strftime("%Y-%m-%d")),
            ("ticker_latest_date", ticker_latest_date.strftime("%Y-%m-%d")),
        ]
    )

    sector_rank = None if sector_ticker_row.empty or pd.isna(sector_ticker_row.iloc[0].get("Sector Rank")) else int(
        sector_ticker_row.iloc[0]["Sector Rank"]
    )
    market_rank = None if ticker_row.empty or pd.isna(ticker_row.iloc[0].get("YTD")) else int(ticker_row.iloc[0]["Rank"])

    return _ReturnsContext(
        ticker=raw_ticker,
        sector=sector,
        latest_market_date=market_latest_date.strftime("%Y-%m-%d"),
        ticker_latest_date=ticker_latest_date.strftime("%Y-%m-%d"),
        price_source=price_source,
        market_cap_source=market_cap_source,
        period_returns=period_returns,
        sector_average_returns=sector_average_returns,
        market_average_returns=market_average_returns,
        sector_rank_ytd=sector_rank,
        sector_count=int(sector_df["YTD"].notna().sum()),
        market_rank_ytd=market_rank,
        market_count=int(returns_df["YTD"].notna().sum()),
        summary_table=summary_table,
        source_table=source_table,
        daily_returns_table=daily_returns_table,
        sector_daily_returns_table=sector_daily_returns_table,
        sector_top_table=sector_top_table,
        sector_bottom_table=sector_bottom_table,
        market_top_table=market_top_table,
        market_bottom_table=market_bottom_table,
        relative_chart_base64=_relative_returns_chart(raw_ticker, ticker_ytd_daily, sector_ytd_daily, market_ytd_daily),
        daily_chart_base64=_daily_return_comparison_chart(raw_ticker, daily_returns_numeric, sector_daily_numeric),
    )


def _run_risk_once(form: dict[str, str]) -> _RiskContext:
    raw_ticker = str(form.get("ticker", "")).strip().upper()
    if not raw_ticker:
        raise ValueError("Provide an S&P 500 ticker for risk analysis.")

    components, components_source = _load_sp500_components_full()
    components = components.drop_duplicates(subset=["Symbol"], keep="last").reset_index(drop=True)
    if raw_ticker not in set(components["Symbol"]):
        raise ValueError(f"{raw_ticker} is not available in the shared S&P 500 universe.")

    sector = str(components.loc[components["Symbol"] == raw_ticker, "Sector"].iloc[0])
    all_symbols = components["Symbol"].tolist()
    start_date = (pd.Timestamp.today().normalize() - pd.DateOffset(years=4, months=1)).strftime("%Y-%m-%d")
    prices, price_source = fetch_sp500_close_prices(all_symbols, start_date=start_date)
    if prices.empty or raw_ticker not in prices.columns:
        raise ValueError(f"Shared price data for {raw_ticker} is unavailable.")
    market_caps, market_cap_source = load_shared_market_caps_for_symbols(all_symbols, start_date=start_date)
    if market_caps is None or market_caps.empty or raw_ticker not in market_caps.columns:
        raise ValueError(f"Shared market cap data for {raw_ticker} is unavailable.")

    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index, errors="coerce")
    prices = prices[~prices.index.isna()].sort_index()
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.loc[:, [symbol for symbol in all_symbols if symbol in prices.columns]]
    prices = prices.dropna(how="all")
    if prices.empty:
        raise ValueError("Shared price data did not return usable rows.")
    market_caps = _clean_market_cap_frame(market_caps)
    market_caps = market_caps.loc[:, [symbol for symbol in all_symbols if symbol in market_caps.columns]]
    market_caps = market_caps.dropna(how="all")
    if market_caps.empty:
        raise ValueError("Shared market cap data did not return usable rows.")

    market_latest_date = pd.Timestamp(prices.index.max()).normalize()
    ticker_series = _clean_close_series(prices[raw_ticker])
    if ticker_series.empty:
        raise ValueError(f"Shared price data for {raw_ticker} is empty.")
    ticker_latest_date = pd.Timestamp(ticker_series.index.max()).normalize()

    daily_returns = prices.pct_change(fill_method=None)
    ticker_daily = _clean_return_series(daily_returns[raw_ticker])
    if ticker_daily.empty:
        raise ValueError(f"Daily return history for {raw_ticker} is unavailable.")

    sector_symbols = components.loc[components["Sector"] == sector, "Symbol"].tolist()
    sector_weight_symbols = [symbol for symbol in sector_symbols if symbol in prices.columns and symbol in market_caps.columns]
    market_weight_symbols = [symbol for symbol in all_symbols if symbol in prices.columns and symbol in market_caps.columns]
    if not sector_weight_symbols:
        raise ValueError(f"Shared market cap data for sector {sector} is unavailable.")
    if not market_weight_symbols:
        raise ValueError("Shared market cap data for the S&P 500 universe is unavailable.")
    sector_daily, _ = _weighted_daily_returns(prices[sector_weight_symbols], market_caps[sector_weight_symbols])
    market_daily, _ = _weighted_daily_returns(prices[market_weight_symbols], market_caps[market_weight_symbols])

    ticker_vol_20d = _annualized_volatility(ticker_daily, 20)
    ticker_vol_60d = _annualized_volatility(ticker_daily, 60)
    ticker_vol_252d = _annualized_volatility(ticker_daily, 252)
    sector_vol_20d = _annualized_volatility(sector_daily, 20)
    sector_vol_60d = _annualized_volatility(sector_daily, 60)
    sector_vol_252d = _annualized_volatility(sector_daily, 252)
    market_vol_20d = _annualized_volatility(market_daily, 20)
    market_vol_60d = _annualized_volatility(market_daily, 60)
    market_vol_252d = _annualized_volatility(market_daily, 252)

    ticker_downside_1y = _downside_volatility(ticker_daily, 252)
    sector_downside_1y = _downside_volatility(sector_daily, 252)
    market_downside_1y = _downside_volatility(market_daily, 252)
    ticker_mdd_1y = _max_drawdown_from_daily_returns(ticker_daily, 252)
    sector_mdd_1y = _max_drawdown_from_daily_returns(sector_daily, 252)
    market_mdd_1y = _max_drawdown_from_daily_returns(market_daily, 252)
    ticker_mdd_3y = _max_drawdown_from_daily_returns(ticker_daily, 756)
    sector_mdd_3y = _max_drawdown_from_daily_returns(sector_daily, 756)
    market_mdd_3y = _max_drawdown_from_daily_returns(market_daily, 756)
    ticker_var_95, ticker_cvar_95 = _historical_var_cvar(ticker_daily, confidence=0.95, window=252)
    sector_var_95, sector_cvar_95 = _historical_var_cvar(sector_daily, confidence=0.95, window=252)
    market_var_95, market_cvar_95 = _historical_var_cvar(market_daily, confidence=0.95, window=252)
    beta_sector_1y, corr_sector_1y = _beta_and_correlation(ticker_daily, sector_daily, window=252)
    beta_market_1y, corr_market_1y = _beta_and_correlation(ticker_daily, market_daily, window=252)

    worst_day_1y = _clean_return_series(ticker_daily).tail(252)
    sector_worst_day_1y = _clean_return_series(sector_daily).tail(252)
    market_worst_day_1y = _clean_return_series(market_daily).tail(252)

    summary_table = pd.DataFrame(
        [
            {
                "Metric": "20D Ann Vol",
                "Ticker": _format_pct(ticker_vol_20d),
                "Sector Cap-Weighted": _format_pct(sector_vol_20d),
                "S&P 500 Cap-Weighted": _format_pct(market_vol_20d),
            },
            {
                "Metric": "60D Ann Vol",
                "Ticker": _format_pct(ticker_vol_60d),
                "Sector Cap-Weighted": _format_pct(sector_vol_60d),
                "S&P 500 Cap-Weighted": _format_pct(market_vol_60d),
            },
            {
                "Metric": "1Y Ann Vol",
                "Ticker": _format_pct(ticker_vol_252d),
                "Sector Cap-Weighted": _format_pct(sector_vol_252d),
                "S&P 500 Cap-Weighted": _format_pct(market_vol_252d),
            },
            {
                "Metric": "1Y Downside Vol",
                "Ticker": _format_pct(ticker_downside_1y),
                "Sector Cap-Weighted": _format_pct(sector_downside_1y),
                "S&P 500 Cap-Weighted": _format_pct(market_downside_1y),
            },
            {
                "Metric": "1Y Max Drawdown",
                "Ticker": _format_pct(ticker_mdd_1y),
                "Sector Cap-Weighted": _format_pct(sector_mdd_1y),
                "S&P 500 Cap-Weighted": _format_pct(market_mdd_1y),
            },
            {
                "Metric": "3Y Max Drawdown",
                "Ticker": _format_pct(ticker_mdd_3y),
                "Sector Cap-Weighted": _format_pct(sector_mdd_3y),
                "S&P 500 Cap-Weighted": _format_pct(market_mdd_3y),
            },
            {
                "Metric": "95% 1D VaR",
                "Ticker": _format_pct(ticker_var_95),
                "Sector Cap-Weighted": _format_pct(sector_var_95),
                "S&P 500 Cap-Weighted": _format_pct(market_var_95),
            },
            {
                "Metric": "95% 1D CVaR",
                "Ticker": _format_pct(ticker_cvar_95),
                "Sector Cap-Weighted": _format_pct(sector_cvar_95),
                "S&P 500 Cap-Weighted": _format_pct(market_cvar_95),
            },
            {
                "Metric": "Worst Day (1Y)",
                "Ticker": _format_pct(float(worst_day_1y.min()) if not worst_day_1y.empty else None),
                "Sector Cap-Weighted": _format_pct(float(sector_worst_day_1y.min()) if not sector_worst_day_1y.empty else None),
                "S&P 500 Cap-Weighted": _format_pct(float(market_worst_day_1y.min()) if not market_worst_day_1y.empty else None),
            },
            {
                "Metric": "Beta vs Sector (1Y)",
                "Ticker": _format_metric(beta_sector_1y, 2) if beta_sector_1y is not None else "-",
                "Sector Cap-Weighted": "1.00",
                "S&P 500 Cap-Weighted": "-",
            },
            {
                "Metric": "Beta vs S&P 500 (1Y)",
                "Ticker": _format_metric(beta_market_1y, 2) if beta_market_1y is not None else "-",
                "Sector Cap-Weighted": "-",
                "S&P 500 Cap-Weighted": "1.00",
            },
            {
                "Metric": "Corr vs Sector (1Y)",
                "Ticker": _format_metric(corr_sector_1y, 2) if corr_sector_1y is not None else "-",
                "Sector Cap-Weighted": "1.00",
                "S&P 500 Cap-Weighted": "-",
            },
            {
                "Metric": "Corr vs S&P 500 (1Y)",
                "Ticker": _format_metric(corr_market_1y, 2) if corr_market_1y is not None else "-",
                "Sector Cap-Weighted": "-",
                "S&P 500 Cap-Weighted": "1.00",
            },
        ]
    )

    risk_rows: list[dict[str, object]] = []
    for symbol in all_symbols:
        if symbol not in prices.columns:
            continue
        symbol_series = _clean_close_series(prices[symbol])
        if symbol_series.empty:
            continue
        symbol_daily = _clean_return_series(daily_returns[symbol])
        if symbol_daily.empty:
            continue
        risk_rows.append(
            {
                "Symbol": symbol,
                "Sector": str(components.loc[components["Symbol"] == symbol, "Sector"].iloc[0]),
                "latest_date": pd.Timestamp(symbol_series.index.max()).normalize(),
                "ann_vol_1y": _annualized_volatility(symbol_daily, 252),
                "max_drawdown_1y": _max_drawdown_from_daily_returns(symbol_daily, 252),
            }
        )

    risk_df = pd.DataFrame(risk_rows)
    if risk_df.empty:
        raise ValueError("Could not build risk ranking tables from shared data.")

    risk_df["Vol Rank"] = (
        risk_df["ann_vol_1y"]
        .rank(method="min", ascending=False)
        .where(risk_df["ann_vol_1y"].notna())
        .astype("Int64")
    )
    risk_df = risk_df.sort_values(["ann_vol_1y", "Symbol"], ascending=[False, True], na_position="last").reset_index(drop=True)
    sector_df = risk_df[risk_df["Sector"] == sector].copy()
    sector_df["Sector Vol Rank"] = (
        sector_df["ann_vol_1y"]
        .rank(method="min", ascending=False)
        .where(sector_df["ann_vol_1y"].notna())
        .astype("Int64")
    )

    ticker_row = risk_df.loc[risk_df["Symbol"] == raw_ticker]
    sector_ticker_row = sector_df.loc[sector_df["Symbol"] == raw_ticker]
    sector_rank = None if sector_ticker_row.empty or pd.isna(sector_ticker_row.iloc[0].get("Sector Vol Rank")) else int(
        sector_ticker_row.iloc[0]["Sector Vol Rank"]
    )
    market_rank = None if ticker_row.empty or pd.isna(ticker_row.iloc[0].get("Vol Rank")) else int(ticker_row.iloc[0]["Vol Rank"])

    source_table = _metadata_table(
        [
            ("ticker", raw_ticker),
            ("sector", sector),
            ("components_source", components_source),
            ("price_source", price_source),
            ("market_cap_source", market_cap_source),
            ("weighting_method", "cap_weighted"),
            ("shared_universe_size", len(all_symbols)),
            ("sector_member_count", len(sector_symbols)),
            ("sector_weight_member_count", len(sector_weight_symbols)),
            ("market_weight_member_count", len(market_weight_symbols)),
            ("latest_market_date", market_latest_date.strftime("%Y-%m-%d")),
            ("ticker_latest_date", ticker_latest_date.strftime("%Y-%m-%d")),
        ]
    )

    return _RiskContext(
        ticker=raw_ticker,
        sector=sector,
        latest_market_date=market_latest_date.strftime("%Y-%m-%d"),
        ticker_latest_date=ticker_latest_date.strftime("%Y-%m-%d"),
        price_source=price_source,
        market_cap_source=market_cap_source,
        ticker_vol_20d=ticker_vol_20d,
        ticker_vol_60d=ticker_vol_60d,
        ticker_vol_252d=ticker_vol_252d,
        ticker_max_drawdown_1y=ticker_mdd_1y,
        ticker_max_drawdown_3y=ticker_mdd_3y,
        beta_sector_1y=beta_sector_1y,
        beta_market_1y=beta_market_1y,
        var_95_1d=ticker_var_95,
        cvar_95_1d=ticker_cvar_95,
        sector_vol_rank_1y=sector_rank,
        sector_count=int(sector_df["ann_vol_1y"].notna().sum()),
        market_vol_rank_1y=market_rank,
        market_count=int(risk_df["ann_vol_1y"].notna().sum()),
        summary_table=summary_table,
        source_table=source_table,
        recent_shock_table=_recent_risk_shock_table(raw_ticker, ticker_daily, sector_daily, market_daily),
        sector_high_vol_table=_risk_rank_table_display(sector_df.dropna(subset=["ann_vol_1y"]).head(10), rank_col="Sector Vol Rank"),
        sector_low_vol_table=_risk_rank_table_display(
            sector_df.dropna(subset=["ann_vol_1y"]).sort_values(["ann_vol_1y", "Symbol"], ascending=[True, True]).head(10),
            rank_col="Sector Vol Rank",
        ),
        market_high_vol_table=_risk_rank_table_display(risk_df.dropna(subset=["ann_vol_1y"]).head(10), rank_col="Vol Rank"),
        market_low_vol_table=_risk_rank_table_display(
            risk_df.dropna(subset=["ann_vol_1y"]).sort_values(["ann_vol_1y", "Symbol"], ascending=[True, True]).head(10),
            rank_col="Vol Rank",
        ),
        drawdown_chart_base64=_drawdown_comparison_chart(raw_ticker, ticker_daily, sector_daily, market_daily),
        volatility_chart_base64=_rolling_volatility_chart(raw_ticker, ticker_daily, sector_daily, market_daily),
        commentary=_risk_commentary(
            ticker=raw_ticker,
            sector=sector,
            ticker_vol_252d=ticker_vol_252d,
            sector_vol_252d=sector_vol_252d,
            market_vol_252d=market_vol_252d,
            ticker_max_drawdown_1y=ticker_mdd_1y,
            beta_market_1y=beta_market_1y,
            var_95_1d=ticker_var_95,
        ),
    )


def _trend_regime(close_series: pd.Series) -> str:
    clean = _clean_close_series(close_series)
    if len(clean) < 120:
        return "판별 보류 (insufficient history)"
    sma20 = clean.rolling(20).mean().iloc[-1]
    sma60 = clean.rolling(60).mean().iloc[-1]
    sma120 = clean.rolling(120).mean().iloc[-1]
    last_price = float(clean.iloc[-1])
    if not all(np.isfinite(v) for v in [sma20, sma60, sma120, last_price]):
        return "판별 보류 (insufficient history)"
    if last_price > sma20 > sma60 > sma120:
        return "상승 추세 우위 (bull trend)"
    if last_price < sma20 < sma60 < sma120:
        return "하락 추세 우위 (bear trend)"
    if sma20 > sma60 and last_price >= sma60:
        return "상승 전환 시도 (early upturn)"
    if sma20 < sma60 and last_price <= sma60:
        return "하락 압력 지속 (downtrend pressure)"
    return "혼합 추세 (mixed trend)"


def _volatility_regime(ticker_daily: pd.Series) -> str:
    rolling_vol = _rolling_volatility_pct(ticker_daily, 20, 252)
    if rolling_vol.empty:
        return "판별 보류 (insufficient history)"
    current_vol = float(rolling_vol.iloc[-1])
    median_vol = float(rolling_vol.median())
    if not np.isfinite(current_vol) or not np.isfinite(median_vol) or median_vol <= 0.0:
        return "판별 보류 (insufficient history)"
    if current_vol >= median_vol * 1.2:
        return "고변동성 (high volatility)"
    if current_vol <= median_vol * 0.85:
        return "저변동성 (calm volatility)"
    return "중립 변동성 (normal volatility)"


def _beta_regime(beta_market_60d: float | None) -> str:
    if beta_market_60d is None or not np.isfinite(beta_market_60d):
        return "판별 보류 (insufficient history)"
    if beta_market_60d >= 1.2:
        return "공격적 베타 (high beta)"
    if beta_market_60d <= 0.85:
        return "방어적 베타 (defensive beta)"
    return "시장유사 베타 (market-like beta)"


def _overall_regime_label(trend_label: str, volatility_label: str, beta_label: str) -> str:
    if "하락" in trend_label and "고변동성" in volatility_label:
        return "스트레스 국면 (risk-off stress)"
    if "상승" in trend_label and "저변동성" in volatility_label and "공격적" not in beta_label:
        return "안정 상승 국면 (stable uptrend)"
    if "상승" in trend_label and "공격적" in beta_label:
        return "고베타 추세 국면 (high-beta trend)"
    if "방어적" in beta_label and "저변동성" in volatility_label:
        return "방어 안정 국면 (defensive calm)"
    return "혼합 국면 (mixed regime)"


def _cumulative_residual_pct(daily_residual: pd.Series) -> pd.Series:
    clean = _clean_return_series(daily_residual)
    if clean.empty:
        return pd.Series(dtype=float)
    return (clean.cumsum() * 100.0).rename(clean.name or "Residual Return")


def _factor_beta_chart(ticker: str, beta_market_curve: pd.Series, beta_sector_curve: pd.Series) -> str:
    chart_df = pd.concat(
        [
            beta_market_curve.rename("Beta vs S&P 500"),
            beta_sector_curve.rename("Beta vs Sector"),
        ],
        axis=1,
    ).dropna(how="all").tail(252)

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    if not chart_df.empty:
        if "Beta vs S&P 500" in chart_df.columns:
            ax.plot(chart_df.index, chart_df["Beta vs S&P 500"], color="#0f4c81", linewidth=2.0, label="Beta vs S&P 500")
        if "Beta vs Sector" in chart_df.columns:
            ax.plot(chart_df.index, chart_df["Beta vs Sector"], color="#b26a00", linewidth=2.0, label="Beta vs Sector")
    ax.axhline(1.0, color="#6c7b88", linewidth=1.0, linestyle="--")
    _apply_year_month_axis(ax)
    ax.set_title(f"{ticker} Rolling 60-Day Beta")
    ax.set_ylabel("Beta")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return _render_chart_base64(fig)


def _factor_residual_chart(ticker: str, residual_market_daily: pd.Series, residual_sector_daily: pd.Series) -> str:
    chart_df = pd.concat(
        [
            _cumulative_residual_pct(residual_market_daily).rename("Residual vs S&P 500"),
            _cumulative_residual_pct(residual_sector_daily).rename("Residual vs Sector"),
        ],
        axis=1,
    ).dropna(how="all").tail(126)

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    if not chart_df.empty:
        if "Residual vs S&P 500" in chart_df.columns:
            ax.plot(chart_df.index, chart_df["Residual vs S&P 500"], color="#0f4c81", linewidth=2.0, label="Residual vs S&P 500")
        if "Residual vs Sector" in chart_df.columns:
            ax.plot(chart_df.index, chart_df["Residual vs Sector"], color="#2e7d32", linewidth=2.0, label="Residual vs Sector")
    ax.axhline(0.0, color="#6c7b88", linewidth=1.0)
    _apply_year_month_axis(ax)
    ax.set_title(f"{ticker} Cumulative Residual Return")
    ax.set_ylabel("Residual Return (%)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return _render_chart_base64(fig)


def _factor_commentary(
    *,
    ticker: str,
    sector: str,
    beta_market_60d: float | None,
    corr_market_60d: float | None,
    residual_market_20d: float | None,
    residual_sector_20d: float | None,
    regime_overall: str,
) -> str:
    parts: list[str] = []
    if beta_market_60d is not None:
        if beta_market_60d >= 1.2:
            parts.append(f"{ticker}는 최근 60거래일 기준으로 시장보다 큰 폭으로 반응하는 고베타(high beta) 성격이 강합니다.")
        elif beta_market_60d <= 0.85:
            parts.append(f"{ticker}는 최근 60거래일 기준으로 시장보다 덜 흔들리는 방어형(defensive) 반응을 보였습니다.")
        else:
            parts.append(f"{ticker}의 최근 60거래일 베타(beta)는 시장 평균에 가까워, S&P 500과 비교적 비슷한 민감도로 움직였습니다.")
    if corr_market_60d is not None:
        if corr_market_60d >= 0.8:
            parts.append("시장과의 동행성(correlation)이 높아 개별 종목 이슈보다 시장 요인의 설명력이 큰 구간으로 읽힙니다.")
        elif corr_market_60d <= 0.45:
            parts.append("시장과의 동행성(correlation)이 높지 않아 최근에는 종목 고유 요인(idiosyncratic factor)의 비중이 상대적으로 커 보입니다.")
    if residual_market_20d is not None:
        if residual_market_20d > 0.03:
            parts.append("최근 20거래일 잔차수익률(residual return)이 플러스여서 시장 공통 요인을 제거한 뒤에도 초과성과가 남았습니다.")
        elif residual_market_20d < -0.03:
            parts.append("최근 20거래일 잔차수익률(residual return)이 마이너스여서 시장을 따라간 부분을 제외하면 종목 고유 성과는 약했습니다.")
    if residual_sector_20d is not None:
        if residual_sector_20d > 0.02:
            parts.append(f"{sector} 섹터 내부 비교에서도 잔차성과가 양호해, 최근에는 섹터 대표주보다 상대적으로 강한 편에 가깝습니다.")
        elif residual_sector_20d < -0.02:
            parts.append(f"{sector} 섹터 내부 비교에서는 아직 종목 고유 성과가 섹터 평균보다 약한 편입니다.")
    parts.append(f"종합 국면(regime)은 현재 {regime_overall}으로 요약됩니다.")
    return " ".join(parts)


def _run_factor_once(form: dict[str, str]) -> _FactorContext:
    raw_ticker = str(form.get("ticker", "")).strip().upper()
    if not raw_ticker:
        raise ValueError("Provide an S&P 500 ticker for factor and regime analysis.")

    components, components_source = _load_sp500_components_full()
    components = components.drop_duplicates(subset=["Symbol"], keep="last").reset_index(drop=True)
    if raw_ticker not in set(components["Symbol"]):
        raise ValueError(f"{raw_ticker} is not available in the shared S&P 500 universe.")

    sector = str(components.loc[components["Symbol"] == raw_ticker, "Sector"].iloc[0])
    all_symbols = components["Symbol"].tolist()
    start_date = (pd.Timestamp.today().normalize() - pd.DateOffset(years=4, months=1)).strftime("%Y-%m-%d")
    prices, price_source = fetch_sp500_close_prices(all_symbols, start_date=start_date)
    if prices.empty or raw_ticker not in prices.columns:
        raise ValueError(f"Shared price data for {raw_ticker} is unavailable.")
    market_caps, market_cap_source = load_shared_market_caps_for_symbols(all_symbols, start_date=start_date)
    if market_caps is None or market_caps.empty or raw_ticker not in market_caps.columns:
        raise ValueError(f"Shared market cap data for {raw_ticker} is unavailable.")

    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index, errors="coerce")
    prices = prices[~prices.index.isna()].sort_index()
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.loc[:, [symbol for symbol in all_symbols if symbol in prices.columns]]
    prices = prices.dropna(how="all")
    if prices.empty:
        raise ValueError("Shared price data did not return usable rows.")

    market_caps = _clean_market_cap_frame(market_caps)
    market_caps = market_caps.loc[:, [symbol for symbol in all_symbols if symbol in market_caps.columns]]
    market_caps = market_caps.dropna(how="all")
    if market_caps.empty:
        raise ValueError("Shared market cap data did not return usable rows.")

    market_latest_date = pd.Timestamp(prices.index.max()).normalize()
    ticker_series = _clean_close_series(prices[raw_ticker])
    if ticker_series.empty:
        raise ValueError(f"Shared price data for {raw_ticker} is empty.")
    ticker_latest_date = pd.Timestamp(ticker_series.index.max()).normalize()

    daily_returns = prices.pct_change(fill_method=None)
    ticker_daily = _clean_return_series(daily_returns[raw_ticker])
    if ticker_daily.empty:
        raise ValueError(f"Daily return history for {raw_ticker} is unavailable.")

    sector_symbols = components.loc[components["Sector"] == sector, "Symbol"].tolist()
    sector_weight_symbols = [symbol for symbol in sector_symbols if symbol in prices.columns and symbol in market_caps.columns]
    market_weight_symbols = [symbol for symbol in all_symbols if symbol in prices.columns and symbol in market_caps.columns]
    if not sector_weight_symbols:
        raise ValueError(f"Shared market cap data for sector {sector} is unavailable.")
    if not market_weight_symbols:
        raise ValueError("Shared market cap data for the S&P 500 universe is unavailable.")

    sector_daily, sector_member_count = _weighted_daily_returns(prices[sector_weight_symbols], market_caps[sector_weight_symbols])
    market_daily, market_member_count = _weighted_daily_returns(prices[market_weight_symbols], market_caps[market_weight_symbols])

    beta_market_curve = _rolling_beta_series(ticker_daily, market_daily, window=60)
    beta_sector_curve = _rolling_beta_series(ticker_daily, sector_daily, window=60)
    corr_market_curve = _rolling_correlation_series(ticker_daily, market_daily, window=60)
    corr_sector_curve = _rolling_correlation_series(ticker_daily, sector_daily, window=60)

    beta_market_60d = float(beta_market_curve.iloc[-1]) if not beta_market_curve.empty else None
    beta_sector_60d = float(beta_sector_curve.iloc[-1]) if not beta_sector_curve.empty else None
    corr_market_60d = float(corr_market_curve.iloc[-1]) if not corr_market_curve.empty else None
    corr_sector_60d = float(corr_sector_curve.iloc[-1]) if not corr_sector_curve.empty else None

    market_joined = pd.concat(
        [ticker_daily.rename("ticker"), _clean_return_series(market_daily).rename("market")],
        axis=1,
    ).dropna()
    sector_joined = pd.concat(
        [ticker_daily.rename("ticker"), _clean_return_series(sector_daily).rename("sector")],
        axis=1,
    ).dropna()

    residual_market_daily = pd.Series(dtype=float)
    if beta_market_60d is not None and not market_joined.empty:
        residual_market_daily = (market_joined["ticker"] - (beta_market_60d * market_joined["market"])).rename("Residual vs S&P 500")

    residual_sector_daily = pd.Series(dtype=float)
    if beta_sector_60d is not None and not sector_joined.empty:
        residual_sector_daily = (sector_joined["ticker"] - (beta_sector_60d * sector_joined["sector"])).rename("Residual vs Sector")

    residual_market_20d = float(residual_market_daily.tail(20).sum()) if not residual_market_daily.empty else None
    residual_sector_20d = float(residual_sector_daily.tail(20).sum()) if not residual_sector_daily.empty else None

    regime_trend = _trend_regime(ticker_series)
    regime_volatility = _volatility_regime(ticker_daily)
    regime_beta = _beta_regime(beta_market_60d)
    regime_overall = _overall_regime_label(regime_trend, regime_volatility, regime_beta)

    summary_table = pd.DataFrame(
        [
            {
                "지표": "60D 베타 vs S&P 500",
                "현재값": _format_metric(beta_market_60d, 2) if beta_market_60d is not None else "-",
                "핵심 해석": "1보다 크면 시장보다 민감, 1보다 작으면 방어적입니다.",
            },
            {
                "지표": "60D 베타 vs 섹터",
                "현재값": _format_metric(beta_sector_60d, 2) if beta_sector_60d is not None else "-",
                "핵심 해석": "섹터 내부에서 얼마나 공격적 또는 방어적인지 보여줍니다.",
            },
            {
                "지표": "60D 상관 vs S&P 500",
                "현재값": _format_metric(corr_market_60d, 2) if corr_market_60d is not None else "-",
                "핵심 해석": "높을수록 시장 요인 설명력이 큽니다.",
            },
            {
                "지표": "60D 상관 vs 섹터",
                "현재값": _format_metric(corr_sector_60d, 2) if corr_sector_60d is not None else "-",
                "핵심 해석": "높을수록 섹터 공통 요인과 함께 움직입니다.",
            },
            {
                "지표": "20D 잔차수익률 vs S&P 500",
                "현재값": _format_pct(residual_market_20d),
                "핵심 해석": "시장 공통 움직임을 제거한 뒤 남는 종목 고유 성과입니다.",
            },
            {
                "지표": "20D 잔차수익률 vs 섹터",
                "현재값": _format_pct(residual_sector_20d),
                "핵심 해석": "섹터 내부 상대성과를 간단히 점검하는 값입니다.",
            },
            {"지표": "추세 국면 (Trend Regime)", "현재값": regime_trend, "핵심 해석": "이동평균 배열과 현재 가격 위치를 함께 읽습니다."},
            {"지표": "변동성 국면 (Volatility Regime)", "현재값": regime_volatility, "핵심 해석": "현재 20일 변동성이 자신의 최근 1년 중앙값 대비 어느 수준인지 봅니다."},
            {"지표": "베타 국면 (Beta Regime)", "현재값": regime_beta, "핵심 해석": "시장 민감도의 성격을 요약한 라벨입니다."},
            {"지표": "종합 국면 (Overall Regime)", "현재값": regime_overall, "핵심 해석": "추세, 변동성, 베타를 합친 실험적 요약입니다."},
        ]
    )

    interpretation_table = pd.DataFrame(
        [
            {
                "개념": "베타 (beta)",
                "이 페이지에서의 의미": "최근 60거래일 동안 시장 또는 섹터가 1 움직일 때 종목이 얼마나 크게 반응했는지를 봅니다.",
            },
            {
                "개념": "상관계수 (correlation)",
                "이 페이지에서의 의미": "같은 방향으로 함께 움직이는 정도입니다. 높을수록 공통 요인의 영향이 큽니다.",
            },
            {
                "개념": "잔차수익률 (residual return)",
                "이 페이지에서의 의미": "시장/섹터 설명분을 제외한 뒤 남는 종목 고유 수익률입니다. 플러스면 상대 초과성과, 마이너스면 상대 열위로 읽을 수 있습니다.",
            },
            {
                "개념": "추세 국면 (trend regime)",
                "이 페이지에서의 의미": "20, 60, 120일 이동평균과 현재 가격의 상대 위치로 방향성을 요약합니다.",
            },
            {
                "개념": "변동성 국면 (volatility regime)",
                "이 페이지에서의 의미": "현재 변동성이 평소보다 뜨거운지, 차분한지를 빠르게 보여줍니다.",
            },
            {
                "개념": "종합 국면 (overall regime)",
                "이 페이지에서의 의미": "추세, 변동성, 베타를 함께 묶어 현재 환경을 읽기 쉬운 문장으로 압축한 실험적 라벨입니다.",
            },
        ]
    )

    recent_factor_table = pd.concat(
        [
            ticker_daily.rename(f"{raw_ticker} Daily Return"),
            _clean_return_series(market_daily).rename("S&P 500 Return"),
            residual_market_daily.rename("Residual vs S&P 500"),
            _clean_return_series(sector_daily).rename(f"{sector} Sector Return"),
            residual_sector_daily.rename("Residual vs Sector"),
            sector_member_count.rename("Sector Members Used"),
            market_member_count.rename("S&P 500 Members Used"),
        ],
        axis=1,
    ).dropna(how="all").tail(20).reset_index()
    if recent_factor_table.empty:
        recent_factor_table = pd.DataFrame(
            columns=[
                "Date",
                f"{raw_ticker} Daily Return",
                "S&P 500 Return",
                "Residual vs S&P 500",
                f"{sector} Sector Return",
                "Residual vs Sector",
                "Sector Members Used",
                "S&P 500 Members Used",
            ]
        )
    else:
        first_col = recent_factor_table.columns[0]
        if first_col != "Date":
            recent_factor_table = recent_factor_table.rename(columns={first_col: "Date"})
        recent_factor_table["Date"] = pd.to_datetime(recent_factor_table["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        for col in [
            f"{raw_ticker} Daily Return",
            "S&P 500 Return",
            "Residual vs S&P 500",
            f"{sector} Sector Return",
            "Residual vs Sector",
        ]:
            if col in recent_factor_table.columns:
                recent_factor_table[col] = recent_factor_table[col].map(_format_pct)
        for col in ["Sector Members Used", "S&P 500 Members Used"]:
            if col in recent_factor_table.columns:
                recent_factor_table[col] = (
                    pd.to_numeric(recent_factor_table[col], errors="coerce")
                    .fillna(0)
                    .astype(int)
                )

    source_table = _metadata_table(
        [
            ("ticker", raw_ticker),
            ("sector", sector),
            ("components_source", components_source),
            ("price_source", price_source),
            ("market_cap_source", market_cap_source),
            ("beta_window", "60 trading days"),
            ("residual_window", "20 trading days"),
            ("shared_universe_size", len(all_symbols)),
            ("sector_member_count", len(sector_symbols)),
            ("sector_weight_member_count", len(sector_weight_symbols)),
            ("market_weight_member_count", len(market_weight_symbols)),
            ("latest_market_date", market_latest_date.strftime("%Y-%m-%d")),
            ("ticker_latest_date", ticker_latest_date.strftime("%Y-%m-%d")),
        ]
    )

    commentary = _factor_commentary(
        ticker=raw_ticker,
        sector=sector,
        beta_market_60d=beta_market_60d,
        corr_market_60d=corr_market_60d,
        residual_market_20d=residual_market_20d,
        residual_sector_20d=residual_sector_20d,
        regime_overall=regime_overall,
    )

    return _FactorContext(
        ticker=raw_ticker,
        sector=sector,
        latest_market_date=market_latest_date.strftime("%Y-%m-%d"),
        ticker_latest_date=ticker_latest_date.strftime("%Y-%m-%d"),
        price_source=price_source,
        market_cap_source=market_cap_source,
        beta_sector_60d=beta_sector_60d,
        beta_market_60d=beta_market_60d,
        corr_sector_60d=corr_sector_60d,
        corr_market_60d=corr_market_60d,
        residual_market_20d=residual_market_20d,
        residual_sector_20d=residual_sector_20d,
        regime_trend=regime_trend,
        regime_volatility=regime_volatility,
        regime_beta=regime_beta,
        regime_overall=regime_overall,
        summary_table=summary_table,
        source_table=source_table,
        interpretation_table=interpretation_table,
        recent_factor_table=recent_factor_table,
        beta_chart_base64=_factor_beta_chart(raw_ticker, beta_market_curve, beta_sector_curve),
        residual_chart_base64=_factor_residual_chart(raw_ticker, residual_market_daily, residual_sector_daily),
        commentary=commentary,
    )


def _render_chart_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=135)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _price_forecast_chart(result: StockForecastResult) -> str:
    hist = result.close_history.copy()
    row = result.summary.iloc[0]
    forecast_date = pd.to_datetime(row["forecast_date"])
    forecast_price = float(row["predicted_price"])
    tail = hist.tail(220)

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.plot(tail.index, tail["close"], color="#1f77b4", linewidth=2.0, label="Historical Close")
    ax.scatter([forecast_date], [forecast_price], color="#c62828", s=60, marker="X", label="Forecast")
    ax.set_title("Close Price and 10-Day Ahead Forecast")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    return _render_chart_base64(fig)


def _model_weight_chart(result: StockForecastResult) -> str:
    scores = result.model_scores.copy()
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.bar(scores["model"], scores["weight"] * 100.0, color="#2e7d32")
    ax.set_title("Ensemble Weights (Inverse MAE)")
    ax.set_ylabel("Weight (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.2)
    return _render_chart_base64(fig)


def _format_metric(value: object, ndigits: int = 4) -> str:
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(float(value)):
            return "-"
        return f"{float(value):,.{ndigits}f}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,d}"
    return str(value)


def _safe_table(df: pd.DataFrame, max_rows: int = 400) -> str:
    table_html = df.head(max_rows).to_html(index=False, border=0, classes="data-table")
    return f'<div class="table-wrap">{table_html}</div>'


def _stacked_table_html(
    df: pd.DataFrame,
    *,
    max_rows: int = 400,
    pinned_columns: list[str] | None = None,
    max_tables: int = 3,
    min_chunk_columns: int = 4,
    section_title: str = "표",
) -> str:
    head = df.head(max_rows).copy()
    if head.empty:
        return _safe_table(head, max_rows=max_rows)

    columns = [str(col) for col in head.columns]
    pinned = [col for col in (pinned_columns or []) if col in columns]
    remaining = [col for col in columns if col not in pinned]
    if not remaining:
        return _safe_table(head, max_rows=max_rows)

    chunk_size = max(min_chunk_columns, (len(remaining) + max(1, max_tables) - 1) // max(1, max_tables))
    if len(remaining) <= chunk_size:
        return _safe_table(head, max_rows=max_rows)

    chunks = [remaining[idx : idx + chunk_size] for idx in range(0, len(remaining), chunk_size)]
    total = len(chunks)
    sections: list[str] = []
    for idx, cols in enumerate(chunks, start=1):
        label = f"{section_title} ({idx}/{total})" if total > 1 else section_title
        section_df = head.loc[:, pinned + cols]
        sections.append(
            "<div class=\"stacked-table-block\">"
            f"<h4>{html.escape(label)}</h4>"
            f"{_safe_table(section_df, max_rows=max_rows)}"
            "</div>"
        )
    return "<div class=\"stacked-table-group\">" + "".join(sections) + "</div>"


def _metadata_table(rows: list[tuple[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": metric, "value": "-" if value in (None, "") else str(value)}
            for metric, value in rows
        ]
    )


def _build_forecast_source_table(form: dict[str, str], result: StockForecastResult) -> pd.DataFrame:
    summary = result.summary.iloc[0] if not result.summary.empty else {}
    prices_csv_path = form.get("prices_csv_path", "").strip()
    use_sample = form.get("use_sample", "") == "on"
    input_mode = "online"
    price_source = result.price_source or "online_unknown"
    rows: list[tuple[str, object]] = []

    if prices_csv_path:
        input_mode = "local_csv"
        price_source = "local_prices_csv"
        rows.append(("prices_csv_path", prices_csv_path))
    elif use_sample:
        input_mode = "sample"
        price_source = "sample_prices"
    else:
        rows.append(("requested_start_date", form.get("start_date", "").strip() or f"today-{form.get('history_years', '8')}y"))
        rows.append(("requested_end_date", form.get("end_date", "").strip() or "today"))

    rows = [
        ("input_mode", input_mode),
        ("price_source", price_source),
        ("ticker", summary.get("ticker", form.get("ticker", ""))),
        ("as_of_date", summary.get("as_of_date", "-")),
        *rows,
    ]
    return _metadata_table(rows)


def _build_financial_source_table(
    *,
    ticker: str,
    primary_source: str,
    secondary_source: str | None = None,
    tertiary_source: str | None = None,
    reason: str | None = None,
    warning: str | None = None,
) -> pd.DataFrame:
    rows: list[tuple[str, object]] = [("ticker", ticker), ("primary_source", primary_source)]
    if secondary_source:
        rows.append(("secondary_source", secondary_source))
    if tertiary_source:
        rows.append(("tertiary_source", tertiary_source))
    if reason:
        rows.append(("fallback_reason", reason))
    if warning:
        rows.append(("warning", warning))
    return _metadata_table(rows)


def _build_financial_provider_status_table(
    *,
    yfinance_info: str,
    yfinance_statements: str,
    price_history: str,
    sec_fallback: str,
    fmp_metrics: str,
    derived_metrics: str,
) -> pd.DataFrame:
    return _metadata_table(
        [
            ("yfinance_info", yfinance_info),
            ("yfinance_statements", yfinance_statements),
            ("price_history", price_history),
            ("sec_fallback", sec_fallback),
            ("fmp_metrics", fmp_metrics),
            ("derived_metrics", derived_metrics),
        ]
    )


def _sample_prices() -> pd.DataFrame:
    start = pd.Timestamp("2019-01-02")
    today = pd.Timestamp.today().normalize()
    dates = pd.bdate_range(start, today)
    if len(dates) < 700:
        dates = pd.bdate_range(start, periods=700)
    rng = np.random.default_rng(123)
    log_returns = rng.normal(loc=0.00025, scale=0.012, size=len(dates))
    close = 100.0 * np.exp(np.cumsum(log_returns))
    return pd.DataFrame({"date": dates, "close": close})


def _run_once(form: dict[str, str]) -> _RunContext:
    ticker = form.get("ticker", "").strip().upper() or None
    horizon = int(form.get("forecast_horizon", "10").strip() or "10")
    history_years = int(form.get("history_years", "8").strip() or "8")
    start_date = form.get("start_date", "").strip() or None
    end_date = form.get("end_date", "").strip() or None
    insecure_ssl = form.get("insecure_ssl", "") == "on"
    ca_bundle_path = form.get("ca_bundle_path", "").strip() or None
    configure_ssl(insecure_ssl=insecure_ssl, ca_bundle=ca_bundle_path)

    use_sample = form.get("use_sample", "") == "on"
    auto_save = form.get("auto_save", "on") == "on"
    out_base = form.get("output_dir", "outputs/stock_forecast").strip() or "outputs/stock_forecast"
    prices_csv_path = form.get("prices_csv_path", "").strip()

    out_dir: Path | None = None
    saved_dir: str | None = None
    if auto_save:
        out_dir = Path(out_base) / datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
        ensure_writable_dir(out_dir)
        saved_dir = str(out_dir)

    if prices_csv_path:
        local_prices = load_price_data_csv(prices_csv_path)
        result = run_ticker_stock_forecast_pipeline(
            ticker=ticker or "LOCAL",
            horizon_days=horizon,
            history_years=history_years,
            output_dir=out_dir,
            price_data=local_prices,
            insecure_ssl=insecure_ssl,
            ca_bundle=ca_bundle_path,
        )
    elif use_sample:
        prices = _sample_prices()
        result = run_ticker_stock_forecast_pipeline(
            ticker=ticker or "SAMPLE",
            horizon_days=horizon,
            history_years=history_years,
            output_dir=out_dir,
            price_data=prices,
            insecure_ssl=insecure_ssl,
            ca_bundle=ca_bundle_path,
        )
    else:
        if not ticker:
            raise ValueError("Provide ticker, or set Local Prices CSV Path Override.")
        result = run_ticker_stock_forecast_pipeline(
            ticker=ticker,
            horizon_days=horizon,
            start_date=start_date,
            end_date=end_date,
            history_years=history_years,
            output_dir=out_dir,
            insecure_ssl=insecure_ssl,
            ca_bundle=ca_bundle_path,
        )

    return _RunContext(
        result=result,
        saved_dir=saved_dir,
        source_table=_build_forecast_source_table(form, result),
    )


def _load_walk_forward_close_series(
    form: dict[str, str],
) -> tuple[str, pd.Series, str, str]:
    ticker = form.get("ticker", "").strip().upper()
    history_years = int(form.get("history_years", "8").strip() or "8")
    start_date = form.get("start_date", "").strip() or None
    end_date = form.get("end_date", "").strip() or None
    insecure_ssl = form.get("insecure_ssl", "") == "on"
    ca_bundle_path = form.get("ca_bundle_path", "").strip() or None
    prices_csv_path = form.get("prices_csv_path", "").strip()
    use_sample = form.get("use_sample", "") == "on"

    configure_ssl(insecure_ssl=insecure_ssl, ca_bundle=ca_bundle_path)

    if prices_csv_path:
        local_prices = load_price_data_csv(prices_csv_path)
        close = _normalize_close_prices(local_prices)
        return ticker or "LOCAL", close, "local_csv", "local_prices_csv"

    if use_sample:
        close = _normalize_close_prices(_sample_prices())
        return ticker or "SAMPLE", close, "sample", "sample_prices"

    if not ticker:
        raise ValueError("Provide ticker, or set Local Prices CSV Path Override.")

    close, price_source = _fetch_close_prices_with_source(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        history_years=history_years,
        insecure_ssl=insecure_ssl,
        ca_bundle=ca_bundle_path,
    )
    return ticker, close, "online", price_source


def _build_walk_forward_source_table(
    form: dict[str, str],
    *,
    ticker: str,
    input_mode: str,
    price_source: str,
    split_count: int,
) -> pd.DataFrame:
    rows: list[tuple[str, object]] = [
        ("input_mode", input_mode),
        ("price_source", price_source),
        ("ticker", ticker),
        ("evaluation_splits", split_count),
        ("requested_start_date", form.get("start_date", "").strip() or f"today-{form.get('history_years', '8')}y"),
        ("requested_end_date", form.get("end_date", "").strip() or "today"),
    ]
    prices_csv_path = form.get("prices_csv_path", "").strip()
    if prices_csv_path:
        rows.append(("prices_csv_path", prices_csv_path))
    return _metadata_table(rows)


def _walk_forward_validation_chart(
    split_df: pd.DataFrame,
    *,
    ticker: str,
    horizon_days: int,
) -> str:
    chart = split_df.copy()
    chart["as_of_date"] = pd.to_datetime(chart["as_of_date"], errors="coerce")
    chart = chart.dropna(subset=["as_of_date"])

    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    ax.plot(
        chart["as_of_date"],
        chart["predicted_return_pct"],
        color="#0f4c81",
        linewidth=1.8,
        marker="o",
        markersize=3.5,
        label="Predicted Return",
    )
    ax.plot(
        chart["as_of_date"],
        chart["actual_return_pct"],
        color="#b26a00",
        linewidth=1.8,
        marker="o",
        markersize=3.5,
        label="Realized Return",
    )
    _apply_year_month_axis(ax)
    ax.axhline(0.0, color="#8a98a8", linewidth=1.0)
    ax.set_title(f"{ticker} Walk-Forward {horizon_days}-Day Return Check")
    ax.set_ylabel("Return (%)")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    return _render_chart_base64(fig)


def _walk_forward_diagnostics_chart(split_df: pd.DataFrame) -> str:
    chart = split_df.copy()
    chart["as_of_date"] = pd.to_datetime(chart["as_of_date"], errors="coerce")
    chart = chart.dropna(subset=["as_of_date"])
    if chart.empty:
        fig, ax = plt.subplots(figsize=(7.0, 3.9))
        ax.set_title("Walk-Forward Diagnostics")
        fig.tight_layout()
        return _render_chart_base64(fig)

    chart["rolling_hit_rate_pct"] = chart["directional_hit"].rolling(5, min_periods=1).mean() * 100.0
    chart["rolling_abs_error_pct"] = chart["absolute_error_pct"].rolling(5, min_periods=1).mean()

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.6), sharex=True)
    ax_top, ax_bottom = axes

    ax_top.bar(
        chart["as_of_date"],
        chart["absolute_error_pct"],
        width=5.0,
        color="#d7e3f2",
        edgecolor="#0f4c81",
        linewidth=0.6,
        label="Absolute Error",
    )
    ax_top.plot(
        chart["as_of_date"],
        chart["rolling_abs_error_pct"],
        color="#c62828",
        linewidth=1.7,
        label="5-Split Avg Error",
    )
    ax_top.set_ylabel("Error (%)")
    ax_top.grid(alpha=0.2)
    ax_top.legend(loc="best")

    hit_colors = ["#2e7d32" if bool(v) else "#c62828" for v in chart["directional_hit"].tolist()]
    ax_bottom.scatter(
        chart["as_of_date"],
        chart["rolling_hit_rate_pct"],
        c=hit_colors,
        s=28,
        alpha=0.85,
        label="Latest Split Direction",
    )
    ax_bottom.plot(
        chart["as_of_date"],
        chart["rolling_hit_rate_pct"],
        color="#0f4c81",
        linewidth=1.7,
        label="5-Split Hit Rate",
    )
    ax_bottom.axhline(50.0, color="#8a98a8", linewidth=1.0, linestyle="--")
    ax_bottom.set_ylim(0.0, 100.0)
    ax_bottom.set_ylabel("Hit Rate (%)")
    ax_bottom.grid(alpha=0.2)
    ax_bottom.legend(loc="best")
    _apply_year_month_axis(ax_bottom)
    fig.tight_layout()
    return _render_chart_base64(fig)


def _build_walk_forward_model_specs(random_state: int = 7) -> list[tuple[str, object]]:
    specs: list[tuple[str, object]] = []
    for model_name, model in _build_model_specs(random_state=random_state):
        if model_name == "elastic_net":
            model.set_params(
                model__alphas=np.logspace(-4, 0, 8),
                model__cv=3,
                model__max_iter=2000,
            )
        elif model_name == "random_forest":
            model.set_params(
                n_estimators=48,
                min_samples_leaf=5,
                n_jobs=1,
            )
        elif model_name == "gradient_boosting":
            model.set_params(
                n_estimators=60,
                max_depth=2,
                learning_rate=0.05,
            )
        specs.append((model_name, model))
    return specs


def _build_walk_forward_direction_model_specs(random_state: int = 7) -> list[tuple[str, object]]:
    specs: list[tuple[str, object]] = []
    for model_name, model in _build_direction_model_specs(random_state=random_state):
        if model_name == "logistic":
            model.set_params(
                model__Cs=6,
                model__cv=3,
                model__max_iter=2500,
            )
        elif model_name == "random_forest_cls":
            model.set_params(
                n_estimators=64,
                min_samples_leaf=5,
                n_jobs=1,
            )
        elif model_name == "gradient_boosting_cls":
            model.set_params(
                n_estimators=80,
                learning_rate=0.05,
                max_depth=2,
            )
        specs.append((model_name, model))
    return specs


def _walk_forward_commentary(
    *,
    hit_rate: float | None,
    mae_return: float | None,
    rmse_return: float | None,
    skill_vs_naive: float | None,
    bias_return: float | None,
    corr: float | None,
) -> str:
    parts: list[str] = []

    if hit_rate is not None:
        if hit_rate >= 0.58:
            parts.append("방향성 적중률이 58% 이상이면 단순한 동전던지기보다 유의미한 방향성 우위를 시사합니다.")
        elif hit_rate >= 0.52:
            parts.append("방향성 적중률은 약한 우위 구간에 가까워 보이지만, 아직 강한 신호로 보기는 이릅니다.")
        else:
            parts.append("방향성 적중률이 50% 안팎이면 현재 예측 신호는 방향 판별력 측면에서 보수적으로 해석하는 편이 좋습니다.")

    if skill_vs_naive is not None:
        if skill_vs_naive > 0.1:
            parts.append("naive zero-return 기준보다 오차가 낮아, 최소한 최근 구간에서는 예측 모델이 단순 무예측보다 낫습니다.")
        elif skill_vs_naive > 0.0:
            parts.append("naive zero-return 기준보다 약간 낫지만, 개선 폭은 아직 제한적입니다.")
        else:
            parts.append("naive zero-return 기준보다 개선이 없거나 오히려 뒤처져, 지금은 예측 출력보다 리스크 관리 참고치로 보는 편이 안전합니다.")

    if bias_return is not None:
        if bias_return >= 0.01:
            parts.append("평균 편향이 플러스면 최근에는 모델이 실제보다 낙관적으로 예측하는 경향이 있었습니다.")
        elif bias_return <= -0.01:
            parts.append("평균 편향이 마이너스면 최근에는 모델이 실제보다 보수적으로 예측하는 경향이 있었습니다.")
        else:
            parts.append("평균 편향은 크지 않아 낙관/비관 한쪽으로 치우친 예측 습관은 두드러지지 않습니다.")

    if corr is not None:
        if corr >= 0.35:
            parts.append("예측 수익률과 실현 수익률의 상관이 비교적 높아, 강약을 구분하는 능력은 어느 정도 유지되는 편입니다.")
        elif corr >= 0.15:
            parts.append("예측 수익률과 실현 수익률의 상관은 약한 편이라, 순위 신호로는 제한적으로만 활용하는 편이 좋습니다.")
        else:
            parts.append("예측 수익률과 실현 수익률의 상관이 낮으면, 숫자 크기보다 방향성이나 리스크 한도 중심으로 읽는 편이 더 낫습니다.")

    if mae_return is not None and rmse_return is not None:
        parts.append(
            f"현재 평균 절대오차는 {_format_pct(mae_return)}, RMSE는 {_format_pct(rmse_return)} 수준으로, "
            "개별 예측값 자체보다는 신호의 일관성과 편향을 함께 보는 것이 중요합니다."
        )

    return " ".join(parts)


def _run_walk_forward_validation_once(form: dict[str, str]) -> _WalkForwardContext:
    horizon_days = int(form.get("forecast_horizon", "10").strip() or "10")
    history_years = int(form.get("history_years", "8").strip() or "8")
    min_train_rows = int(form.get("wf_min_train_rows", "252").strip() or "252")
    step_size = int(form.get("wf_step_size", "21").strip() or "21")
    max_splits = int(form.get("wf_max_splits", "4").strip() or "4")
    auto_save = form.get("auto_save", "on") == "on"
    out_base = form.get("output_dir", "outputs/walk_forward_validation").strip() or "outputs/walk_forward_validation"

    if horizon_days <= 0:
        raise ValueError("Forecast horizon must be positive.")
    if min_train_rows < 80:
        raise ValueError("Minimum train rows should be at least 80 for walk-forward validation.")
    if step_size <= 0:
        raise ValueError("Step size must be positive.")
    if max_splits <= 0:
        raise ValueError("Max splits must be positive.")

    ticker, close, input_mode, price_source = _load_walk_forward_close_series(form)

    try:
        dataset, _, _, _ = _build_supervised_dataset(
            close=close,
            horizon_days=horizon_days,
            ticker=ticker,
        )
    except ValueError as exc:
        can_retry_with_wider_range = (
            input_mode == "online"
            and (form.get("start_date", "").strip() or form.get("end_date", "").strip())
            and "Feature table is empty after preprocessing" in str(exc)
        )
        if not can_retry_with_wider_range:
            raise
        insecure_ssl = form.get("insecure_ssl", "") == "on"
        ca_bundle_path = form.get("ca_bundle_path", "").strip() or None
        close_retry, source_retry = _fetch_close_prices_with_source(
            ticker=ticker,
            start_date=None,
            end_date=None,
            history_years=history_years,
            insecure_ssl=insecure_ssl,
            ca_bundle=ca_bundle_path,
        )
        close = close_retry
        price_source = f"{source_retry}+auto_widened_range"
        dataset, _, _, _ = _build_supervised_dataset(
            close=close,
            horizon_days=horizon_days,
            ticker=ticker,
        )

    if len(dataset) <= min_train_rows:
        raise ValueError(
            "Not enough usable rows for walk-forward validation. "
            "Use a wider date range, a lower horizon, or a smaller minimum train window."
        )

    candidate_indices = list(range(min_train_rows, len(dataset), step_size))
    if not candidate_indices:
        raise ValueError("No walk-forward split could be created with the current settings.")
    if len(candidate_indices) > max_splits:
        candidate_indices = candidate_indices[-max_splits:]

    feature_cols = [c for c in dataset.columns if c != "target"]
    close_positions = {pd.Timestamp(idx).normalize(): pos for pos, idx in enumerate(close.index)}
    split_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    default_no_trade_threshold = 0.60

    for split_number, test_idx in enumerate(candidate_indices, start=1):
        train_df = dataset.iloc[:test_idx].copy()
        if len(train_df) <= max(60, horizon_days):
            continue

        inner_split = max(int(len(train_df) * 0.8), 120)
        inner_split = min(inner_split, len(train_df) - max(30, horizon_days))
        if inner_split <= 0:
            continue

        x_train = train_df.iloc[:inner_split][feature_cols]
        y_train = train_df.iloc[:inner_split]["target"]
        x_valid = train_df.iloc[inner_split:][feature_cols]
        y_valid = train_df.iloc[inner_split:]["target"]
        x_all = train_df[feature_cols]
        y_all = train_df["target"]
        x_test = dataset.iloc[[test_idx]][feature_cols]
        y_train_dir = (y_train > 0.0).astype(int)
        y_valid_dir = (y_valid > 0.0).astype(int)
        y_all_dir = (y_all > 0.0).astype(int)

        eval_rows: list[dict[str, object]] = []
        as_of_date = pd.Timestamp(dataset.index[test_idx]).normalize()
        as_of_pos = close_positions.get(as_of_date)
        if as_of_pos is None or (as_of_pos + horizon_days) >= len(close.index):
            continue

        future_date = pd.Timestamp(close.index[as_of_pos + horizon_days]).normalize()
        entry_price = float(close.iloc[as_of_pos])
        actual_future_close = float(close.iloc[as_of_pos + horizon_days])

        for model_name, base_model in _build_walk_forward_model_specs(random_state=7):
            model_train = clone(base_model)
            model_train.fit(x_train, y_train)
            valid_pred = np.asarray(model_train.predict(x_valid), dtype=float)
            valid_mae = float(np.mean(np.abs(y_valid.to_numpy(dtype=float) - valid_pred)))

            model_full = clone(base_model)
            model_full.fit(x_all, y_all)
            pred_log_return = float(model_full.predict(x_test)[0])
            eval_rows.append(
                {
                    "model": model_name,
                    "validation_mae": valid_mae,
                    "predicted_log_return": pred_log_return,
                }
            )

        score_df = pd.DataFrame(eval_rows).sort_values("validation_mae").reset_index(drop=True)
        score_df["weight"] = _inverse_error_weights(score_df["validation_mae"].to_numpy(dtype=float))
        ensemble_log_return = float((score_df["predicted_log_return"] * score_df["weight"]).sum())
        predicted_future_close = float(entry_price * np.exp(ensemble_log_return))
        validation_mae_log = float((score_df["validation_mae"] * score_df["weight"]).sum()) if not score_df.empty else np.nan

        predicted_return = float(np.exp(ensemble_log_return) - 1.0)
        actual_return = float(actual_future_close / entry_price - 1.0)
        signed_error = predicted_return - actual_return
        directional_hit = bool(np.sign(predicted_return) == np.sign(actual_return))

        direction_prob_up = 0.5
        direction_confidence = 0.5
        classification_hit = False
        trade_taken = False
        trade_signal = "No-Trade"
        signal_return = 0.0
        if y_train_dir.nunique() >= 2 and y_valid_dir.nunique() >= 2 and y_all_dir.nunique() >= 2:
            direction_eval_rows: list[dict[str, object]] = []
            for model_name, base_model in _build_walk_forward_direction_model_specs(random_state=7):
                model_train = clone(base_model)
                model_train.fit(x_train, y_train_dir)
                valid_prob = np.asarray(model_train.predict_proba(x_valid)[:, 1], dtype=float)
                valid_brier = float(np.mean(np.square(valid_prob - y_valid_dir.to_numpy(dtype=float))))

                model_full = clone(base_model)
                model_full.fit(x_all, y_all_dir)
                pred_prob_up = float(model_full.predict_proba(x_test)[0, 1])
                direction_eval_rows.append(
                    {
                        "model": model_name,
                        "validation_brier": valid_brier,
                        "predicted_prob_up": pred_prob_up,
                    }
                )
            direction_score_df = pd.DataFrame(direction_eval_rows).sort_values("validation_brier").reset_index(drop=True)
            direction_score_df["weight"] = _inverse_error_weights(direction_score_df["validation_brier"].to_numpy(dtype=float))
            direction_prob_up = float((direction_score_df["predicted_prob_up"] * direction_score_df["weight"]).sum())

        direction_confidence = float(max(direction_prob_up, 1.0 - direction_prob_up))
        classification_hit = bool((direction_prob_up >= 0.5) == (actual_return >= 0.0))
        magnitude_threshold = float(np.exp(max(validation_mae_log, 0.0)) - 1.0) if np.isfinite(validation_mae_log) else 0.0
        trade_taken = bool(
            direction_confidence >= default_no_trade_threshold
            and abs(predicted_return) >= max(0.0035, magnitude_threshold)
        )
        if trade_taken:
            trade_signal = "Long" if direction_prob_up >= 0.5 else "Short"
            signal_return = actual_return if trade_signal == "Long" else -actual_return
        regime_labels = classify_regime_from_feature_row(dataset.iloc[test_idx][feature_cols])

        split_rows.append(
            {
                "split": split_number,
                "as_of_date": as_of_date.strftime("%Y-%m-%d"),
                "realized_date": future_date.strftime("%Y-%m-%d"),
                "train_rows": int(len(train_df)),
                "predicted_return_pct": predicted_return * 100.0,
                "actual_return_pct": actual_return * 100.0,
                "predicted_price": predicted_future_close,
                "actual_price": actual_future_close,
                "signed_error_pct": signed_error * 100.0,
                "absolute_error_pct": abs(signed_error) * 100.0,
                "directional_hit": directional_hit,
                "direction_prob_up_pct": direction_prob_up * 100.0,
                "direction_confidence_pct": direction_confidence * 100.0,
                "classification_hit": classification_hit,
                "trade_taken": trade_taken,
                "trade_signal": trade_signal,
                "signal_return_pct": signal_return * 100.0,
                "trend_regime": regime_labels["trend_regime"],
                "volatility_regime": regime_labels["volatility_regime"],
                "beta_regime": regime_labels["beta_regime"],
                "overall_regime": regime_labels["overall_regime"],
            }
        )

        for row in score_df.to_dict(orient="records"):
            model_pred_return = float(np.exp(float(row["predicted_log_return"])) - 1.0)
            model_rows.append(
                {
                    "split": split_number,
                    "as_of_date": as_of_date.strftime("%Y-%m-%d"),
                    "model": str(row["model"]),
                    "validation_mae": float(row["validation_mae"]),
                    "weight": float(row["weight"]),
                    "predicted_return_pct": model_pred_return * 100.0,
                    "actual_return_pct": actual_return * 100.0,
                    "absolute_error_pct": abs(model_pred_return - actual_return) * 100.0,
                    "directional_hit": bool(np.sign(model_pred_return) == np.sign(actual_return)),
                }
            )

    if not split_rows:
        raise ValueError("Walk-forward validation could not generate any usable split. Relax the settings and try again.")

    split_df = pd.DataFrame(split_rows)
    model_df = pd.DataFrame(model_rows)

    predicted_return = split_df["predicted_return_pct"].astype(float) / 100.0
    actual_return = split_df["actual_return_pct"].astype(float) / 100.0
    error_return = predicted_return - actual_return
    abs_actual_baseline = float(np.mean(np.abs(actual_return.to_numpy(dtype=float))))

    mae_return = float(np.mean(np.abs(error_return.to_numpy(dtype=float))))
    rmse_return = float(np.sqrt(np.mean(np.square(error_return.to_numpy(dtype=float)))))
    bias_return = float(np.mean(error_return.to_numpy(dtype=float)))
    direction_hit_rate = float(split_df["directional_hit"].astype(float).mean())
    classification_hit_rate = float(split_df["classification_hit"].astype(float).mean())
    trade_mask = split_df["trade_taken"].astype(bool)
    trade_coverage_rate = float(trade_mask.mean()) if len(split_df) else None
    trade_hit_rate = float(split_df.loc[trade_mask, "classification_hit"].astype(float).mean()) if trade_mask.any() else None
    skill_vs_naive = (
        float(1.0 - (mae_return / max(abs_actual_baseline, 1e-8)))
        if np.isfinite(abs_actual_baseline) and abs_actual_baseline > 0.0
        else None
    )
    return_correlation = None
    if len(split_df) >= 2:
        corr_val = predicted_return.corr(actual_return)
        return_correlation = float(corr_val) if pd.notna(corr_val) else None

    summary_table = pd.DataFrame(
        [
            {"metric": "Evaluation Splits", "value": f"{len(split_df):,d}"},
            {"metric": "Forecast Horizon", "value": f"{horizon_days} business days"},
            {"metric": "Minimum Train Rows", "value": f"{min_train_rows:,d}"},
            {"metric": "Step Size", "value": f"{step_size:,d} rows"},
            {"metric": "Mean Absolute Error", "value": _format_pct(mae_return)},
            {"metric": "RMSE", "value": _format_pct(rmse_return)},
            {"metric": "Regression Directional Hit Rate", "value": _format_pct(direction_hit_rate)},
            {"metric": "Classification Hit Rate", "value": _format_pct(classification_hit_rate)},
            {"metric": "No-Trade Coverage", "value": _format_pct(trade_coverage_rate)},
            {"metric": "Traded Hit Rate", "value": _format_pct(trade_hit_rate) if trade_hit_rate is not None else "-"},
            {"metric": "Mean Bias", "value": _format_pct(bias_return)},
            {"metric": "Skill vs Naive Zero-Return", "value": _format_pct(skill_vs_naive) if skill_vs_naive is not None else "-"},
            {"metric": "Prediction/Realization Correlation", "value": _format_metric(return_correlation, 3) if return_correlation is not None else "-"},
        ]
    )

    interpretation_table = pd.DataFrame(
        [
            {
                "지표": "방향성 적중률 (Directional Hit Rate)",
                "현재값": _format_pct(direction_hit_rate),
                "해석 포인트": "회귀 예측값의 부호만 봤을 때의 방향 적중률입니다. 대략 55%를 넘기면 방향성 우위가 있다고 볼 여지가 커집니다.",
            },
            {
                "지표": "분류 적중률 (Classification Hit Rate)",
                "현재값": _format_pct(classification_hit_rate),
                "해석 포인트": "상승/하락 분류기(classifier)가 별도로 방향을 얼마나 잘 골랐는지 보여줍니다. 회귀 적중률과 비교해 개선 여지를 볼 수 있습니다.",
            },
            {
                "지표": "거래 커버리지 (No-Trade Coverage)",
                "현재값": _format_pct(trade_coverage_rate),
                "해석 포인트": "no-trade 필터를 통과해 실제로 신호를 낸 비중입니다. 낮을수록 보수적이고, 높을수록 더 자주 거래합니다.",
            },
            {
                "지표": "선별 거래 적중률 (Traded Hit Rate)",
                "현재값": _format_pct(trade_hit_rate) if trade_hit_rate is not None else "-",
                "해석 포인트": "필터를 통과한 구간만 봤을 때의 적중률입니다. coverage와 함께 읽어야 의미가 선명해집니다.",
            },
            {
                "지표": "평균절대오차 (MAE)",
                "현재값": _format_pct(mae_return),
                "해석 포인트": "낮을수록 좋습니다. 예측 선행수익률과 실제 선행수익률 사이의 평균 간격입니다.",
            },
            {
                "지표": "나이브 대비 스킬 (Skill vs Naive)",
                "현재값": _format_pct(skill_vs_naive) if skill_vs_naive is not None else "-",
                "해석 포인트": "플러스면 0% 수익률을 찍는 단순 기준보다 평균절대오차 기준으로 더 낫다는 뜻입니다.",
            },
            {
                "지표": "평균 편향 (Mean Bias)",
                "현재값": _format_pct(bias_return),
                "해석 포인트": "플러스면 예측이 대체로 낙관적이었고, 마이너스면 보수적으로 치우친 편입니다.",
            },
            {
                "지표": "예측/실현 상관 (Prediction/Realization Correlation)",
                "현재값": _format_metric(return_correlation, 3) if return_correlation is not None else "-",
                "해석 포인트": "높을수록 강한 구간과 약한 구간을 상대적으로 더 잘 구분했다는 뜻입니다.",
            },
        ]
    )

    model_table = pd.DataFrame()
    if not model_df.empty:
        model_table = (
            model_df.groupby("model", as_index=False)
            .agg(
                avg_validation_mae=("validation_mae", "mean"),
                avg_weight=("weight", "mean"),
                out_of_sample_mae_pct=("absolute_error_pct", "mean"),
                hit_rate=("directional_hit", "mean"),
            )
            .sort_values("out_of_sample_mae_pct")
            .reset_index(drop=True)
        )
        model_table["avg_validation_mae"] = model_table["avg_validation_mae"].map(lambda v: _format_metric(v, 5))
        model_table["avg_weight"] = model_table["avg_weight"].map(_format_pct)
        model_table["out_of_sample_mae_pct"] = model_table["out_of_sample_mae_pct"].map(lambda v: f"{float(v):,.2f}%")
        model_table["hit_rate"] = model_table["hit_rate"].map(_format_pct)
        model_table = model_table.rename(
            columns={
                "model": "Model",
                "avg_validation_mae": "Avg Validation MAE",
                "avg_weight": "Avg Ensemble Weight",
                "out_of_sample_mae_pct": "Out-of-Sample MAE",
                "hit_rate": "Directional Hit Rate",
            }
        )

    confidence_series = split_df["direction_confidence_pct"].astype(float) / 100.0
    threshold_rows: list[dict[str, object]] = []
    for threshold in [0.50, 0.55, 0.60, 0.65]:
        mask = confidence_series >= threshold
        hit_rate = float(split_df.loc[mask, "classification_hit"].astype(float).mean()) if mask.any() else None
        avg_signal_return = float(split_df.loc[mask, "signal_return_pct"].astype(float).mean()) if mask.any() else None
        threshold_rows.append(
            {
                "Confidence Threshold": f"{threshold * 100.0:.0f}%",
                "Trades": int(mask.sum()),
                "Coverage": _format_pct(float(mask.mean()) if len(mask) else None),
                "Hit Rate": _format_pct(hit_rate) if hit_rate is not None else "-",
                "Avg Signal Return": f"{avg_signal_return:,.2f}%" if avg_signal_return is not None and np.isfinite(avg_signal_return) else "-",
            }
        )
    threshold_table = pd.DataFrame(threshold_rows)

    regime_rows: list[dict[str, object]] = []
    for regime_name, group in split_df.groupby("overall_regime", dropna=False):
        group_trade_mask = group["trade_taken"].astype(bool)
        regime_trade_hit = float(group.loc[group_trade_mask, "classification_hit"].astype(float).mean()) if group_trade_mask.any() else None
        regime_signal_return = float(group.loc[group_trade_mask, "signal_return_pct"].astype(float).mean()) if group_trade_mask.any() else None
        regime_rows.append(
            {
                "Overall Regime": str(regime_name),
                "Splits": int(len(group)),
                "Regression Hit": _format_pct(float(group["directional_hit"].astype(float).mean())),
                "Classification Hit": _format_pct(float(group["classification_hit"].astype(float).mean())),
                "Trade Coverage": _format_pct(float(group_trade_mask.mean())),
                "Traded Hit": _format_pct(regime_trade_hit) if regime_trade_hit is not None else "-",
                "Avg Signal Return": f"{regime_signal_return:,.2f}%" if regime_signal_return is not None and np.isfinite(regime_signal_return) else "-",
                "Avg Abs Error": f"{float(group['absolute_error_pct'].astype(float).mean()):,.2f}%",
            }
        )
    regime_table = pd.DataFrame(regime_rows).sort_values(["Splits", "Overall Regime"], ascending=[False, True]).reset_index(drop=True)

    split_table = split_df.iloc[::-1].reset_index(drop=True).copy()
    split_table["predicted_price"] = split_table["predicted_price"].map(lambda v: _format_metric(v, 2))
    split_table["actual_price"] = split_table["actual_price"].map(lambda v: _format_metric(v, 2))
    for col in [
        "predicted_return_pct",
        "actual_return_pct",
        "signed_error_pct",
        "absolute_error_pct",
        "direction_prob_up_pct",
        "direction_confidence_pct",
        "signal_return_pct",
    ]:
        split_table[col] = split_table[col].map(lambda v: f"{float(v):,.2f}%")
    split_table["directional_hit"] = split_table["directional_hit"].map(lambda v: "Hit" if v else "Miss")
    split_table["classification_hit"] = split_table["classification_hit"].map(lambda v: "Hit" if v else "Miss")
    split_table["trade_taken"] = split_table["trade_taken"].map(lambda v: "Trade" if v else "No-Trade")
    split_table = split_table.rename(
        columns={
            "split": "Split",
            "as_of_date": "As Of",
            "realized_date": "Realized Date",
            "train_rows": "Train Rows",
            "predicted_return_pct": "Predicted Return",
            "actual_return_pct": "Realized Return",
            "predicted_price": "Predicted Price",
            "actual_price": "Realized Price",
            "signed_error_pct": "Signed Error",
            "absolute_error_pct": "Absolute Error",
            "directional_hit": "Regression Direction",
            "direction_prob_up_pct": "Prob Up",
            "direction_confidence_pct": "Confidence",
            "classification_hit": "Classification",
            "trade_taken": "Trade Filter",
            "trade_signal": "Signal",
            "signal_return_pct": "Signal Return",
            "overall_regime": "Overall Regime",
        }
    )

    forecast_chart_base64 = _walk_forward_validation_chart(
        split_df,
        ticker=ticker,
        horizon_days=horizon_days,
    )
    diagnostics_chart_base64 = _walk_forward_diagnostics_chart(split_df)
    commentary = _walk_forward_commentary(
        hit_rate=direction_hit_rate,
        mae_return=mae_return,
        rmse_return=rmse_return,
        skill_vs_naive=skill_vs_naive,
        bias_return=bias_return,
        corr=return_correlation,
    )

    saved_dir: str | None = None
    if auto_save:
        out_dir = Path(out_base) / datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
        ensure_writable_dir(out_dir)
        saved_dir = str(out_dir)
        summary_table.to_csv(out_dir / "walk_forward_summary.csv", index=False, encoding="utf-8-sig")
        interpretation_table.to_csv(out_dir / "walk_forward_interpretation.csv", index=False, encoding="utf-8-sig")
        split_df.to_csv(out_dir / "walk_forward_splits.csv", index=False, encoding="utf-8-sig")
        model_df.to_csv(out_dir / "walk_forward_models.csv", index=False, encoding="utf-8-sig")
        threshold_table.to_csv(out_dir / "walk_forward_thresholds.csv", index=False, encoding="utf-8-sig")
        regime_table.to_csv(out_dir / "walk_forward_regimes.csv", index=False, encoding="utf-8-sig")
        (out_dir / "walk_forward_commentary.txt").write_text(commentary, encoding="utf-8")

    source_table = _build_walk_forward_source_table(
        form,
        ticker=ticker,
        input_mode=input_mode,
        price_source=price_source,
        split_count=int(len(split_df)),
    )
    source_table = pd.concat(
        [
            source_table,
            _metadata_table(
                [
                    ("feature_stack", "price + market/sector relative + factor/regime"),
                    ("direction_models", "logistic + random_forest_cls + gradient_boosting_cls"),
                    ("default_no_trade_threshold", f"{int(default_no_trade_threshold * 100)}%"),
                ]
            ),
        ],
        ignore_index=True,
    )

    return _WalkForwardContext(
        ticker=ticker,
        price_source=price_source,
        input_mode=input_mode,
        horizon_days=horizon_days,
        evaluation_splits=int(len(split_df)),
        min_train_rows=min_train_rows,
        step_size=step_size,
        max_splits=max_splits,
        train_start_date=pd.Timestamp(dataset.index[0]).strftime("%Y-%m-%d"),
        latest_as_of_date=str(split_df.iloc[-1]["as_of_date"]),
        latest_realized_date=str(split_df.iloc[-1]["realized_date"]),
        direction_hit_rate=direction_hit_rate,
        mae_return=mae_return,
        rmse_return=rmse_return,
        bias_return=bias_return,
        skill_vs_naive=skill_vs_naive,
        return_correlation=return_correlation,
        classification_hit_rate=classification_hit_rate,
        trade_coverage_rate=trade_coverage_rate,
        trade_hit_rate=trade_hit_rate,
        summary_table=summary_table,
        source_table=source_table,
        interpretation_table=interpretation_table,
        split_table=split_table,
        model_table=model_table,
        threshold_table=threshold_table,
        regime_table=regime_table,
        forecast_chart_base64=forecast_chart_base64,
        diagnostics_chart_base64=diagnostics_chart_base64,
        commentary=commentary,
        saved_dir=saved_dir,
    )


def _format_fin_value(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if not np.isfinite(f):
            return "-"
        if abs(f) >= 100:
            return f"{f:,.0f}"
        return f"{f:,.4f}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,d}"
    return str(value)


def _short_error_message(exc: Exception) -> str:
    msg = " ".join(str(exc).split())
    if not msg:
        return exc.__class__.__name__
    return msg[:120]


def _metrics_table(metrics: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"metric": name, "value": _format_fin_value(value)} for name, value in metrics.items()]
    )


def _request_verify_value(ca_bundle_path: str | None, insecure_ssl: bool) -> bool | str:
    if ca_bundle_path:
        return str(ca_bundle_path)
    if insecure_ssl:
        return False
    return True


_SEC_COMPANY_NAME_CACHE: list[tuple[str, str]] | None = None
_TICKER_TOKEN_RE = re.compile(r"^[A-Za-z0-9\.^/\-_=]{1,15}$")
_LOCAL_COMPANY_ALIAS = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "nvidia": "NVDA",
    "amazon": "AMZN",
    "tesla": "TSLA",
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "meta": "META",
    "netflix": "NFLX",
    "애플": "AAPL",
    "마이크로소프트": "MSFT",
    "엔비디아": "NVDA",
    "아마존": "AMZN",
    "테슬라": "TSLA",
    "구글": "GOOGL",
    "알파벳": "GOOGL",
    "메타": "META",
    "넷플릭스": "NFLX",
}


def _normalize_search_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _is_likely_ticker_symbol(text: str) -> bool:
    return bool(_TICKER_TOKEN_RE.fullmatch(str(text).strip()))


def _sec_company_name_index(verify: bool | str) -> list[tuple[str, str]]:
    global _SEC_COMPANY_NAME_CACHE
    if _SEC_COMPANY_NAME_CACHE is not None:
        return _SEC_COMPANY_NAME_CACHE

    payload = _sec_get_json(SEC_TICKERS_URL, verify=verify)
    if payload is None:
        return []

    entries = payload.values() if isinstance(payload, dict) else []
    parsed: list[tuple[str, str]] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker", "")).strip().upper()
        title = str(item.get("title", "")).strip()
        if not ticker or not title:
            continue
        parsed.append((ticker, title))

    if parsed:
        _SEC_COMPANY_NAME_CACHE = parsed
    return parsed


def _score_company_match(query_norm: str, title_norm: str) -> int:
    if not query_norm or not title_norm:
        return -1

    score = 0
    if query_norm == title_norm:
        score += 120
    elif title_norm.startswith(query_norm):
        score += 90
    elif query_norm in title_norm:
        score += 70

    terms = [t for t in query_norm.split() if len(t) >= 2]
    if not terms:
        terms = [query_norm]

    matched = sum(1 for term in terms if term in title_norm)
    if matched == 0 and query_norm not in title_norm:
        return -1

    score += matched * 8
    score -= max(0, len(title_norm) - len(query_norm)) // 20
    return score


def _search_ticker_from_sec(query: str, *, verify: bool | str) -> tuple[str, str, str] | None:
    q_raw = str(query).strip()
    q_norm = _normalize_search_text(q_raw)
    if len(q_norm) < 2:
        return None

    best: tuple[int, str, str] | None = None
    for ticker, title in _sec_company_name_index(verify=verify):
        title_norm = _normalize_search_text(title)
        score = _score_company_match(q_norm, title_norm)
        if score < 0:
            continue
        if ticker == q_raw.upper():
            score += 80
        if best is None or score > best[0]:
            best = (score, ticker, title)

    if best is None:
        return None
    return best[1], best[2], "sec_company_tickers"


def _search_ticker_from_yfinance(
    query: str,
    *,
    ca_bundle_path: str | None,
    insecure_ssl: bool,
) -> tuple[str, str, str] | None:
    if yf is None or not hasattr(yf, "Search"):
        return None

    kwargs: dict[str, object] = {"max_results": 8, "news_count": 0}
    session = _build_yfinance_session(ca_bundle=ca_bundle_path, insecure_ssl=insecure_ssl)
    if session is not None:
        kwargs["session"] = session

    try:
        search = yf.Search(query, **kwargs)
    except TypeError:
        kwargs.pop("session", None)
        try:
            search = yf.Search(query, **kwargs)
        except Exception:
            return None
    except Exception:
        return None

    quotes = getattr(search, "quotes", None)
    if not isinstance(quotes, list):
        return None

    for item in quotes:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol", "")).strip().upper()
        if not symbol or not _is_likely_ticker_symbol(symbol):
            continue

        quote_type = str(item.get("quoteType", "")).strip().upper()
        if quote_type and quote_type not in {"EQUITY", "ETF", "INDEX"}:
            continue

        title = str(item.get("shortname") or item.get("longname") or item.get("displayName") or symbol).strip()
        return symbol, title, "yfinance_search"

    return None


def _resolve_ticker_input(
    query: str,
    *,
    ca_bundle_path: str | None,
    insecure_ssl: bool,
) -> tuple[str | None, str | None, bool]:
    raw = str(query or "").strip()
    if not raw:
        return None, None, False

    upper_raw = raw.upper()

    alias_ticker = _LOCAL_COMPANY_ALIAS.get(raw.lower())
    if alias_ticker:
        if alias_ticker != upper_raw:
            return alias_ticker, f"Resolved ticker: '{raw}' -> '{alias_ticker}' (local_alias)", False
        return alias_ticker, None, False

    if _is_likely_ticker_symbol(raw) and raw == upper_raw and " " not in raw:
        return upper_raw, None, False

    verify = _request_verify_value(ca_bundle_path=ca_bundle_path, insecure_ssl=insecure_ssl)

    sec_match = _search_ticker_from_sec(raw, verify=verify)
    if sec_match is not None:
        ticker, title, source = sec_match
        if ticker != upper_raw:
            return ticker, f"Resolved ticker: '{raw}' -> '{ticker}' ({title}, {source})", False
        return ticker, None, False

    yf_match = _search_ticker_from_yfinance(raw, ca_bundle_path=ca_bundle_path, insecure_ssl=insecure_ssl)
    if yf_match is not None:
        ticker, title, source = yf_match
        if ticker != upper_raw:
            return ticker, f"Resolved ticker: '{raw}' -> '{ticker}' ({title}, {source})", False
        return ticker, None, False

    if _is_likely_ticker_symbol(upper_raw) and " " not in raw:
        return upper_raw, None, False

    return None, f"Could not resolve company name '{raw}' to a ticker symbol.", True


def _to_number(value: object) -> float | int | None:
    if value is None:
        return None

    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return f if np.isfinite(f) else None

    s = str(value).strip()
    if not s or s.lower() in {"none", "null", "na", "n/a", "-"}:
        return None

    s = s.replace(",", "")
    try:
        f = float(s)
    except Exception:
        return None

    if not np.isfinite(f):
        return None
    return int(f) if f.is_integer() else f


def _normalize_statement_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    try:
        out.columns = [pd.to_datetime(c) for c in out.columns]
        out = out.sort_index(axis=1, ascending=False)
    except Exception:
        pass
    return out


def _pick_latest_value_df(df: pd.DataFrame, candidates: list[str]) -> object:
    if df is None or df.empty or len(df.columns) == 0:
        return None
    latest_col = df.columns[0]
    for name in candidates:
        if name in df.index:
            return _to_number(df.loc[name, latest_col])
    return None


def _pick_recent_values_df(df: pd.DataFrame, candidates: list[str], periods: int) -> list[float]:
    if df is None or df.empty or len(df.columns) == 0:
        return []

    selected = next((name for name in candidates if name in df.index), None)
    if selected is None:
        return []

    values: list[float] = []
    for col in list(df.columns[: max(1, periods)]):
        value = _to_number(df.loc[selected, col])
        if value is not None:
            values.append(float(value))
    return values


def _statement_looks_quarterly(df: pd.DataFrame) -> bool:
    if df is None or df.empty or len(df.columns) < 2:
        return False
    try:
        first = pd.to_datetime(df.columns[0])
        second = pd.to_datetime(df.columns[1])
    except Exception:
        return False
    return abs((first - second).days) <= 120


def _pick_ttm_value_df(df: pd.DataFrame, candidates: list[str]) -> float | None:
    periods = 4 if _statement_looks_quarterly(df) else 1
    values = _pick_recent_values_df(df, candidates, periods=periods)
    if not values:
        return None
    return float(sum(values)) if periods > 1 else float(values[0])


def _pick_average_value_df(df: pd.DataFrame, candidates: list[str], periods: int) -> float | None:
    values = _pick_recent_values_df(df, candidates, periods=periods)
    if not values:
        return None
    return float(sum(values) / len(values))
def _statement_preview_from_df(df: pd.DataFrame, row_map: dict[str, list[str]], periods: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["line_item"])

    cols = list(df.columns[: max(1, periods)])
    rows: list[dict[str, object]] = []
    for label, candidates in row_map.items():
        selected = None
        for c in candidates:
            if c in df.index:
                selected = c
                break
        if selected is None:
            continue

        row: dict[str, object] = {"line_item": label}
        has_any = False
        for col in cols:
            col_name = str(pd.to_datetime(col).date()) if isinstance(col, (pd.Timestamp, datetime)) else str(col)
            value = _to_number(df.loc[selected, col])
            if value is not None:
                has_any = True
            row[col_name] = _format_fin_value(value)
        if has_any:
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["line_item"])
    return pd.DataFrame(rows)


SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
FMP_BASE_URL = "https://financialmodelingprep.com/stable"


def _resolve_fmp_api_key(api_key: str | None) -> str | None:
    key = str(api_key).strip() if api_key else ""
    if not key:
        key = str(os.getenv("FMP_API_KEY", "")).strip()
    return key or None


def _fmp_get_json(endpoint: str, *, symbol: str, api_key: str | None, verify: bool | str) -> dict[str, object] | None:
    if not api_key:
        return None

    session = requests.Session()
    try:
        resp = session.get(
            f"{FMP_BASE_URL}/{endpoint}",
            params={"symbol": symbol, "apikey": api_key},
            timeout=20,
            verify=verify,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return None

    if isinstance(payload, list):
        if payload and isinstance(payload[0], dict):
            return payload[0]
        return None
    return payload if isinstance(payload, dict) else None


def _first_number(source: dict[str, object] | None, *keys: str) -> float | int | None:
    if not isinstance(source, dict):
        return None
    for key in keys:
        value = _to_number(source.get(key))
        if value is not None:
            return value
    return None

def _lookup_number(source: object, *keys: str) -> float | int | None:
    if source is None:
        return None
    for key in keys:
        value = None
        if isinstance(source, dict):
            value = source.get(key)
        else:
            try:
                value = getattr(source, key)
            except Exception:
                value = None
        numeric = _to_number(value)
        if numeric is not None:
            return numeric
    return None


def _load_common_price_history(ticker: str, *, start_date: str) -> tuple[pd.Series, str | None]:
    try:
        prices, source = fetch_sp500_close_prices([ticker], start_date)
    except Exception:
        return pd.Series(dtype=float), None

    if source == "fallback" or prices.empty or ticker not in prices.columns:
        return pd.Series(dtype=float), None

    close = pd.Series(prices[ticker], dtype=float).dropna()
    if close.empty:
        return pd.Series(dtype=float), None

    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close.sort_index()
    return close, str(source)




    return None



def _fetch_fmp_metric_overrides(
    *,
    ticker: str,
    verify: bool | str,
    api_key: str | None,
) -> tuple[dict[str, object], bool]:
    key_metrics_ttm = _fmp_get_json("key-metrics-ttm", symbol=ticker, api_key=api_key, verify=verify)
    ratios_ttm = _fmp_get_json("ratios-ttm", symbol=ticker, api_key=api_key, verify=verify)
    quote = _fmp_get_json("quote", symbol=ticker, api_key=api_key, verify=verify)

    metrics = {
        "Market Cap": _first_number(quote, "marketCap") or _first_number(key_metrics_ttm, "marketCapTTM", "marketCap"),
        "PER (Trailing)": _first_number(quote, "pe") or _first_number(ratios_ttm, "priceEarningsRatioTTM", "priceEarningsRatio") or _first_number(key_metrics_ttm, "peRatioTTM", "peRatio"),
        "PER (Forward)": _first_number(quote, "forwardPE"),
        "PBR": _first_number(quote, "priceToBook") or _first_number(ratios_ttm, "priceToBookRatioTTM", "priceToBookRatio") or _first_number(key_metrics_ttm, "pbRatioTTM", "pbRatio"),
        "EPS (Trailing)": _first_number(quote, "eps") or _first_number(key_metrics_ttm, "netIncomePerShareTTM", "epsTTM"),
        "ROE": _first_number(ratios_ttm, "returnOnEquityTTM", "returnOnEquity") or _first_number(key_metrics_ttm, "roeTTM", "roe"),
        "Debt/Equity": _first_number(ratios_ttm, "debtEquityRatioTTM", "debtEquityRatio", "debtToEquityTTM", "debtToEquity"),
        "Current Ratio": _first_number(ratios_ttm, "currentRatioTTM", "currentRatio"),
        "Dividend Yield": _first_number(quote, "dividendYield") or _first_number(ratios_ttm, "dividendYieldTTM", "dividendYield"),
        "52W High": _first_number(quote, "yearHigh", "52WeekHigh"),
        "52W Low": _first_number(quote, "yearLow", "52WeekLow"),
    }
    used = any(value is not None for value in metrics.values())
    return metrics, used


def _merge_missing_metrics(base_metrics: dict[str, object], supplement_metrics: dict[str, object]) -> tuple[dict[str, object], bool]:
    merged = dict(base_metrics)
    used = False
    for key, value in supplement_metrics.items():
        if merged.get(key) is None and value is not None:
            merged[key] = value
            used = True
    return merged, used


def _sec_headers() -> dict[str, str]:
    return {
        "User-Agent": "Keumj Stock Forecast Lab/1.0 (finance-data-support@example.com)",
        "Accept": "application/json",
    }


def _sec_get_json(url: str, *, verify: bool | str) -> dict[str, object] | None:
    session = requests.Session()
    try:
        resp = session.get(url, headers=_sec_headers(), timeout=20, verify=verify)
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _sec_find_cik_for_ticker(ticker: str, *, verify: bool | str) -> str | None:
    payload = _sec_get_json(SEC_TICKERS_URL, verify=verify)
    if payload is None:
        return None

    target = ticker.strip().upper()
    entries = payload.values() if isinstance(payload, dict) else []
    for item in entries:
        if not isinstance(item, dict):
            continue
        if str(item.get("ticker", "")).strip().upper() != target:
            continue
        cik_raw = item.get("cik_str")
        try:
            return str(int(cik_raw)).zfill(10)
        except Exception:
            return None
    return None


def _sec_pick_series(
    us_gaap: dict[str, object],
    concepts: list[str],
    *,
    units: tuple[str, ...],
) -> dict[pd.Timestamp, float]:
    allowed_forms = {"10-K", "10-Q", "20-F", "40-F"}
    for concept in concepts:
        node = us_gaap.get(concept)
        if not isinstance(node, dict):
            continue
        units_node = node.get("units")
        if not isinstance(units_node, dict):
            continue

        for unit in units:
            values = units_node.get(unit)
            if not isinstance(values, list):
                continue

            latest_by_end: dict[pd.Timestamp, tuple[str, float]] = {}
            for row in values:
                if not isinstance(row, dict):
                    continue
                form = str(row.get("form", "")).upper()
                if form and form not in allowed_forms:
                    continue

                end_raw = str(row.get("end", "")).strip()
                value = _to_number(row.get("val"))
                if not end_raw or value is None:
                    continue

                try:
                    end_ts = pd.Timestamp(end_raw).normalize()
                except Exception:
                    continue

                filed = str(row.get("filed", ""))
                prev = latest_by_end.get(end_ts)
                if prev is None or filed > prev[0]:
                    latest_by_end[end_ts] = (filed, float(value))

            if latest_by_end:
                return {k: v for k, (_, v) in latest_by_end.items()}

    return {}


def _sec_rows_to_df(rows: dict[str, dict[pd.Timestamp, float]]) -> pd.DataFrame:
    all_dates = sorted({d for m in rows.values() for d in m.keys()}, reverse=True)
    if not all_dates:
        return pd.DataFrame()

    data: dict[str, dict[pd.Timestamp, float]] = {}
    for row_name, series in rows.items():
        if not series:
            continue
        data[row_name] = {d: series.get(d, np.nan) for d in all_dates}

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data).T
    df.columns = pd.to_datetime(df.columns)
    return _normalize_statement_df(df)


def _fetch_sec_statement_frames(
    *,
    ticker: str,
    verify: bool | str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    cik = _sec_find_cik_for_ticker(ticker, verify=verify)
    if not cik:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), ""

    facts_payload = _sec_get_json(SEC_COMPANYFACTS_URL.format(cik=cik), verify=verify)
    if not facts_payload:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), ""

    facts = facts_payload.get("facts") if isinstance(facts_payload, dict) else None
    us_gaap = facts.get("us-gaap") if isinstance(facts, dict) else None
    if not isinstance(us_gaap, dict):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), str(facts_payload.get("entityName") or "")

    revenue = _sec_pick_series(us_gaap, ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"], units=("USD",))
    gross_profit = _sec_pick_series(us_gaap, ["GrossProfit"], units=("USD",))
    operating_income = _sec_pick_series(us_gaap, ["OperatingIncomeLoss"], units=("USD",))
    net_income = _sec_pick_series(us_gaap, ["NetIncomeLoss", "ProfitLoss"], units=("USD",))
    diluted_eps = _sec_pick_series(us_gaap, ["EarningsPerShareDiluted"], units=("USD/shares",))

    cash = _sec_pick_series(us_gaap, ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"], units=("USD",))
    assets = _sec_pick_series(us_gaap, ["Assets"], units=("USD",))
    current_assets = _sec_pick_series(us_gaap, ["AssetsCurrent"], units=("USD",))
    liabilities = _sec_pick_series(us_gaap, ["Liabilities"], units=("USD",))
    current_liabilities = _sec_pick_series(us_gaap, ["LiabilitiesCurrent"], units=("USD",))
    equity = _sec_pick_series(us_gaap, ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"], units=("USD",))
    debt = _sec_pick_series(us_gaap, ["Debt", "LongTermDebtAndFinanceLeaseLiabilities", "LongTermDebtNoncurrent", "LongTermDebt"], units=("USD",))

    ocf = _sec_pick_series(us_gaap, ["NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"], units=("USD",))
    icf = _sec_pick_series(us_gaap, ["NetCashProvidedByUsedInInvestingActivities", "NetCashProvidedByUsedInInvestingActivitiesContinuingOperations"], units=("USD",))
    fcfin = _sec_pick_series(us_gaap, ["NetCashProvidedByUsedInFinancingActivities", "NetCashProvidedByUsedInFinancingActivitiesContinuingOperations"], units=("USD",))
    capex = _sec_pick_series(us_gaap, ["PaymentsToAcquirePropertyPlantAndEquipment", "CapitalExpendituresIncurredButNotYetPaid"], units=("USD",))
    free_cf: dict[pd.Timestamp, float] = {}
    for d, v in ocf.items():
        cap = capex.get(d)
        if cap is None:
            continue
        free_cf[d] = float(v) - abs(float(cap))

    income_df = _sec_rows_to_df(
        {
            "Total Revenue": revenue,
            "Gross Profit": gross_profit,
            "Operating Income": operating_income,
            "Net Income": net_income,
            "Diluted EPS": diluted_eps,
        }
    )
    balance_df = _sec_rows_to_df(
        {
            "Cash And Cash Equivalents": cash,
            "Total Assets": assets,
            "Total Liabilities Net Minority Interest": liabilities,
            "Current Assets": current_assets,
            "Current Liabilities": current_liabilities,
            "Stockholders Equity": equity,
            "Total Debt": debt,
        }
    )
    cashflow_df = _sec_rows_to_df(
        {
            "Operating Cash Flow": ocf,
            "Investing Cash Flow": icf,
            "Financing Cash Flow": fcfin,
            "Free Cash Flow": free_cf,
            "Capital Expenditure": capex,
        }
    )

    company_name = str(facts_payload.get("entityName") or "")
    return income_df, balance_df, cashflow_df, company_name


def _build_financial_context_yfinance(
    *,
    ticker: str,
    periods: int,
    out_dir: Path | None,
    saved_dir: str | None,
    fallback_reason: str | None,
    ca_bundle_path: str | None,
    insecure_ssl: bool,
    overview_fallback: dict[str, object] | None = None,
    fmp_api_key: str | None = None,
) -> _FinancialContext:
    if yf is None:
        raise ValueError(
            "yfinance is not installed. Install yfinance to load online financial data."
        )

    verify_value = _request_verify_value(ca_bundle_path=ca_bundle_path, insecure_ssl=insecure_ssl)

    yf_session = _build_yfinance_session(ca_bundle=ca_bundle_path, insecure_ssl=insecure_ssl)
    ticker_obj = yf.Ticker(ticker, session=yf_session)

    try:
        info = ticker_obj.info or {}
        info_error: str | None = None
    except Exception as exc:
        info = {}
        info_error = _short_error_message(exc)

    def _safe_statement_df(attr_name: str) -> pd.DataFrame:
        try:
            return _normalize_statement_df(getattr(ticker_obj, attr_name, pd.DataFrame()))
        except Exception:
            return pd.DataFrame()

    income = _safe_statement_df("income_stmt")
    if income.empty:
        income = _safe_statement_df("financials")
    if income.empty:
        income = _safe_statement_df("quarterly_income_stmt")

    balance = _safe_statement_df("balance_sheet")
    if balance.empty:
        balance = _safe_statement_df("quarterly_balance_sheet")

    cashflow = _safe_statement_df("cashflow")
    if cashflow.empty:
        cashflow = _safe_statement_df("quarterly_cashflow")

    sec_company_name = ""
    used_sec_fallback = False
    if income.empty and balance.empty and cashflow.empty:
        sec_income, sec_balance, sec_cashflow, sec_company_name = _fetch_sec_statement_frames(
            ticker=ticker,
            verify=verify_value,
        )
        if not sec_income.empty:
            income = sec_income
        if not sec_balance.empty:
            balance = sec_balance
        if not sec_cashflow.empty:
            cashflow = sec_cashflow
        used_sec_fallback = not income.empty or not balance.empty or not cashflow.empty

    overview = overview_fallback or {}

    try:
        fast_info = ticker_obj.fast_info or {}
    except Exception:
        fast_info = {}
    if fast_info and not isinstance(fast_info, dict):
        try:
            fast_info = dict(fast_info)
        except Exception:
            fast_info = {}

    used_common_price_loader = False
    try:
        price_history = ticker_obj.history(period="1y", auto_adjust=False)
    except Exception:
        price_history = pd.DataFrame()

    close_series = price_history.get("Close") if isinstance(price_history, pd.DataFrame) else None
    clean_close = pd.Series(dtype=float)
    if close_series is not None:
        clean_close = pd.Series(close_series).dropna()

    if clean_close.empty:
        fallback_close, _ = _load_common_price_history(
            ticker,
            start_date=(pd.Timestamp.today().normalize() - pd.Timedelta(days=370)).strftime("%Y-%m-%d"),
        )
        if not fallback_close.empty:
            clean_close = fallback_close
            used_common_price_loader = True

    recent_close = None
    history_52w_high = None
    history_52w_low = None
    if not clean_close.empty:
        recent_close = float(clean_close.iloc[-1])
        history_52w_high = float(clean_close.max())
        history_52w_low = float(clean_close.min())

    market_cap = (
        _to_number(info.get("marketCap"))
        or _lookup_number(fast_info, "marketCap", "market_cap")
        or _to_number(overview.get("MarketCapitalization"))
    )
    trailing_eps = _to_number(info.get("trailingEps")) or _to_number(overview.get("EPS"))
    latest_price = (
        _lookup_number(info, "currentPrice", "regularMarketPrice", "previousClose")
        or _lookup_number(fast_info, "lastPrice", "last_price", "regularMarketPrice", "previousClose", "regularMarketPreviousClose")
        or recent_close
    )
    year_high = (
        _to_number(info.get("fiftyTwoWeekHigh"))
        or _lookup_number(fast_info, "yearHigh", "year_high")
        or history_52w_high
        or _to_number(overview.get("52WeekHigh"))
    )
    year_low = (
        _to_number(info.get("fiftyTwoWeekLow"))
        or _lookup_number(fast_info, "yearLow", "year_low")
        or history_52w_low
        or _to_number(overview.get("52WeekLow"))
    )

    latest_eps_stmt = _pick_latest_value_df(income, ["Diluted EPS", "Basic EPS"])
    ttm_eps_stmt = _pick_ttm_value_df(income, ["Diluted EPS", "Basic EPS"])
    latest_net_income = _pick_latest_value_df(income, ["Net Income"])
    ttm_net_income = _pick_ttm_value_df(income, ["Net Income"])
    latest_equity = _pick_latest_value_df(balance, ["Stockholders Equity", "Total Stockholder Equity"])
    average_equity = _pick_average_value_df(
        balance,
        ["Stockholders Equity", "Total Stockholder Equity"],
        periods=(4 if _statement_looks_quarterly(balance) else 2),
    )
    latest_debt = _pick_latest_value_df(balance, ["Total Debt", "Long Term Debt"])
    current_assets = _pick_latest_value_df(balance, ["Current Assets", "Total Current Assets"])
    current_liabilities = _pick_latest_value_df(
        balance,
        ["Current Liabilities", "Total Current Liabilities", "Current Debt And Capital Lease Obligation"],
    )

    implied_shares = _lookup_number(info, "sharesOutstanding", "impliedSharesOutstanding") or _lookup_number(
        fast_info,
        "shares",
        "sharesOutstanding",
        "shareCount",
    )
    if implied_shares is None and latest_eps_stmt not in (None, 0) and latest_net_income is not None:
        implied_shares = abs(float(latest_net_income) / float(latest_eps_stmt))
    if implied_shares is None and ttm_eps_stmt not in (None, 0) and ttm_net_income is not None:
        implied_shares = abs(float(ttm_net_income) / float(ttm_eps_stmt))

    metrics = {
        "Market Cap": market_cap,
        "PER (Trailing)": _to_number(info.get("trailingPE")) or _to_number(overview.get("PERatio")),
        "PER (Forward)": _to_number(info.get("forwardPE")) or _to_number(overview.get("ForwardPE")),
        "PBR": _to_number(info.get("priceToBook")) or _to_number(overview.get("PriceToBookRatio")),
        "EPS (Trailing)": trailing_eps,
        "ROE": _to_number(info.get("returnOnEquity")) or _to_number(overview.get("ReturnOnEquityTTM")),
        "Debt/Equity": _to_number(info.get("debtToEquity")) or _to_number(overview.get("DebtToEquity")),
        "Current Ratio": _to_number(info.get("currentRatio")) or _to_number(overview.get("CurrentRatio")),
        "Dividend Yield": _to_number(info.get("dividendYield")) or _to_number(overview.get("DividendYield")),
        "52W High": year_high,
        "52W Low": year_low,
    }
    fmp_metrics, _ = _fetch_fmp_metric_overrides(
        ticker=ticker,
        verify=verify_value,
        api_key=_resolve_fmp_api_key(fmp_api_key),
    )
    metrics, used_fmp = _merge_missing_metrics(metrics, fmp_metrics)

    derived_market_cap = (
        float(latest_price) * float(implied_shares)
        if metrics.get("Market Cap") is None and latest_price is not None and implied_shares is not None
        else None
    )
    effective_market_cap = metrics.get("Market Cap") if metrics.get("Market Cap") is not None else derived_market_cap

    derived_metrics: dict[str, object] = {
        "Market Cap": derived_market_cap,
        "PER (Trailing)": (float(latest_price) / float(ttm_eps_stmt)) if metrics.get("PER (Trailing)") is None and latest_price is not None and ttm_eps_stmt not in (None, 0) else None,
        "PBR": (float(effective_market_cap) / float(latest_equity)) if metrics.get("PBR") is None and effective_market_cap is not None and latest_equity not in (None, 0) else None,
        "EPS (Trailing)": ttm_eps_stmt if metrics.get("EPS (Trailing)") is None else None,
        "ROE": (float(ttm_net_income) / float(average_equity)) if metrics.get("ROE") is None and ttm_net_income is not None and average_equity not in (None, 0) else None,
        "Debt/Equity": ((float(latest_debt) / float(latest_equity)) * 100.0) if metrics.get("Debt/Equity") is None and latest_debt is not None and latest_equity not in (None, 0) else None,
        "Current Ratio": (float(current_assets) / float(current_liabilities)) if metrics.get("Current Ratio") is None and current_assets is not None and current_liabilities not in (None, 0) else None,
        "52W High": history_52w_high if metrics.get("52W High") is None else None,
        "52W Low": history_52w_low if metrics.get("52W Low") is None else None,
    }
    metrics, used_derived = _merge_missing_metrics(metrics, derived_metrics)
    provider_status_table = _build_financial_provider_status_table(
        yfinance_info=("ok" if bool(info) else f"error: {info_error}" if info_error else "empty"),
        yfinance_statements="ok" if not (income.empty and balance.empty and cashflow.empty) else "empty",
        price_history="ok" if not price_history.empty else ("shared_sql" if used_common_price_loader else "empty"),
        sec_fallback="used" if used_sec_fallback else ("not needed" if not (income.empty and balance.empty and cashflow.empty) else "not available"),
        fmp_metrics="used" if used_fmp else "not used",
        derived_metrics="used" if used_derived else "not used",
    )

    income_table = _statement_preview_from_df(
        income,
        row_map={
            "Revenue": ["Total Revenue", "Revenue"],
            "Gross Profit": ["Gross Profit"],
            "Operating Income": ["Operating Income"],
            "Net Income": ["Net Income"],
            "Diluted EPS": ["Diluted EPS", "Basic EPS"],
        },
        periods=periods,
    )
    balance_table = _statement_preview_from_df(
        balance,
        row_map={
            "Cash": ["Cash And Cash Equivalents", "Cash"],
            "Total Assets": ["Total Assets"],
            "Total Liabilities": ["Total Liabilities Net Minority Interest", "Total Liab"],
            "Stockholders Equity": ["Stockholders Equity", "Total Stockholder Equity"],
            "Total Debt": ["Total Debt", "Long Term Debt"],
        },
        periods=periods,
    )
    cashflow_table = _statement_preview_from_df(
        cashflow,
        row_map={
            "Operating Cash Flow": ["Operating Cash Flow"],
            "Investing Cash Flow": ["Investing Cash Flow"],
            "Financing Cash Flow": ["Financing Cash Flow"],
            "Free Cash Flow": ["Free Cash Flow"],
            "Capital Expenditure": ["Capital Expenditure"],
        },
        periods=periods,
    )

    summary_table = pd.DataFrame(
        [
            {
                "line_item": "Revenue",
                "latest_value": _format_fin_value(_pick_latest_value_df(income, ["Total Revenue", "Revenue"])),
            },
            {
                "line_item": "Operating Income",
                "latest_value": _format_fin_value(_pick_latest_value_df(income, ["Operating Income"])),
            },
            {
                "line_item": "Net Income",
                "latest_value": _format_fin_value(_pick_latest_value_df(income, ["Net Income"])),
            },
            {
                "line_item": "Total Assets",
                "latest_value": _format_fin_value(_pick_latest_value_df(balance, ["Total Assets"])),
            },
            {
                "line_item": "Total Liabilities",
                "latest_value": _format_fin_value(
                    _pick_latest_value_df(balance, ["Total Liabilities Net Minority Interest", "Total Liab"])
                ),
            },
            {
                "line_item": "Operating Cash Flow",
                "latest_value": _format_fin_value(_pick_latest_value_df(cashflow, ["Operating Cash Flow"])),
            },
            {
                "line_item": "Free Cash Flow",
                "latest_value": _format_fin_value(_pick_latest_value_df(cashflow, ["Free Cash Flow"])),
            },
        ]
    )

    no_statements = income_table.empty and balance_table.empty and cashflow_table.empty
    if no_statements and not any(v is not None for v in metrics.values()):
        raise ValueError(
            "yfinance and SEC financial data were unavailable for this ticker. Wait a few minutes and retry."
        )

    extra_sources: list[str] = []
    if used_sec_fallback:
        extra_sources.append("sec_companyfacts")
    if used_fmp:
        extra_sources.append("fmp_metrics")
    if used_derived:
        extra_sources.append("derived_metrics")

    if out_dir is not None:
        _metrics_table(metrics).to_csv(out_dir / "financial_metrics.csv", index=False)
        summary_table.to_csv(out_dir / "financial_summary.csv", index=False)
        if not income.empty:
            income.to_csv(out_dir / "income_statement_raw.csv")
        if not balance.empty:
            balance.to_csv(out_dir / "balance_sheet_raw.csv")
        if not cashflow.empty:
            cashflow.to_csv(out_dir / "cashflow_statement_raw.csv")
        source_note = "source=yfinance\n"
        for idx, source_name in enumerate(extra_sources, start=2):
            source_note += f"source{idx}={source_name}\n"
        if fallback_reason:
            source_note += f"reason={fallback_reason}\n"
        if no_statements:
            source_note += "warning=statement_data_unavailable_provider_limits\n"
        (out_dir / "financial_source.txt").write_text(source_note, encoding="utf-8")

    return _FinancialContext(
        ticker=ticker,
        company_name=str(
            info.get("longName")
            or info.get("shortName")
            or sec_company_name
            or overview.get("Name")
            or overview.get("Symbol")
            or ticker
        ),
        currency=str(info.get("currency") or overview.get("Currency") or ""),
        metrics=metrics,
        summary_table=summary_table,
        income_table=income_table,
        balance_table=balance_table,
        cashflow_table=cashflow_table,
        saved_dir=saved_dir,
        source_table=_build_financial_source_table(
            ticker=ticker,
            primary_source="yfinance",
            secondary_source=extra_sources[0] if len(extra_sources) > 0 else None,
            tertiary_source=extra_sources[1] if len(extra_sources) > 1 else None,
            reason=fallback_reason,
            warning="statement_data_unavailable_provider_limits" if no_statements else None,
        ),
        provider_status_table=provider_status_table,
    )


def _run_financial_once(form: dict[str, str]) -> _FinancialContext:
    ticker = form.get("ticker", "").strip().upper()
    if not ticker:
        raise ValueError("Provide ticker for financial statements page.")

    periods = int(form.get("statement_periods", "4").strip() or "4")
    periods = max(1, min(periods, 8))

    insecure_ssl = form.get("insecure_ssl", "") == "on"
    ca_bundle_path = form.get("ca_bundle_path", "").strip() or None
    configure_ssl(insecure_ssl=insecure_ssl, ca_bundle=ca_bundle_path)

    auto_save = form.get("auto_save", "on") == "on"
    out_base = form.get("output_dir", "outputs/stock_forecast_finance").strip() or "outputs/stock_forecast_finance"

    out_dir: Path | None = None
    saved_dir: str | None = None
    if auto_save:
        out_dir = Path(out_base) / datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
        ensure_writable_dir(out_dir)
        saved_dir = str(out_dir)

    return _build_financial_context_yfinance(
        ticker=ticker,
        periods=periods,
        out_dir=out_dir,
        saved_dir=saved_dir,
        fallback_reason=None,
        ca_bundle_path=ca_bundle_path,
        insecure_ssl=insecure_ssl,
        overview_fallback=None,
        fmp_api_key=form.get("fmp_api_key", "").strip() or None,
    )


def _nav(active: str, *, enable_technical_page: bool = False, is_sub_page: bool = False) -> str:
    c1 = "active" if active == "page1" else ""
    c2 = "active" if active == "page2" else ""
    c3 = "active" if active == "page3" else ""
    c4 = "active" if active == "page4" else ""
    c5 = "active" if active == "page5" else ""
    cf = "active" if active == "factor" else ""
    c6 = "active" if active == "page6" else ""
    c8 = "active" if active == "page8" else ""
    sync_script = ""
    if enable_technical_page:
        sync_script = """
        <script>
          (function () {
            if (window.__keumjStockSyncInstalled) {
              return;
            }
            window.__keumjStockSyncInstalled = true;
            let lastCommandId = null;

            async function pollExternalCommand() {
              try {
                const res = await fetch("/external_command_state", { cache: "no-store" });
                if (res.ok) {
                  const data = await res.json();
                  const commandId = Number(data.command_id || 0);
                  if (lastCommandId === null) {
                    lastCommandId = commandId;
                  } else if (commandId > lastCommandId && data.navigate_url) {
                    lastCommandId = commandId;
                    const nextUrl = String(data.navigate_url);
                    const currentUrl = window.location.pathname + window.location.search;
                    if (currentUrl !== nextUrl) {
                      window.location.href = nextUrl;
                      return;
                    }
                  } else {
                    lastCommandId = commandId;
                  }
                }
              } catch (err) {
              }
              window.setTimeout(pollExternalCommand, 1200);
            }

            window.setTimeout(pollExternalCommand, 1200);
          })();
        </script>
        """
    if is_sub_page: # Sub-pages should not render their own navigation
        return ""
    links = [
        f'<a class="{c1}" href="/forecast">주가 예측</a>',
        f'<a class="{c2}" href="/page2">재무제표·밸류에이션</a>',
    ]
    if enable_technical_page:
        links.extend(
            [
                f'<a class="{c3}" href="/page3">기술적 분석</a>',
                f'<a class="{c4}" href="/page4">수익률 비교</a>',
                f'<a class="{c5}" href="/page5">리스크 대시보드</a>',
                f'<a class="{cf}" href="/factor-regime">팩터·레짐 랩</a>',
                f'<a class="{c6}" href="/page6">의사결정 대시보드</a>',
                f'<a class="{c8}" href="/page8">워크포워드 검증</a>',
            ]
        )
    if os.getenv("ENABLE_MACRO", "").strip().lower() in {"1", "true", "yes", "on"}:
        links.append('<a class="" href="/macro/overview">거시분석</a>')
    return '<div class="nav">' + "".join(links) + "</div>" + sync_script


def _base_css(is_sub_page: bool = False) -> str:
    return """
    :root {
      """ + _shared_theme_root_css() + """
    }
    .latest-inline {
      margin-top: 8px; padding: 8px 10px; border-radius: 8px;
      background: #eef4fb; border: 1px solid #c7d9ee; color: #24425f; font-size: 12px; line-height: 1.45;
    }
    body { margin: 0; background: var(--bg); color: var(--text); font-family: "Segoe UI", "Noto Sans KR", sans-serif; }
    .wrap { max-width: 1460px; margin: 0 auto; padding: 20px; }
    h1 { margin: 0 0 10px; font-size: 24px; }
    .page-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; margin-bottom: 10px; }
    .page-head h1 { margin: 0; }
    .page-credit { color: var(--muted); font-size: 11px; white-space: nowrap; padding-top: 4px; }
    .sub { color: var(--muted); margin-bottom: 14px; }
    .card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 14px; }
    .nav { display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap; }
    .nav a { text-decoration: none; color: var(--brand); border: 1px solid var(--line); background: #fff; border-radius: 999px; padding: 7px 12px; font-size: 13px; }
    .nav a.active { background: var(--brand); color: #fff; border-color: var(--brand); }
    .nav a.refresh { color: #111; border-color: #111; }
    .nav a.refresh.active { background: #111; color: #fff; border-color: #111; }
    .form-grid { display: grid; grid-template-columns: repeat(4, minmax(180px, 1fr)); gap: 10px 12px; }
    .form-grid label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 4px; }
    .form-grid input[type="text"], .form-grid input[type="number"] {
      width: 100%; box-sizing: border-box; padding: 8px; border: 1px solid var(--line); border-radius: 6px;
    }
    .row { display: flex; flex-wrap: wrap; gap: 14px; align-items: center; margin-top: 10px; }
    button { background: var(--brand); border: 0; color: #fff; padding: 10px 16px; border-radius: 8px; cursor: pointer; font-weight: 600; }
    .notice { margin-top: 10px; border-radius: 8px; padding: 10px; }
    .notice.ok { background: var(--ok-bg); border: 1px solid var(--ok-line); }
    .notice.err { background: var(--err-bg); border: 1px solid var(--err-line); }
    .metrics { margin-top: 12px; display: grid; grid-template-columns: repeat(4, minmax(170px, 1fr)); gap: 10px; }
    .metric { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 10px; }
    .metric span { display: block; font-size: 12px; color: var(--muted); }
    .metric strong { display: block; margin-top: 5px; font-size: 17px; }
    .charts { margin-top: 12px; display: grid; grid-template-columns: repeat(2, minmax(290px, 1fr)); gap: 10px; }
    .charts img { width: 100%; height: auto; }
    .chart-grid { margin-top: 12px; display: grid; grid-template-columns: repeat(2, minmax(280px, 1fr)); gap: 10px; }
    .chart-card { border: 1px solid var(--line); border-radius: 10px; background: #fff; padding: 10px; }
    .chart-card h4 { margin: 0 0 8px 0; }
    .chart-card img { width: 100%; height: auto; border-radius: 6px; }
    .chart-desc { margin: 8px 0 0 0; font-size: 12px; color: var(--muted); line-height: 1.4; }
    .tables { margin-top: 12px; display: grid; grid-template-columns: 1fr; gap: 10px; }
    .table-grid { margin-top: 12px; display: grid; grid-template-columns: repeat(2, minmax(320px, 1fr)); gap: 10px; }
    .table-wrap { width: 100%; max-width: 100%; min-width: 0; overflow-x: auto; -webkit-overflow-scrolling: touch; }
    .data-table { width: max-content; min-width: 100%; border-collapse: collapse; font-size: 13px; line-height: 1.45; }
    .data-table th, .data-table td { border: 1px solid var(--line); padding: 8px; text-align: left; vertical-align: top; white-space: normal; overflow-wrap: break-word; word-break: keep-all; }
    .stacked-table-group { display: grid; gap: 12px; }
    .stacked-table-block h4 { margin: 4px 0 8px; font-size: 13px; color: var(--muted); }
    pre { margin: 0; white-space: pre-wrap; word-break: break-word; }
    @media (max-width: 980px) {
      .form-grid { grid-template-columns: repeat(2, minmax(160px, 1fr)); }
      .metrics { grid-template-columns: repeat(2, minmax(160px, 1fr)); }
      .charts { grid-template-columns: 1fr; }
      .chart-grid { grid-template-columns: 1fr; }
      .table-grid { grid-template-columns: 1fr; }
    }
    """


def _shared_theme_root_css() -> str:
    return """
      --bg: #f3f5f7;
      --card: #ffffff;
      --line: #d4dde8;
      --text: #1f2937;
      --muted: #5f6b7a;
      --brand: #0f4c81;
      --accent: #0f4c81;
      --ok-bg: #e8f7ee;
      --ok-line: #99d5af;
      --err-bg: #fff2f2;
      --err-line: #efadad;
    """.strip()


def _page_head(title: str, is_sub_page: bool = False) -> str:
    return (
        "<div class=\"page-head\">"
        f"<h1>{html.escape(title)}</h1>"
        "<div class=\"page-credit\">Keumj 제작</div>"
        "</div>"
    )


def _html_page(
    form: dict[str, str],
    *,
    ctx: _RunContext | None,
    error: str | None,
    ticker_note: str | None = None,
    ticker_note_error: bool = False,
    is_sub_page: bool = False,
    enable_technical_page: bool = False,
) -> str:
    use_sample_checked = "checked" if form.get("use_sample", "") == "on" else ""
    auto_save_checked = "checked" if form.get("auto_save", "on") == "on" else ""
    insecure_ssl_checked = "checked" if form.get("insecure_ssl", "") == "on" else ""
    defaults = {
        "ticker": form.get("ticker", ""),
        "forecast_horizon": form.get("forecast_horizon", "10"),
        "history_years": form.get("history_years", "8"),
        "start_date": form.get("start_date", "2025-12-31"),
        "end_date": form.get("end_date", datetime.utcnow().strftime("%Y-%m-%d")),
        "output_dir": form.get("output_dir", "outputs/stock_forecast"),
        "prices_csv_path": form.get("prices_csv_path", ""),
        "ca_bundle_path": form.get("ca_bundle_path", ""),
    }

    info_html = ""
    if ctx is not None and ctx.saved_dir:
        info_html += f'<div class="notice ok">예측 결과를 <code>{html.escape(ctx.saved_dir)}</code>에 저장했습니다.</div>'
    if error:
        info_html += f'<div class="notice err"><pre>{html.escape(error)}</pre></div>'
    if ticker_note:
        css = "err" if ticker_note_error else "ok"
        info_html += f'<div class="notice {css}"><pre>{html.escape(ticker_note)}</pre></div>'
    if ctx is None and not error:
        info_html += (
            "<div class=\"notice ok\">"
            "주가 예측 페이지는 선택한 티커 또는 선택 입력인 로컬 가격 CSV를 바탕으로 "
            "10영업일 앙상블 예측을 수행합니다. 오프라인 데모가 필요할 때만 샘플 데이터를 사용해 주세요."
            "</div>"
        )

    metric_html = ""
    charts_html = ""
    tables_html = ""
    if ctx is not None:
        summary = ctx.result.summary.iloc[0]
        metric_html = f"""
        <div class="metrics">
          <div class="metric"><span>티커</span><strong>{html.escape(str(summary.get("ticker", "-")))}</strong></div>
          <div class="metric"><span>기준일</span><strong>{html.escape(str(summary.get("as_of_date", "-")))}</strong></div>
          <div class="metric"><span>예측일</span><strong>{html.escape(str(summary.get("forecast_date", "-")))}</strong></div>
          <div class="metric"><span>예측 기간(영업일)</span><strong>{_format_metric(summary.get("horizon_days"), 0)}</strong></div>
          <div class="metric"><span>최근 종가</span><strong>{_format_metric(summary.get("last_close"), 2)}</strong></div>
          <div class="metric"><span>예측 주가</span><strong>{_format_metric(summary.get("predicted_price"), 2)}</strong></div>
          <div class="metric"><span>기대 수익률</span><strong>{_format_metric(summary.get("expected_return_pct"), 2)}%</strong></div>
          <div class="metric"><span>상승 확률</span><strong>{_format_metric(summary.get("direction_prob_up_pct"), 2)}%</strong></div>
          <div class="metric"><span>방향 신뢰도</span><strong>{_format_metric(summary.get("direction_confidence_pct"), 2)}%</strong></div>
          <div class="metric"><span>판단 신호</span><strong>{html.escape(str(summary.get("direction_signal", "-")))}</strong></div>
          <div class="metric"><span>거래 필터</span><strong>{html.escape(str(summary.get("trade_filter", "-")))}</strong></div>
          <div class="metric"><span>앙상블 로그수익률</span><strong>{_format_metric(summary.get("ensemble_predicted_log_return"), 5)}</strong></div>
        </div>
        """

        charts_html = f"""
        <div class="charts">
          <div class="card"><h3>주가 예측</h3><img src="data:image/png;base64,{_price_forecast_chart(ctx.result)}" alt="price forecast chart" /></div>
          <div class="card"><h3>모델 가중치</h3><img src="data:image/png;base64,{_model_weight_chart(ctx.result)}" alt="model weight chart" /></div>
        </div>
        """

        importance = (
            ctx.result.feature_importance.dropna(subset=["importance"])
            .sort_values(["model", "importance"], ascending=[True, False])
            .groupby("model", as_index=False)
            .head(12)
            .reset_index(drop=True)
        )
        tables_html = f"""
        <div class="tables">
          <div class="card"><h3>데이터 소스 메타데이터</h3>{_safe_table(ctx.source_table)}</div>
          <div class="card"><h3>예측 요약</h3>{_stacked_table_html(ctx.result.summary, pinned_columns=["ticker", "as_of_date", "forecast_date"], max_tables=3, min_chunk_columns=3, section_title="예측 요약")}</div>
          <div class="card"><h3>모델 점수</h3>{_stacked_table_html(ctx.result.model_scores, pinned_columns=["model"], max_tables=2, min_chunk_columns=3, section_title="모델 점수")}</div>
          <div class="card"><h3>방향성 점수</h3>{_stacked_table_html(ctx.result.direction_scores, pinned_columns=["model"], max_tables=2, min_chunk_columns=3, section_title="방향성 점수")}</div>
          <div class="card"><h3>현재 장세 스냅샷</h3>{_stacked_table_html(ctx.result.regime_snapshot, max_tables=2, min_chunk_columns=3, section_title="현재 장세 스냅샷")}</div>
          <div class="card"><h3>상위 피처 중요도(모델별)</h3>{_stacked_table_html(importance, pinned_columns=["model", "feature"], max_tables=2, min_chunk_columns=2, section_title="상위 피처 중요도")}</div>
        </div>
        """

    head_content = ""
    if not is_sub_page:
        head_content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Stock Analysis Lab | S&P 500 - 주가 예측</title>
  <style>{_base_css()}</style>
</head>
<body>"""

    body_content = f"""
  <div class="wrap">
    {_page_head(_APP_TITLE)}
    <div class="sub">선택한 티커 또는 로컬 가격 CSV를 바탕으로 10영업일 앙상블 주가 예측을 수행하는 페이지</div>
    {_nav("page1", enable_technical_page=enable_technical_page)}
    <form class="card" method="post" action="/run">
      <div class="form-grid">
        <div><label>티커</label><input type="text" name="ticker" value="{html.escape(defaults["ticker"])}" /></div>
        <div><label>예측 기간(영업일)</label><input type="number" min="1" name="forecast_horizon" value="{html.escape(defaults["forecast_horizon"])}" /></div>
        <div><label>과거 데이터 연수</label><input type="number" min="2" name="history_years" value="{html.escape(defaults["history_years"])}" /></div>
        <div><label>시작일(선택)</label><input type="text" name="start_date" value="{html.escape(defaults["start_date"])}" /></div>
        <div><label>종료일(선택)</label><input type="text" name="end_date" value="{html.escape(defaults["end_date"])}" /></div>
        <div><label>출력 기본 폴더</label><input type="text" name="output_dir" value="{html.escape(defaults["output_dir"])}" /></div>
        <div><label>로컬 가격 CSV 경로(선택)</label><input type="text" name="prices_csv_path" value="{html.escape(defaults["prices_csv_path"])}" /></div>
        <div><label>CA 번들 경로(선택)</label><input type="text" name="ca_bundle_path" value="{html.escape(defaults["ca_bundle_path"])}" /></div>
      </div>
      <div class="row">
        <label><input type="checkbox" name="use_sample" {use_sample_checked} /> 샘플 가격 데이터 사용(오프라인)</label>
        <label><input type="checkbox" name="auto_save" {auto_save_checked} /> 결과 자동 저장</label>
        <label><input type="checkbox" name="insecure_ssl" {insecure_ssl_checked} /> 임시로 SSL 검증 완화</label>
        <button type="submit" name="intent" value="run">주가 예측 실행</button>
        <button type="submit" name="intent" value="resolve_ticker">회사명으로 티커 찾기</button>
      </div>
    </form>
    {info_html}
    {metric_html}
    {charts_html}
    {tables_html}
  </div>"""

    if is_sub_page:
        return body_content
    return head_content + body_content + "</body>\n</html>\n"


def _html_walk_forward_page(
    form: dict[str, str],
    *,
    ctx: _WalkForwardContext | None,
    error: str | None,
    ticker_note: str | None = None,
    is_sub_page: bool = False,
    ticker_note_error: bool = False,
) -> str:
    defaults = {
        "ticker": form.get("ticker", ""),
        "forecast_horizon": form.get("forecast_horizon", "10"),
        "history_years": form.get("history_years", "8"),
        "start_date": form.get("start_date", "2025-12-31"),
        "end_date": form.get("end_date", datetime.utcnow().strftime("%Y-%m-%d")),
        "wf_min_train_rows": form.get("wf_min_train_rows", "252"),
        "wf_step_size": form.get("wf_step_size", "21"),
        "wf_max_splits": form.get("wf_max_splits", "4"),
        "output_dir": form.get("output_dir", "outputs/walk_forward_validation"),
        "prices_csv_path": form.get("prices_csv_path", ""),
        "ca_bundle_path": form.get("ca_bundle_path", ""),
    }
    use_sample_checked = "checked" if form.get("use_sample", "") == "on" else ""
    auto_save_checked = "checked" if form.get("auto_save", "on") == "on" else ""
    insecure_ssl_checked = "checked" if form.get("insecure_ssl", "") == "on" else ""

    info_html = ""
    if ctx is not None and ctx.saved_dir:
        info_html += f'<div class="notice ok">워크포워드 결과를 <code>{html.escape(ctx.saved_dir)}</code>에 저장했습니다.</div>'
    if error:
        info_html += f'<div class="notice err"><pre>{html.escape(error)}</pre></div>'
    if ticker_note:
        css = "err" if ticker_note_error else "ok"
        info_html += f'<div class="notice {css}"><pre>{html.escape(ticker_note)}</pre></div>'
    if ctx is None and not error:
        info_html += (
            "<div class=\"notice ok\">"
            "워크-포워드 검증(Walk-Forward Validation)은 여러 과거 시점으로 시간을 되감아, "
            "각 시점에 실제로 사용 가능했을 정보만으로 예측을 다시 수행합니다. "
            "단 한 번의 최신 예측이 아니라 예측 엔진의 반복적 신뢰도를 점검하는 데 가장 적합한 페이지입니다. "
            "실행 시간이 길게 느껴지면 먼저 Max Splits를 낮춰 주세요."
            "</div>"
        )

    metric_html = ""
    explanation_html = ""
    charts_html = ""
    tables_html = ""
    if ctx is not None:
        metric_html = f"""
        <div class="metrics">
          <div class="metric"><span>Ticker</span><strong>{html.escape(ctx.ticker)}</strong></div>
          <div class="metric"><span>Price Source</span><strong>{html.escape(ctx.price_source)}</strong></div>
          <div class="metric"><span>평가 분할 수 (Splits)</span><strong>{ctx.evaluation_splits:,d}</strong></div>
          <div class="metric"><span>예측 수평선 (Horizon)</span><strong>{ctx.horizon_days:,d}d</strong></div>
          <div class="metric"><span>회귀 방향 적중률</span><strong>{_format_pct(ctx.direction_hit_rate)}</strong></div>
          <div class="metric"><span>분류 적중률</span><strong>{_format_pct(ctx.classification_hit_rate)}</strong></div>
          <div class="metric"><span>거래 커버리지</span><strong>{_format_pct(ctx.trade_coverage_rate)}</strong></div>
          <div class="metric"><span>선별 거래 적중률</span><strong>{_format_pct(ctx.trade_hit_rate) if ctx.trade_hit_rate is not None else "-"}</strong></div>
          <div class="metric"><span>평균절대오차 (MAE)</span><strong>{_format_pct(ctx.mae_return)}</strong></div>
          <div class="metric"><span>RMSE</span><strong>{_format_pct(ctx.rmse_return)}</strong></div>
          <div class="metric"><span>나이브 대비 스킬</span><strong>{_format_pct(ctx.skill_vs_naive) if ctx.skill_vs_naive is not None else "-"}</strong></div>
          <div class="metric"><span>평균 편향 (Bias)</span><strong>{_format_pct(ctx.bias_return)}</strong></div>
          <div class="metric"><span>수익률 상관 (Correlation)</span><strong>{_format_metric(ctx.return_correlation, 3) if ctx.return_correlation is not None else "-"}</strong></div>
          <div class="metric"><span>최근 As-Of 날짜</span><strong>{html.escape(ctx.latest_as_of_date)}</strong></div>
          <div class="metric"><span>최근 실현 날짜</span><strong>{html.escape(ctx.latest_realized_date)}</strong></div>
        </div>
        """
        explanation_html = f"""
        <div class="card">
          <h3>워크-포워드 검증 (Walk-Forward Validation) 읽는 법</h3>
          <p>각 분할(split)은 특정 시점(as-of date)까지 확보 가능했던 과거 데이터만으로 동일한 예측 구조를 다시 학습한 뒤, 다음 {ctx.horizon_days} 영업일의 움직임을 예측하고 실제 결과와 비교합니다.</p>
          <ul>
            <li><b>회귀 방향 적중률</b>은 가격 예측 회귀모형의 부호만 봤을 때 방향을 얼마나 맞히는지 보여줍니다.</li>
            <li><b>분류 적중률 (Classification Hit Rate)</b>은 상승/하락 분류기(classifier)가 별도로 방향을 고르는 능력을 따로 점검합니다.</li>
            <li><b>거래 커버리지</b>와 <b>선별 거래 적중률</b>은 no-trade 필터를 통과한 신호만 따로 보면 품질이 좋아지는지를 보여줍니다.</li>
            <li><b>평균절대오차 (MAE)</b>와 <b>RMSE</b>는 예측 선행수익률과 실제 선행수익률의 거리감을 측정합니다.</li>
            <li><b>나이브 대비 스킬 (Skill vs Naive)</b>은 0% 수익률을 가정한 단순 기준과 비교한 값으로, 플러스면 모델이 적어도 평균 오차 측면에서는 더 낫다는 뜻입니다.</li>
            <li><b>편향 (Bias)</b>은 모델이 구조적으로 너무 낙관적이었는지, 혹은 지나치게 보수적이었는지 보여줍니다.</li>
            <li><b>레짐별 요약 (Regime Summary)</b>은 어떤 시장 환경에서 예측이 잘 맞고, 어떤 환경에서 쉽게 깨지는지를 보여줍니다.</li>
          </ul>
          <p>{html.escape(ctx.commentary)}</p>
        </div>
        """
        charts_html = f"""
        <div class="charts">
          <div class="card"><h3>예측 선행수익률 vs 실제 선행수익률</h3><img src="data:image/png;base64,{ctx.forecast_chart_base64}" alt="walk-forward validation chart" /></div>
          <div class="card"><h3>오차와 롤링 적중률</h3><img src="data:image/png;base64,{ctx.diagnostics_chart_base64}" alt="walk-forward diagnostics chart" /></div>
        </div>
        """
        model_table_html = _safe_table(ctx.model_table) if not ctx.model_table.empty else "<p class='hint'>Model-level diagnostics were not available.</p>"
        tables_html = f"""
        <div class="tables">
          <div class="card"><h3>검증 요약</h3>{_safe_table(ctx.summary_table)}</div>
          <div class="card"><h3>해석 가이드</h3>{_safe_table(ctx.interpretation_table)}</div>
          <div class="card"><h3>No-Trade Threshold Summary</h3>{_safe_table(ctx.threshold_table)}</div>
          <div class="card"><h3>Regime Summary</h3>{_safe_table(ctx.regime_table)}</div>
          <div class="card"><h3>데이터 소스 메타데이터</h3>{_safe_table(ctx.source_table)}</div>
          <div class="card"><h3>모델 진단</h3>{model_table_html}</div>
          <div class="card"><h3>분할별 결과</h3>{_stacked_table_html(ctx.split_table, max_rows=120, pinned_columns=["Split", "As Of", "Realized Date"], max_tables=3, min_chunk_columns=4, section_title="분할별 결과")}</div>
        </div>
        """

    head_content = ""
    if not is_sub_page:
        head_content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Stock Analysis Lab | S&P 500 - 워크포워드 검증</title>
  <style>{_base_css()}</style>
</head>
<body>"""

    body_content = f"""
  <div class="wrap">
    {_page_head(_APP_TITLE)}
    <div class="sub">워크-포워드 검증으로 예측 엔진의 반복 성능을 점검하고 결과를 쉽게 해석하는 페이지</div>
    {_nav("page8", enable_technical_page=True)}
    <form class="card" method="post" action="/run_walk_forward">
      <div class="form-grid">
        <div><label>티커</label><input type="text" name="ticker" value="{html.escape(defaults["ticker"])}" /></div>
        <div><label>예측 기간(영업일)</label><input type="number" min="1" name="forecast_horizon" value="{html.escape(defaults["forecast_horizon"])}" /></div>
        <div><label>과거 데이터 연수</label><input type="number" min="2" name="history_years" value="{html.escape(defaults["history_years"])}" /></div>
        <div><label>최소 학습 행 수</label><input type="number" min="80" name="wf_min_train_rows" value="{html.escape(defaults["wf_min_train_rows"])}" /></div>
        <div><label>분할 간격</label><input type="number" min="1" name="wf_step_size" value="{html.escape(defaults["wf_step_size"])}" /></div>
        <div><label>최대 분할 수</label><input type="number" min="1" name="wf_max_splits" value="{html.escape(defaults["wf_max_splits"])}" /></div>
        <div><label>시작일(선택)</label><input type="text" name="start_date" value="{html.escape(defaults["start_date"])}" /></div>
        <div><label>종료일(선택)</label><input type="text" name="end_date" value="{html.escape(defaults["end_date"])}" /></div>
        <div><label>출력 기본 폴더</label><input type="text" name="output_dir" value="{html.escape(defaults["output_dir"])}" /></div>
        <div><label>로컬 가격 CSV 경로(선택)</label><input type="text" name="prices_csv_path" value="{html.escape(defaults["prices_csv_path"])}" /></div>
        <div><label>CA 번들 경로(선택)</label><input type="text" name="ca_bundle_path" value="{html.escape(defaults["ca_bundle_path"])}" /></div>
      </div>
      <div class="row">
        <label><input type="checkbox" name="use_sample" {use_sample_checked} /> 샘플 가격 데이터 사용(오프라인)</label>
        <label><input type="checkbox" name="auto_save" {auto_save_checked} /> 결과 자동 저장</label>
        <label><input type="checkbox" name="insecure_ssl" {insecure_ssl_checked} /> 임시로 SSL 검증 완화</label>
        <button type="submit" name="intent" value="run">워크포워드 검증 실행</button>
        <button type="submit" name="intent" value="resolve_ticker">회사명으로 티커 찾기</button>
      </div>
    </form>
    {info_html}
    {metric_html}
    {explanation_html}
    {charts_html}
    {tables_html}
  </div>
</body>
</html>
"""


def _html_financial_page(
    form: dict[str, str],
    *,
    ctx: _FinancialContext | None,
    error: str | None,
    ticker_note: str | None = None,
    ticker_note_error: bool = False,
    is_sub_page: bool = False,
    enable_technical_page: bool = False,
) -> str:
    defaults = {
        "ticker": form.get("ticker", ""),
        "statement_periods": form.get("statement_periods", "4"),
        "output_dir": form.get("output_dir", "outputs/stock_forecast_finance"),
        "ca_bundle_path": form.get("ca_bundle_path", ""),
        "fmp_api_key": form.get("fmp_api_key", ""),
    }
    auto_save_checked = "checked" if form.get("auto_save", "on") == "on" else ""
    insecure_ssl_checked = "checked" if form.get("insecure_ssl", "") == "on" else ""
    info_html = ""
    if ctx is not None and ctx.saved_dir:
        info_html += f'<div class="notice ok">결과를 <code>{html.escape(ctx.saved_dir)}</code>에 저장했습니다.</div>'
    if error:
        info_html += f'<div class="notice err"><pre>{html.escape(error)}</pre></div>'
    if ticker_note:
        css = "err" if ticker_note_error else "ok"
        info_html += f'<div class="notice {css}"><pre>{html.escape(ticker_note)}</pre></div>'
    if ctx is None and not error:
        info_html += (
            "<div class=\"notice ok\">"
            "재무제표·밸류에이션 페이지는 다른 탭과 별도로 실행됩니다. "
            "티커를 입력한 뒤 실행 버튼을 누르면 최신 재무 스냅샷을 불러옵니다."
            "</div>"
        )

    metric_html = ""
    tables_html = ""
    if ctx is not None:
        metric_html = f"""
        <div class="metrics">
          <div class="metric"><span>티커</span><strong>{html.escape(ctx.ticker)}</strong></div>
          <div class="metric"><span>회사명</span><strong>{html.escape(str(ctx.company_name or '-'))}</strong></div>
          <div class="metric"><span>통화</span><strong>{html.escape(str(ctx.currency or '-'))}</strong></div>
          <div class="metric"><span>PER (Trailing)</span><strong>{_format_fin_value(ctx.metrics.get("PER (Trailing)"))}</strong></div>
          <div class="metric"><span>PER (Forward)</span><strong>{_format_fin_value(ctx.metrics.get("PER (Forward)"))}</strong></div>
          <div class="metric"><span>PBR</span><strong>{_format_fin_value(ctx.metrics.get("PBR"))}</strong></div>
          <div class="metric"><span>시가총액</span><strong>{_format_fin_value(ctx.metrics.get("Market Cap"))}</strong></div>
          <div class="metric"><span>ROE</span><strong>{_format_fin_value(ctx.metrics.get("ROE"))}</strong></div>
        </div>
        """

        tables_html = f"""
        <div class="tables">
          <div class="card"><h3>데이터 소스 메타데이터</h3>{_safe_table(ctx.source_table)}</div>
          <div class="card"><h3>제공자 상태</h3>{_safe_table(ctx.provider_status_table)}</div>
          <div class="card"><h3>재무 지표</h3>{_safe_table(_metrics_table(ctx.metrics))}</div>
          <div class="card"><h3>최신 재무 요약</h3>{_safe_table(ctx.summary_table)}</div>
          <div class="card"><h3>손익계산서(최근 기간)</h3>{_safe_table(ctx.income_table)}</div>
          <div class="card"><h3>재무상태표(최근 기간)</h3>{_safe_table(ctx.balance_table)}</div>
          <div class="card"><h3>현금흐름표(최근 기간)</h3>{_safe_table(ctx.cashflow_table)}</div>
        </div>
        """

    head_content = ""
    if not is_sub_page:
        head_content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Stock Analysis Lab | S&P 500 - 재무제표·밸류에이션</title>
  <style>{_base_css()}</style>
</head>
<body>"""

    body_content = f"""
  <div class="wrap">
    {_page_head(_APP_TITLE)}
    <div class="sub">티커 기준 재무제표와 밸류에이션 지표(PER/PBR)를 한 번에 확인하는 페이지</div>
    {_nav("page2", enable_technical_page=enable_technical_page)}
    <form class="card" method="post" action="/run_financial">
      <div class="form-grid">
        <div><label>티커</label><input type="text" name="ticker" value="{html.escape(defaults["ticker"])}" /></div>
        <div><label>조회 기간 수</label><input type="number" min="1" name="statement_periods" value="{html.escape(defaults["statement_periods"])}" /></div>
        <div><label>출력 기본 폴더</label><input type="text" name="output_dir" value="{html.escape(defaults["output_dir"])}" /></div>
        <div><label>CA 번들 경로(선택)</label><input type="text" name="ca_bundle_path" value="{html.escape(defaults["ca_bundle_path"])}" /></div>
        <div><label>FMP API 키(선택)</label><input type="text" name="fmp_api_key" value="{html.escape(defaults["fmp_api_key"])}" /></div>
      </div>
      <div class="row">
        <label><input type="checkbox" name="auto_save" {auto_save_checked} /> 결과 자동 저장</label>
        <label><input type="checkbox" name="insecure_ssl" {insecure_ssl_checked} /> 임시로 SSL 검증 완화</label>
        <button type="submit" name="intent" value="run">재무 스냅샷 실행</button>
        <button type="submit" name="intent" value="resolve_ticker">회사명으로 티커 찾기</button>
      </div>
    </form>
    {info_html}
    {metric_html}
    {tables_html}
  </div>
"""

    if is_sub_page:
        return body_content
    return head_content + body_content + "</body>\n</html>\n"


def _normalize_technical_action(action: str) -> str:
    act = str(action or "all").strip().lower()
    return act if act in {"ma", "candle", "rsi", "macd", "all"} else "all"


def _html_technical_page(
    form: dict[str, str],
    *,
    ctx: ta_web_gui._RunContext | None,
    error: str | None,
    ticker_note: str | None = None,
    is_sub_page: bool = False,
    ticker_note_error: bool = False,
) -> str:
    defaults = {
        "ticker": form.get("ticker", ""),
        "output_dir": form.get("output_dir", "outputs/technical_analysis"),
        "action": _normalize_technical_action(form.get("action", "all")),
    }
    use_sample_checked = "checked" if form.get("use_sample", "") == "on" else ""
    auto_save_checked = "checked" if form.get("auto_save", "on") == "on" else ""

    info_html = ""
    if ctx is not None and ctx.saved_dir:
        info_html += f'<div class="notice ok">기술적 분석 결과를 <code>{html.escape(ctx.saved_dir)}</code>에 저장했습니다.</div>'
    if error:
        info_html += f'<div class="notice err"><pre>{html.escape(error)}</pre></div>'
    if ticker_note:
        css = "err" if ticker_note_error else "ok"
        info_html += f'<div class="notice {css}"><pre>{html.escape(ticker_note)}</pre></div>'
    if ctx is None and not error:
        info_html += (
            "<div class=\"notice ok\">"
            "기술적 분석 페이지는 선택한 티커의 최신 OHLCV 데이터를 불러와 "
            "차트별 진단 결과를 보여줍니다. 오프라인 미리보기가 필요할 때만 샘플 데이터를 사용해 주세요."
            "</div>"
        )

    metric_html = ""
    table_html = ""
    charts_html = ""
    if ctx is not None:
        metric_html = f"""
        <div class="metrics">
          <div class="metric"><span>티커</span><strong>{html.escape(ctx.ticker)}</strong></div>
          <div class="metric"><span>데이터 소스</span><strong>{html.escape(ctx.source)}</strong></div>
          <div class="metric"><span>행 수</span><strong>{ctx.rows:,d}</strong></div>
          <div class="metric"><span>기간</span><strong>{html.escape(ctx.first_date)} ~ {html.escape(ctx.last_date)}</strong></div>
          <div class="metric"><span>실행 항목</span><strong>{html.escape(ctx.action)}</strong></div>
          <div class="metric"><span>룩백 목표</span><strong>{ta_web_gui.LOOKBACK_ROWS:,d}</strong></div>
        </div>
        """

        cards = []
        for chart in ctx.charts:
            cards.append(
                f"<div class=\"chart-card\"><h4>{html.escape(chart.title)}</h4>"
                f"<img src=\"data:image/png;base64,{chart.image_base64}\" alt=\"{html.escape(chart.title)}\" />"
                f"<p class=\"chart-desc\">{html.escape(chart.description)}</p></div>"
            )
        charts_html = f"<div class=\"chart-grid\">{''.join(cards)}</div>" if cards else ""

        table_html = f"""
        <div class="tables">
          <div class="card"><h3>데이터 소스 메타데이터</h3>{_safe_table(ctx.source_table)}</div>
          <div class="card"><h3>실행 요약</h3>{_safe_table(ctx.summary_table)}</div>
        </div>
        """

    head_content = ""
    if not is_sub_page:
        head_content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Stock Analysis Lab | S&P 500 - 기술적 분석</title>
  <style>{_base_css()}</style>
</head>
<body>"""

    body_content = f"""
  <div class="wrap">
    {_page_head(_APP_TITLE)}
    <div class="sub">이동평균선, 캔들, RSI, MACD 차트로 기술적 흐름을 점검하는 페이지</div>
    {_nav("page3", enable_technical_page=True)}
    <form class="card" method="post" action="/run_technical">
      <div class="form-grid">
        <div><label>티커</label><input type="text" name="ticker" value="{html.escape(defaults["ticker"])}" /></div>
        <div><label>출력 기본 폴더</label><input type="text" name="output_dir" value="{html.escape(defaults["output_dir"])}" /></div>
      </div>
      <div class="row">
        <label><input type="checkbox" name="use_sample" {use_sample_checked} /> 샘플 가격 데이터 사용(오프라인)</label>
        <label><input type="checkbox" name="auto_save" {auto_save_checked} /> CSV/차트 자동 저장</label>
      </div>
      <div class="row">
        <button type="submit" name="action" value="ma">이동평균선</button>
        <button type="submit" name="action" value="candle">캔들차트</button>
        <button type="submit" name="action" value="rsi">RSI</button>
        <button type="submit" name="action" value="macd">MACD</button>
        <button type="submit" name="action" value="all">전체 실행</button>
        <button type="submit" name="intent" value="resolve_ticker">회사명으로 티커 찾기</button>
      </div>
    </form>
    {info_html}
    {metric_html}
    {table_html}
    {charts_html}
  </div>
"""

    if is_sub_page:
        return body_content
    return head_content + body_content + "</body>\n</html>\n"


def _html_returns_page(
    form: dict[str, str],
    *,
    ctx: _ReturnsContext | None,
    error: str | None,
    ticker_note: str | None = None,
    is_sub_page: bool = False,
    ticker_note_error: bool = False,
) -> str:
    defaults = {
        "ticker": form.get("ticker", ""),
    }

    info_html = ""
    if error:
        info_html += f'<div class="notice err"><pre>{html.escape(error)}</pre></div>'
    if ticker_note:
        css = "err" if ticker_note_error else "ok"
        info_html += f'<div class="notice {css}"><pre>{html.escape(ticker_note)}</pre></div>'
    if ctx is None and not error:
        info_html += (
            "<div class=\"notice ok\">"
            "수익률 비교 페이지는 공유 S&P 500 SQLite/캐시 데이터를 바탕으로 "
            "선택 종목을 섹터와 전체 지수에 비교합니다. 현재 실험실에서 선택한 티커를 그대로 재사용해도 됩니다."
            "</div>"
        )

    metric_html = ""
    charts_html = ""
    tables_html = ""
    ranking_html = ""
    if ctx is not None:
        sector_rank_text = "-" if ctx.sector_rank_ytd is None else f"{ctx.sector_rank_ytd:,d} / {ctx.sector_count:,d}"
        market_rank_text = "-" if ctx.market_rank_ytd is None else f"{ctx.market_rank_ytd:,d} / {ctx.market_count:,d}"
        metric_html = f"""
        <div class="metrics">
          <div class="metric"><span>Ticker</span><strong>{html.escape(ctx.ticker)}</strong></div>
          <div class="metric"><span>Sector</span><strong>{html.escape(ctx.sector)}</strong></div>
          <div class="metric"><span>Market Latest Date</span><strong>{html.escape(ctx.latest_market_date)}</strong></div>
          <div class="metric"><span>Ticker Latest Date</span><strong>{html.escape(ctx.ticker_latest_date)}</strong></div>
          <div class="metric"><span>3Y Return</span><strong>{_format_pct(ctx.period_returns.get("3Y"))}</strong></div>
          <div class="metric"><span>1Y Return</span><strong>{_format_pct(ctx.period_returns.get("1Y"))}</strong></div>
          <div class="metric"><span>6M Return</span><strong>{_format_pct(ctx.period_returns.get("6M"))}</strong></div>
          <div class="metric"><span>1M Return</span><strong>{_format_pct(ctx.period_returns.get("1M"))}</strong></div>
          <div class="metric"><span>YTD Return</span><strong>{_format_pct(ctx.period_returns.get("YTD"))}</strong></div>
          <div class="metric"><span>MTD Return</span><strong>{_format_pct(ctx.period_returns.get("MTD"))}</strong></div>
          <div class="metric"><span>Sector YTD Rank</span><strong>{html.escape(sector_rank_text)}</strong></div>
          <div class="metric"><span>S&P 500 YTD Rank</span><strong>{html.escape(market_rank_text)}</strong></div>
        </div>
        """

        charts_html = f"""
        <div class="charts">
          <div class="card"><h3>YTD 기준 100 지수</h3><img src="data:image/png;base64,{ctx.relative_chart_base64}" alt="ytd base 100 index chart" /></div>
          <div class="card"><h3>최근 일간 수익률 비교</h3><img src="data:image/png;base64,{ctx.daily_chart_base64}" alt="daily return comparison chart" /></div>
        </div>
        """

        tables_html = f"""
        <div class="tables">
          <div class="card"><h3>기간별 수익률 비교</h3>{_safe_table(ctx.summary_table)}</div>
          <div class="table-grid">
            <div class="card"><h3>최근 10영업일 일간 수익률 ({html.escape(ctx.ticker)})</h3>{_safe_table(ctx.daily_returns_table)}</div>
            <div class="card"><h3>최근 10영업일 일간 수익률 ({html.escape(ctx.sector)} 섹터)</h3>{_safe_table(ctx.sector_daily_returns_table)}</div>
          </div>
          <div class="card"><h3>데이터 소스 메타데이터</h3>{_safe_table(ctx.source_table)}</div>
        </div>
        """

        ranking_html = f"""
        <div class="table-grid">
          <div class="card"><h3>섹터 YTD 상위 10개</h3>{_safe_table(ctx.sector_top_table)}</div>
          <div class="card"><h3>섹터 YTD 하위 10개</h3>{_safe_table(ctx.sector_bottom_table)}</div>
          <div class="card"><h3>S&amp;P 500 YTD 상위 10개</h3>{_safe_table(ctx.market_top_table)}</div>
          <div class="card"><h3>S&amp;P 500 YTD 하위 10개</h3>{_safe_table(ctx.market_bottom_table)}</div>
        </div>
        """

    head_content = ""
    if not is_sub_page:
        head_content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Stock Analysis Lab | S&P 500 - 수익률 비교</title>
  <style>{_base_css()}</style>
</head>
<body>"""

    body_content = f"""
  <div class="wrap">
    {_page_head(_APP_TITLE)}
    <div class="sub">공유 S&P 500 데이터로 종목 수익률과 섹터·지수 상대강도를 비교하는 페이지</div>
    {_nav("page4", enable_technical_page=True)}
    <form class="card" method="post" action="/run_returns">
      <div class="form-grid">
        <div><label>Ticker</label><input type="text" name="ticker" value="{html.escape(defaults["ticker"])}" /></div>
      </div>
      <div class="row">
        <button type="submit" name="intent" value="run">수익률 비교 실행</button>
        <button type="submit" name="intent" value="resolve_ticker">회사명으로 티커 찾기</button>
      </div>
    </form>
    {info_html}
    {metric_html}
    {charts_html}
    {tables_html}
    {ranking_html}
  </div>"""

    if is_sub_page:
        return body_content
    return head_content + body_content + "</body>\n</html>\n"


def _html_risk_page(
    form: dict[str, str],
    *,
    ctx: _RiskContext | None,
    error: str | None,
    ticker_note: str | None = None,
    is_sub_page: bool = False,
    ticker_note_error: bool = False,
) -> str:
    defaults = {
        "ticker": form.get("ticker", ""),
    }

    info_html = ""
    if error:
        info_html += f'<div class="notice err"><pre>{html.escape(error)}</pre></div>'
    if ticker_note:
        css = "err" if ticker_note_error else "ok"
        info_html += f'<div class="notice {css}"><pre>{html.escape(ticker_note)}</pre></div>'
    if ctx is None and not error:
        info_html += (
            "<div class=\"notice ok\">"
            "리스크 대시보드는 공유 S&P 500 SQLite 데이터를 이용해 선택 종목의 "
            "변동성, 낙폭, 시장 민감도를 측정합니다. 현재 실험실에서 선택한 티커를 그대로 재사용해도 됩니다."
            "</div>"
        )

    metric_html = ""
    commentary_html = ""
    charts_html = ""
    tables_html = ""
    ranking_html = ""
    if ctx is not None:
        sector_rank_text = "-" if ctx.sector_vol_rank_1y is None else f"{ctx.sector_vol_rank_1y:,d} / {ctx.sector_count:,d}"
        market_rank_text = "-" if ctx.market_vol_rank_1y is None else f"{ctx.market_vol_rank_1y:,d} / {ctx.market_count:,d}"
        metric_html = f"""
        <div class="metrics">
          <div class="metric"><span>Ticker</span><strong>{html.escape(ctx.ticker)}</strong></div>
          <div class="metric"><span>Sector</span><strong>{html.escape(ctx.sector)}</strong></div>
          <div class="metric"><span>Market Latest Date</span><strong>{html.escape(ctx.latest_market_date)}</strong></div>
          <div class="metric"><span>Ticker Latest Date</span><strong>{html.escape(ctx.ticker_latest_date)}</strong></div>
          <div class="metric"><span>20D Ann Vol</span><strong>{_format_pct(ctx.ticker_vol_20d)}</strong></div>
          <div class="metric"><span>60D Ann Vol</span><strong>{_format_pct(ctx.ticker_vol_60d)}</strong></div>
          <div class="metric"><span>1Y Ann Vol</span><strong>{_format_pct(ctx.ticker_vol_252d)}</strong></div>
          <div class="metric"><span>1Y Max Drawdown</span><strong>{_format_pct(ctx.ticker_max_drawdown_1y)}</strong></div>
          <div class="metric"><span>3Y Max Drawdown</span><strong>{_format_pct(ctx.ticker_max_drawdown_3y)}</strong></div>
          <div class="metric"><span>Beta vs Sector</span><strong>{_format_metric(ctx.beta_sector_1y, 2) if ctx.beta_sector_1y is not None else "-"}</strong></div>
          <div class="metric"><span>Beta vs S&P 500</span><strong>{_format_metric(ctx.beta_market_1y, 2) if ctx.beta_market_1y is not None else "-"}</strong></div>
          <div class="metric"><span>95% 1D VaR</span><strong>{_format_pct(ctx.var_95_1d)}</strong></div>
          <div class="metric"><span>95% 1D CVaR</span><strong>{_format_pct(ctx.cvar_95_1d)}</strong></div>
          <div class="metric"><span>Sector 1Y Vol Rank</span><strong>{html.escape(sector_rank_text)}</strong></div>
          <div class="metric"><span>S&P 500 1Y Vol Rank</span><strong>{html.escape(market_rank_text)}</strong></div>
        </div>
        """
        commentary_html = f"""
        <div class="card">
          <h3>리스크 해석</h3>
          <p>{html.escape(ctx.commentary)}</p>
        </div>
        """
        charts_html = f"""
        <div class="charts">
          <div class="card"><h3>1년 낙폭 비교</h3><img src="data:image/png;base64,{ctx.drawdown_chart_base64}" alt="drawdown comparison chart" /></div>
          <div class="card"><h3>20일 롤링 연율화 변동성</h3><img src="data:image/png;base64,{ctx.volatility_chart_base64}" alt="rolling volatility chart" /></div>
        </div>
        """
        tables_html = f"""
        <div class="tables">
          <div class="card"><h3>변동성·낙폭 요약</h3>{_safe_table(ctx.summary_table)}</div>
          <div class="card"><h3>최근 20영업일 충격 점검</h3>{_safe_table(ctx.recent_shock_table)}</div>
          <div class="card"><h3>데이터 소스 메타데이터</h3>{_safe_table(ctx.source_table)}</div>
        </div>
        """
        ranking_html = f"""
        <div class="table-grid">
          <div class="card"><h3>섹터 내 1년 변동성 상위</h3>{_safe_table(ctx.sector_high_vol_table)}</div>
          <div class="card"><h3>섹터 내 1년 변동성 하위</h3>{_safe_table(ctx.sector_low_vol_table)}</div>
          <div class="card"><h3>S&amp;P 500 1년 변동성 상위</h3>{_safe_table(ctx.market_high_vol_table)}</div>
          <div class="card"><h3>S&amp;P 500 1년 변동성 하위</h3>{_safe_table(ctx.market_low_vol_table)}</div>
        </div>
        """

    head_content = ""
    if not is_sub_page:
        head_content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Stock Analysis Lab | S&P 500 - 리스크 대시보드</title>
  <style>{_base_css()}</style>
</head>
<body>"""

    body_content = f"""
  <div class="wrap">
    {_page_head(_APP_TITLE)}
    <div class="sub">공유 S&P 500 데이터로 변동성, 낙폭, 베타를 점검하는 리스크 페이지</div>
    {_nav("page5", enable_technical_page=True)}
    <form class="card" method="post" action="/run_risk">
      <div class="form-grid">
        <div><label>티커</label><input type="text" name="ticker" value="{html.escape(defaults["ticker"])}" /></div>
      </div>
      <div class="row">
        <button type="submit" name="intent" value="run">리스크 대시보드 실행</button>
        <button type="submit" name="intent" value="resolve_ticker">회사명으로 티커 찾기</button>
      </div>
    </form>
    {info_html}
    {metric_html}
    {commentary_html}
    {charts_html}
    {tables_html}
    {ranking_html}
  </div>"""

    if is_sub_page:
        return body_content
    return head_content + body_content + "</body>\n</html>\n"


def _html_factor_page(
    form: dict[str, str],
    *,
    ctx: _FactorContext | None,
    error: str | None,
    ticker_note: str | None = None,
    is_sub_page: bool = False,
    ticker_note_error: bool = False,
) -> str:
    defaults = {
        "ticker": form.get("ticker", ""),
    }

    info_html = ""
    if error:
        info_html += f'<div class="notice err"><pre>{html.escape(error)}</pre></div>'
    if ticker_note:
        css = "err" if ticker_note_error else "ok"
        info_html += f'<div class="notice {css}"><pre>{html.escape(ticker_note)}</pre></div>'
    if ctx is None and not error:
        info_html += (
            "<div class=\"notice ok\">"
            "팩터·레짐 랩은 선택 종목을 시장(S&P 500)과 섹터 요인으로 분해해, "
            "최근 60거래일의 민감도와 최근 20거래일의 잔차성과를 실험적으로 읽는 페이지입니다. "
            "리스크 대시보드 다음 단계에서 현재 주가 움직임이 시장 영향인지, 섹터 영향인지, 혹은 종목 고유 흐름인지 가늠할 때 유용합니다."
            "</div>"
        )

    metric_html = ""
    explanation_html = """
        <div class="card">
          <h3>팩터·레짐 랩 읽는 법</h3>
          <p>이 페이지는 종목의 일별 수익률을 시장(S&amp;P 500)과 섹터 수익률에 비교해 <b>베타(beta)</b>, <b>상관계수(correlation)</b>, <b>잔차수익률(residual return)</b>, 그리고 현재 <b>국면(regime)</b>을 함께 읽도록 만든 실험적 페이지입니다.</p>
          <ul>
            <li><b>베타 (beta)</b>는 시장 또는 섹터가 1 움직일 때 종목이 평균적으로 얼마나 크게 반응했는지 보여줍니다.</li>
            <li><b>상관계수 (correlation)</b>는 함께 움직이는 강도를, <b>잔차수익률 (residual return)</b>은 공통 요인을 제거한 뒤 남는 종목 고유 성과를 의미합니다.</li>
            <li><b>추세 국면 (trend regime)</b>, <b>변동성 국면 (volatility regime)</b>, <b>베타 국면 (beta regime)</b>을 합쳐 현재 환경을 한 줄로 요약합니다.</li>
          </ul>
        </div>
    """
    charts_html = ""
    tables_html = ""
    if ctx is not None:
        metric_html = f"""
        <div class="metrics">
          <div class="metric"><span>Ticker</span><strong>{html.escape(ctx.ticker)}</strong></div>
          <div class="metric"><span>Sector</span><strong>{html.escape(ctx.sector)}</strong></div>
          <div class="metric"><span>Market Latest Date</span><strong>{html.escape(ctx.latest_market_date)}</strong></div>
          <div class="metric"><span>Ticker Latest Date</span><strong>{html.escape(ctx.ticker_latest_date)}</strong></div>
          <div class="metric"><span>60D Beta vs S&amp;P 500</span><strong>{_format_metric(ctx.beta_market_60d, 2) if ctx.beta_market_60d is not None else "-"}</strong></div>
          <div class="metric"><span>60D Beta vs Sector</span><strong>{_format_metric(ctx.beta_sector_60d, 2) if ctx.beta_sector_60d is not None else "-"}</strong></div>
          <div class="metric"><span>60D Corr vs S&amp;P 500</span><strong>{_format_metric(ctx.corr_market_60d, 2) if ctx.corr_market_60d is not None else "-"}</strong></div>
          <div class="metric"><span>60D Corr vs Sector</span><strong>{_format_metric(ctx.corr_sector_60d, 2) if ctx.corr_sector_60d is not None else "-"}</strong></div>
          <div class="metric"><span>20D Residual vs S&amp;P 500</span><strong>{_format_pct(ctx.residual_market_20d)}</strong></div>
          <div class="metric"><span>20D Residual vs Sector</span><strong>{_format_pct(ctx.residual_sector_20d)}</strong></div>
          <div class="metric"><span>Trend Regime</span><strong>{html.escape(ctx.regime_trend)}</strong></div>
          <div class="metric"><span>Volatility Regime</span><strong>{html.escape(ctx.regime_volatility)}</strong></div>
          <div class="metric"><span>Beta Regime</span><strong>{html.escape(ctx.regime_beta)}</strong></div>
          <div class="metric"><span>Overall Regime</span><strong>{html.escape(ctx.regime_overall)}</strong></div>
        </div>
        """
        explanation_html = f"""
        <div class="card">
          <h3>팩터·레짐 랩 읽는 법</h3>
          <p>이 페이지는 종목의 일별 수익률을 시장(S&amp;P 500)과 섹터 수익률에 비교해 <b>베타(beta)</b>, <b>상관계수(correlation)</b>, <b>잔차수익률(residual return)</b>, 그리고 현재 <b>국면(regime)</b>을 함께 읽도록 만든 실험적 페이지입니다.</p>
          <ul>
            <li><b>베타 (beta)</b>는 시장 또는 섹터가 1 움직일 때 종목이 평균적으로 얼마나 크게 반응했는지 보여줍니다.</li>
            <li><b>상관계수 (correlation)</b>는 함께 움직이는 강도를, <b>잔차수익률 (residual return)</b>은 공통 요인을 제거한 뒤 남는 종목 고유 성과를 의미합니다.</li>
            <li><b>추세 국면 (trend regime)</b>, <b>변동성 국면 (volatility regime)</b>, <b>베타 국면 (beta regime)</b>을 합쳐 현재 환경을 한 줄로 요약합니다.</li>
          </ul>
          <p>{html.escape(ctx.commentary)}</p>
        </div>
        """
        charts_html = f"""
        <div class="charts">
          <div class="card"><h3>Rolling 60-Day Beta</h3><img src="data:image/png;base64,{ctx.beta_chart_base64}" alt="rolling beta chart" /></div>
          <div class="card"><h3>Cumulative Residual Return</h3><img src="data:image/png;base64,{ctx.residual_chart_base64}" alt="residual return chart" /></div>
        </div>
        """
        tables_html = f"""
        <div class="tables">
          <div class="card"><h3>팩터 요약</h3>{_safe_table(ctx.summary_table)}</div>
          <div class="card"><h3>해석 가이드</h3>{_safe_table(ctx.interpretation_table)}</div>
          <div class="card"><h3>최근 20거래일 분해표</h3>{_safe_table(ctx.recent_factor_table, max_rows=120)}</div>
          <div class="card"><h3>데이터 소스 메타데이터</h3>{_safe_table(ctx.source_table)}</div>
        </div>
        """

    head_content = ""
    if not is_sub_page:
        head_content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Stock Analysis Lab | S&P 500 - 팩터·레짐 랩</title>
  <style>{_base_css()}</style>
</head>
<body>"""

    body_content = f"""
  <div class="wrap">
    {_page_head(_APP_TITLE)}
    <div class="sub">시장과 섹터 요인 분해를 통해 종목의 현재 국면을 읽는 실험용 팩터·레짐 페이지</div>
    {_nav("factor", enable_technical_page=True)}
    <form class="card" method="post" action="/run_factor">
      <div class="form-grid">
        <div><label>티커</label><input type="text" name="ticker" value="{html.escape(defaults["ticker"])}" /></div>
      </div>
      <div class="row">
        <button type="submit" name="intent" value="run">팩터·레짐 랩 실행</button>
        <button type="submit" name="intent" value="resolve_ticker">회사명으로 티커 찾기</button>
      </div>
    </form>
    {info_html}
    {metric_html}
    {explanation_html}
    {charts_html}
    {tables_html}
  </div>"""

    if is_sub_page:
        return body_content
    return head_content + body_content + "</body>\n</html>\n"


def _html_decision_page(
    form: dict[str, str],
    *,
    ctx: _DecisionContext | None,
    error: str | None,
    ticker_note: str | None = None,
    is_sub_page: bool = False,
    ticker_note_error: bool = False,
) -> str:
    defaults = {
        "ticker": form.get("ticker", ""),
    }

    info_html = ""
    if error:
        info_html += f'<div class="notice err"><pre>{html.escape(error)}</pre></div>'
    if ticker_note:
        css = "err" if ticker_note_error else "ok"
        info_html += f'<div class="notice {css}"><pre>{html.escape(ticker_note)}</pre></div>'
    if ctx is None and not error:
        info_html += (
            "<div class=\"notice ok\">"
            "의사결정 대시보드는 추세, 모멘텀, 상대강도, 리스크, 선택적인 재무 지표를 함께 묶어 "
            "종합 판단을 돕는 페이지입니다. 현재 실험실에서 선택한 티커를 그대로 재사용해도 됩니다."
            "</div>"
        )

    metric_html = ""
    commentary_html = ""
    charts_html = ""
    tables_html = ""
    reason_html = ""
    if ctx is not None:
        metric_html = f"""
        <div class="metrics">
          <div class="metric"><span>Ticker</span><strong>{html.escape(ctx.ticker)}</strong></div>
          <div class="metric"><span>Sector</span><strong>{html.escape(ctx.sector)}</strong></div>
          <div class="metric"><span>Latest Market Date</span><strong>{html.escape(ctx.latest_market_date)}</strong></div>
          <div class="metric"><span>Recommendation</span><strong>{html.escape(ctx.recommendation)}</strong></div>
          <div class="metric"><span>Confidence</span><strong>{html.escape(ctx.confidence_label)}</strong></div>
          <div class="metric"><span>Total Score</span><strong>{ctx.total_score:+.2f}</strong></div>
        </div>
        """
        commentary_html = f"""
        <div class="card">
          <h3>최종 판단</h3>
          <p>{html.escape(ctx.final_commentary)}</p>
        </div>
        """
        charts_html = f"""
        <div class="charts">
          <div class="card"><h3>의사결정 점수 분해</h3><img src="data:image/png;base64,{ctx.score_chart_base64}" alt="decision score chart" /></div>
          <div class="card"><h3>추세·볼린저 맥락</h3><img src="data:image/png;base64,{ctx.trend_chart_base64}" alt="trend context chart" /></div>
        </div>
        """
        reason_html = f"""
        <div class="table-grid">
          <div class="card"><h3>사는 쪽 근거</h3>{_html_reason_list(ctx.bullish_reasons)}</div>
          <div class="card"><h3>파는 쪽 근거</h3>{_html_reason_list(ctx.bearish_reasons)}</div>
          <div class="card"><h3>추가 확인 포인트</h3>{_html_reason_list(ctx.watch_items)}</div>
        </div>
        """
        tables_html = f"""
        <div class="tables">
          <div class="card"><h3>점수표</h3>{_safe_table(ctx.score_table)}</div>
          <div class="card"><h3>신호 스냅샷</h3>{_safe_table(ctx.signal_table)}</div>
          <div class="card"><h3>데이터 소스 메타데이터</h3>{_safe_table(ctx.source_table)}</div>
        </div>
        """

    head_content = ""
    if not is_sub_page:
        head_content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Stock Analysis Lab | S&P 500 - 의사결정 대시보드</title>
  <style>{_base_css()}</style>
</head>
<body>"""

    body_content = f"""
  <div class="wrap">
    {_page_head(_APP_TITLE)}
    <div class="sub">추세, 모멘텀, 상대강도, 리스크, 밸류에이션을 함께 읽는 종합 판단 페이지</div>
    {_nav("page6", enable_technical_page=True)}
    <form class="card" method="post" action="/run_decision">
      <div class="form-grid">
        <div><label>티커</label><input type="text" name="ticker" value="{html.escape(defaults["ticker"])}" /></div>
      </div>
      <div class="row">
        <button type="submit" name="intent" value="run">의사결정 대시보드 실행</button>
        <button type="submit" name="intent" value="resolve_ticker">회사명으로 티커 찾기</button>
      </div>
    </form>
    {info_html}
    {metric_html}
    {commentary_html}
    {charts_html}
    {reason_html}
    {tables_html}
  </div>
"""

    if is_sub_page:
        return body_content
    return head_content + body_content + "</body>\n</html>\n"


def _latest_date_from_csv(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        raw = pd.read_csv(path, usecols=[0])
    except Exception:
        return None
    if raw.empty:
        return None
    dates = pd.to_datetime(raw.iloc[:, 0], errors="coerce").dropna()
    if dates.empty:
        return None
    return pd.Timestamp(dates.max()).normalize().strftime("%Y-%m-%d")


def _project_root_dir() -> Path:
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


def _refresh_subprocess_command(root_dir: Path) -> tuple[list[str], str]:
    if getattr(sys, "frozen", False):
        exe_path = Path(sys.executable).resolve()
        return [str(exe_path), "refresh"], f"{exe_path} refresh"

    batch_path = root_dir / "refresh_stock_data.bat"
    if not batch_path.exists() or not batch_path.is_file():
        return [], f"refresh_stock_data.bat not found ({batch_path})"
    return ["cmd.exe", "/c", str(batch_path)], str(batch_path)


def _sp500_sqlite_max_date(sqlite_path: Path) -> str | None:
    if not sqlite_path.exists() or not sqlite_path.is_file():
        return None
    try:
        with sqlite3.connect(sqlite_path) as conn:
            row = conn.execute("SELECT MAX(date) FROM prices").fetchone()
    except Exception:
        return None
    if not row or row[0] is None:
        return None
    return str(row[0])


def _csv_row_count(path: Path) -> int:
    if not path.exists() or not path.is_file():
        return 0
    try:
        raw = pd.read_csv(path)
    except Exception:
        return 0
    return int(len(raw))


def _collect_post_refresh_items(root_dir: Path) -> list[dict[str, object]]:
    data_dir = root_dir / "data"
    shared_db_root = data_dir / "sp500_shared_db"
    sp500_sqlite_path = shared_db_root / "sp500_shared_prices.sqlite"

    items: list[dict[str, object]] = []
    metrics_csv = data_dir / "sp500_all_metrics_prices.csv"
    mcap_csv = data_dir / "sp500_market_caps.csv"
    items.extend(
        [
            {
                "dataset": "sp500_all_metrics_prices",
                "latest_date": _latest_date_from_csv(metrics_csv),
                "rows": _csv_row_count(metrics_csv),
                "source": "refresh_stock_data.bat",
                "path": str(metrics_csv.resolve()),
            },
            {
                "dataset": "sp500_market_caps",
                "latest_date": _latest_date_from_csv(mcap_csv),
                "rows": _csv_row_count(mcap_csv),
                "source": "refresh_stock_data.bat",
                "path": str(mcap_csv.resolve()),
            },
            {
                "dataset": "sp500_shared_prices_sqlite",
                "latest_date": _sp500_sqlite_max_date(sp500_sqlite_path),
                "rows": 0,
                "source": "refresh_stock_data.bat",
                "path": str(sp500_sqlite_path.resolve()),
            },
        ]
    )
    return items


def _html_refresh_page(*, enable_technical_page: bool = True, is_sub_page: bool = False) -> str:
    return f"""<!doctype html>
<html lang="en">
<head> # Removed this section as it's now handled by the main GUI
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{_APP_TITLE} - 데이터 갱신</title>
  <style>
    {_base_css()}
    .small {{ font-size: 12px; color: var(--muted); }}
    .split-grid {{ margin-top: 12px; display: grid; grid-template-columns: repeat(2, minmax(360px, 1fr)); gap: 10px; }}
    .pane {{ background: #fff; border: 1px solid var(--line); border-radius: 10px; padding: 12px; }}
    .pane h3 {{ margin: 0 0 8px 0; }}
    .line-list {{ height: 480px; overflow: auto; border: 1px solid var(--line); border-radius: 8px; background: #fff; padding: 8px; }}
    .line {{ font-family: Consolas, "Courier New", monospace; font-size: 12px; line-height: 1.45; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; border-bottom: 1px solid #eef2f7; padding: 3px 2px; }}
    .line:last-child {{ border-bottom: 0; }}
    @media (max-width: 1180px) {{
      .split-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    {_page_head(_APP_TITLE)}
    <div class="sub">증분 다운로드와 SQLite 업데이트 진행 상황을 확인하는 페이지</div>
    {_nav("page7", enable_technical_page=enable_technical_page)}
    <form class="card" method="post" action="/run_refresh">
      <div class="row">
        <button type="submit">증분 갱신 시작</button>
      </div>
      <p id="refresh-meta" class="small">상태: 대기 / 실행 ID: - / 시작: - / 종료: -</p>
    </form>
    <div class="split-grid">
      <div class="pane">
        <h3>진행경과</h3>
        <div id="refresh-log" class="line-list"><div class="line">아직 로그가 없습니다.</div></div>
      </div>
      <div class="pane">
        <h3>업데이트목록</h3>
        <div id="refresh-updates" class="line-list"><div class="line">아직 업데이트 항목이 없습니다.</div></div>
      </div>
    </div>
  </div>
  <script>
    const metaEl = document.getElementById("refresh-meta");
    const latestEl = document.getElementById("refresh-latest");
    const logEl = document.getElementById("refresh-log");
    const updatesEl = document.getElementById("refresh-updates");
    let lastLogCount = -1;
    let lastUpdateKey = "";

    function esc(value) {{
      return String(value ?? "-")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }}

    async function pollStatus() {{
      try {{
        const res = await fetch("/refresh_status", {{ cache: "no-store" }});
        if (!res.ok) {{
          metaEl.textContent = "상태: 오류 (상태 조회 엔드포인트에 접근할 수 없습니다)";
          return;
        }}
        const data = await res.json();
        const statusText = data.status || "대기";
        const runId = data.run_id || "-";
        const started = data.started_at || "-";
        const finished = data.finished_at || "-";
        metaEl.textContent = "상태: " + statusText + " / 실행 ID: " + runId + " / 시작: " + started + " / 종료: " + finished;

        const logCount = Number(data.log_count || 0);
        if (logCount !== lastLogCount) {{
          lastLogCount = logCount;
          const logs = Array.isArray(data.logs) ? data.logs : [];
          if (logs.length === 0) {{
            logEl.innerHTML = "<div class='line'>아직 로그가 없습니다.</div>";
          }} else {{
            logEl.innerHTML = logs.map((line) => "<div class='line'>" + esc(line) + "</div>").join("");
          }}
          logEl.scrollTop = logEl.scrollHeight;
        }}

        const items = Array.isArray(data.updated_items) ? data.updated_items : [];
        const updateKey = JSON.stringify(items);
        if (updateKey !== lastUpdateKey) {{
          lastUpdateKey = updateKey;
          if (items.length === 0) {{
            updatesEl.innerHTML = "<div class='line'>아직 업데이트 항목이 없습니다.</div>";
          }} else {{
            updatesEl.innerHTML = items.map((item) => {{
              const line = (item.dataset || "-")
                + " | latest=" + (item.latest_date || "-")
                + " | rows=" + (item.rows ?? "-")
                + " | source=" + (item.source || "-");
              return "<div class='line'>" + esc(line) + "</div>";
            }}).join("");
          }}
        }}
      }} catch (err) {{
        metaEl.textContent = "상태: 오류 (" + String(err) + ")";
      }}
    }}

    pollStatus();
    setInterval(pollStatus, 2200);
  </script>
</body>
</html>
"""


def _html_refresh_history_page(is_sub_page: bool = False) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{_APP_TITLE} - 데이터 갱신 이력</title>
  <style>
    {_base_css()}
    .table-wrap {{ overflow: auto; max-height: 360px; border: 1px solid var(--line); border-radius: 10px; background: #fff; }}
    .caption {{ margin: 0 0 8px 0; color: var(--muted); font-size: 12px; }}
  </style>
</head>
<body>
  <div class="wrap">
    {_page_head("데이터 갱신 이력")}
    <div class="sub">실행 요약과 업데이트된 데이터 날짜를 확인하는 페이지</div>
    <div class="card">
      <h3>실행 요약</h3>
      <p id="history-generated" class="caption">생성 시각: -</p>
      <div class="table-wrap">
        <table class="data-table">
          <thead>
            <tr>
              <th>Run ID</th>
              <th>Status</th>
              <th>Started</th>
              <th>Finished</th>
              <th>SP500 Old Max Date</th>
              <th>SP500 New Max Date</th>
              <th>Financial SQLite Added</th>
              <th>SP500 SQLite Added</th>
              <th>SP500 MCap Updates</th>
              <th>Error</th>
            </tr>
          </thead>
          <tbody id="run-history-body"></tbody>
        </table>
      </div>
    </div>
    <div class="card">
      <h3>업데이트 항목(최신 날짜 포함)</h3>
      <div class="table-wrap">
        <table class="data-table">
          <thead>
            <tr>
              <th>Run ID</th>
              <th>Dataset</th>
              <th>Latest Date</th>
              <th>Rows</th>
              <th>Source</th>
              <th>Path</th>
            </tr>
          </thead>
          <tbody id="update-history-body"></tbody>
        </table>
      </div>
    </div>
  </div>
  <script>
    function esc(value) {{
      return String(value ?? "-")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }}

    async function pollHistory() {{
      try {{
        const res = await fetch("/refresh_history_data", {{ cache: "no-store" }});
        if (!res.ok) {{
          return;
        }}
        const data = await res.json();
        const runs = Array.isArray(data.runs) ? data.runs : [];

        document.getElementById("history-generated").textContent = "생성 시각: " + (data.generated_at || "-");

        const runRows = runs.map((run) => `
          <tr>
            <td>${{esc(run.run_id)}}</td>
            <td>${{esc(run.status)}}</td>
            <td>${{esc(run.started_at)}}</td>
            <td>${{esc(run.finished_at)}}</td>
            <td>${{esc(run.sp500_old_max_date)}}</td>
            <td>${{esc(run.sp500_new_max_date)}}</td>
            <td>${{esc(run.financial_sqlite_rows_added)}}</td>
            <td>${{esc(run.sp500_sqlite_added_rows)}}</td>
            <td>${{esc(run.sp500_sqlite_market_cap_updates)}}</td>
            <td>${{esc(run.error_message)}}</td>
          </tr>
        `).join("");
        document.getElementById("run-history-body").innerHTML = runRows || "<tr><td colspan='10'>아직 갱신 이력이 없습니다.</td></tr>";

        const updateRows = [];
        for (const run of runs) {{
          const items = Array.isArray(run.updated_items) ? run.updated_items : [];
          for (const item of items) {{
            updateRows.push(`
              <tr>
                <td>${{esc(run.run_id)}}</td>
                <td>${{esc(item.dataset)}}</td>
                <td>${{esc(item.latest_date)}}</td>
                <td>${{esc(item.rows)}}</td>
                <td>${{esc(item.source)}}</td>
                <td>${{esc(item.path)}}</td>
              </tr>
            `);
          }}
        }}
        document.getElementById("update-history-body").innerHTML = updateRows.join("") || "<tr><td colspan='6'>아직 업데이트된 데이터셋이 없습니다.</td></tr>";
      }} catch (err) {{
        return;
      }}
    }}

    pollHistory();
    setInterval(pollHistory, 2000);
  </script>
</body>
</html>
"""


def _browser_target_host(host: str) -> str:
    clean = str(host or "").strip()
    if clean in {"", "0.0.0.0", "::"}:
        return "localhost"
    return clean


def _schedule_browser_open(host: str, port: int) -> None:
    target_url = f"http://{_browser_target_host(host)}:{int(port)}"

    def _open() -> None:
        try:
            webbrowser.open(target_url, new=2)
        except Exception:
            return

    timer = threading.Timer(0.6, _open)
    timer.daemon = True
    timer.start()


def launch_stock_forecast_web_gui(
    host: str = "0.0.0.0",
    port: int = 8512,
    *,
    enable_technical_page: bool = False,
    open_browser: bool = False,
) -> None:
    class Handler(BaseHTTPRequestHandler):
        state_form: dict[str, str] = {
            "ticker": "",
            "forecast_horizon": "10",
            "history_years": "8",
            "start_date": "2025-12-31",
            "end_date": datetime.utcnow().strftime("%Y-%m-%d"),
            "output_dir": "outputs/stock_forecast",
            "prices_csv_path": "",
            "use_sample": "",
            "auto_save": "on",
            "insecure_ssl": "",
            "ca_bundle_path": "",
        }
        state_ctx: _RunContext | None = None
        state_error: str | None = None

        state_fin_form: dict[str, str] = {
            "ticker": "",
            "statement_periods": "4",
            "output_dir": "outputs/stock_forecast_finance",
            "auto_save": "on",
            "insecure_ssl": "",
            "ca_bundle_path": "",
            "fmp_api_key": "",
        }
        state_fin_ctx: _FinancialContext | None = None
        state_fin_error: str | None = None
        state_fin_cache: dict[str, _FinancialContext] = {}

        state_ticker_note: str | None = None
        state_ticker_note_error: bool = False

        state_ta_form: dict[str, str] = {
            "ticker": "",
            "output_dir": "outputs/technical_analysis",
            "use_sample": "",
            "auto_save": "on",
            "action": "all",
        }
        state_ta_ctx: ta_web_gui._RunContext | None = None
        state_ta_error: str | None = None
        state_ta_cache: ta_web_gui._CachedData | None = None

        state_ret_form: dict[str, str] = {
            "ticker": "",
        }
        state_ret_ctx: _ReturnsContext | None = None
        state_ret_error: str | None = None

        state_risk_form: dict[str, str] = {
            "ticker": "",
        }
        state_risk_ctx: _RiskContext | None = None
        state_risk_error: str | None = None

        state_dec_form: dict[str, str] = {
            "ticker": "",
        }
        state_dec_ctx: _DecisionContext | None = None
        state_dec_error: str | None = None

        state_wfv_form: dict[str, str] = {
            "ticker": "",
            "forecast_horizon": "10",
            "history_years": "8",
            "start_date": "2025-12-31",
            "end_date": datetime.utcnow().strftime("%Y-%m-%d"),
            "wf_min_train_rows": "252",
            "wf_step_size": "21",
            "wf_max_splits": "4",
            "output_dir": "outputs/walk_forward_validation",
            "prices_csv_path": "",
            "use_sample": "",
            "auto_save": "on",
            "insecure_ssl": "",
            "ca_bundle_path": "",
        }
        state_wfv_ctx: _WalkForwardContext | None = None
        state_wfv_error: str | None = None
        state_factor_form: dict[str, str] = {
            "ticker": "",
        }
        state_factor_ctx: _FactorContext | None = None
        state_factor_error: str | None = None

        refresh_lock = threading.Lock()
        state_refresh_running: bool = False
        state_refresh_status: str = "idle"
        state_refresh_run_id: int = 0
        state_refresh_started_at: str | None = None
        state_refresh_finished_at: str | None = None
        state_refresh_error: str | None = None
        state_refresh_logs: list[str] = []
        state_refresh_live_items: list[dict[str, object]] = []
        state_refresh_history: list[dict[str, object]] = []
        state_external_command_id: int = 0
        state_external_ticker: str = ""
        state_external_navigate_url: str = ""
        state_external_updated_at: str | None = None

        def do_GET(self) -> None:  # noqa: N802
            parsed_url = urlparse(self.path)
            path = parsed_url.path
            query = parse_qs(parsed_url.query)

            if enable_technical_page and path == "/external_command_state":
                self._send_json(
                    {
                        "ok": True,
                        "command_id": self.state_external_command_id,
                        "ticker": self.state_external_ticker,
                        "navigate_url": self.state_external_navigate_url,
                        "updated_at": self.state_external_updated_at,
                    }
                )
                return
            if enable_technical_page and path == "/external_select":
                ticker = str(query.get("ticker", [""])[0]).strip().upper()
                target = str(query.get("target", ["/decision"])[0]).strip() or "/decision" # Changed default target
                if target not in {"/page6", "/decision", "/forecast", "/page1"}:
                    target = "/page6"
                if not ticker:
                    self._send_json({"ok": False, "error": "ticker is required"}, status=400)
                    return
                self.__class__.state_external_command_id += 1
                self.__class__.state_external_ticker = ticker
                self.__class__.state_external_navigate_url = f"{target}?ticker={ticker}&intent=run"
                self.__class__.state_external_updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self._send_json(
                    {
                        "ok": True,
                        "command_id": self.state_external_command_id,
                        "ticker": self.state_external_ticker,
                        "navigate_url": self.state_external_navigate_url,
                        "updated_at": self.state_external_updated_at,
                    }
                )
                return

            # 외부 티커 요청 처리 (예: ?ticker=AAPL&intent=run)
            ticker_arg = query.get("ticker", [None])[0] # This logic needs to be moved to main web_gui
            if ticker_arg and path in {"/", "/index.html", "/forecast", "/page1", "/page6", "/decision"}:
                clean_t = ticker_arg.strip().upper()
                decision_entry = path in {"/", "/index.html", "/page6", "/decision"}
                self.__class__.state_form["ticker"] = clean_t
                self.__class__._sync_cross_page_tickers(clean_t)

                if query.get("intent", [""])[0] == "run":
                    try:
                        if decision_entry:
                            if self.state_ret_ctx is None or self.state_ret_ctx.ticker != clean_t:
                                self.__class__.state_ret_form = {"ticker": clean_t}
                                self.__class__.state_ret_ctx = _run_returns_once(self.state_ret_form)
                                self.__class__.state_ret_error = None
                            if self.state_risk_ctx is None or self.state_risk_ctx.ticker != clean_t:
                                self.__class__.state_risk_form = {"ticker": clean_t}
                                self.__class__.state_risk_ctx = _run_risk_once(self.state_risk_form)
                                self.__class__.state_risk_error = None
                            fin_ctx = self.state_fin_ctx if self.state_fin_ctx is not None and self.state_fin_ctx.ticker.strip().upper() == clean_t else None
                            self.__class__.state_dec_ctx = _run_decision_once(
                                self.state_dec_form,
                                returns_ctx=self.state_ret_ctx,
                                risk_ctx=self.state_risk_ctx,
                                fin_ctx=fin_ctx,
                            )
                            self.__class__.state_dec_error = None
                        else:
                            # 기본 예측 페이지 요청은 forecast 컨텍스트를 즉시 계산합니다.
                            self.__class__.state_ctx = _run_once(self.__class__.state_form)
                            self.__class__.state_error = None
                    except Exception as exc:
                        if decision_entry:
                            self.__class__.state_dec_ctx = None
                            if isinstance(exc, ValueError):
                                self.__class__.state_dec_error = str(exc)
                            else:
                                self.__class__.state_dec_error = traceback.format_exc()
                        else:
                            self.__class__.state_ctx = None
                            self.__class__.state_error = str(exc)

                # 루트 접근 시 분석 결과 페이지로 자동 전환
                if path in {"/", "/index.html"}:
                    path = "/page6"

            if path in ("/", "/index.html"): # This should be handled by the main web_gui
                self._send_html(
                    _html_financial_page(
                        self.state_fin_form,
                        ctx=self.state_fin_ctx,
                        error=self.state_fin_error,
                        ticker_note=self.state_ticker_note,
                        ticker_note_error=self.state_ticker_note_error,
                        enable_technical_page=enable_technical_page,
                    )
                )
                return
            if path in ("/forecast", "/stock-forecast"): # This should be handled by the main web_gui
                self._send_html(
                    _html_page(
                        self.state_form,
                        ctx=self.state_ctx,
                        error=self.state_error,
                        ticker_note=self.state_ticker_note,
                        ticker_note_error=self.state_ticker_note_error,
                        enable_technical_page=enable_technical_page,
                    )
                )
                return
            if path in ("/page2", "/financials"): # This should be handled by the main web_gui
                self._send_html(
                    _html_financial_page(
                        self.state_fin_form,
                        ctx=self.state_fin_ctx,
                        error=self.state_fin_error,
                        ticker_note=self.state_ticker_note,
                        ticker_note_error=self.state_ticker_note_error,
                        enable_technical_page=enable_technical_page,
                    )
                )
                return
            if enable_technical_page and path in ("/page3", "/technical"): # This should be handled by the main web_gui
                self._send_html(
                    _html_technical_page(
                        self.state_ta_form,
                        ctx=self.state_ta_ctx,
                        error=self.state_ta_error,
                        ticker_note=self.state_ticker_note,
                        ticker_note_error=self.state_ticker_note_error,
                    )
                )
                return
            if enable_technical_page and path in ("/page4", "/returns"): # This should be handled by the main web_gui
                self._send_html(
                    _html_returns_page(
                        self.state_ret_form,
                        ctx=self.state_ret_ctx,
                        error=self.state_ret_error,
                        ticker_note=self.state_ticker_note,
                        ticker_note_error=self.state_ticker_note_error,
                    )
                )
                return
            if enable_technical_page and path in ("/page5", "/risk"): # This should be handled by the main web_gui
                self._send_html(
                    _html_risk_page(
                        self.state_risk_form,
                        ctx=self.state_risk_ctx,
                        error=self.state_risk_error,
                        ticker_note=self.state_ticker_note,
                        ticker_note_error=self.state_ticker_note_error,
                    )
                )
                return
            if enable_technical_page and path in ("/factor-regime", "/factor", "/page5b"): # This should be handled by the main web_gui
                self._send_html(
                    _html_factor_page(
                        self.state_factor_form,
                        ctx=self.state_factor_ctx,
                        error=self.state_factor_error,
                        ticker_note=self.state_ticker_note,
                        ticker_note_error=self.state_ticker_note_error,
                    )
                )
                return
            if enable_technical_page and path in ("/page6", "/decision"): # This should be handled by the main web_gui
                self._send_html(
                    _html_decision_page(
                        self.state_dec_form,
                        ctx=self.state_dec_ctx,
                        error=self.state_dec_error,
                        ticker_note=self.state_ticker_note,
                        ticker_note_error=self.state_ticker_note_error,
                    )
                )
                return
            if enable_technical_page and path in ("/page7", "/refresh-data"): # This should be handled by the main web_gui
                self._send_html(_html_refresh_page(enable_technical_page=True))
                return
            if enable_technical_page and path in ("/page8", "/walk-forward-validation"): # This should be handled by the main web_gui
                self._send_html(
                    _html_walk_forward_page(
                        self.state_wfv_form,
                        ctx=self.state_wfv_ctx,
                        error=self.state_wfv_error,
                        ticker_note=self.state_ticker_note,
                        ticker_note_error=self.state_ticker_note_error,
                    )
                )
                return
            if enable_technical_page and path == "/refresh_status": # This should be handled by the main web_gui
                self._send_json(self._refresh_status_payload())
                return
            if enable_technical_page and path == "/refresh_history_data": # This should be handled by the main web_gui
                self._send_json(self._refresh_history_payload())
                return
            if enable_technical_page and path == "/refresh_history_window": # This should be handled by the main web_gui
                self._send_html(_html_refresh_history_page())
                return

            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

        def do_POST(self) -> None:  # noqa: N802 # This should be handled by the main web_gui
            path = urlparse(self.path).path
            if path == "/run": # This should be handled by the main web_gui
                self._handle_forecast_run()
                return
            if path == "/run_financial": # This should be handled by the main web_gui
                self._handle_financial_run()
                return
            if enable_technical_page and path == "/run_technical": # This should be handled by the main web_gui
                self._handle_technical_run()
                return
            if enable_technical_page and path == "/run_returns": # This should be handled by the main web_gui
                self._handle_returns_run()
                return
            if enable_technical_page and path == "/run_risk": # This should be handled by the main web_gui
                self._handle_risk_run()
                return
            if enable_technical_page and path == "/run_factor": # This should be handled by the main web_gui
                self._handle_factor_run()
                return
            if enable_technical_page and path == "/run_decision": # This should be handled by the main web_gui
                self._handle_decision_run()
                return
            if enable_technical_page and path == "/run_refresh": # This should be handled by the main web_gui
                self._handle_refresh_run()
                return
            if enable_technical_page and path == "/run_walk_forward": # This should be handled by the main web_gui
                self._handle_walk_forward_run()
                return

            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

        def _read_form(self) -> dict[str, str]:
            length = int(self.headers.get("Content-Length", "0"))
            payload = self.rfile.read(length).decode("utf-8", errors="ignore")
            parsed = parse_qs(payload)
            return {k: v[0] for k, v in parsed.items()}

        @staticmethod
        def _financial_cache_key(fin_form: dict[str, str]) -> str:
            return "|".join(
                [
                    fin_form.get("ticker", "").strip().upper(),
                    fin_form.get("statement_periods", "4").strip() or "4",
                    fin_form.get("ca_bundle_path", "").strip(),
                    fin_form.get("insecure_ssl", "").strip(),
                    _resolve_fmp_api_key(fin_form.get("fmp_api_key", "")) or "",
                ]
            )

        @classmethod
        def _sync_cross_page_tickers(cls, ticker: str) -> None:
            clean_ticker = str(ticker or "").strip().upper()
            if not clean_ticker:
                return

            prev_forecast_ticker = ""
            if cls.state_ctx is not None:
                try:
                    prev_forecast_ticker = str(cls.state_ctx.result.summary.iloc[0].get("ticker", "")).strip().upper()
                except Exception:
                    prev_forecast_ticker = ""
            prev_fin_ticker = cls.state_fin_ctx.ticker.strip().upper() if cls.state_fin_ctx is not None else ""
            prev_ta_ticker = cls.state_ta_ctx.ticker.strip().upper() if cls.state_ta_ctx is not None else ""
            prev_ret_ticker = cls.state_ret_ctx.ticker.strip().upper() if cls.state_ret_ctx is not None else ""
            prev_risk_ticker = cls.state_risk_ctx.ticker.strip().upper() if cls.state_risk_ctx is not None else ""
            prev_factor_ticker = cls.state_factor_ctx.ticker.strip().upper() if cls.state_factor_ctx is not None else ""
            prev_dec_ticker = cls.state_dec_ctx.ticker.strip().upper() if cls.state_dec_ctx is not None else ""
            prev_wfv_ticker = cls.state_wfv_ctx.ticker.strip().upper() if cls.state_wfv_ctx is not None else ""

            cls.state_form = {
                **cls.state_form,
                "ticker": clean_ticker,
            }

            cls.state_ta_form = {
                "ticker": clean_ticker,
                "output_dir": cls.state_ta_form.get("output_dir", "outputs/technical_analysis"),
                "use_sample": cls.state_ta_form.get("use_sample", ""),
                "auto_save": cls.state_ta_form.get("auto_save", "on"),
                "action": cls.state_ta_form.get("action", "all"),
            }
            cls.state_ret_form = {
                "ticker": clean_ticker,
            }
            cls.state_risk_form = {
                "ticker": clean_ticker,
            }
            cls.state_factor_form = {
                "ticker": clean_ticker,
            }
            cls.state_dec_form = {
                "ticker": clean_ticker,
            }
            cls.state_wfv_form = {
                **cls.state_wfv_form,
                "ticker": clean_ticker,
            }
            cls.state_dec_ctx = None
            cls.state_dec_error = None

            if clean_ticker not in {"LOCAL", "SAMPLE"}:
                cls.state_fin_form = {
                    "ticker": clean_ticker,
                    "statement_periods": cls.state_fin_form.get("statement_periods", "4"),
                    "output_dir": cls.state_fin_form.get("output_dir", "outputs/stock_forecast_finance"),
                    "auto_save": cls.state_fin_form.get("auto_save", cls.state_form.get("auto_save", "on")),
                    "insecure_ssl": cls.state_fin_form.get("insecure_ssl", cls.state_form.get("insecure_ssl", "")),
                    "ca_bundle_path": cls.state_fin_form.get("ca_bundle_path", cls.state_form.get("ca_bundle_path", "")),
                    "fmp_api_key": cls.state_fin_form.get("fmp_api_key", ""),
                }
            else:
                cls.state_fin_form = {
                    **cls.state_fin_form,
                    "ticker": "",
                }
            if prev_forecast_ticker and prev_forecast_ticker != clean_ticker:
                cls.state_ctx = None
                cls.state_error = None
            if prev_fin_ticker and prev_fin_ticker != clean_ticker:
                cls.state_fin_ctx = None
                cls.state_fin_error = None
            if prev_ta_ticker and prev_ta_ticker != clean_ticker:
                cls.state_ta_ctx = None
                cls.state_ta_error = None
            if prev_ret_ticker and prev_ret_ticker != clean_ticker:
                cls.state_ret_ctx = None
                cls.state_ret_error = None
            if prev_risk_ticker and prev_risk_ticker != clean_ticker:
                cls.state_risk_ctx = None
                cls.state_risk_error = None
            if prev_factor_ticker and prev_factor_ticker != clean_ticker:
                cls.state_factor_ctx = None
                cls.state_factor_error = None
            if prev_dec_ticker and prev_dec_ticker != clean_ticker:
                cls.state_dec_ctx = None
                cls.state_dec_error = None
            if prev_wfv_ticker and prev_wfv_ticker != clean_ticker:
                cls.state_wfv_ctx = None
                cls.state_wfv_error = None

        def _handle_forecast_run(self) -> None:
            form = self._read_form()
            for checkbox in ["use_sample", "auto_save", "insecure_ssl"]:
                if checkbox not in form:
                    form[checkbox] = ""

            intent = form.get("intent", "run").strip().lower() or "run"
            insecure_ssl = form.get("insecure_ssl", "") == "on"
            ca_bundle_path = form.get("ca_bundle_path", "").strip() or None

            prev_ticker = self.state_form.get("ticker", "").strip().upper()
            input_ticker_raw = form.get("ticker", "").strip()
            resolved_ticker, ticker_note, ticker_note_error = _resolve_ticker_input(
                input_ticker_raw,
                ca_bundle_path=ca_bundle_path,
                insecure_ssl=insecure_ssl,
            )
            effective_ticker = (resolved_ticker or prev_ticker).strip().upper()

            self.__class__.state_ticker_note = ticker_note
            self.__class__.state_ticker_note_error = ticker_note_error

            self.__class__.state_form = {
                "ticker": effective_ticker,
                "forecast_horizon": form.get("forecast_horizon", "10"),
                "history_years": form.get("history_years", "8"),
                "start_date": form.get("start_date", ""),
                "end_date": form.get("end_date", datetime.utcnow().strftime("%Y-%m-%d")),
                "output_dir": form.get("output_dir", "outputs/stock_forecast"),
                "prices_csv_path": form.get("prices_csv_path", ""),
                "use_sample": form.get("use_sample", ""),
                "auto_save": form.get("auto_save", ""),
                "insecure_ssl": form.get("insecure_ssl", ""),
                "ca_bundle_path": form.get("ca_bundle_path", ""),
            }
            self.__class__._sync_cross_page_tickers(effective_ticker)

            if intent == "resolve_ticker":
                self.__class__.state_error = None
                self._send_html(
                    _html_page(
                        self.state_form,
                        ctx=self.state_ctx,
                        error=self.state_error,
                        ticker_note=self.state_ticker_note,
                        ticker_note_error=self.state_ticker_note_error,
                        enable_technical_page=enable_technical_page,
                    )
                )
                return

            if ticker_note_error and not effective_ticker:
                self.__class__.state_ctx = None
                self.__class__.state_error = ticker_note or "Provide ticker, or set Local Prices CSV Path Override."
                self._send_html(
                    _html_page(
                        self.state_form,
                        ctx=self.state_ctx,
                        error=self.state_error,
                        ticker_note=self.state_ticker_note,
                        ticker_note_error=self.state_ticker_note_error,
                        enable_technical_page=enable_technical_page,
                    )
                )
                return

            try:
                self.__class__.state_ctx = _run_once(self.state_form)
                self.__class__.state_error = None
            except Exception as exc:
                self.__class__.state_ctx = None
                out_dir = Path(self.state_form.get("output_dir", "outputs/stock_forecast"))
                hint = security_hint(exc, output_dir=out_dir)
                if isinstance(exc, ValueError):
                    self.__class__.state_error = str(exc)
                else:
                    self.__class__.state_error = f"{hint}\n\nRaw error: {exc}" if hint else traceback.format_exc()

            if self.state_ctx is not None:
                inferred_ticker = str(self.state_ctx.result.summary.iloc[0].get("ticker", "")).strip().upper()
                requested_ticker = self.state_form.get("ticker", "").strip().upper()
                financial_ticker = requested_ticker or inferred_ticker

                if financial_ticker:
                    self.__class__._sync_cross_page_tickers(financial_ticker)

            self._send_html(
                _html_page(
                    self.state_form,
                    ctx=self.state_ctx,
                    error=self.state_error,
                    ticker_note=self.state_ticker_note,
                    ticker_note_error=self.state_ticker_note_error,
                    enable_technical_page=enable_technical_page,
                )
            )

        def _handle_financial_run(self) -> None:
            form = self._read_form()
            for checkbox in ["auto_save", "insecure_ssl"]:
                if checkbox not in form:
                    form[checkbox] = ""

            intent = form.get("intent", "run").strip().lower() or "run"
            insecure_ssl = form.get("insecure_ssl", "") == "on"
            ca_bundle_path = form.get("ca_bundle_path", "").strip() or None

            fallback_ticker = self.state_form.get("ticker", "").strip().upper()
            raw_ticker = form.get("ticker", "").strip()
            resolved_ticker, ticker_note, ticker_note_error = _resolve_ticker_input(
                raw_ticker,
                ca_bundle_path=ca_bundle_path,
                insecure_ssl=insecure_ssl,
            )
            effective_ticker = (resolved_ticker or fallback_ticker).strip().upper()

            self.__class__.state_ticker_note = ticker_note
            self.__class__.state_ticker_note_error = ticker_note_error

            self.__class__.state_fin_form = {
                "ticker": effective_ticker,
                "statement_periods": form.get("statement_periods", "4"),
                "output_dir": form.get("output_dir", "outputs/stock_forecast_finance"),
                "auto_save": form.get("auto_save", ""),
                "insecure_ssl": form.get("insecure_ssl", ""),
                "ca_bundle_path": form.get("ca_bundle_path", ""),
                "fmp_api_key": form.get("fmp_api_key", ""),
            }
            entered_ticker = self.state_fin_form.get("ticker", "").strip().upper()
            if entered_ticker:
                self.__class__._sync_cross_page_tickers(entered_ticker)

            if intent == "resolve_ticker":
                self.__class__.state_fin_error = None
                self._send_html(
                    _html_financial_page(
                        self.state_fin_form,
                        ctx=self.state_fin_ctx,
                        error=self.state_fin_error,
                        ticker_note=self.state_ticker_note,
                        ticker_note_error=self.state_ticker_note_error,
                        enable_technical_page=enable_technical_page,
                    )
                )
                return

            if ticker_note_error and not effective_ticker:
                self.__class__.state_fin_ctx = None
                self.__class__.state_fin_error = ticker_note or "Provide ticker for financial statements page."
                self._send_html(
                    _html_financial_page(
                        self.state_fin_form,
                        ctx=self.state_fin_ctx,
                        error=self.state_fin_error,
                        ticker_note=self.state_ticker_note,
                        ticker_note_error=self.state_ticker_note_error,
                        enable_technical_page=enable_technical_page,
                    )
                )
                return

            cache_key = self._financial_cache_key(self.state_fin_form)
            cached = self.state_fin_cache.get(cache_key)
            if cached is not None:
                self.__class__.state_fin_ctx = cached
                self.__class__.state_fin_error = None
            else:
                try:
                    fin_ctx = _run_financial_once(self.state_fin_form)
                    self.__class__.state_fin_ctx = fin_ctx
                    self.__class__.state_fin_cache[cache_key] = fin_ctx
                    self.__class__.state_fin_error = None
                except Exception as exc:
                    self.__class__.state_fin_ctx = None
                    out_dir = Path(self.state_fin_form.get("output_dir", "outputs/stock_forecast_finance"))
                    hint = security_hint(exc, output_dir=out_dir)
                    if isinstance(exc, ValueError):
                        self.__class__.state_fin_error = str(exc)
                    else:
                        self.__class__.state_fin_error = (
                            f"{hint}\n\nRaw error: {exc}" if hint else traceback.format_exc()
                        )

            self._send_html(
                _html_financial_page(
                    self.state_fin_form,
                    ctx=self.state_fin_ctx,
                    error=self.state_fin_error,
                    ticker_note=self.state_ticker_note,
                    ticker_note_error=self.state_ticker_note_error,
                    enable_technical_page=enable_technical_page,
                )
            )

        def _handle_technical_run(self) -> None:
            form = self._read_form()
            for checkbox in ["use_sample", "auto_save"]:
                if checkbox not in form:
                    form[checkbox] = ""

            intent = form.get("intent", "run").strip().lower() or "run"
            raw_ticker = form.get("ticker", "").strip()
            resolved_ticker, ticker_note, ticker_note_error = _resolve_ticker_input(
                raw_ticker,
                ca_bundle_path=None,
                insecure_ssl=False,
            )

            self.__class__.state_ticker_note = ticker_note
            self.__class__.state_ticker_note_error = ticker_note_error

            fallback_ticker = self.state_form.get("ticker", "").strip().upper()
            effective_ticker = (resolved_ticker or fallback_ticker).strip().upper()
            action = _normalize_technical_action(form.get("action", "all"))

            self.__class__.state_ta_form = {
                "ticker": effective_ticker,
                "output_dir": form.get("output_dir", "outputs/technical_analysis"),
                "use_sample": form.get("use_sample", ""),
                "auto_save": form.get("auto_save", ""),
                "action": action,
            }

            self.__class__._sync_cross_page_tickers(effective_ticker)

            if intent == "resolve_ticker":
                self.__class__.state_ta_error = None
                self._send_html(
                    _html_technical_page(
                        self.state_ta_form,
                        ctx=self.state_ta_ctx,
                        error=self.state_ta_error,
                        ticker_note=self.state_ticker_note,
                        ticker_note_error=self.state_ticker_note_error,
                    )
                )
                return

            if ticker_note_error and not effective_ticker:
                self.__class__.state_ta_ctx = None
                self.__class__.state_ta_error = ticker_note or "Provide ticker for technical analysis."
                self._send_html(
                    _html_technical_page(
                        self.state_ta_form,
                        ctx=self.state_ta_ctx,
                        error=self.state_ta_error,
                        ticker_note=self.state_ticker_note,
                        ticker_note_error=self.state_ticker_note_error,
                    )
                )
                return

            try:
                ta_ctx, ta_cache = ta_web_gui._run_analysis(
                    form=self.state_ta_form,
                    action=action,
                    cache=self.state_ta_cache,
                )
                self.__class__.state_ta_ctx = ta_ctx
                self.__class__.state_ta_cache = ta_cache
                self.__class__.state_ta_error = None
            except Exception as exc:
                self.__class__.state_ta_ctx = None
                out_dir = Path(self.state_ta_form.get("output_dir", "outputs/technical_analysis"))
                hint = security_hint(exc, output_dir=out_dir)
                if isinstance(exc, ValueError):
                    self.__class__.state_ta_error = str(exc)
                else:
                    self.__class__.state_ta_error = f"{hint}\n\nRaw error: {exc}" if hint else traceback.format_exc()

            self._send_html(
                _html_technical_page(
                    self.state_ta_form,
                    ctx=self.state_ta_ctx,
                    error=self.state_ta_error,
                    ticker_note=self.state_ticker_note,
                    ticker_note_error=self.state_ticker_note_error,
                )
            )

        def _handle_returns_run(self) -> None:
            form = self._read_form()
            intent = form.get("intent", "run").strip().lower() or "run"
            raw_ticker = form.get("ticker", "").strip()
            resolved_ticker, ticker_note, ticker_note_error = _resolve_ticker_input(
                raw_ticker,
                ca_bundle_path=None,
                insecure_ssl=False,
            )

            self.__class__.state_ticker_note = ticker_note
            self.__class__.state_ticker_note_error = ticker_note_error

            fallback_ticker = self.state_form.get("ticker", "").strip().upper()
            effective_ticker = (resolved_ticker or fallback_ticker).strip().upper()
            self.__class__.state_ret_form = {
                "ticker": effective_ticker,
            }

            self.__class__._sync_cross_page_tickers(effective_ticker)

            if intent == "resolve_ticker":
                self.__class__.state_ret_error = None
                self._send_html(
                    _html_returns_page(
                        self.state_ret_form,
                        ctx=self.state_ret_ctx,
                        error=self.state_ret_error,
                        ticker_note=self.state_ticker_note,
                        ticker_note_error=self.state_ticker_note_error,
                    )
                )
                return

            if ticker_note_error and not effective_ticker:
                self.__class__.state_ret_ctx = None
                self.__class__.state_ret_error = ticker_note or "Provide an S&P 500 ticker for return analysis."
                self._send_html(
                    _html_returns_page(
                        self.state_ret_form,
                        ctx=self.state_ret_ctx,
                        error=self.state_ret_error,
                        ticker_note=self.state_ticker_note,
                        ticker_note_error=self.state_ticker_note_error,
                    )
                )
                return

            try:
                self.__class__.state_ret_ctx = _run_returns_once(self.state_ret_form)
                self.__class__.state_ret_error = None
            except Exception as exc:
                self.__class__.state_ret_ctx = None
                if isinstance(exc, ValueError):
                    self.__class__.state_ret_error = str(exc)
                else:
                    self.__class__.state_ret_error = traceback.format_exc()

            self._send_html(
                _html_returns_page(
                    self.state_ret_form,
                    ctx=self.state_ret_ctx,
                    error=self.state_ret_error,
                    ticker_note=self.state_ticker_note,
                    ticker_note_error=self.state_ticker_note_error,
                )
            )

        def _handle_risk_run(self) -> None:
            form = self._read_form()
            intent = form.get("intent", "run").strip().lower() or "run"
            raw_ticker = form.get("ticker", "").strip()
            resolved_ticker, ticker_note, ticker_note_error = _resolve_ticker_input(
                raw_ticker,
                ca_bundle_path=None,
                insecure_ssl=False,
            )

            self.__class__.state_ticker_note = ticker_note
            self.__class__.state_ticker_note_error = ticker_note_error

            fallback_ticker = self.state_form.get("ticker", "").strip().upper()
            effective_ticker = (resolved_ticker or fallback_ticker).strip().upper()
            self.__class__.state_risk_form = {
                "ticker": effective_ticker,
            }

            self.__class__._sync_cross_page_tickers(effective_ticker)

            if intent == "resolve_ticker":
                self.__class__.state_risk_error = None
                self._send_html(
                    _html_risk_page(
                        self.state_risk_form,
                        ctx=self.state_risk_ctx,
                        error=self.state_risk_error,
                        ticker_note=self.state_ticker_note,
                        ticker_note_error=self.state_ticker_note_error,
                    )
                )
                return

            if ticker_note_error and not effective_ticker:
                self.__class__.state_risk_ctx = None
                self.__class__.state_risk_error = ticker_note or "Provide an S&P 500 ticker for risk analysis."
                self._send_html(
                    _html_risk_page(
                        self.state_risk_form,
                        ctx=self.state_risk_ctx,
                        error=self.state_risk_error,
                        ticker_note=self.state_ticker_note,
                        ticker_note_error=self.state_ticker_note_error,
                    )
                )
                return

            try:
                self.__class__.state_risk_ctx = _run_risk_once(self.state_risk_form)
                self.__class__.state_risk_error = None
            except Exception as exc:
                self.__class__.state_risk_ctx = None
                if isinstance(exc, ValueError):
                    self.__class__.state_risk_error = str(exc)
                else:
                    self.__class__.state_risk_error = traceback.format_exc()

            self._send_html(
                _html_risk_page(
                    self.state_risk_form,
                    ctx=self.state_risk_ctx,
                    error=self.state_risk_error,
                    ticker_note=self.state_ticker_note,
                    ticker_note_error=self.state_ticker_note_error,
                )
            )

        def _handle_factor_run(self) -> None:
            form = self._read_form()
            intent = form.get("intent", "run").strip().lower() or "run"
            raw_ticker = form.get("ticker", "").strip()
            resolved_ticker, ticker_note, ticker_note_error = _resolve_ticker_input(
                raw_ticker,
                ca_bundle_path=None,
                insecure_ssl=False,
            )

            self.__class__.state_ticker_note = ticker_note
            self.__class__.state_ticker_note_error = ticker_note_error

            fallback_ticker = self.state_form.get("ticker", "").strip().upper()
            effective_ticker = (resolved_ticker or fallback_ticker).strip().upper()
            self.__class__.state_factor_form = {
                "ticker": effective_ticker,
            }

            self.__class__._sync_cross_page_tickers(effective_ticker)

            if intent == "resolve_ticker":
                self.__class__.state_factor_error = None
                self._send_html(
                    _html_factor_page(
                        self.state_factor_form,
                        ctx=self.state_factor_ctx,
                        error=self.state_factor_error,
                        ticker_note=self.state_ticker_note,
                        ticker_note_error=self.state_ticker_note_error,
                    )
                )
                return

            if ticker_note_error and not effective_ticker:
                self.__class__.state_factor_ctx = None
                self.__class__.state_factor_error = ticker_note or "Provide an S&P 500 ticker for factor and regime analysis."
                self._send_html(
                    _html_factor_page(
                        self.state_factor_form,
                        ctx=self.state_factor_ctx,
                        error=self.state_factor_error,
                        ticker_note=self.state_ticker_note,
                        ticker_note_error=self.state_ticker_note_error,
                    )
                )
                return

            try:
                self.__class__.state_factor_ctx = _run_factor_once(self.state_factor_form)
                self.__class__.state_factor_error = None
            except Exception as exc:
                self.__class__.state_factor_ctx = None
                if isinstance(exc, ValueError):
                    self.__class__.state_factor_error = str(exc)
                else:
                    self.__class__.state_factor_error = traceback.format_exc()

            self._send_html(
                _html_factor_page(
                    self.state_factor_form,
                    ctx=self.state_factor_ctx,
                    error=self.state_factor_error,
                    ticker_note=self.state_ticker_note,
                    ticker_note_error=self.state_ticker_note_error,
                )
            )

        def _handle_decision_run(self) -> None:
            form = self._read_form()
            intent = form.get("intent", "run").strip().lower() or "run"
            raw_ticker = form.get("ticker", "").strip()
            resolved_ticker, ticker_note, ticker_note_error = _resolve_ticker_input(
                raw_ticker,
                ca_bundle_path=None,
                insecure_ssl=False,
            )

            self.__class__.state_ticker_note = ticker_note
            self.__class__.state_ticker_note_error = ticker_note_error

            fallback_ticker = self.state_form.get("ticker", "").strip().upper()
            effective_ticker = (resolved_ticker or fallback_ticker).strip().upper()
            self.__class__.state_dec_form = {
                "ticker": effective_ticker,
            }

            self.__class__._sync_cross_page_tickers(effective_ticker)

            if intent == "resolve_ticker":
                self.__class__.state_dec_error = None
                self._send_html(
                    _html_decision_page(
                        self.state_dec_form,
                        ctx=self.state_dec_ctx,
                        error=self.state_dec_error,
                        ticker_note=self.state_ticker_note,
                        ticker_note_error=self.state_ticker_note_error,
                    )
                )
                return

            if ticker_note_error and not effective_ticker:
                self.__class__.state_dec_ctx = None
                self.__class__.state_dec_error = ticker_note or "Provide an S&P 500 ticker for decision analysis."
                self._send_html(
                    _html_decision_page(
                        self.state_dec_form,
                        ctx=self.state_dec_ctx,
                        error=self.state_dec_error,
                        ticker_note=self.state_ticker_note,
                        ticker_note_error=self.state_ticker_note_error,
                    )
                )
                return

            try:
                if self.state_ret_ctx is None or self.state_ret_ctx.ticker != effective_ticker:
                    self.__class__.state_ret_form = {"ticker": effective_ticker}
                    self.__class__.state_ret_ctx = _run_returns_once(self.state_ret_form)
                    self.__class__.state_ret_error = None
                if self.state_risk_ctx is None or self.state_risk_ctx.ticker != effective_ticker:
                    self.__class__.state_risk_form = {"ticker": effective_ticker}
                    self.__class__.state_risk_ctx = _run_risk_once(self.state_risk_form)
                    self.__class__.state_risk_error = None

                fin_ctx = self.state_fin_ctx if self.state_fin_ctx is not None and self.state_fin_ctx.ticker.strip().upper() == effective_ticker else None
                self.__class__.state_dec_ctx = _run_decision_once(
                    self.state_dec_form,
                    returns_ctx=self.state_ret_ctx,
                    risk_ctx=self.state_risk_ctx,
                    fin_ctx=fin_ctx,
                )
                self.__class__.state_dec_error = None
            except Exception as exc:
                self.__class__.state_dec_ctx = None
                if isinstance(exc, ValueError):
                    self.__class__.state_dec_error = str(exc)
                else:
                    self.__class__.state_dec_error = traceback.format_exc()

            self._send_html(
                _html_decision_page(
                    self.state_dec_form,
                    ctx=self.state_dec_ctx,
                    error=self.state_dec_error,
                    ticker_note=self.state_ticker_note,
                    ticker_note_error=self.state_ticker_note_error,
                )
            )

        @classmethod
        def _append_refresh_log(cls, message: str) -> None:
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line = f"[{stamp}] {message}"
            with cls.refresh_lock:
                cls.state_refresh_logs.append(line)
                if len(cls.state_refresh_logs) > 600:
                    cls.state_refresh_logs = cls.state_refresh_logs[-600:]

        @classmethod
        def _upsert_refresh_live_item(
            cls,
            *,
            dataset: str,
            latest_date: str | None,
            rows: int,
            source: str,
            path: str,
        ) -> None:
            with cls.refresh_lock:
                for item in cls.state_refresh_live_items:
                    if str(item.get("dataset", "")) == dataset:
                        item["latest_date"] = latest_date
                        item["rows"] = int(rows)
                        item["source"] = source
                        item["path"] = path
                        return
                cls.state_refresh_live_items.append(
                    {
                        "dataset": dataset,
                        "latest_date": latest_date,
                        "rows": int(rows),
                        "source": source,
                        "path": path,
                    }
                )

        @classmethod
        def _refresh_status_payload(cls) -> dict[str, object]:
            root_dir = _project_root_dir()
            metrics_csv = root_dir / "data" / "sp500_all_metrics_prices.csv"
            latest_date = _latest_date_from_csv(metrics_csv)
            latest_summary = f"가격 데이터 최신일: {latest_date or '-'}"
            with cls.refresh_lock:
                return {
                    "status": cls.state_refresh_status,
                    "run_id": cls.state_refresh_run_id,
                    "running": cls.state_refresh_running,
                    "started_at": cls.state_refresh_started_at,
                    "finished_at": cls.state_refresh_finished_at,
                    "error": cls.state_refresh_error,
                    "log_count": len(cls.state_refresh_logs),
                    "logs": list(cls.state_refresh_logs[-220:]),
                    "updated_items": [dict(item) for item in cls.state_refresh_live_items],
                    "history_count": len(cls.state_refresh_history),
                    "latest_summary": latest_summary,
                }

        @classmethod
        def _refresh_history_payload(cls) -> dict[str, object]:
            with cls.refresh_lock:
                runs: list[dict[str, object]] = []
                for row in cls.state_refresh_history[:120]:
                    copied = dict(row)
                    copied["updated_items"] = [dict(item) for item in row.get("updated_items", [])]  # type: ignore[arg-type]
                    runs.append(copied)
            return {
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "runs": runs,
            }

        @classmethod
        def _start_refresh_job(cls) -> tuple[bool, str]:
            with cls.refresh_lock:
                if cls.state_refresh_running:
                    return False, "A refresh job is already running."
                cls.state_refresh_run_id += 1
                run_id = cls.state_refresh_run_id
                cls.state_refresh_running = True
                cls.state_refresh_status = "running"
                cls.state_refresh_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cls.state_refresh_finished_at = None
                cls.state_refresh_error = None
                cls.state_refresh_logs = []
                cls.state_refresh_live_items = []

            thread = threading.Thread(target=cls._run_refresh_job, args=(run_id,), daemon=True)
            thread.start()
            cls._append_refresh_log(f"Run {run_id} started.")
            return True, f"Run {run_id} started."

        @classmethod
        def _run_refresh_job(cls, run_id: int) -> None:
            status = "success"
            error_message: str | None = None
            updated_items: list[dict[str, object]] = []
            financial_rows_added = 0
            sp500_rows_added = 0
            sp500_market_cap_updates = 0
            root_dir = _project_root_dir()
            refresh_cmd, refresh_label = _refresh_subprocess_command(root_dir)
            sp500_db_path = root_dir / "data" / "sp500_shared_db" / "sp500_shared_prices.sqlite"
            sp500_old_max_date: str | None = _sp500_sqlite_max_date(sp500_db_path)
            sp500_new_max_date: str | None = sp500_old_max_date

            if not refresh_cmd:
                status = "error"
                error_message = f"FileNotFoundError: {refresh_label}"
                cls._append_refresh_log(f"[error] {error_message}")
            else:
                cls._append_refresh_log(f"Executing refresh: {refresh_label}")
                exit_code = 1
                try:
                    proc = subprocess.Popen(
                        refresh_cmd,
                        cwd=str(root_dir),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        bufsize=1,
                    )
                    if proc.stdout is not None:
                        for line in proc.stdout:
                            stripped = line.rstrip("\r\n")
                            if not stripped:
                                continue
                            cls._append_refresh_log(stripped)

                            m_sp500 = re.search(r"SQLite added rows=(\d+), market_cap_updates=(\d+)", stripped)
                            if m_sp500:
                                sp500_rows_added = int(m_sp500.group(1))
                                sp500_market_cap_updates = int(m_sp500.group(2))

                            m_dates = re.search(r"SQLite date range: old_max=([^,]+), new_max=(.+)$", stripped)
                            if m_dates:
                                sp500_old_max_date = m_dates.group(1).strip()
                                sp500_new_max_date = m_dates.group(2).strip()

                    exit_code = int(proc.wait())
                except Exception as exc:
                    status = "error"
                    error_message = f"{type(exc).__name__}: {exc}"
                    cls._append_refresh_log(f"[error] {error_message}")
                else:
                    if exit_code != 0:
                        status = "error"
                        error_message = f"{refresh_label} exited with code {exit_code}"
                        cls._append_refresh_log(f"[error] {error_message}")
                    else:
                        cls._append_refresh_log("Refresh finished successfully.")

                sp500_new_max_date = _sp500_sqlite_max_date(sp500_db_path) or sp500_new_max_date
                updated_items = _collect_post_refresh_items(root_dir)
                for item in updated_items:
                    cls._upsert_refresh_live_item(
                        dataset=str(item.get("dataset", "")),
                        latest_date=str(item.get("latest_date", "")) or None,
                        rows=int(item.get("rows", 0) or 0),
                        source=str(item.get("source", "")),
                        path=str(item.get("path", "")),
                    )

            finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            history_row = {
                "run_id": run_id,
                "status": status,
                "started_at": cls.state_refresh_started_at,
                "finished_at": finished_at,
                "sp500_old_max_date": sp500_old_max_date,
                "sp500_new_max_date": sp500_new_max_date,
                "financial_sqlite_rows_added": financial_rows_added,
                "sp500_sqlite_added_rows": sp500_rows_added,
                "sp500_sqlite_market_cap_updates": sp500_market_cap_updates,
                "error_message": error_message,
                "updated_items": updated_items,
            }

            with cls.refresh_lock:
                cls.state_refresh_running = False
                cls.state_refresh_status = status
                cls.state_refresh_finished_at = finished_at
                cls.state_refresh_error = error_message
                if status == "success":
                    cls.state_refresh_live_items = [dict(item) for item in updated_items]
                cls.state_refresh_history.insert(0, history_row)
                if len(cls.state_refresh_history) > 300:
                    cls.state_refresh_history = cls.state_refresh_history[:300]

            cls._append_refresh_log(f"Run {run_id} finished with status={status}.")

        def _handle_refresh_run(self) -> None:
            started, message = self.__class__._start_refresh_job()
            if not started:
                self.__class__._append_refresh_log(message)
            self._send_html(_html_refresh_page(enable_technical_page=True))

        def _handle_walk_forward_run(self) -> None:
            form = self._read_form()
            for checkbox in ["use_sample", "auto_save", "insecure_ssl"]:
                if checkbox not in form:
                    form[checkbox] = ""

            intent = form.get("intent", "run").strip().lower() or "run"
            insecure_ssl = form.get("insecure_ssl", "") == "on"
            ca_bundle_path = form.get("ca_bundle_path", "").strip() or None

            fallback_ticker = self.state_form.get("ticker", "").strip().upper()
            raw_ticker = form.get("ticker", "").strip()
            resolved_ticker, ticker_note, ticker_note_error = _resolve_ticker_input(
                raw_ticker,
                ca_bundle_path=ca_bundle_path,
                insecure_ssl=insecure_ssl,
            )
            effective_ticker = (resolved_ticker or fallback_ticker).strip().upper()

            self.__class__.state_ticker_note = ticker_note
            self.__class__.state_ticker_note_error = ticker_note_error

            self.__class__.state_wfv_form = {
                "ticker": effective_ticker,
                "forecast_horizon": form.get("forecast_horizon", "10"),
                "history_years": form.get("history_years", "8"),
                "start_date": form.get("start_date", ""),
                "end_date": form.get("end_date", datetime.utcnow().strftime("%Y-%m-%d")),
                "wf_min_train_rows": form.get("wf_min_train_rows", "252"),
                "wf_step_size": form.get("wf_step_size", "21"),
                "wf_max_splits": form.get("wf_max_splits", "4"),
                "output_dir": form.get("output_dir", "outputs/walk_forward_validation"),
                "prices_csv_path": form.get("prices_csv_path", ""),
                "use_sample": form.get("use_sample", ""),
                "auto_save": form.get("auto_save", ""),
                "insecure_ssl": form.get("insecure_ssl", ""),
                "ca_bundle_path": form.get("ca_bundle_path", ""),
            }
            if effective_ticker:
                self.__class__._sync_cross_page_tickers(effective_ticker)

            if intent == "resolve_ticker":
                self.__class__.state_wfv_error = None
                self._send_html(
                    _html_walk_forward_page(
                        self.state_wfv_form,
                        ctx=self.state_wfv_ctx,
                        error=self.state_wfv_error,
                        ticker_note=self.state_ticker_note,
                        ticker_note_error=self.state_ticker_note_error,
                    )
                )
                return

            if ticker_note_error and not effective_ticker and not form.get("prices_csv_path", "").strip() and form.get("use_sample", "") != "on":
                self.__class__.state_wfv_ctx = None
                self.__class__.state_wfv_error = ticker_note or "Provide ticker, or set Local Prices CSV Path Override."
                self._send_html(
                    _html_walk_forward_page(
                        self.state_wfv_form,
                        ctx=self.state_wfv_ctx,
                        error=self.state_wfv_error,
                        ticker_note=self.state_ticker_note,
                        ticker_note_error=self.state_ticker_note_error,
                    )
                )
                return

            try:
                self.__class__.state_wfv_ctx = _run_walk_forward_validation_once(self.state_wfv_form)
                self.__class__.state_wfv_error = None
            except Exception as exc:
                self.__class__.state_wfv_ctx = None
                out_dir = Path(self.state_wfv_form.get("output_dir", "outputs/walk_forward_validation"))
                hint = security_hint(exc, output_dir=out_dir)
                if isinstance(exc, ValueError):
                    self.__class__.state_wfv_error = str(exc)
                else:
                    self.__class__.state_wfv_error = f"{hint}\n\nRaw error: {exc}" if hint else traceback.format_exc()

            self._send_html(
                _html_walk_forward_page(
                    self.state_wfv_form,
                    ctx=self.state_wfv_ctx,
                    error=self.state_wfv_error,
                    ticker_note=self.state_ticker_note,
                    ticker_note_error=self.state_ticker_note_error,
                )
            )

        def _send_html(self, body: str) -> None:
            payload = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_json(self, payload_obj: dict[str, object], status: int = 200) -> None:
            payload = json.dumps(payload_obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            return
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"{_APP_TITLE} listening on http://{host}:{port}")
    # Removed server.serve_forever() and browser opening logic as this is now a module

# Expose functions and dataclasses for external use
__all__ = [
    "_RunContext", "_FinancialContext", "_ReturnsContext", "_RiskContext", "_DecisionContext", "_FactorContext", "_WalkForwardContext",
    "_run_once", "_run_financial_once", "_run_returns_once", "_run_risk_once", "_run_factor_once", "_run_decision_once", "_run_walk_forward_validation_once",
    "_html_page", "_html_financial_page", "_html_technical_page", "_html_returns_page", "_html_risk_page", "_html_factor_page", "_html_decision_page", "_html_walk_forward_page",
    "_resolve_ticker_input",
    "ta_web_gui", # Expose technical_analysis module
]

# No __main__ block here, as it's not a standalone executable anymore
