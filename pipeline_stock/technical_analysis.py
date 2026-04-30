from __future__ import annotations

import base64
import html
import io
import os
import re
import traceback
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import patches

try:
    import FinanceDataReader as fdr
except Exception:  # pragma: no cover - optional dependency
    fdr = None

try:
    import yfinance as yf
except Exception:  # pragma: no cover - optional dependency
    yf = None

from pipeline_common.notebook_data import (
    fetch_sp500_close_prices,
    load_btc_close,
    load_dxy_series,
    load_fx_close,
    load_index_series,
)
from pipeline_common.shared_sp500_prices_sql import load_shared_ohlcv_for_symbol
from pipeline_common.security import ensure_writable_dir, security_hint

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOOKBACK_ROWS = 1400
TECH_CACHE_DIR = Path(os.getenv("KEUMJ_TECH_OHLCV_CACHE_DIR", "data/technical_ohlcv_cache"))


def _technical_cache_path(ticker: str) -> Path:
    safe = re.sub(r"[^A-Z0-9._-]+", "_", str(ticker).strip().upper())
    return TECH_CACHE_DIR / f"{safe}.csv"


def _load_ohlcv_cache(ticker: str) -> pd.DataFrame:
    path = _technical_cache_path(ticker)
    if not path.exists() or not path.is_file():
        return pd.DataFrame()

    try:
        raw = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    cols = {str(c).lower(): c for c in raw.columns}
    date_col = cols.get("date") or raw.columns[0]
    selected = [date_col]
    for name in ["open", "high", "low", "close", "volume"]:
        col = cols.get(name)
        if col is None:
            return pd.DataFrame()
        selected.append(col)

    frame = raw[selected].copy()
    frame.columns = ["date", "open", "high", "low", "close", "volume"]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date")
    if frame.empty:
        return pd.DataFrame()

    out = frame.set_index(frame["date"].dt.normalize())[["open", "high", "low", "close", "volume"]]
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def _save_ohlcv_cache(ticker: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return

    out = df.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()]
    if out.empty:
        return

    out = out[["open", "high", "low", "close", "volume"]]
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.index = out.index.tz_localize(None).normalize()
    out.index.name = "date"

    path = _technical_cache_path(ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, encoding="utf-8")


def _unique_nonempty(values: list[str | None]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        txt = str(value or "").strip()
        if not txt or txt in seen:
            continue
        seen.add(txt)
        out.append(txt)
    return out


def _normalize_symbol_token(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value or "").upper())


def _date_column_name(columns: list[object]) -> object | None:
    for col in columns:
        low = str(col).strip().lower()
        if low in {"date", "datetime", "timestamp", "time"}:
            return col
    return columns[0] if columns else None


def _ohlcv_base_csv_candidates() -> list[str]:
    return []


def _pick_col_by_alias(raw: pd.DataFrame, aliases: list[str]) -> object | None:
    cols = {str(c).strip().lower(): c for c in raw.columns}
    for alias in aliases:
        col = cols.get(alias.lower())
        if col is not None:
            return col
    return None


def _find_wide_metric_col(raw: pd.DataFrame, ticker: str, aliases: list[str]) -> object | None:
    ticker_low = str(ticker).strip().lower()
    ticker_norm = _normalize_symbol_token(ticker_low)
    sep = r"[\s_\-\./:]*"

    for col in raw.columns:
        low = str(col).strip().lower()
        for alias in aliases:
            a = re.escape(alias.lower())
            t = re.escape(ticker_low)
            if re.match(fr"^{t}{sep}{a}$", low) or re.match(fr"^{a}{sep}{t}$", low):
                return col

    for col in raw.columns:
        low = str(col).strip().lower()
        norm = _normalize_symbol_token(low)
        if ticker_norm and ticker_norm not in norm:
            continue
        for alias in aliases:
            alias_norm = _normalize_symbol_token(alias)
            if alias_norm and alias_norm in norm:
                return col

    return None


def _series_from_col(raw: pd.DataFrame, col: object) -> pd.Series:
    data = raw[col]
    if isinstance(data, pd.DataFrame):
        return data.iloc[:, 0]
    return data


def _extract_ohlcv_long(raw: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    if raw is None or raw.empty:
        return None

    date_col = _date_column_name(list(raw.columns))
    symbol_col = _pick_col_by_alias(raw, ["symbol", "ticker", "code", "asset"])
    if date_col is None or symbol_col is None:
        return None

    open_col = _pick_col_by_alias(raw, ["open", "o"])
    high_col = _pick_col_by_alias(raw, ["high", "h"])
    low_col = _pick_col_by_alias(raw, ["low", "l"])
    close_col = _pick_col_by_alias(raw, ["close", "adj close", "adjclose", "c"])
    volume_col = _pick_col_by_alias(raw, ["volume", "vol", "v"])

    if open_col is None or high_col is None or low_col is None or close_col is None:
        return None

    ticker_norm = _normalize_symbol_token(ticker)
    frame = raw[[date_col, symbol_col, open_col, high_col, low_col, close_col] + ([volume_col] if volume_col is not None else [])].copy()
    symbols = frame[symbol_col].astype(str).map(_normalize_symbol_token)
    frame = frame[symbols == ticker_norm]
    if frame.empty:
        return None

    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame = frame.dropna(subset=[date_col])
    if frame.empty:
        return None

    out = pd.DataFrame(index=frame[date_col].dt.normalize())
    out["open"] = pd.to_numeric(frame[open_col], errors="coerce").to_numpy()
    out["high"] = pd.to_numeric(frame[high_col], errors="coerce").to_numpy()
    out["low"] = pd.to_numeric(frame[low_col], errors="coerce").to_numpy()
    out["close"] = pd.to_numeric(frame[close_col], errors="coerce").to_numpy()
    if volume_col is not None:
        out["volume"] = pd.to_numeric(frame[volume_col], errors="coerce").fillna(0.0).to_numpy()
    else:
        out["volume"] = 0.0

    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out if not out.empty else None


def _extract_ohlcv_wide(raw: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    if raw is None or raw.empty:
        return None

    date_col = _date_column_name(list(raw.columns))
    if date_col is None:
        return None

    open_col = _find_wide_metric_col(raw, ticker, ["open", "o"])
    high_col = _find_wide_metric_col(raw, ticker, ["high", "h"])
    low_col = _find_wide_metric_col(raw, ticker, ["low", "l"])
    close_col = _find_wide_metric_col(raw, ticker, ["close", "adj close", "adjclose", "c"])
    volume_col = _find_wide_metric_col(raw, ticker, ["volume", "vol", "v"])

    if open_col is None or high_col is None or low_col is None or close_col is None:
        return None

    frame = raw[[date_col, open_col, high_col, low_col, close_col] + ([volume_col] if volume_col is not None else [])].copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame = frame.dropna(subset=[date_col])
    if frame.empty:
        return None

    out = pd.DataFrame(index=frame[date_col].dt.normalize())
    out["open"] = pd.to_numeric(frame[open_col], errors="coerce").to_numpy()
    out["high"] = pd.to_numeric(frame[high_col], errors="coerce").to_numpy()
    out["low"] = pd.to_numeric(frame[low_col], errors="coerce").to_numpy()
    out["close"] = pd.to_numeric(frame[close_col], errors="coerce").to_numpy()
    if volume_col is not None:
        out["volume"] = pd.to_numeric(frame[volume_col], errors="coerce").fillna(0.0).to_numpy()
    else:
        out["volume"] = 0.0

    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out if not out.empty else None


def _extract_ohlcv_multi(raw: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    if raw is None or raw.empty or not isinstance(raw.columns, pd.MultiIndex):
        return None

    ticker_low = str(ticker).strip().lower()

    def _match_level(val: object, targets: list[str]) -> bool:
        txt = str(val).strip().lower()
        return txt in {t.lower() for t in targets}

    date_col: tuple[object, object] | None = None
    for col in raw.columns:
        a = str(col[0]).strip().lower()
        b = str(col[1]).strip().lower()
        if a in {"date", "datetime", "timestamp", "time"} or b in {"date", "datetime", "timestamp", "time"}:
            date_col = col
            break

    # yfinance MultiIndex CSV often stores the date index as the first unnamed column.
    if date_col is None and len(raw.columns) > 0:
        first_col = raw.columns[0]
        try:
            probe = pd.to_datetime(_series_from_col(raw, first_col), errors="coerce")
            if probe.notna().any():
                date_col = first_col
        except Exception:
            pass

    def _find_metric(metric_aliases: list[str]) -> tuple[object, object] | None:
        for col in raw.columns:
            a = str(col[0]).strip().lower()
            b = str(col[1]).strip().lower()
            if (a == ticker_low and _match_level(b, metric_aliases)) or (b == ticker_low and _match_level(a, metric_aliases)):
                return col
        return None

    open_col = _find_metric(["open", "o"])
    high_col = _find_metric(["high", "h"])
    low_col = _find_metric(["low", "l"])
    close_col = _find_metric(["close", "adj close", "adjclose", "c"])
    volume_col = _find_metric(["volume", "vol", "v"])

    if date_col is None or open_col is None or high_col is None or low_col is None or close_col is None:
        return None

    date_series = _series_from_col(raw, date_col)
    idx = pd.to_datetime(date_series, errors="coerce")
    valid = ~idx.isna()
    if not valid.any():
        return None

    out = pd.DataFrame(index=idx[valid].dt.normalize())
    out["open"] = pd.to_numeric(_series_from_col(raw, open_col)[valid], errors="coerce").to_numpy()
    out["high"] = pd.to_numeric(_series_from_col(raw, high_col)[valid], errors="coerce").to_numpy()
    out["low"] = pd.to_numeric(_series_from_col(raw, low_col)[valid], errors="coerce").to_numpy()
    out["close"] = pd.to_numeric(_series_from_col(raw, close_col)[valid], errors="coerce").to_numpy()
    if volume_col is not None:
        out["volume"] = pd.to_numeric(_series_from_col(raw, volume_col)[valid], errors="coerce").fillna(0.0).to_numpy()
    else:
        out["volume"] = 0.0

    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out if not out.empty else None


def _load_shared_db_ohlcv_seed(
    ticker: str,
    *,
    start_ts: pd.Timestamp | None,
) -> tuple[pd.DataFrame | None, str | None]:
    ticker_clean = str(ticker).strip().upper()
    if not ticker_clean:
        return None, None

    out, source = load_shared_ohlcv_for_symbol(ticker_clean, start_date=start_ts)
    if out is None or out.empty:
        return None, None
    return _normalize_ohlcv(out, tail_rows=None), source or "shared_db"


def _load_local_ohlcv_seed(
    ticker: str,
    *,
    start_ts: pd.Timestamp | None,
) -> tuple[pd.DataFrame | None, str | None]:
    return None, None


def _sample_seed_from_ticker(ticker: str) -> int:
    txt = (ticker or "SAMPLE").upper()
    return 1000 + sum((i + 1) * ord(ch) for i, ch in enumerate(txt)) % 1_000_000


def _sample_base_from_ticker(ticker: str) -> float:
    seed = _sample_seed_from_ticker(ticker)
    return 40.0 + float(seed % 260)


@dataclass
class _ChartItem:
    title: str
    image_base64: str
    description: str


@dataclass
class _RunContext:
    ticker: str
    source: str
    rows: int
    first_date: str
    last_date: str
    action: str
    summary_table: pd.DataFrame
    source_table: pd.DataFrame
    charts: list[_ChartItem]
    saved_dir: str | None
    notice: str | None


@dataclass
class _CachedData:
    key: tuple[str, bool, bool, str, bool]
    df: pd.DataFrame
    source: str


def _default_form() -> dict[str, str]:
    return {
        "ticker": "AAPL",
        "action": "all",
        "output_dir": "outputs/technical_analysis",
        "auto_save": "on",
        "use_sample": "",
    }


def _base_css() -> str:
    return """
    :root {
      color-scheme: light;
    }
    body {
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: linear-gradient(180deg, #f5f7fb 0%, #eef2f7 100%);
      color: #1f2937;
    }
    .wrap {
      width: 100%;
      max-width: 1460px;
      margin: 0 auto;
      padding: 24px;
    }
    h1, h2, h3, h4 {
      margin: 0 0 12px 0;
    }
    .sub, .hint {
      color: #52606d;
    }
    .card {
      max-width: 100%;
      min-width: 0;
      overflow-x: auto;
      background: rgba(255, 255, 255, 0.92);
      border: 1px solid rgba(148, 163, 184, 0.35);
      border-radius: 16px;
      padding: 18px;
      margin: 18px 0;
      box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
    }
    .grid {
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    }
    label {
      display: block;
      font-weight: 700;
      margin-bottom: 6px;
    }
    input[type='text'], select {
      width: 100%;
      box-sizing: border-box;
      padding: 10px 12px;
      border: 1px solid #cbd5e1;
      border-radius: 10px;
      background: #fff;
    }
    button {
      padding: 10px 16px;
      border: 0;
      border-radius: 10px;
      background: #0f766e;
      color: #fff;
      font-weight: 700;
      cursor: pointer;
    }
    button:hover {
      background: #115e59;
    }
    .error, .warn {
      border-radius: 12px;
      padding: 12px 14px;
      margin: 14px 0;
    }
    .error { background: #fef2f2; color: #991b1b; }
    .warn { background: #fffbeb; color: #92400e; }
    .chart-grid {
      display: grid;
      gap: 16px;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    }
    .chart-card img {
      width: 100%;
      height: auto;
      border-radius: 12px;
      border: 1px solid rgba(148, 163, 184, 0.35);
      background: #fff;
    }
    table {
      width: max-content;
      min-width: 100%;
      border-collapse: collapse;
      font-size: 0.95rem;
    }
    th, td {
      border-bottom: 1px solid #e2e8f0;
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
      white-space: nowrap;
    }
    th {
      background: #f8fafc;
    }
    pre {
      white-space: pre-wrap;
      word-break: break-word;
      margin: 0;
    }
    """


def _env_flag(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "")).strip().lower()
    if raw == "":
        return default
    return raw in {"1", "true", "yes", "on"}


def _fig_to_png_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=135)
    plt.close(fig)
    return buf.getvalue()


def _png_bytes_to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _sample_ohlcv(rows: int, base: float = 100.0, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=rows)
    ret = rng.normal(0.0003, 0.015, size=rows)
    close = base * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0.0, 0.003, size=rows))
    spread = np.abs(rng.normal(0.006, 0.003, size=rows))
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * (1.0 - spread)
    low = np.maximum(0.01, low)
    volume = rng.integers(1_000_000, 15_000_000, size=rows)
    out = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )
    return out


def _normalize_ohlcv(frame: pd.DataFrame, *, tail_rows: int | None = LOOKBACK_ROWS) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise ValueError("No market data returned from provider")

    df = frame.copy()
    if isinstance(df.columns, pd.MultiIndex):
        if len(df.columns.levels) >= 1:
            df.columns = [str(c[0]) for c in df.columns]

    colmap = {str(c).lower(): c for c in df.columns}
    required = {
        "open": ["open", "시가"],
        "high": ["high", "고가"],
        "low": ["low", "저가"],
        "close": ["close", "adj close", "종가"],
        "volume": ["volume", "거래량"],
    }

    picked: dict[str, str] = {}
    for k, aliases in required.items():
        for a in aliases:
            if a in colmap:
                picked[k] = colmap[a]
                break

    missing = [k for k in ["open", "high", "low", "close"] if k not in picked]
    if missing:
        raise ValueError(f"OHLC columns missing from provider response: {missing}")

    data = pd.DataFrame(index=pd.to_datetime(df.index))
    for k in ["open", "high", "low", "close"]:
        data[k] = pd.to_numeric(df[picked[k]], errors="coerce")

    if "volume" in picked:
        data["volume"] = pd.to_numeric(df[picked["volume"]], errors="coerce").fillna(0.0)
    else:
        data["volume"] = 0.0

    data = data.sort_index().dropna(subset=["open", "high", "low", "close"])
    if tail_rows is not None:
        data = data.tail(tail_rows)
    if data.empty:
        raise ValueError("No valid OHLC rows after preprocessing")
    return data


def _close_series_to_ohlcv_frame(close: pd.Series) -> pd.DataFrame:
    series = pd.Series(close, dtype=float).dropna().sort_index()
    if series.empty:
        raise ValueError("No close data available from common loader")

    df = pd.DataFrame(index=pd.to_datetime(series.index))
    df["close"] = series.astype(float)
    df["open"] = df["close"].shift(1).fillna(df["close"])
    span = (df[["open", "close"]].max(axis=1) - df[["open", "close"]].min(axis=1)).abs()
    buffer = df["close"].abs() * 0.002 + span * 0.35
    df["high"] = df[["open", "close"]].max(axis=1) + buffer
    df["low"] = (df[["open", "close"]].min(axis=1) - buffer).clip(lower=0.0)
    df["volume"] = 0.0
    return _normalize_ohlcv(df[["open", "high", "low", "close", "volume"]])


def _load_stock_close_from_local_sources(
    ticker: str,
    *,
    start_ts: pd.Timestamp,
) -> tuple[pd.Series | None, str | None]:
    df, source = _load_shared_db_ohlcv_seed(ticker, start_ts=start_ts)
    if df is None or df.empty or "close" not in df.columns:
        return None, None

    close = pd.Series(df["close"], dtype=float).dropna().sort_index()
    if close.empty:
        return None, None
    return close, source


def _fetch_common_loader_ohlcv(ticker: str) -> tuple[pd.DataFrame, str]:
    ticker_clean = str(ticker).strip().upper()
    start = (pd.Timestamp.today().normalize() - pd.DateOffset(years=8)).strftime("%Y-%m-%d")

    if ticker_clean in {"BTC/USD", "BTC-USD"}:
        close, source = load_btc_close(start=start)
    elif ticker_clean in {"DX-Y.NYB", "DXY"}:
        close, source = load_dxy_series(start=start)
    elif "/" in ticker_clean:
        base_map = {"KRW/USD": 0.0008, "JPY/USD": 0.009, "USD/KRW": 1300.0, "USD/JPY": 150.0}
        close, source = load_fx_close(
            ticker_clean,
            start=start,
            fallback_base=base_map.get(ticker_clean, 1.0),
        )
    elif ticker_clean in {"US500", "HSI"}:
        seed_map = {"US500": 19, "HSI": 23}
        close, source = load_index_series(ticker_clean, start=start, seed=seed_map.get(ticker_clean, 0))
    else:
        try:
            common_prices, common_source = fetch_sp500_close_prices([ticker_clean], start)
            if common_prices is not None and not common_prices.empty and ticker_clean in common_prices.columns:
                close = pd.Series(common_prices[ticker_clean], dtype=float).dropna().sort_index()
                if not close.empty:
                    return _close_series_to_ohlcv_frame(close), f"common:{common_source}"
        except Exception:
            pass

        raise ValueError(f"No shared SQLite OHLCV data found for '{ticker_clean}'")

    if source == "fallback":
        raise ValueError(f"Common loader only produced synthetic fallback for '{ticker_clean}'")

    return _close_series_to_ohlcv_frame(close), f"common:{source}"


def _fetch_ohlcv_data(
    *,
    ticker: str,
    use_sample: bool,
    insecure_ssl: bool = False,
    ca_bundle_path: str | None = None,
    require_live_ohlcv: bool = False,
) -> tuple[pd.DataFrame, str]:
    if use_sample:
        return _sample_ohlcv(LOOKBACK_ROWS, base=_sample_base_from_ticker(ticker), seed=_sample_seed_from_ticker(ticker)), "sample"

    ticker_clean = str(ticker).strip().upper()

    try:
        shared_seed_df, shared_seed_src = _load_shared_db_ohlcv_seed(ticker_clean, start_ts=None)
        if shared_seed_df is not None:
            return shared_seed_df.tail(LOOKBACK_ROWS), shared_seed_src or "shared_db"
    except Exception:
        pass

    cached = _load_ohlcv_cache(ticker_clean)
    had_cache = not cached.empty
    latest_bday = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=1)[0]
    if had_cache and pd.Timestamp(cached.index.max()).normalize() >= pd.Timestamp(latest_bday).normalize():
        return cached.tail(LOOKBACK_ROWS), "cache"

    fetch_start: pd.Timestamp | None = None
    if had_cache:
        fetch_start = pd.Timestamp(cached.index.max()).normalize() + pd.Timedelta(days=1)

    def _merge_with_cache(fresh: pd.DataFrame, source: str) -> tuple[pd.DataFrame, str]:
        if fresh is None or fresh.empty:
            if had_cache:
                return cached.tail(LOOKBACK_ROWS), "cache"
            raise ValueError("No new OHLCV rows returned")

        merged = fresh.copy() if not had_cache else pd.concat([cached, fresh], axis=0)
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        _save_ohlcv_cache(ticker_clean, merged)

        if had_cache and fetch_start is not None:
            return merged.tail(LOOKBACK_ROWS), f"{source}+incremental"
        return merged.tail(LOOKBACK_ROWS), source

    errors: list[str] = []

    try:
        shared_seed_df, shared_seed_src = _load_shared_db_ohlcv_seed(ticker_clean, start_ts=fetch_start)
        if shared_seed_df is not None:
            return _merge_with_cache(shared_seed_df, shared_seed_src or "shared_db")
    except Exception as exc:
        errors.append(f"shared_db: {exc}")

    if not require_live_ohlcv:
        try:
            common_df, common_source = _fetch_common_loader_ohlcv(ticker_clean)
            common_full = _normalize_ohlcv(common_df, tail_rows=None)
            return _merge_with_cache(common_full, common_source)
        except Exception as exc:
            errors.append(f"common_loader: {exc}")
    else:
        errors.append("common_loader: disabled by require_live_ohlcv")

    detail = " | ".join(errors) if errors else "no provider available"
    raise ValueError(
        "Market data download failed. "
        f"Ticker='{ticker_clean}', lookback={LOOKBACK_ROWS}. Provider errors: {detail}"
    )
def _calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _calc_macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def _calc_bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower


def _chart_description(name: str, df: pd.DataFrame) -> str:
    close = df["close"].astype(float)

    if name == "moving_average":
        c = float(close.iloc[-1])
        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma60 = float(close.rolling(60).mean().iloc[-1])
        ma120 = float(close.rolling(120).mean().iloc[-1])
        rsi = float(_calc_rsi(close, 14).iloc[-1])
        bb_mid, bb_upper, bb_lower = _calc_bollinger(close, 20, 2.0)
        upper = float(bb_upper.iloc[-1])
        lower = float(bb_lower.iloc[-1])
        if np.isfinite(ma20) and np.isfinite(ma60) and np.isfinite(upper) and np.isfinite(lower):
            if c > ma20 and ma20 > ma60:
                trend = "단기/중기 상승 추세"
            elif c < ma20 and ma20 < ma60:
                trend = "단기/중기 하락 추세"
            else:
                trend = "혼조(추세 전환 가능 구간)"
            band_width = upper - lower
            band_pos = (c - lower) / band_width if band_width > 0 else np.nan
            dist_upper = ((upper - c) / c) * 100.0 if c > 0 else np.nan
            dist_lower = ((c - lower) / c) * 100.0 if c > 0 else np.nan

            leaning = "중립"
            reasons: list[str] = []
            buy_score = 0
            sell_score = 0

            if c >= ma20:
                sell_score += 1
                reasons.append("종가가 MA20 위에 있어 단기 강세 쪽")
            else:
                buy_score += 1
                reasons.append("종가가 MA20 아래에 있어 눌림 구간 쪽")

            if np.isfinite(ma120):
                if ma20 > ma60 > ma120:
                    sell_score += 1
                    reasons.append("MA20 > MA60 > MA120 정렬로 추세는 상승 우위")
                elif ma20 < ma60 < ma120:
                    buy_score += 1
                    reasons.append("MA20 < MA60 < MA120 정렬로 추세는 약세 우위")

            if np.isfinite(band_pos):
                if band_pos <= 0.25:
                    buy_score += 2
                    reasons.append("볼린저 밴드 하단에 가까워 기술적 반등 후보")
                elif band_pos <= 0.45:
                    buy_score += 1
                    reasons.append("밴드 중하단에 있어 매수 쪽으로 약간 기움")
                elif band_pos >= 0.75:
                    sell_score += 2
                    reasons.append("볼린저 밴드 상단에 가까워 단기 과열 경계")
                elif band_pos >= 0.55:
                    sell_score += 1
                    reasons.append("밴드 중상단에 있어 매도 쪽으로 약간 기움")

            if np.isfinite(rsi):
                if rsi <= 35:
                    buy_score += 1
                    reasons.append(f"RSI(14) {rsi:,.1f}로 과매도권에 근접")
                elif rsi >= 65:
                    sell_score += 1
                    reasons.append(f"RSI(14) {rsi:,.1f}로 과매수권에 근접")

            if buy_score >= sell_score + 2:
                leaning = "매수 타이밍"
            elif sell_score >= buy_score + 2:
                leaning = "매도 타이밍"
            elif buy_score > sell_score:
                leaning = "약한 매수 우위"
            elif sell_score > buy_score:
                leaning = "약한 매도 우위"

            reason_text = "; ".join(reasons[:3])
            return (
                f"종가 {c:,.2f}, MA20 {ma20:,.2f}, MA60 {ma60:,.2f}, "
                f"볼린저 상단 {upper:,.2f}, 하단 {lower:,.2f}입니다. {trend}이며 "
                f"현재는 {leaning} 쪽에 더 가깝습니다. "
                f"상단까지 {dist_upper:,.2f}%, 하단까지 {dist_lower:,.2f}% 거리이고, "
                f"판단 근거는 {reason_text}입니다."
            )
        return "이동평균 계산 구간이 부족해 추세 판별이 제한됩니다."

    if name == "candlestick":
        row = df.iloc[-1]
        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])
        kind = "양봉" if c >= o else "음봉"
        pct = ((c / o) - 1.0) * 100.0 if o != 0 else np.nan
        body = abs(c - o)
        full_range = max(h - l, 1e-8)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        body_ratio = body / full_range
        if lower_wick > body * 1.5 and c >= o:
            pattern = "아랫꼬리가 긴 반등형"
            leaning = "매수 쪽"
        elif upper_wick > body * 1.5 and c < o:
            pattern = "윗꼬리가 긴 저항형"
            leaning = "매도 쪽"
        elif body_ratio >= 0.6 and c >= o:
            pattern = "실체가 큰 상승 장악형에 가까운 양봉"
            leaning = "매수 쪽"
        elif body_ratio >= 0.6 and c < o:
            pattern = "실체가 큰 하락 압력형 음봉"
            leaning = "매도 쪽"
        else:
            pattern = "방향성이 강하지 않은 중립 봉"
            leaning = "중립"
        if np.isfinite(pct):
            return (
                f"최근 봉은 {kind}이며 시가 {o:,.2f} → 종가 {c:,.2f} ({pct:+.2f}%), "
                f"고가 {h:,.2f}, 저가 {l:,.2f}입니다. 봉 형태는 {pattern}로 읽히며 "
                f"단기적으로는 {leaning}에 더 가까운 신호입니다."
            )
        return (
            f"최근 봉은 {kind}이며 시가 {o:,.2f}, 종가 {c:,.2f}, 고가 {h:,.2f}, 저가 {l:,.2f}입니다. "
            f"봉 형태는 {pattern}로 해석되며 단기적으로는 {leaning}에 더 가깝습니다."
        )

    if name == "rsi":
        rsi = float(_calc_rsi(close, 14).iloc[-1])
        if np.isfinite(rsi):
            if rsi >= 70:
                zone = "과매수 구간"
                leaning = "매도 타이밍"
            elif rsi <= 30:
                zone = "과매도 구간"
                leaning = "매수 타이밍"
            else:
                zone = "중립 구간"
                if rsi >= 60:
                    leaning = "약한 매도 우위"
                elif rsi <= 40:
                    leaning = "약한 매수 우위"
                else:
                    leaning = "중립"
            return (
                f"RSI(14) 최신값은 {rsi:,.2f}로 {zone}에 위치합니다. "
                f"RSI 기준으로는 현재 {leaning} 쪽에 더 가깝고, "
                f"{70 - rsi:,.2f}p 위면 과매수, {rsi - 30:,.2f}p 아래면 과매도 경계입니다."
            )
        return "RSI 계산에 필요한 데이터가 부족합니다."

    if name == "macd":
        macd, signal, hist = _calc_macd(close)
        m = float(macd.iloc[-1])
        s = float(signal.iloc[-1])
        h = float(hist.iloc[-1])
        if np.isfinite(m) and np.isfinite(s) and np.isfinite(h):
            hist_phase = "유지"
            if len(hist) >= 2:
                prev_h = float(hist.iloc[-2])
                if np.isfinite(prev_h):
                    hist_phase = "확대" if abs(h) >= abs(prev_h) else "축소"
            if m > s and h > 0:
                phase = "상승 모멘텀 우세"
                leaning = "매수 쪽"
            elif m < s and h < 0:
                phase = "하락 모멘텀 우세"
                leaning = "매도 쪽"
            else:
                phase = "모멘텀 혼조"
                leaning = "중립"
            gap = abs(m - s)
            return (
                f"MACD {m:,.4f}, Signal {s:,.4f}, Hist {h:,.4f}로 {phase} 상태입니다. "
                f"시그널선과의 간격은 {gap:,.4f}이며 현재는 {leaning}에 더 가깝습니다. "
                f"다만 히스토그램이 {hist_phase}되는지 함께 보면 추세 강도 판단에 도움이 됩니다."
            )
        return "MACD 계산에 필요한 데이터가 부족합니다."

    return "차트 해석 정보가 없습니다."


def _chart_moving_average(df: pd.DataFrame, ticker: str) -> bytes:
    tail = df.tail(300).copy()
    tail["MA20"] = tail["close"].rolling(20).mean()
    tail["MA60"] = tail["close"].rolling(60).mean()
    tail["MA120"] = tail["close"].rolling(120).mean()
    _, tail["BB_Upper"], tail["BB_Lower"] = _calc_bollinger(tail["close"], 20, 2.0)

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.fill_between(
        tail.index,
        tail["BB_Lower"].to_numpy(dtype=float),
        tail["BB_Upper"].to_numpy(dtype=float),
        color="#90caf9",
        alpha=0.18,
        label="Bollinger(20,2)",
    )
    ax.plot(tail.index, tail["close"], label="Close", linewidth=1.6, color="#1565c0")
    ax.plot(tail.index, tail["MA20"], label="MA20", linewidth=1.2, color="#ef6c00")
    ax.plot(tail.index, tail["MA60"], label="MA60", linewidth=1.2, color="#2e7d32")
    ax.plot(tail.index, tail["MA120"], label="MA120", linewidth=1.2, color="#6a1b9a")
    ax.plot(tail.index, tail["BB_Upper"], linewidth=0.9, color="#42a5f5", alpha=0.9)
    ax.plot(tail.index, tail["BB_Lower"], linewidth=0.9, color="#42a5f5", alpha=0.9)
    ax.set_title(f"{ticker} - Moving Averages + Bollinger Bands (latest 300 bars)")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    return _fig_to_png_bytes(fig)


def _chart_candlestick(df: pd.DataFrame, ticker: str) -> bytes:
    tail = df.tail(180).copy()
    x = np.arange(len(tail))

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    body_width = 0.62

    o = tail["open"].to_numpy(dtype=float)
    h = tail["high"].to_numpy(dtype=float)
    l = tail["low"].to_numpy(dtype=float)
    c = tail["close"].to_numpy(dtype=float)

    median_range = float(np.nanmedian(h - l)) if len(tail) else 1.0
    min_body = max(median_range * 0.01, 1e-8)

    for i in range(len(tail)):
        up = c[i] >= o[i]
        color = "#2e7d32" if up else "#c62828"
        ax.vlines(x[i], l[i], h[i], color=color, linewidth=1.0, alpha=0.95)
        lower = min(o[i], c[i])
        height = max(abs(c[i] - o[i]), min_body)
        rect = patches.Rectangle(
            (x[i] - body_width / 2.0, lower),
            body_width,
            height,
            facecolor=color,
            edgecolor=color,
            linewidth=0.8,
            alpha=0.9,
        )
        ax.add_patch(rect)

    step = max(len(tail) // 8, 1)
    ticks = np.arange(0, len(tail), step)
    labels = [tail.index[i].strftime("%Y-%m-%d") for i in ticks]

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_xlim(-1, len(tail))
    ax.set_title(f"{ticker} - Candlestick (latest 180 bars)")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.2)
    return _fig_to_png_bytes(fig)


def _chart_rsi(df: pd.DataFrame, ticker: str) -> bytes:
    tail = df.tail(300).copy()
    rsi = _calc_rsi(tail["close"], period=14)

    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    ax.plot(tail.index, rsi, color="#6a1b9a", linewidth=1.4, label="RSI(14)")
    ax.axhline(70, color="#c62828", linestyle="--", linewidth=1.0)
    ax.axhline(30, color="#2e7d32", linestyle="--", linewidth=1.0)
    ax.set_ylim(0, 100)
    ax.set_title(f"{ticker} - RSI(14)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    return _fig_to_png_bytes(fig)


def _chart_macd(df: pd.DataFrame, ticker: str) -> bytes:
    tail = df.tail(300).copy()
    macd, signal, hist = _calc_macd(tail["close"])

    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    colors = np.where(hist >= 0, "#2e7d32", "#c62828")
    ax.bar(tail.index, hist, color=colors, alpha=0.45, width=1.0, label="Histogram")
    ax.plot(tail.index, macd, color="#1565c0", linewidth=1.3, label="MACD")
    ax.plot(tail.index, signal, color="#ef6c00", linewidth=1.2, label="Signal")
    ax.axhline(0.0, color="#555", linewidth=0.8)
    ax.set_title(f"{ticker} - MACD(12,26,9)")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    return _fig_to_png_bytes(fig)


def _summary(df: pd.DataFrame, ticker: str, source: str) -> pd.DataFrame:
    close = df["close"]
    ma20 = close.rolling(20).mean().iloc[-1]
    ma60 = close.rolling(60).mean().iloc[-1]
    ma120 = close.rolling(120).mean().iloc[-1]
    rsi14 = _calc_rsi(close, 14).iloc[-1]
    macd, signal, hist = _calc_macd(close)

    rows = [
        {"metric": "ticker", "value": ticker},
        {"metric": "source", "value": source},
        {"metric": "rows", "value": int(len(df))},
        {"metric": "first_date", "value": df.index.min().date().isoformat()},
        {"metric": "last_date", "value": df.index.max().date().isoformat()},
        {"metric": "last_close", "value": f"{float(close.iloc[-1]):,.4f}"},
        {"metric": "MA20", "value": f"{float(ma20):,.4f}" if np.isfinite(ma20) else "-"},
        {"metric": "MA60", "value": f"{float(ma60):,.4f}" if np.isfinite(ma60) else "-"},
        {"metric": "MA120", "value": f"{float(ma120):,.4f}" if np.isfinite(ma120) else "-"},
        {"metric": "RSI14", "value": f"{float(rsi14):,.4f}" if np.isfinite(rsi14) else "-"},
        {"metric": "MACD", "value": f"{float(macd.iloc[-1]):,.4f}" if np.isfinite(macd.iloc[-1]) else "-"},
        {"metric": "MACD Signal", "value": f"{float(signal.iloc[-1]):,.4f}" if np.isfinite(signal.iloc[-1]) else "-"},
        {"metric": "MACD Hist", "value": f"{float(hist.iloc[-1]):,.4f}" if np.isfinite(hist.iloc[-1]) else "-"},
    ]
    return pd.DataFrame(rows)


def _safe_table(df: pd.DataFrame, max_rows: int = 400) -> str:
    return df.head(max_rows).to_html(index=False, border=0, classes="data-table")


def _build_source_table(
    *,
    ticker: str,
    source: str,
    rows: int,
    first_date: str,
    last_date: str,
    notice: str | None,
) -> pd.DataFrame:
    items: list[tuple[str, object]] = [
        ("ticker", ticker),
        ("price_source", source),
        ("rows", rows),
        ("first_date", first_date),
        ("last_date", last_date),
    ]
    if notice:
        items.append(("fallback_notice", notice))
    return pd.DataFrame(
        [{"metric": metric, "value": str(value)} for metric, value in items]
    )

def _save_outputs(
    *,
    out_dir: Path | None,
    ticker: str,
    df: pd.DataFrame,
    charts: list[tuple[str, bytes]],
) -> str | None:
    if out_dir is None:
        return None

    ensure_writable_dir(out_dir)
    run_dir = out_dir / datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
    ensure_writable_dir(run_dir)

    safe_ticker = ticker.replace("/", "_").replace("\\", "_").replace(" ", "_")
    csv_path = run_dir / f"{safe_ticker}_ohlcv_tail_{LOOKBACK_ROWS}.csv"
    out = df.copy().reset_index().rename(columns={"index": "date"})
    out.to_csv(csv_path, index=False, encoding="utf-8-sig")

    for name, png in charts:
        fname = f"{safe_ticker}_{name}.png"
        (run_dir / fname).write_bytes(png)

    return str(run_dir.resolve())


def _run_analysis(
    *,
    form: dict[str, str],
    action: str,
    cache: _CachedData | None,
) -> tuple[_RunContext, _CachedData | None]:
    ticker = form.get("ticker", "").strip().upper()
    use_sample = form.get("use_sample", "") == "on"
    auto_save = form.get("auto_save", "on") == "on"
    output_dir = form.get("output_dir", "outputs/technical_analysis").strip() or "outputs/technical_analysis"

    if not ticker and not use_sample:
        raise ValueError("Provide ticker or enable 'Use sample prices (offline)'.")

    ticker_final = ticker or "SAMPLE"

    key = (ticker_final, use_sample)

    source_notice: str | None = None
    cache_out: _CachedData | None = cache

    if cache is not None and cache.key == key:
        df = cache.df.copy()
        source = cache.source
    else:
        df, source = _fetch_ohlcv_data(
            ticker=ticker_final,
            use_sample=use_sample,
        )
        if source == "cache":
            source_notice = "Loaded from local OHLCV cache (already up-to-date for latest business day)."
        cache_out = _CachedData(key=key, df=df.copy(), source=source)

    chart_bytes: list[tuple[str, bytes]] = []
    if action in {"ma", "all"}:
        chart_bytes.append(("moving_average", _chart_moving_average(df, ticker_final)))
    if action in {"candle", "all"}:
        chart_bytes.append(("candlestick", _chart_candlestick(df, ticker_final)))
    if action in {"rsi", "all"}:
        chart_bytes.append(("rsi", _chart_rsi(df, ticker_final)))
    if action in {"macd", "all"}:
        chart_bytes.append(("macd", _chart_macd(df, ticker_final)))

    out_base = Path(output_dir)
    saved_dir = _save_outputs(
        out_dir=out_base if auto_save else None,
        ticker=ticker_final,
        df=df,
        charts=chart_bytes,
    )

    chart_map = {
        "moving_average": "Moving Average",
        "candlestick": "Candlestick",
        "rsi": "RSI(14)",
        "macd": "MACD",
    }
    charts = [
        _ChartItem(
            title=chart_map.get(name, name),
            image_base64=_png_bytes_to_b64(data),
            description=_chart_description(name, df),
        )
        for name, data in chart_bytes
    ]

    first_date = df.index.min().date().isoformat()
    last_date = df.index.max().date().isoformat()
    ctx = _RunContext(
        ticker=ticker_final,
        source=source,
        rows=int(len(df)),
        first_date=first_date,
        last_date=last_date,
        action=action,
        summary_table=_summary(df, ticker_final, source),
        source_table=_build_source_table(
            ticker=ticker_final,
            source=source,
            rows=int(len(df)),
            first_date=first_date,
            last_date=last_date,
            notice=source_notice,
        ),
        charts=charts,
        saved_dir=saved_dir,
        notice=source_notice,
    )
    return ctx, cache_out

def _render_page(form: dict[str, str], ctx: _RunContext | None, error: str | None) -> str:
    f = _default_form()
    f.update(form)

    use_sample_checked = "checked" if f.get("use_sample", "") == "on" else ""
    auto_save_checked = "checked" if f.get("auto_save", "on") == "on" else ""

    err_html = ""
    if error:
        err_html = f"<div class='error'><pre>{html.escape(error)}</pre></div>"

    result_html = ""
    if ctx is not None:
        notice_html = f"<div class='warn'><pre>{html.escape(ctx.notice)}</pre></div>" if ctx.notice else ""
        cards = []
        for c in ctx.charts:
            cards.append(
                f"<div class='chart-card'><h4>{html.escape(c.title)}</h4><img src='data:image/png;base64,{c.image_base64}' alt='{html.escape(c.title)}' /><p class='chart-desc'>{html.escape(c.description)}</p></div>"
            )
        chart_html = "\n".join(cards) if cards else "<p class='hint'>No chart generated.</p>"

        saved_line = f"<p class='hint'><b>Saved:</b> {html.escape(ctx.saved_dir)}</p>" if ctx.saved_dir else ""
        result_html = f"""
{notice_html}
<div class='card'>
  <h3>Run Summary</h3>
  <p class='hint'>
    <b>Ticker:</b> {html.escape(ctx.ticker)}<br>
    <b>Source:</b> {html.escape(ctx.source)}<br>
    <b>Rows:</b> {ctx.rows} (fixed target={LOOKBACK_ROWS})<br>
    <b>Range:</b> {html.escape(ctx.first_date)} ~ {html.escape(ctx.last_date)}
  </p>
  {saved_line}
  {_safe_table(ctx.summary_table)}
</div>
<div class='card'>
  <h3>Data Source Metadata</h3>
  {_safe_table(ctx.source_table)}
</div>
<div class='card'>
  <h3>Charts</h3>
  <div class='chart-grid'>
    {chart_html}
  </div>
</div>
"""

    return f"""<!doctype html>
<html lang='ko'>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <title>Ticker Technical Analysis GUI</title>
  <style>
    {_base_css()}
  </style>
</head>
<body>
  <div class='wrap'>
    <h1>Ticker Technical Analysis</h1>
    <div class='sub'>Ticker input + button-based charts (Moving Average, Candlestick, RSI, MACD). Data window is fixed to latest {LOOKBACK_ROWS} rows.</div>

    {err_html}

    <form class='card' method='post' action='/run'>
      <div class='grid'>
        <div>
          <label>Ticker</label>
          <input type='text' name='ticker' value='{html.escape(f.get("ticker", "AAPL"))}' placeholder='AAPL, TSLA, MSFT, ...' />
        </div>
        <div>
          <label>Output Base Directory</label>
          <input type='text' name='output_dir' value='{html.escape(f.get("output_dir", "outputs/technical_analysis"))}' />
        </div>
      </div>
      <div class='check-row'>
        <label><input type='checkbox' name='use_sample' {use_sample_checked} /> Use sample prices (offline)</label>
        <label><input type='checkbox' name='auto_save' {auto_save_checked} /> Auto-save CSV/charts</label>
      </div>
      <div class='btn-row'>
        <button type='submit' name='action' value='ma'>Moving Average</button>
        <button type='submit' name='action' value='candle'>Candlestick</button>
        <button type='submit' name='action' value='rsi'>RSI</button>
        <button type='submit' name='action' value='macd'>MACD</button>
        <button type='submit' name='action' value='all' class='secondary'>Run All</button>
      </div>
      <p class='hint' style='margin-top:10px;'>The system always uses the most recent available date and keeps only the latest {LOOKBACK_ROWS} rows.</p>
    </form>

    {result_html}
  </div>
</body>
</html>
"""

def _send_html(handler: BaseHTTPRequestHandler, body: str, status: int = 200) -> None:
    data = body.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _redirect(handler: BaseHTTPRequestHandler, location: str = "/") -> None:
    handler.send_response(303)
    handler.send_header("Location", location)
    handler.end_headers()


def run_web_gui(host: str = "127.0.0.1", port: int = 8792) -> None:
    class Handler(BaseHTTPRequestHandler):
        state_form: dict[str, str] = _default_form()
        state_ctx: _RunContext | None = None
        state_error: str | None = None
        state_cache: _CachedData | None = None

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/":
                _send_html(self, "<h3>Not Found</h3>", status=404)
                return
            page = _render_page(self.__class__.state_form, self.__class__.state_ctx, self.__class__.state_error)
            _send_html(self, page)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/run":
                _send_html(self, "<h3>Not Found</h3>", status=404)
                return

            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8", errors="replace")
            data = parse_qs(raw)
            form = {
                "ticker": data.get("ticker", [""])[0],
                "output_dir": data.get("output_dir", ["outputs/technical_analysis"])[0],
                "use_sample": "on" if "use_sample" in data else "",
                "auto_save": "on" if "auto_save" in data else "",
                "action": data.get("action", ["all"])[0],
            }

            try:
                action = form.get("action", "all").strip().lower()
                if action not in {"ma", "candle", "rsi", "macd", "all"}:
                    action = "all"
                ctx, cache = _run_analysis(form=form, action=action, cache=self.__class__.state_cache)
                self.__class__.state_form = form
                self.__class__.state_ctx = ctx
                self.__class__.state_cache = cache
                self.__class__.state_error = None
            except Exception as exc:
                out_dir = Path(form.get("output_dir", "outputs/technical_analysis"))
                hint = security_hint(exc, output_dir=(PROJECT_ROOT / out_dir).resolve())
                raw_err = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
                self.__class__.state_form = form
                self.__class__.state_ctx = None
                self.__class__.state_error = f"{hint}\n\nRaw error: {raw_err}" if hint else raw_err

            _redirect(self, "/")

        def log_message(self, fmt: str, *args: object) -> None:  # noqa: D401
            return

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"[technical-analysis-web-gui] http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def launch_web_gui(host: str = "127.0.0.1", port: int = 8792) -> None:
    run_web_gui(host=host, port=port)


if __name__ == "__main__":
    launch_web_gui()
