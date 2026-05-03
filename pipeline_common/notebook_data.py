from __future__ import annotations

import os
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .shared_sp500_prices_sql import (
    TREASURY_DATASET,
    load_financial_market_frame,
    load_financial_market_series,
)
from .shared_sp500_prices_sql import load_shared_close_prices_for_symbols

try:
    from fredapi import Fred
except Exception:  # pragma: no cover - optional dependency
    Fred = None

try:
    import FinanceDataReader as fdr
except Exception:  # pragma: no cover - optional dependency
    fdr = None

try:
    import yfinance as yf
except Exception:  # pragma: no cover - optional dependency
    yf = None

try:
    from curl_cffi import requests as curl_requests
except Exception:  # pragma: no cover - optional dependency
    curl_requests = None


DEFAULT_SP500_COMPONENTS = pd.DataFrame(
    {
        "Symbol": ["AAPL", "MSFT", "NVDA", "AMZN", "JPM", "XOM", "UNH", "PG", "HD", "META"],
        "Sector": [
            "Information Technology",
            "Information Technology",
            "Information Technology",
            "Consumer Discretionary",
            "Financials",
            "Energy",
            "Health Care",
            "Consumer Staples",
            "Consumer Discretionary",
            "Communication Services",
        ],
    }
)


def _prefer_live_data() -> bool:
    # Default to local cached datasets (SQLite/CSV) so analysis code uses the
    # repository-managed market store unless live refresh is explicitly requested.
    raw = str(os.getenv("KEUMJ_PREFER_LIVE_DATA", "0")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _running_on_render() -> bool:
    return bool(os.getenv("RENDER") or os.getenv("RENDER_SERVICE_ID"))


def _sp500_local_csv_fallback_enabled() -> bool:
    # Render can run with only the shared SQLite file plus the small components
    # CSV. Local/dev keeps the historical CSV fallback unless explicitly off.
    return _env_bool("KEUMJ_SP500_LOCAL_CSV_FALLBACK", not _running_on_render())


def _sp500_synthetic_fallback_enabled() -> bool:
    # Synthetic data is useful for notebooks/dev demos, but hosted analysis
    # should surface missing production data instead of silently inventing it.
    return _env_bool("KEUMJ_SP500_SYNTHETIC_FALLBACK", not _running_on_render())




def _insecure_ssl_enabled() -> bool:
    raw = str(os.getenv("KEUMJ_INSECURE_SSL", "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _resolve_ca_bundle_path() -> str | None:
    ca = str(os.getenv("KEUMJ_CA_BUNDLE", "")).strip() or str(os.getenv("REQUESTS_CA_BUNDLE", "")).strip()
    return ca or None




def _business_index(start: str, min_periods: int = 1000) -> pd.DatetimeIndex:
    today = pd.Timestamp.today().normalize()
    idx = pd.bdate_range(start=start, end=today)
    if len(idx) < 120:
        idx = pd.bdate_range(end=today, periods=min_periods)
    return idx


def make_gbm_series(
    name: str,
    *,
    start: str,
    base: float,
    drift: float,
    vol: float,
    seed: int,
    min_periods: int = 1000,
) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = _business_index(start=start, min_periods=min_periods)
    rets = rng.normal(drift, vol, size=len(idx))
    levels = base * np.exp(np.cumsum(rets))
    return pd.Series(levels, index=idx, name=name)


def _read_local_series(
    csv_path: str,
    *,
    value_col_candidates: list[str],
    name: str,
    start: str | None,
) -> pd.Series | None:
    path = Path(csv_path)
    if not path.exists() or not path.is_file():
        return None

    try:
        raw = pd.read_csv(path)
    except Exception as exc:
        warnings.warn(f"Local CSV read failed ({path}): {exc}")
        return None

    if raw.empty:
        return None

    cols_lower = {str(c).lower(): c for c in raw.columns}
    date_col = cols_lower.get("date") or cols_lower.get("datetime")
    if date_col is None:
        date_col = raw.columns[0]

    value_col = None
    for cand in value_col_candidates:
        mapped = cols_lower.get(cand.lower())
        if mapped is not None:
            value_col = mapped
            break
    if value_col is None:
        value_col = raw.columns[-1]

    out = raw[[date_col, value_col]].copy()
    out.columns = ["date", "value"]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna().sort_values("date")
    if out.empty:
        return None

    s = pd.Series(out["value"].values, index=out["date"].dt.normalize(), name=name)
    s = s[~s.index.duplicated(keep="last")]
    if start:
        s = s[s.index >= pd.Timestamp(start)]
    return s if not s.empty else None


def _unique_nonempty(values: list[str | None]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        vv = str(v or "").strip()
        if not vv or vv in seen:
            continue
        seen.add(vv)
        out.append(vv)
    return out


def _load_financial_market_series_db(
    dataset: str,
    *,
    series_id: str,
    start: str,
) -> tuple[pd.Series | None, str | None]:
    series, source = load_financial_market_series(dataset, series_id=series_id, start_date=start)
    if series is None or series.empty:
        return None, None
    return series, source or "sqlite"


def _load_financial_market_frame_db(
    dataset: str,
    *,
    series_ids: list[str],
    start: str,
) -> tuple[pd.DataFrame | None, str | None]:
    frame, source = load_financial_market_frame(dataset, series_ids=series_ids, start_date=start)
    if frame is None or frame.empty:
        return None, None
    return frame, source or "sqlite"


def _normalize_yield_df(df: pd.DataFrame, series_ids: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=series_ids)

    out = df.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()]
    out = out.sort_index()
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out.groupby(out.index).last()

    missing = [s for s in series_ids if s not in out.columns]
    if missing:
        fb = make_fallback_yield_df(series_ids, start=str(out.index.min().date()) if len(out.index) else "2015-01-01")
        fb = fb.reindex(out.index)
        for sid in missing:
            out[sid] = fb[sid]

    out = out[series_ids].ffill().bfill().dropna()
    return out


def _default_yield_base(series_ids: list[str]) -> np.ndarray:
    # Covers DGS short-to-long tenor set commonly used in notebooks.
    default_map = {
        "DGS1MO": 5.4,
        "DGS3MO": 5.35,
        "DGS6MO": 5.25,
        "DGS1": 5.05,
        "DGS2": 4.75,
        "DGS3": 4.55,
        "DGS5": 4.35,
        "DGS7": 4.25,
        "DGS10": 4.20,
        "DGS20": 4.35,
        "DGS30": 4.25,
    }
    return np.array([default_map.get(s, 4.0) for s in series_ids], dtype=float)


def make_fallback_yield_df(series_ids: list[str], start: str = "2015-01-01", seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = _business_index(start=start, min_periods=2500)
    base = _default_yield_base(series_ids)

    common = np.cumsum(rng.normal(0, 0.01, size=(len(idx), 1)), axis=0)
    slope = np.cumsum(rng.normal(0, 0.005, size=(len(idx), 1)), axis=0)
    shape = np.linspace(0.8, -0.4, len(series_ids)).reshape(1, -1)
    noise = rng.normal(0, 0.03, size=(len(idx), len(series_ids)))

    data = base + common + slope @ shape + noise
    return pd.DataFrame(data, index=idx, columns=series_ids)


def load_local_yield_df(series_ids: list[str], start: str = "2015-01-01") -> tuple[pd.DataFrame | None, str | None]:
    candidates = _unique_nonempty(
        [
            os.getenv("YIELD_CURVE_CSV_PATH", ""),
            "data/treasury_yields.csv",
            "data/yield_curve.csv",
            "data/ust_yields.csv",
        ]
    )

    for csv_path in candidates:
        path = Path(csv_path)
        if not path.exists() or not path.is_file():
            continue

        try:
            raw = pd.read_csv(path)
        except Exception as exc:
            warnings.warn(f"Local yield CSV read failed ({path}): {exc}")
            continue

        if raw.empty:
            continue

        cols_lower = {str(c).lower(): c for c in raw.columns}
        date_col = cols_lower.get("date") or cols_lower.get("datetime")
        if date_col is None:
            date_col = raw.columns[0]

        # Wide format: date + DGS columns.
        if any(s in raw.columns for s in series_ids):
            df = raw.copy()
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col]).set_index(date_col)
            norm = _normalize_yield_df(df, series_ids)
            norm = norm[norm.index >= pd.Timestamp(start)]
            if not norm.empty:
                return norm, f"local_csv:{csv_path}"

        # Long format: date, series_id, value.
        sid_col = cols_lower.get("series_id") or cols_lower.get("series") or cols_lower.get("ticker")
        val_col = cols_lower.get("value") or cols_lower.get("yield") or cols_lower.get("close")
        if sid_col is not None and val_col is not None:
            df = raw[[date_col, sid_col, val_col]].copy()
            df.columns = ["date", "series_id", "value"]
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["series_id"] = df["series_id"].astype(str).str.upper()
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna()
            if not df.empty:
                piv = df.pivot_table(index="date", columns="series_id", values="value", aggfunc="last")
                norm = _normalize_yield_df(piv, series_ids)
                norm = norm[norm.index >= pd.Timestamp(start)]
                if not norm.empty:
                    return norm, f"local_csv:{csv_path}"

    return None, None


def load_yield_curve_df(series_ids: list[str], start: str = "2012-01-02") -> tuple[pd.DataFrame, str]:
    live_first = _prefer_live_data()
    sqlite_df, sqlite_source = _load_financial_market_frame_db(TREASURY_DATASET, series_ids=series_ids, start=start)
    local_df, local_source = load_local_yield_df(series_ids, start=start)
    if not live_first and sqlite_df is not None and not sqlite_df.empty:
        return sqlite_df, sqlite_source or "sqlite"
    if not live_first and local_df is not None and not local_df.empty:
        return local_df, local_source or "local_csv"

    if Fred is not None:
        try:
            fred = Fred(api_key=os.getenv("FRED_API_KEY"))
            data = {sid: fred.get_series(sid, observation_start=start) for sid in series_ids}
            df = pd.DataFrame(data)
            df = _normalize_yield_df(df, series_ids)
            df = df[df.index >= pd.Timestamp(start)]
            if not df.empty:
                return df, "fred"
        except Exception as exc:
            warnings.warn(f"FRED load failed, trying fallback: {exc}")

    if sqlite_df is not None and not sqlite_df.empty:
        return sqlite_df, sqlite_source or "sqlite"
    if local_df is not None and not local_df.empty:
        return local_df, local_source or "local_csv"

    return make_fallback_yield_df(series_ids, start=start), "fallback"

def load_dxy_series(start: str = "2012-01-02") -> tuple[pd.Series, str]:
    live_first = _prefer_live_data()
    sqlite_series, sqlite_source = _load_financial_market_series_db("dxy", series_id="DXY", start=start)
    candidates = _unique_nonempty([
        os.getenv("DXY_CSV_PATH", ""),
        "data/dxy.csv",
        "data/DXY.csv",
    ])
    if not live_first and sqlite_series is not None:
        return sqlite_series, sqlite_source or "sqlite"
    if not live_first:
        for csv_path in candidates:
            s = _read_local_series(csv_path, value_col_candidates=["close", "dxy", "value", "price"], name="DXY", start=start)
            if s is not None:
                return s, f"local_csv:{csv_path}"

    if Fred is not None:
        try:
            fred = Fred(api_key=os.getenv("FRED_API_KEY"))
            s = fred.get_series("DTWEXBGS", observation_start=start).dropna()
            if not s.empty:
                s.name = "DXY"
                return s, "fred"
        except Exception as exc:
            warnings.warn(f"FRED load failed, trying fallback: {exc}")

    if fdr is not None:
        try:
            df = fdr.DataReader("DX-Y.NYB", start)
            if isinstance(df, pd.DataFrame) and not df.empty and "Close" in df.columns:
                s = df["Close"].dropna().astype(float)
                s.name = "DXY"
                if not s.empty:
                    return s, "fdr:DX-Y.NYB"
        except Exception as exc:
            warnings.warn(f"DXY load via FDR failed, trying fallback: {exc}")

    if sqlite_series is not None:
        return sqlite_series, sqlite_source or "sqlite"
    for csv_path in candidates:
        s = _read_local_series(csv_path, value_col_candidates=["close", "dxy", "value", "price"], name="DXY", start=start)
        if s is not None:
            return s, f"local_csv:{csv_path}"

    return make_gbm_series("DXY", start=start, base=100.0, drift=0.00005, vol=0.004, seed=7, min_periods=1000), "fallback"

def load_btc_close(start: str = "2012-01-02") -> tuple[pd.Series, str]:
    live_first = _prefer_live_data()
    sqlite_series, sqlite_source = _load_financial_market_series_db("btc", series_id="BTC_Close", start=start)
    candidates = _unique_nonempty([
        os.getenv("BTC_CSV_PATH", ""),
        "data/btc_usd.csv",
        "data/BTC_USD.csv",
        "data/btc.csv",
    ])
    if not live_first and sqlite_series is not None:
        return sqlite_series, sqlite_source or "sqlite"
    if not live_first:
        for csv_path in candidates:
            s = _read_local_series(csv_path, value_col_candidates=["close", "btc_close", "value", "price"], name="BTC_Close", start=start)
            if s is not None:
                return s, f"local_csv:{csv_path}"

    if fdr is not None:
        try:
            df = fdr.DataReader("BTC/USD", start)
            if isinstance(df, pd.DataFrame) and not df.empty and "Close" in df.columns:
                s = df["Close"].dropna().astype(float)
                s.name = "BTC_Close"
                return s, "fdr:BTC/USD"
        except Exception as exc:
            warnings.warn(f"BTC load via FDR failed, trying fallback: {exc}")

    if sqlite_series is not None:
        return sqlite_series, sqlite_source or "sqlite"
    for csv_path in candidates:
        s = _read_local_series(csv_path, value_col_candidates=["close", "btc_close", "value", "price"], name="BTC_Close", start=start)
        if s is not None:
            return s, f"local_csv:{csv_path}"

    return make_gbm_series("BTC_Close", start=start, base=8000.0, drift=0.0006, vol=0.03, seed=11, min_periods=1000), "fallback"

def _symbol_variants(symbol: str, invert: bool) -> list[tuple[str, bool]]:
    out = [(symbol, invert)]
    if "/" in symbol:
        base, quote = symbol.split("/", 1)
        rev = f"{quote}/{base}"
        if rev != symbol:
            out.append((rev, not invert))
    return out


def _fx_local_candidates(symbol: str) -> list[str]:
    token = symbol.replace("/", "_").replace("-", "_").upper()
    lower = token.lower()
    return _unique_nonempty(
        [
            os.getenv(f"FX_{token}_CSV_PATH", ""),
            os.getenv(f"{token}_CSV_PATH", ""),
            f"data/fx_{lower}.csv",
            f"data/{lower}.csv",
            f"data/{token}.csv",
        ]
    )


def load_fx_close(
    symbol: str,
    *,
    invert: bool = False,
    start: str = "2012-01-02",
    fallback_base: float = 1.0,
    seed: int = 0,
) -> tuple[pd.Series, str]:
    live_first = _prefer_live_data()
    variants = _symbol_variants(symbol, invert)
    sqlite_variants: list[tuple[str, bool, str]] = []
    for sym, apply_invert in variants:
        dataset = f"fx_{sym.replace('/', '_').replace('-', '_').lower()}"
        sqlite_variants.append((dataset, apply_invert, sym))

    if not live_first:
        for dataset, apply_invert, sym in sqlite_variants:
            s, source = _load_financial_market_series_db(dataset, series_id=sym, start=start)
            if s is None:
                continue
            if apply_invert:
                s = 1.0 / s.replace(0, np.nan)
                s = s.dropna()
            if not s.empty:
                s.name = symbol
                return s, source or "sqlite"
        for sym, apply_invert in variants:
            for csv_path in _fx_local_candidates(sym):
                s = _read_local_series(csv_path, value_col_candidates=["close", "price", "value"], name=symbol, start=start)
                if s is None:
                    continue
                if apply_invert:
                    s = 1.0 / s.replace(0, np.nan)
                    s = s.dropna()
                if not s.empty:
                    s.name = symbol
                    return s, f"local_csv:{csv_path}"

    if fdr is not None:
        for sym, apply_invert in variants:
            try:
                df = fdr.DataReader(sym, start)
                if isinstance(df, pd.DataFrame) and not df.empty and "Close" in df.columns:
                    s = df["Close"].dropna().astype(float)
                    if apply_invert:
                        s = 1.0 / s.replace(0, np.nan)
                        s = s.dropna()
                    if not s.empty:
                        s.name = symbol
                        return s, f"fdr:{sym}"
            except Exception:
                pass

    for dataset, apply_invert, sym in sqlite_variants:
        s, source = _load_financial_market_series_db(dataset, series_id=sym, start=start)
        if s is None:
            continue
        if apply_invert:
            s = 1.0 / s.replace(0, np.nan)
            s = s.dropna()
        if not s.empty:
            s.name = symbol
            return s, source or "sqlite"

    for sym, apply_invert in variants:
        for csv_path in _fx_local_candidates(sym):
            s = _read_local_series(csv_path, value_col_candidates=["close", "price", "value"], name=symbol, start=start)
            if s is None:
                continue
            if apply_invert:
                s = 1.0 / s.replace(0, np.nan)
                s = s.dropna()
            if not s.empty:
                s.name = symbol
                return s, f"local_csv:{csv_path}"

    warnings.warn(f"Load failed for {symbol}; using fallback synthetic series.")
    return make_gbm_series(symbol, start=start, base=fallback_base, drift=0.00003, vol=0.003, seed=seed, min_periods=1400), "fallback"

def _index_local_candidates(symbol: str) -> list[str]:
    token = symbol.replace("/", "_").replace("-", "_").upper()
    lower = token.lower()
    return _unique_nonempty(
        [
            os.getenv(f"INDEX_{token}_CSV_PATH", ""),
            os.getenv(f"{token}_CSV_PATH", ""),
            f"data/index_{lower}.csv",
            f"data/{lower}.csv",
            f"data/{token}.csv",
        ]
    )


def load_index_series(symbol: str, start: str = "2012-01-02", seed: int = 0) -> tuple[pd.Series, str]:
    live_first = _prefer_live_data()
    sqlite_series, sqlite_source = _load_financial_market_series_db(
        f"index_{symbol.replace('/', '_').replace('-', '_').lower()}",
        series_id=str(symbol).strip().upper(),
        start=start,
    )
    candidates = _index_local_candidates(symbol)
    if not live_first and sqlite_series is not None:
        sqlite_series.name = symbol
        return sqlite_series, sqlite_source or "sqlite"
    if not live_first:
        for csv_path in candidates:
            s = _read_local_series(csv_path, value_col_candidates=["close", "adj close", "price", "value"], name=symbol, start=start)
            if s is not None:
                return s, f"local_csv:{csv_path}"

    if fdr is not None:
        try:
            df = fdr.DataReader(symbol, start)
            if isinstance(df, pd.DataFrame) and not df.empty and "Close" in df.columns:
                return df["Close"].dropna().astype(float), f"fdr:{symbol}"
        except Exception as exc:
            warnings.warn(f"Index load failed for {symbol}: {exc}")

    if sqlite_series is not None:
        sqlite_series.name = symbol
        return sqlite_series, sqlite_source or "sqlite"
    for csv_path in candidates:
        s = _read_local_series(csv_path, value_col_candidates=["close", "adj close", "price", "value"], name=symbol, start=start)
        if s is not None:
            return s, f"local_csv:{csv_path}"

    base = 4500.0 if symbol == "US500" else 22000.0
    return make_gbm_series(symbol, start=start, base=base, drift=0.0002, vol=0.01, seed=seed, min_periods=1500), "fallback"

def _load_components_local(max_symbols: int) -> tuple[pd.DataFrame | None, str | None]:
    candidates = _unique_nonempty([
        os.getenv("SP500_COMPONENTS_CSV_PATH", ""),
        "data/sp500_components_full.csv",
        "data/sp500_components.csv",
    ])
    for csv_path in candidates:
        path = Path(csv_path)
        if not path.exists() or not path.is_file():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        cols = {str(c).lower(): c for c in df.columns}
        sym = cols.get("symbol")
        sec = cols.get("sector")
        if sym is None or sec is None:
            continue
        out = df[[sym, sec]].copy()
        out.columns = ["Symbol", "Sector"]
        out = out.dropna().drop_duplicates("Symbol")
        if not out.empty:
            return out.head(max_symbols).reset_index(drop=True), f"local_csv:{csv_path}"
    return None, None


def load_sp500_components(max_symbols: int = 80) -> tuple[pd.DataFrame, str]:
    live_first = _prefer_live_data()
    local_df, local_src = _load_components_local(max_symbols)
    if not live_first and local_df is not None:
        return local_df, local_src or "local_csv"

    if fdr is not None:
        try:
            comps = fdr.StockListing("S&P500")[["Symbol", "Sector"]].dropna().drop_duplicates("Symbol")
            return comps.head(max_symbols).reset_index(drop=True), "fdr"
        except Exception as exc:
            warnings.warn(f"StockListing failed, fallback used: {exc}")

    if local_df is not None:
        return local_df, local_src or "local_csv"

    return DEFAULT_SP500_COMPONENTS.copy().head(max_symbols).reset_index(drop=True), "fallback"


def _normalize_symbols(symbols: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        clean = str(symbol).strip().upper()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        out.append(clean)
    return out


def _safe_symbol_filename(symbol: str) -> str:
    return re.sub(r"[^A-Z0-9._-]+", "_", str(symbol).strip().upper())


def _sp500_shared_db_root() -> Path:
    return Path(os.getenv("KEUMJ_SP500_DB_DIR", "data/sp500_shared_db"))


def _sp500_shared_prices_dir() -> Path:
    return _sp500_shared_db_root() / "prices"


def _sp500_shared_start_date() -> pd.Timestamp:
    raw = str(os.getenv("KEUMJ_SP500_DB_START_DATE", "2015-12-31")).strip() or "2015-12-31"
    try:
        return pd.Timestamp(raw).normalize()
    except Exception:
        return pd.Timestamp("2015-12-31")


def _sp500_shared_symbol_path(symbol: str) -> Path:
    return _sp500_shared_prices_dir() / f"{_safe_symbol_filename(symbol)}.csv"


def _load_prices_shared_db(symbols: list[str], start_date: str) -> tuple[pd.DataFrame | None, str | None]:
    sqlite_df, sqlite_src = load_shared_close_prices_for_symbols(symbols, start_date=start_date)
    if sqlite_df is not None and not sqlite_df.empty:
        return sqlite_df, sqlite_src or "sqlite"
    return None, None




def _load_prices_wide_local(symbols: list[str], start_date: str) -> tuple[pd.DataFrame | None, str | None]:
    candidates = _unique_nonempty([
        os.getenv("SP500_METRICS_CSV_PATH", ""),
        "data/sp500_all_metrics_prices.csv",
    ])
    for csv_path in candidates:
        path = Path(csv_path)
        if not path.exists() or not path.is_file():
            continue
        try:
            raw = pd.read_csv(path)
        except Exception:
            continue
        if raw.empty:
            continue
        cols = {str(c).lower(): c for c in raw.columns}
        date_col = cols.get("date") or cols.get("datetime")
        if date_col is None:
            date_col = raw.columns[0]

        out = raw.copy()
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        out = out.dropna(subset=[date_col]).set_index(date_col).sort_index()
        rename_map = {f"{symbol}_Close": symbol for symbol in symbols if f"{symbol}_Close" in out.columns}
        keep = [s for s in symbols if s in out.columns]
        if rename_map:
            out = out[list(rename_map)].rename(columns=rename_map)
            keep = [s for s in symbols if s in out.columns]
        if not keep:
            continue
        out = out[keep].apply(pd.to_numeric, errors="coerce")
        out = out[out.index >= pd.Timestamp(start_date)].dropna(how="all")
        if not out.empty:
            return out, f"local_csv:{csv_path}"
    return None, None


def _load_prices_panel_local(symbols: list[str], start_date: str) -> tuple[pd.DataFrame | None, str | None]:
    base_dir = Path(os.getenv("SP500_PRICES_PANEL_DIR", "data/sp500_prices"))
    if not base_dir.exists() or not base_dir.is_dir():
        return None, None

    series: list[pd.Series] = []
    for s in symbols:
        csv_path = str(base_dir / f"{s}.csv")
        ss = _read_local_series(csv_path, value_col_candidates=["close", "adj close", "price", "value"], name=s, start=start_date)
        if ss is not None:
            series.append(ss.rename(s))
    if not series:
        return None, None

    out = pd.concat(series, axis=1).sort_index().dropna(how="all")
    if out.empty:
        return None, None
    return out, "local_csv:data/sp500_prices/*.csv"




def fetch_sp500_close_prices(symbols: list[str], start_date: str) -> tuple[pd.DataFrame, str]:
    normalized_symbols = _normalize_symbols(symbols)
    if not normalized_symbols:
        return pd.DataFrame(), "empty"

    shared_cached_df, shared_cached_src = _load_prices_shared_db(normalized_symbols, start_date)
    if shared_cached_df is not None:
        return shared_cached_df, shared_cached_src or "shared_db"

    if _sp500_local_csv_fallback_enabled():
        local_wide_df, local_wide_src = _load_prices_wide_local(normalized_symbols, start_date)
        if local_wide_df is not None:
            return local_wide_df, local_wide_src or "local_csv"

        local_panel_df, local_panel_src = _load_prices_panel_local(normalized_symbols, start_date)
        if local_panel_df is not None:
            return local_panel_df, local_panel_src or "local_csv"

    if not _sp500_synthetic_fallback_enabled():
        return pd.DataFrame(columns=normalized_symbols), "unavailable:shared_sqlite_missing_or_empty"

    rng = np.random.default_rng(123)
    idx = pd.bdate_range(start=start_date, end=pd.Timestamp.today().normalize())
    shocks = rng.normal(0.0003, 0.012, size=(len(idx), len(normalized_symbols)))
    levels = 100.0 * np.exp(np.cumsum(shocks, axis=0))
    return pd.DataFrame(levels, index=idx, columns=normalized_symbols), "fallback"


def load_krx_components(max_symbols: int | None = None, *, components_csv: str | os.PathLike[str] | None = None) -> tuple[pd.DataFrame, str]:
    candidates = _unique_nonempty([
        str(components_csv or ""),
        os.getenv("KRX_COMPONENTS_CSV_PATH", ""),
        "data/krx_components_full.csv",
        "data/krx_components.csv",
    ])
    for csv_path in candidates:
        path = Path(csv_path)
        if not path.exists() or not path.is_file():
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        cols = {str(col).strip().lower(): col for col in frame.columns}
        symbol_col = cols.get("symbol")
        if symbol_col is None:
            continue
        out = frame.copy()
        out["Symbol"] = out[symbol_col].astype(str).str.strip().str.upper()
        sector_col = cols.get("sector")
        if sector_col is None:
            out["Sector"] = "Unknown"
        else:
            out["Sector"] = out[sector_col].astype(str).str.strip().replace({"": "Unknown", "nan": "Unknown"})
        if max_symbols is not None and int(max_symbols) > 0:
            out = out.head(int(max_symbols)).reset_index(drop=True)
        return out, f"csv:{path.as_posix()}"
    return load_sp500_components(max_symbols=max_symbols or 0)


def fetch_krx_close_prices(symbols: list[str], start_date: str) -> tuple[pd.DataFrame, str]:
    from .shared_krx_prices_sql import load_shared_close_prices_for_symbols

    normalized_symbols = _normalize_symbols(symbols)
    if not normalized_symbols:
        return pd.DataFrame(), "empty"
    shared_cached_df, shared_cached_src = load_shared_close_prices_for_symbols(normalized_symbols, start_date=start_date)
    if shared_cached_df is not None and not shared_cached_df.empty:
        return shared_cached_df, shared_cached_src or "krx_shared_db"
    return fetch_sp500_close_prices(symbols, start_date)


__all__ = [
    "make_gbm_series",
    "make_fallback_yield_df",
    "load_yield_curve_df",
    "load_dxy_series",
    "load_btc_close",
    "load_fx_close",
    "load_index_series",
    "load_sp500_components",
    "fetch_sp500_close_prices",
]
