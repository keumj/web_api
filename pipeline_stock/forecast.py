from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import re

import requests
import numpy as np
import pandas as pd

try:
    from curl_cffi import requests as curl_requests
except Exception:  # pragma: no cover - optional dependency
    curl_requests = None

try:
    import yfinance as yf
except Exception:  # pragma: no cover - optional dependency
    yf = None
else:
    try:
        from yfinance import set_tz_cache_location

        _YF_CACHE_DIR = Path(os.getenv("KEUMJ_YFINANCE_CACHE_DIR", "data/.yfinance_cache"))
        _YF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        set_tz_cache_location(str(_YF_CACHE_DIR))
    except Exception:
        pass

from pipeline_common.notebook_data import fetch_sp500_close_prices
from pipeline_common.shared_sp500_prices_sql import load_shared_market_caps_for_symbols

from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _running_on_render() -> bool:
    return bool(os.getenv("RENDER") or os.getenv("RENDER_SERVICE_ID"))


def _forecast_light_mode() -> bool:
    return _env_bool("KEUMJM_FORECAST_LIGHT_MODE", _running_on_render())


def _relative_features_enabled() -> bool:
    return _env_bool("KEUMJ_STOCK_RELATIVE_FEATURES", not _forecast_light_mode())


def _forecast_fast_linear_only() -> bool:
    return _env_bool("KEUMJM_FORECAST_FAST_LINEAR_ONLY", _running_on_render())


@dataclass
class StockForecastResult:
    summary: pd.DataFrame
    model_scores: pd.DataFrame
    close_history: pd.DataFrame
    feature_importance: pd.DataFrame
    direction_scores: pd.DataFrame = field(default_factory=pd.DataFrame)
    regime_snapshot: pd.DataFrame = field(default_factory=pd.DataFrame)
    price_source: str = ""


def _normalize_ticker_token(value: str) -> str:
    return re.sub(r"[^A-Z0-9._-]+", "_", str(value or "").strip().upper())


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


def _clean_return_series(series: pd.Series) -> pd.Series:
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


def _weighted_daily_returns(price_df: pd.DataFrame, market_cap_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
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


def _rolling_beta_series(asset_daily: pd.Series, benchmark_daily: pd.Series, *, window: int) -> pd.Series:
    asset = _clean_return_series(asset_daily).rename("asset")
    benchmark = _clean_return_series(benchmark_daily).rename("benchmark")
    joined = pd.concat([asset, benchmark], axis=1).dropna()
    if len(joined) < window:
        return pd.Series(dtype=float)
    cov = joined["asset"].rolling(window).cov(joined["benchmark"])
    var = joined["benchmark"].rolling(window).var()
    beta = cov / var.replace(0.0, np.nan)
    return beta.dropna()


def _rolling_correlation_series(asset_daily: pd.Series, benchmark_daily: pd.Series, *, window: int) -> pd.Series:
    asset = _clean_return_series(asset_daily).rename("asset")
    benchmark = _clean_return_series(benchmark_daily).rename("benchmark")
    joined = pd.concat([asset, benchmark], axis=1).dropna()
    if len(joined) < window:
        return pd.Series(dtype=float)
    corr = joined["asset"].rolling(window).corr(joined["benchmark"])
    return corr.dropna()


def _rolling_comp_return(daily_returns: pd.Series, window: int) -> pd.Series:
    clean = _clean_return_series(daily_returns)
    if clean.empty:
        return pd.Series(dtype=float)
    return ((1.0 + clean).rolling(window).apply(np.prod, raw=True) - 1.0).dropna()


def _series_regime_labels(feature_row: pd.Series) -> dict[str, str]:
    sma20 = float(feature_row.get("sma_ratio_20", np.nan))
    sma60 = float(feature_row.get("sma_ratio_60", np.nan))
    mom20 = float(feature_row.get("mom_20", np.nan))
    beta60 = float(feature_row.get("beta_market_60", np.nan))
    rel_vol = float(feature_row.get("rel_vol_market_20", np.nan))

    if np.isfinite(sma20) and np.isfinite(sma60) and np.isfinite(mom20):
        if sma20 > 0.01 and sma60 > 0.0 and mom20 > 0.0:
            trend = "상승 추세 (uptrend)"
        elif sma20 < -0.01 and sma60 < 0.0 and mom20 < 0.0:
            trend = "하락 추세 (downtrend)"
        else:
            trend = "혼합 추세 (mixed trend)"
    else:
        trend = "판별 보류 (insufficient history)"

    if np.isfinite(rel_vol):
        if rel_vol >= 0.2:
            volatility = "고변동성 (high volatility)"
        elif rel_vol <= -0.12:
            volatility = "저변동성 (calm volatility)"
        else:
            volatility = "중립 변동성 (normal volatility)"
    else:
        volatility = "판별 보류 (insufficient history)"

    if np.isfinite(beta60):
        if beta60 >= 1.15:
            beta_regime = "공격적 베타 (high beta)"
        elif beta60 <= 0.85:
            beta_regime = "방어적 베타 (defensive beta)"
        else:
            beta_regime = "시장유사 베타 (market-like beta)"
    else:
        beta_regime = "판별 보류 (insufficient history)"

    if "상승" in trend and "고변동성" in volatility:
        overall = "공격적 상승 국면 (high-beta rally)"
    elif "상승" in trend and "저변동성" in volatility:
        overall = "안정 상승 국면 (stable uptrend)"
    elif "하락" in trend and "고변동성" in volatility:
        overall = "스트레스 국면 (risk-off stress)"
    elif "방어적" in beta_regime and "저변동성" in volatility:
        overall = "방어 안정 국면 (defensive calm)"
    else:
        overall = "혼합 국면 (mixed regime)"

    return {
        "trend_regime": trend,
        "volatility_regime": volatility,
        "beta_regime": beta_regime,
        "overall_regime": overall,
    }


def classify_regime_from_feature_row(feature_row: pd.Series) -> dict[str, str]:
    return _series_regime_labels(feature_row)


def _relative_feature_frame(ticker: str, close: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    ticker_clean = str(ticker or "").strip().upper()
    if not ticker_clean or close.empty:
        return pd.DataFrame(index=close.index), pd.DataFrame(columns=["metric", "value"])
    if not _relative_features_enabled():
        return pd.DataFrame(index=close.index), pd.DataFrame(
            [{"metric": "relative_feature_status", "value": "disabled: light forecast mode"}]
        )

    try:
        components, components_source = _load_sp500_components_full()
    except Exception as exc:
        return pd.DataFrame(index=close.index), pd.DataFrame(
            [{"metric": "relative_feature_status", "value": f"disabled: {exc}"}]
        )

    if ticker_clean not in set(components["Symbol"]):
        return pd.DataFrame(index=close.index), pd.DataFrame(
            [
                {"metric": "relative_feature_status", "value": "disabled: ticker outside shared S&P 500 universe"},
                {"metric": "components_source", "value": components_source},
            ]
        )

    sector = str(components.loc[components["Symbol"] == ticker_clean, "Sector"].iloc[0])
    all_symbols = components["Symbol"].tolist()
    start_date = (pd.Timestamp(close.index.min()).normalize() - pd.DateOffset(months=9)).strftime("%Y-%m-%d")
    end_date = pd.Timestamp(close.index.max()).normalize().strftime("%Y-%m-%d")

    try:
        prices, price_source = fetch_sp500_close_prices(all_symbols, start_date=start_date)
        market_caps, market_cap_source = load_shared_market_caps_for_symbols(
            all_symbols,
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as exc:
        return pd.DataFrame(index=close.index), pd.DataFrame(
            [
                {"metric": "relative_feature_status", "value": f"disabled: {exc}"},
                {"metric": "components_source", "value": components_source},
            ]
        )

    if prices is None or prices.empty:
        return pd.DataFrame(index=close.index), pd.DataFrame(
            [{"metric": "relative_feature_status", "value": "disabled: shared prices unavailable"}]
        )

    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index, errors="coerce")
    prices = prices[~prices.index.isna()].sort_index()
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.dropna(how="all")
    prices = prices.loc[:, [symbol for symbol in all_symbols if symbol in prices.columns]]
    if prices.empty or ticker_clean not in prices.columns:
        return pd.DataFrame(index=close.index), pd.DataFrame(
            [{"metric": "relative_feature_status", "value": "disabled: ticker missing from shared price frame"}]
        )

    ticker_close_shared = _clean_close_series(prices[ticker_clean]).reindex(close.index).ffill()
    ticker_daily = _clean_return_series(ticker_close_shared.pct_change(fill_method=None))

    sector_symbols = components.loc[components["Sector"] == sector, "Symbol"].tolist()
    sector_weight_symbols = [symbol for symbol in sector_symbols if symbol in prices.columns]
    market_weight_symbols = [symbol for symbol in all_symbols if symbol in prices.columns]

    market_caps_clean = _clean_market_cap_frame(market_caps if market_caps is not None else pd.DataFrame())
    use_cap_weights = not market_caps_clean.empty

    if use_cap_weights:
        sector_weight_symbols = [symbol for symbol in sector_weight_symbols if symbol in market_caps_clean.columns]
        market_weight_symbols = [symbol for symbol in market_weight_symbols if symbol in market_caps_clean.columns]

    if not sector_weight_symbols or not market_weight_symbols:
        return pd.DataFrame(index=close.index), pd.DataFrame(
            [{"metric": "relative_feature_status", "value": "disabled: not enough sector/market constituents"}]
        )

    if use_cap_weights:
        sector_daily, _ = _weighted_daily_returns(prices[sector_weight_symbols], market_caps_clean[sector_weight_symbols])
        market_daily, _ = _weighted_daily_returns(prices[market_weight_symbols], market_caps_clean[market_weight_symbols])
        weighting_method = "cap_weighted"
        market_cap_source_value = market_cap_source or "sqlite_market_caps"
    else:
        sector_daily = prices[sector_weight_symbols].pct_change(fill_method=None).mean(axis=1, skipna=True).dropna()
        market_daily = prices[market_weight_symbols].pct_change(fill_method=None).mean(axis=1, skipna=True).dropna()
        weighting_method = "equal_weighted_fallback"
        market_cap_source_value = "unavailable"

    market_daily = _clean_return_series(market_daily).reindex(close.index)
    sector_daily = _clean_return_series(sector_daily).reindex(close.index)
    aligned_close = _clean_close_series(close).reindex(close.index).ffill()
    aligned_daily = _clean_return_series(aligned_close.pct_change(fill_method=None)).reindex(close.index)

    asset_vol_20 = aligned_daily.rolling(20).std()
    market_vol_20 = market_daily.rolling(20).std()
    sector_vol_20 = sector_daily.rolling(20).std()

    beta_market_20 = _rolling_beta_series(aligned_daily, market_daily, window=20).reindex(close.index)
    beta_market_60 = _rolling_beta_series(aligned_daily, market_daily, window=60).reindex(close.index)
    beta_sector_20 = _rolling_beta_series(aligned_daily, sector_daily, window=20).reindex(close.index)
    beta_sector_60 = _rolling_beta_series(aligned_daily, sector_daily, window=60).reindex(close.index)
    corr_market_20 = _rolling_correlation_series(aligned_daily, market_daily, window=20).reindex(close.index)
    corr_market_60 = _rolling_correlation_series(aligned_daily, market_daily, window=60).reindex(close.index)
    corr_sector_20 = _rolling_correlation_series(aligned_daily, sector_daily, window=20).reindex(close.index)
    corr_sector_60 = _rolling_correlation_series(aligned_daily, sector_daily, window=60).reindex(close.index)

    residual_market_daily = aligned_daily - (beta_market_20 * market_daily)
    residual_sector_daily = aligned_daily - (beta_sector_20 * sector_daily)

    feature_df = pd.DataFrame(index=close.index)
    feature_df["market_ret_1"] = market_daily.shift(1)
    feature_df["market_ret_5"] = _rolling_comp_return(market_daily, 5).reindex(close.index).shift(1)
    feature_df["market_ret_20"] = _rolling_comp_return(market_daily, 20).reindex(close.index).shift(1)
    feature_df["sector_ret_1"] = sector_daily.shift(1)
    feature_df["sector_ret_5"] = _rolling_comp_return(sector_daily, 5).reindex(close.index).shift(1)
    feature_df["sector_ret_20"] = _rolling_comp_return(sector_daily, 20).reindex(close.index).shift(1)
    feature_df["rel_mom_market_5"] = (aligned_close.pct_change(5) - _rolling_comp_return(market_daily, 5).reindex(close.index)).shift(1)
    feature_df["rel_mom_market_20"] = (aligned_close.pct_change(20) - _rolling_comp_return(market_daily, 20).reindex(close.index)).shift(1)
    feature_df["rel_mom_sector_5"] = (aligned_close.pct_change(5) - _rolling_comp_return(sector_daily, 5).reindex(close.index)).shift(1)
    feature_df["rel_mom_sector_20"] = (aligned_close.pct_change(20) - _rolling_comp_return(sector_daily, 20).reindex(close.index)).shift(1)
    feature_df["beta_market_20"] = beta_market_20.shift(1)
    feature_df["beta_market_60"] = beta_market_60.shift(1)
    feature_df["beta_sector_20"] = beta_sector_20.shift(1)
    feature_df["beta_sector_60"] = beta_sector_60.shift(1)
    feature_df["corr_market_20"] = corr_market_20.shift(1)
    feature_df["corr_market_60"] = corr_market_60.shift(1)
    feature_df["corr_sector_20"] = corr_sector_20.shift(1)
    feature_df["corr_sector_60"] = corr_sector_60.shift(1)
    feature_df["market_vol_20"] = market_vol_20.shift(1)
    feature_df["sector_vol_20"] = sector_vol_20.shift(1)
    feature_df["rel_vol_market_20"] = ((asset_vol_20 / market_vol_20.replace(0.0, np.nan)) - 1.0).shift(1)
    feature_df["rel_vol_sector_20"] = ((asset_vol_20 / sector_vol_20.replace(0.0, np.nan)) - 1.0).shift(1)
    feature_df["residual_market_5"] = residual_market_daily.rolling(5).sum().shift(1)
    feature_df["residual_market_20"] = residual_market_daily.rolling(20).sum().shift(1)
    feature_df["residual_sector_5"] = residual_sector_daily.rolling(5).sum().shift(1)
    feature_df["residual_sector_20"] = residual_sector_daily.rolling(20).sum().shift(1)

    snapshot_row = feature_df.replace([np.inf, -np.inf], np.nan).dropna().tail(1)
    if snapshot_row.empty:
        regime_snapshot = pd.DataFrame(
            [
                {"metric": "relative_feature_status", "value": "enabled_but_latest_snapshot_invalid"},
                {"metric": "sector", "value": sector},
                {"metric": "components_source", "value": components_source},
                {"metric": "price_source", "value": price_source},
                {"metric": "market_cap_source", "value": market_cap_source_value},
                {"metric": "weighting_method", "value": weighting_method},
            ]
        )
    else:
        labels = _series_regime_labels(snapshot_row.iloc[0])
        regime_snapshot = pd.DataFrame(
            [
                {"metric": "relative_feature_status", "value": "enabled"},
                {"metric": "sector", "value": sector},
                {"metric": "components_source", "value": components_source},
                {"metric": "price_source", "value": price_source},
                {"metric": "market_cap_source", "value": market_cap_source_value},
                {"metric": "weighting_method", "value": weighting_method},
                {"metric": "beta_market_60", "value": f"{float(snapshot_row.iloc[0]['beta_market_60']):.3f}"},
                {"metric": "beta_sector_60", "value": f"{float(snapshot_row.iloc[0]['beta_sector_60']):.3f}"},
                {"metric": "rel_mom_market_20", "value": f"{float(snapshot_row.iloc[0]['rel_mom_market_20']) * 100.0:.2f}%"},
                {"metric": "residual_market_20", "value": f"{float(snapshot_row.iloc[0]['residual_market_20']) * 100.0:.2f}%"},
                {"metric": "overall_regime", "value": labels["overall_regime"]},
            ]
        )

    return feature_df.replace([np.inf, -np.inf], np.nan), regime_snapshot


def _build_yfinance_session(*, ca_bundle: str | None = None, insecure_ssl: bool = False):
    if curl_requests is not None:
        session = curl_requests.Session()
    else:
        session = requests.Session()
        session.trust_env = True
    if ca_bundle:
        session.verify = ca_bundle
    elif insecure_ssl:
        session.verify = False
    return session


def _extract_close_long(raw: pd.DataFrame, ticker: str) -> pd.Series | None:
    if raw is None or raw.empty:
        return None

    cols = {str(c).strip().lower(): c for c in raw.columns}
    date_col = cols.get("date") or cols.get("datetime") or cols.get("timestamp") or raw.columns[0]
    symbol_col = cols.get("symbol") or cols.get("ticker") or cols.get("code") or cols.get("asset")
    close_col = cols.get("close") or cols.get("adj close") or cols.get("adjclose") or cols.get("c")
    if symbol_col is None or close_col is None:
        return None

    frame = raw[[date_col, symbol_col, close_col]].copy()
    frame.columns = ["date", "symbol", "close"]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["date", "close"])
    if frame.empty:
        return None

    ticker_norm = _normalize_ticker_token(ticker)
    symbols = frame["symbol"].astype(str).map(_normalize_ticker_token)
    frame = frame[symbols == ticker_norm]
    if frame.empty:
        return None

    close = pd.Series(frame["close"].values, index=frame["date"].dt.normalize(), name="close")
    close = close[~close.index.duplicated(keep="last")].sort_index()
    return close if not close.empty else None


def _extract_close_wide(raw: pd.DataFrame, ticker: str) -> pd.Series | None:
    if raw is None or raw.empty:
        return None

    cols = list(raw.columns)
    date_col = None
    for col in cols:
        low = str(col).strip().lower()
        if low in {"date", "datetime", "timestamp", "time"}:
            date_col = col
            break
    if date_col is None:
        date_col = cols[0]

    ticker_low = str(ticker).strip().lower()
    ticker_norm = _normalize_ticker_token(ticker_low)
    sep = r"[\s_\-\./:]*"

    close_col = None
    for col in cols:
        low = str(col).strip().lower()
        if re.match(fr"^{re.escape(ticker_low)}{sep}(close|adj close|adjclose|c)$", low):
            close_col = col
            break
        if re.match(fr"^(close|adj close|adjclose|c){sep}{re.escape(ticker_low)}$", low):
            close_col = col
            break

    if close_col is None:
        for col in cols:
            low = str(col).strip().lower()
            norm = _normalize_ticker_token(low)
            if ticker_norm and ticker_norm not in norm:
                continue
            if any(alias in norm for alias in ["CLOSE", "ADJCLOSE", "C"]):
                close_col = col
                break

    if close_col is None:
        return None

    frame = raw[[date_col, close_col]].copy()
    frame.columns = ["date", "close"]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["date", "close"])
    if frame.empty:
        return None

    close = pd.Series(frame["close"].values, index=frame["date"].dt.normalize(), name="close")
    close = close[~close.index.duplicated(keep="last")].sort_index()
    return close if not close.empty else None


def _extract_close_multi(raw: pd.DataFrame, ticker: str) -> pd.Series | None:
    if raw is None or raw.empty or not isinstance(raw.columns, pd.MultiIndex):
        return None

    ticker_low = str(ticker).strip().lower()

    def _find_metric(metric_aliases: list[str]) -> tuple[object, object] | None:
        for col in raw.columns:
            a = str(col[0]).strip().lower()
            b = str(col[1]).strip().lower()
            if (a == ticker_low and b in {alias.lower() for alias in metric_aliases}) or (b == ticker_low and a in {alias.lower() for alias in metric_aliases}):
                return col
        return None

    close_col = _find_metric(["close", "adj close", "adjclose", "c"])
    if close_col is None:
        return None

    date_col = None
    for col in raw.columns:
        a = str(col[0]).strip().lower()
        b = str(col[1]).strip().lower()
        if a in {"date", "datetime", "timestamp", "time"} or b in {"date", "datetime", "timestamp", "time"}:
            date_col = col
            break
    if date_col is None:
        date_col = raw.columns[0]

    date_series = raw[date_col]
    if isinstance(date_series, pd.DataFrame):
        date_series = date_series.iloc[:, 0]
    close_series = raw[close_col]
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.iloc[:, 0]

    idx = pd.to_datetime(date_series, errors="coerce")
    close = pd.to_numeric(close_series, errors="coerce")
    valid = ~idx.isna() & ~close.isna()
    if not valid.any():
        return None

    out = pd.Series(close[valid].to_numpy(), index=idx[valid].dt.normalize(), name="close")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out if not out.empty else None


def _load_close_from_local_sources(
    ticker: str,
    *,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> tuple[pd.Series | None, str | None]:
    try:
        common_df, common_source = fetch_sp500_close_prices([ticker], start_ts.strftime("%Y-%m-%d"))
        if common_df is not None and not common_df.empty and ticker in common_df.columns:
            close = pd.Series(common_df[ticker], dtype=float).dropna().sort_index()
            close = close[(close.index >= start_ts.normalize()) & (close.index <= end_ts.normalize())]
            if not close.empty:
                return close, common_source or "common_loader"
    except Exception:
        pass

    return None, None


def _fetch_close_prices_technical_cache(
    ticker: str,
    *,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> tuple[pd.Series, str]:
    safe = re.sub(r"[^A-Z0-9._-]+", "_", str(ticker).strip().upper())
    path = Path(os.getenv("KEUMJ_TECH_OHLCV_CACHE_DIR", "data/technical_ohlcv_cache")) / f"{safe}.csv"
    if not path.exists() or not path.is_file():
        raise ValueError("technical OHLCV cache file not found")

    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError("technical OHLCV cache is empty")

    cols = {str(c).lower(): c for c in raw.columns}
    date_col = cols.get("date") or raw.columns[0]
    close_col = cols.get("close")
    if close_col is None:
        raise ValueError("technical OHLCV cache missing close column")

    frame = raw[[date_col, close_col]].copy()
    frame.columns = ["date", "close"]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna().sort_values("date")
    if frame.empty:
        raise ValueError("technical OHLCV cache has no usable rows")

    close = pd.Series(frame["close"].values, index=frame["date"].dt.normalize(), name="close")
    close = close[~close.index.duplicated(keep="last")]
    close = close[(close.index >= start_ts.normalize()) & (close.index <= end_ts.normalize())]
    if close.empty:
        raise ValueError("technical OHLCV cache has no rows in requested range")

    return close, "technical_ohlcv_cache"


def _fetch_close_prices_with_source(
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
    history_years: int = 8,
    insecure_ssl: bool = False,
    ca_bundle: str | None = None,
) -> tuple[pd.Series, str]:
    ticker_clean = str(ticker).strip().upper()
    if not ticker_clean:
        raise ValueError("ticker must not be empty")

    if start_date is None:
        start_ts = pd.Timestamp.today().normalize() - pd.DateOffset(years=history_years)
    else:
        start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) if end_date else pd.Timestamp.today().normalize()

    errors: list[str] = []

    if yf is not None:
        try:
            raw = yf.download(
                ticker_clean,
                start=start_ts.normalize().strftime("%Y-%m-%d"),
                end=(end_ts.normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=False,
                actions=False,
                threads=False,
            )
            if isinstance(raw, pd.DataFrame) and not raw.empty:
                cols = {str(c).strip().lower(): c for c in raw.columns}
                close_col = cols.get("close") or cols.get("adj close") or cols.get("adjclose")
                if close_col is not None:
                    close = pd.to_numeric(raw[close_col], errors="coerce").dropna()
                    close.index = pd.to_datetime(close.index).normalize()
                    close = close[(close.index >= start_ts.normalize()) & (close.index <= end_ts.normalize())]
                    if not close.empty:
                        close.name = "close"
                        return close, "yfinance"
            errors.append("yfinance download failed")
        except Exception as exc:
            errors.append(f"yfinance download failed: {exc}")
    else:
        errors.append("yfinance unavailable")

    try:
        close, source = _load_close_from_local_sources(ticker_clean, start_ts=start_ts, end_ts=end_ts)
        if close is not None and not close.empty:
            return close, source or "common_loader"
        errors.append("common loader unavailable")
    except Exception as exc:
        errors.append(f"common loader unavailable: {exc}")

    try:
        return _fetch_close_prices_technical_cache(ticker_clean, start_ts=start_ts, end_ts=end_ts)
    except Exception as tech_exc:
        detail = " | ".join(errors) if errors else "no provider available"
        raise ValueError(
            f"{detail}; technical cache fallback failed for '{ticker_clean}': {tech_exc}"
        ) from tech_exc


def fetch_close_prices(
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
    history_years: int = 8,
    insecure_ssl: bool = False,
    ca_bundle: str | None = None,
) -> pd.Series:
    close, _ = _fetch_close_prices_with_source(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        history_years=history_years,
        insecure_ssl=insecure_ssl,
        ca_bundle=ca_bundle,
    )
    return close


def load_price_data_csv(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path).expanduser()
    if not path.exists() or not path.is_file():
        raise ValueError(f"prices-csv file not found: {path}")

    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Failed to read prices-csv: {path}") from exc

    if frame.empty:
        raise ValueError(f"prices-csv is empty: {path}")
    return frame


def run_ticker_stock_forecast_pipeline(
    ticker: str,
    horizon_days: int = 10,
    start_date: str | None = None,
    end_date: str | None = None,
    history_years: int = 8,
    output_dir: Path | None = None,
    price_data: pd.DataFrame | pd.Series | None = None,
    random_state: int = 7,
    insecure_ssl: bool = False,
    ca_bundle: str | None = None,
) -> StockForecastResult:
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")

    ticker_clean = str(ticker).strip().upper()
    if not ticker_clean:
        raise ValueError("ticker must not be empty")

    if price_data is None:
        close, price_source = _fetch_close_prices_with_source(
            ticker=ticker_clean,
            start_date=start_date,
            end_date=end_date,
            history_years=history_years,
            insecure_ssl=insecure_ssl,
            ca_bundle=ca_bundle,
        )
    else:
        close = _normalize_close_prices(price_data)
        price_source = "provided_price_data"

    try:
        dataset, latest_features, latest_date, regime_snapshot = _build_supervised_dataset(
            close=close,
            horizon_days=horizon_days,
            ticker=ticker_clean,
        )
    except ValueError as exc:
        # If user selected a very narrow date window in GUI, retry once with history_years range.
        can_retry_with_wider_range = (
            price_data is None
            and (start_date is not None or end_date is not None)
            and "Feature table is empty after preprocessing" in str(exc)
        )
        if not can_retry_with_wider_range:
            raise

        close_retry, source_retry = _fetch_close_prices_with_source(
            ticker=ticker_clean,
            start_date=None,
            end_date=None,
            history_years=history_years,
            insecure_ssl=insecure_ssl,
            ca_bundle=ca_bundle,
        )
        close = close_retry
        price_source = f"{source_retry}+auto_widened_range"
        dataset, latest_features, latest_date, regime_snapshot = _build_supervised_dataset(
            close=close,
            horizon_days=horizon_days,
            ticker=ticker_clean,
        )
    if len(dataset) < 60:
        raise ValueError("Not enough history to train; provide at least ~60 usable rows after preprocessing")

    split_idx = max(int(len(dataset) * 0.8), 200)
    split_idx = min(split_idx, len(dataset) - max(30, horizon_days))
    if split_idx <= 0:
        raise ValueError("Not enough rows for train/validation split")

    feature_cols = [c for c in dataset.columns if c != "target"]
    x_train = dataset.iloc[:split_idx][feature_cols]
    y_train = dataset.iloc[:split_idx]["target"]
    x_valid = dataset.iloc[split_idx:][feature_cols]
    y_valid = dataset.iloc[split_idx:]["target"]
    x_all = dataset[feature_cols]
    y_all = dataset["target"]
    y_train_dir = (y_train > 0.0).astype(int)
    y_valid_dir = (y_valid > 0.0).astype(int)
    y_all_dir = (y_all > 0.0).astype(int)

    model_specs = _build_model_specs(random_state=random_state)
    eval_rows: list[dict[str, float | str]] = []
    full_fit_models: dict[str, object] = {}
    predictions: dict[str, float] = {}

    for name, model in model_specs:
        model_train = clone(model)
        model_train.fit(x_train, y_train)
        valid_pred = model_train.predict(x_valid)
        mae = float(mean_absolute_error(y_valid, valid_pred))

        model_full = clone(model)
        model_full.fit(x_all, y_all)
        pred_return = float(model_full.predict(latest_features)[0])

        full_fit_models[name] = model_full
        predictions[name] = pred_return
        eval_rows.append(
            {
                "model": name,
                "validation_mae": mae,
                "predicted_log_return": pred_return,
            }
        )

    model_scores = pd.DataFrame(eval_rows).sort_values("validation_mae").reset_index(drop=True)
    model_scores["weight"] = _inverse_error_weights(model_scores["validation_mae"].to_numpy(dtype=float))
    ensemble_return = float((model_scores["predicted_log_return"] * model_scores["weight"]).sum())
    ensemble_validation_mae = float((model_scores["validation_mae"] * model_scores["weight"]).sum()) if not model_scores.empty else np.nan

    direction_rows: list[dict[str, float | str]] = []
    direction_prob_up = 0.5
    direction_confidence = 0.5
    direction_signal = "No-Trade"
    trade_filter = "confidence filter inactive"
    if y_train_dir.nunique() >= 2 and y_valid_dir.nunique() >= 2 and y_all_dir.nunique() >= 2:
        for name, model in _build_direction_model_specs(random_state=random_state):
            model_train = clone(model)
            model_train.fit(x_train, y_train_dir)
            valid_prob = np.asarray(model_train.predict_proba(x_valid)[:, 1], dtype=float)
            valid_brier = float(np.mean(np.square(valid_prob - y_valid_dir.to_numpy(dtype=float))))

            model_full = clone(model)
            model_full.fit(x_all, y_all_dir)
            pred_prob_up = float(model_full.predict_proba(latest_features)[0, 1])
            direction_rows.append(
                {
                    "model": name,
                    "validation_brier": valid_brier,
                    "predicted_prob_up": pred_prob_up,
                }
            )
        direction_scores = pd.DataFrame(direction_rows).sort_values("validation_brier").reset_index(drop=True)
        direction_scores["weight"] = _inverse_error_weights(direction_scores["validation_brier"].to_numpy(dtype=float))
        direction_prob_up = float((direction_scores["predicted_prob_up"] * direction_scores["weight"]).sum())
    else:
        direction_scores = pd.DataFrame(
            [{"model": "direction_classifier", "validation_brier": np.nan, "predicted_prob_up": 0.5, "weight": 1.0}]
        )

    last_close = float(close.iloc[-1])
    forecast_price = float(last_close * np.exp(ensemble_return))
    naive_price = last_close
    forecast_date = pd.bdate_range(latest_date, periods=horizon_days + 1)[-1]
    expected_return_pct = (forecast_price / last_close - 1.0) * 100.0
    direction_confidence = float(max(direction_prob_up, 1.0 - direction_prob_up))
    no_trade_threshold = 0.60
    magnitude_threshold_pct = float((np.exp(max(ensemble_validation_mae, 0.0)) - 1.0) * 100.0) if np.isfinite(ensemble_validation_mae) else 0.0
    if direction_confidence < no_trade_threshold or abs(expected_return_pct) < max(0.35, magnitude_threshold_pct):
        direction_signal = "No-Trade"
        trade_filter = f"confidence<{int(no_trade_threshold * 100)}% or expected edge below validation MAE"
    else:
        direction_signal = "Up" if direction_prob_up >= 0.5 else "Down"
        trade_filter = f"active: confidence>={int(no_trade_threshold * 100)}%"

    summary = pd.DataFrame(
        [
            {
                "ticker": ticker_clean,
                "as_of_date": pd.Timestamp(latest_date).strftime("%Y-%m-%d"),
                "forecast_date": pd.Timestamp(forecast_date).strftime("%Y-%m-%d"),
                "horizon_days": int(horizon_days),
                "last_close": last_close,
                "predicted_price": forecast_price,
                "naive_random_walk_price": naive_price,
                "ensemble_predicted_log_return": ensemble_return,
                "expected_return_pct": expected_return_pct,
                "direction_prob_up_pct": direction_prob_up * 100.0,
                "direction_confidence_pct": direction_confidence * 100.0,
                "direction_signal": direction_signal,
                "trade_filter": trade_filter,
            }
        ]
    )

    importance = _estimate_feature_importance(full_fit_models=full_fit_models, feature_cols=feature_cols)
    close_history = close.to_frame("close")
    close_history.index.name = "date"

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_dir / "forecast_summary.csv", index=False)
        model_scores.to_csv(output_dir / "model_scores.csv", index=False)
        importance.to_csv(output_dir / "feature_importance.csv", index=False)
        direction_scores.to_csv(output_dir / "direction_scores.csv", index=False)
        regime_snapshot.to_csv(output_dir / "regime_snapshot.csv", index=False)
        close_history.tail(500).to_csv(output_dir / "price_history_tail.csv")
        (output_dir / "forecast_source.txt").write_text(f"source={price_source}\n", encoding="utf-8")

    return StockForecastResult(
        summary=summary,
        model_scores=model_scores,
        close_history=close_history,
        feature_importance=importance,
        direction_scores=direction_scores,
        regime_snapshot=regime_snapshot,
        price_source=price_source,
    )


def _build_supervised_dataset(
    close: pd.Series,
    horizon_days: int,
    *,
    ticker: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.DataFrame]:
    ret_1d = close.pct_change()
    feature_df = pd.DataFrame(index=close.index)
    feature_df["ret_lag_1"] = ret_1d.shift(1)
    feature_df["ret_lag_2"] = ret_1d.shift(2)
    feature_df["ret_lag_3"] = ret_1d.shift(3)
    feature_df["ret_lag_5"] = ret_1d.shift(5)
    feature_df["ret_lag_10"] = ret_1d.shift(10)
    feature_df["ret_lag_20"] = ret_1d.shift(20)
    feature_df["ret_lag_60"] = ret_1d.shift(60)
    feature_df["vol_5"] = ret_1d.rolling(5).std().shift(1)
    feature_df["vol_10"] = ret_1d.rolling(10).std().shift(1)
    feature_df["vol_20"] = ret_1d.rolling(20).std().shift(1)
    feature_df["vol_60"] = ret_1d.rolling(60).std().shift(1)
    feature_df["mom_5"] = close.pct_change(5).shift(1)
    feature_df["mom_10"] = close.pct_change(10).shift(1)
    feature_df["mom_20"] = close.pct_change(20).shift(1)
    feature_df["mom_60"] = close.pct_change(60).shift(1)
    feature_df["sma_ratio_5"] = (close / close.rolling(5).mean() - 1.0).shift(1)
    feature_df["sma_ratio_10"] = (close / close.rolling(10).mean() - 1.0).shift(1)
    feature_df["sma_ratio_20"] = (close / close.rolling(20).mean() - 1.0).shift(1)
    feature_df["sma_ratio_60"] = (close / close.rolling(60).mean() - 1.0).shift(1)
    feature_df["rsi_14"] = _compute_rsi(close, window=14).shift(1)
    feature_df["dow_sin"] = np.sin(2.0 * np.pi * feature_df.index.dayofweek / 5.0)
    feature_df["dow_cos"] = np.cos(2.0 * np.pi * feature_df.index.dayofweek / 5.0)
    regime_snapshot = pd.DataFrame(columns=["metric", "value"])
    if ticker:
        relative_feature_df, regime_snapshot = _relative_feature_frame(str(ticker), close)
        feature_df = feature_df.join(relative_feature_df, how="left")

    target = np.log(close.shift(-horizon_days) / close).rename("target")
    dataset = feature_df.join(target).replace([np.inf, -np.inf], np.nan).dropna()
    if dataset.empty:
        valid_target = int(target.notna().sum())
        raise ValueError(
            "Feature table is empty after preprocessing "
            f"(rows={len(close)}, valid_target_rows={valid_target}, horizon_days={horizon_days}). "
            "Try a wider date range or lower forecast horizon."
        )

    latest_date = pd.Timestamp(feature_df.index[-1]).normalize()
    latest_features = feature_df.iloc[[-1]].replace([np.inf, -np.inf], np.nan).dropna()
    if latest_features.empty:
        last_valid = feature_df.replace([np.inf, -np.inf], np.nan).dropna()
        if last_valid.empty:
            raise ValueError("Latest feature row is invalid; provide more historical data")
        latest_features = last_valid.iloc[[-1]]

    return dataset, latest_features, latest_date, regime_snapshot


def _build_model_specs(random_state: int) -> list[tuple[str, object]]:
    light_mode = _forecast_light_mode()
    elastic = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                ElasticNetCV(
                    l1_ratio=[0.1, 0.5, 0.9] if not light_mode else [0.5],
                    alphas=np.logspace(-4, 1, 20 if not light_mode else 8),
                    cv=5 if not light_mode else 3,
                    max_iter=10000 if not light_mode else 3000,
                ),
            ),
        ]
    )
    if _forecast_fast_linear_only():
        return [("elastic_net", elastic)]

    rf = RandomForestRegressor(
        n_estimators=300 if not light_mode else 60,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=1,
    )
    gb = GradientBoostingRegressor(
        max_depth=4,
        learning_rate=0.05,
        n_estimators=300 if not light_mode else 80,
        random_state=random_state,
    )
    return [
        ("elastic_net", elastic),
        ("random_forest", rf),
        ("gradient_boosting", gb),
    ]


def _build_direction_model_specs(random_state: int) -> list[tuple[str, object]]:
    light_mode = _forecast_light_mode()
    logistic = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegressionCV(
                    Cs=10 if not light_mode else 4,
                    cv=5 if not light_mode else 3,
                    scoring="neg_log_loss",
                    max_iter=5000 if not light_mode else 2000,
                    class_weight="balanced",
                    l1_ratios=(0.0,),
                    random_state=random_state,
                ),
            ),
        ]
    )
    if _forecast_fast_linear_only():
        return [("logistic", logistic)]

    rf = RandomForestClassifier(
        n_estimators=250 if not light_mode else 60,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=1,
    )
    gb = GradientBoostingClassifier(
        n_estimators=200 if not light_mode else 80,
        learning_rate=0.05,
        max_depth=2,
        random_state=random_state,
    )
    return [
        ("logistic", logistic),
        ("random_forest_cls", rf),
        ("gradient_boosting_cls", gb),
    ]


def _inverse_error_weights(errors: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(errors, dtype=float)
    if arr.size == 0:
        return np.array([])
    if np.any(~np.isfinite(arr)):
        return np.full(arr.shape, 1.0 / arr.size)
    inv = 1.0 / np.maximum(arr, eps)
    total = float(inv.sum())
    if not np.isfinite(total) or total <= 0:
        return np.full(arr.shape, 1.0 / arr.size)
    return inv / total


def _estimate_feature_importance(full_fit_models: dict[str, object], feature_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name, model in full_fit_models.items():
        if hasattr(model, "feature_importances_"):
            values = np.asarray(getattr(model, "feature_importances_"), dtype=float)
        elif isinstance(model, Pipeline) and hasattr(model.named_steps.get("model"), "coef_"):
            coef = np.asarray(getattr(model.named_steps["model"], "coef_"), dtype=float)
            values = np.abs(coef)
        else:
            values = np.full(len(feature_cols), np.nan)

        if values.shape[0] != len(feature_cols):
            values = np.full(len(feature_cols), np.nan)

        for feature, importance in zip(feature_cols, values):
            rows.append(
                {
                    "model": model_name,
                    "feature": feature,
                    "importance": float(importance) if np.isfinite(importance) else np.nan,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["model", "feature", "importance"])
    return out.sort_values(["model", "importance"], ascending=[True, False]).reset_index(drop=True)


def _compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    both_zero = (avg_gain == 0.0) & (avg_loss == 0.0)
    only_gain = (avg_gain > 0.0) & (avg_loss == 0.0)
    only_loss = (avg_gain == 0.0) & (avg_loss > 0.0)

    rsi = rsi.where(~both_zero, 50.0)
    rsi = rsi.where(~only_gain, 100.0)
    rsi = rsi.where(~only_loss, 0.0)
    return rsi


def _normalize_close_prices(price_data: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(price_data, pd.Series):
        close = price_data.copy()
        close.index = pd.to_datetime(close.index, utc=True).tz_convert(None).normalize()
        close = pd.to_numeric(close, errors="coerce").dropna()
        close = close[close > 0].sort_index()
        close.name = "close"
        return close

    if price_data.empty:
        raise ValueError("price_data is empty")

    frame = price_data.copy()
    date_col = "date" if "date" in frame.columns else None
    close_col = "close" if "close" in frame.columns else None

    if date_col is None:
        frame.index = pd.to_datetime(frame.index, utc=True).tz_convert(None).normalize()
    else:
        frame[date_col] = pd.to_datetime(frame[date_col], utc=True).dt.tz_convert(None).dt.normalize()
        frame = frame.set_index(date_col)

    if close_col is None:
        numeric_cols = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
        if not numeric_cols:
            raise ValueError("price_data must include at least one numeric column")
        close_col = str(numeric_cols[0])

    close = pd.to_numeric(frame[close_col], errors="coerce").dropna()
    close = close[close > 0].sort_index()
    close.name = "close"
    return close





# Backward-compatible alias
run_stock_forecast_pipeline = run_ticker_stock_forecast_pipeline





