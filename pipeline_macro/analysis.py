from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_common.notebook_data import (
    fetch_sp500_close_prices,
    load_local_yield_df,
    load_sp500_components,
    make_fallback_yield_df,
    make_gbm_series,
)


DEFAULT_START_DATE = "2024-01-01"
DEFAULT_LOOKBACK_DAYS = 504


@dataclass
class MacroDashboard:
    as_of_date: str
    regime_label: str
    risk_level: str
    equity_bias: str
    summary: pd.DataFrame
    scores: pd.DataFrame
    indicators: pd.DataFrame
    rates: pd.DataFrame
    risk_assets: pd.DataFrame
    dollar_commodities: pd.DataFrame
    sector_playbook: pd.DataFrame
    sources: pd.DataFrame
    market_series: pd.DataFrame
    rate_series: pd.DataFrame
    risk_series: pd.DataFrame
    commodity_series: pd.DataFrame


def _latest_value(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.iloc[-1]) if not clean.empty else np.nan


def _return_pct(series: pd.Series, window: int) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) <= window:
        return np.nan
    base = float(clean.iloc[-(window + 1)])
    latest = float(clean.iloc[-1])
    if base == 0 or not np.isfinite(base):
        return np.nan
    return (latest / base - 1.0) * 100.0


def _change(series: pd.Series, window: int) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) <= window:
        return np.nan
    return float(clean.iloc[-1] - clean.iloc[-(window + 1)])


def _zscore(series: pd.Series, window: int = 252) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna().tail(window)
    if len(clean) < 20:
        return np.nan
    std = float(clean.std(ddof=1))
    if std <= 0 or not np.isfinite(std):
        return np.nan
    return float((clean.iloc[-1] - clean.mean()) / std)


def _score_from_z(z: float, *, inverse: bool = False) -> float:
    if not np.isfinite(z):
        return 50.0
    score = 50.0 + np.clip(z, -2.0, 2.0) * 20.0
    if inverse:
        score = 100.0 - score
    return float(np.clip(score, 0.0, 100.0))


def _cap_weighted_series(prices: pd.DataFrame, symbols: list[str]) -> pd.Series:
    frame = prices.reindex(columns=symbols).apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if frame.empty:
        return pd.Series(dtype=float, name="S&P 500")
    daily = frame.pct_change(fill_method=None)
    market_daily = daily.mean(axis=1, skipna=True).dropna()
    if market_daily.empty:
        return pd.Series(dtype=float, name="S&P 500")
    return (1.0 + market_daily).cumprod().rename("S&P 500")


def _read_local_series(path: Path, name: str, start: str) -> pd.Series | None:
    if not path.is_file():
        return None
    try:
        raw = pd.read_csv(path)
    except Exception:
        return None
    if raw.empty:
        return None
    cols = {str(c).strip().lower(): c for c in raw.columns}
    date_col = cols.get("date") or cols.get("datetime") or raw.columns[0]
    value_col = (
        cols.get("close")
        or cols.get("adj close")
        or cols.get("adj_close")
        or cols.get("price")
        or cols.get("value")
        or raw.columns[-1]
    )
    out = raw[[date_col, value_col]].copy()
    out.columns = ["date", "value"]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna().sort_values("date")
    if out.empty:
        return None
    series = pd.Series(out["value"].values, index=out["date"].dt.normalize(), name=name)
    series = series[~series.index.duplicated(keep="last")]
    series = series[series.index >= pd.Timestamp(start)]
    return series.dropna() if not series.dropna().empty else None


def _load_local_or_fallback(
    name: str,
    *,
    env_name: str,
    filenames: list[str],
    start: str,
    base: float,
    drift: float,
    vol: float,
    seed: int,
) -> tuple[pd.Series, str]:
    candidates = [os.getenv(env_name, "").strip(), *filenames]
    for raw_path in candidates:
        if not raw_path:
            continue
        series = _read_local_series(Path(raw_path), name, start)
        if series is not None:
            return series, f"local_csv:{raw_path}"
    return make_gbm_series(name, start=start, base=base, drift=drift, vol=vol, seed=seed, min_periods=1000), "fallback"


def _simple_regime(growth: float, inflation: float, policy: float, risk: float) -> tuple[str, str, str]:
    if risk < 35 and growth < 45:
        return "Risk-off Slowdown", "High", "Defensive"
    if growth >= 55 and inflation < 60 and risk >= 50:
        return "Goldilocks / Risk-on", "Low", "Constructive"
    if inflation >= 65 and policy >= 60:
        return "Inflation Pressure / Tight Policy", "Medium-High", "Selective"
    if growth < 45 and inflation < 55:
        return "Disinflation Slowdown", "Medium", "Quality"
    if risk >= 60:
        return "Liquidity-led Risk-on", "Medium", "Constructive"
    return "Mixed Macro", "Medium", "Neutral"


def _sector_bias_table(growth: float, inflation: float, policy: float, risk: float) -> pd.DataFrame:
    rows = [
        ("Information Technology", "성장/유동성 민감", 0.35 * growth + 0.35 * risk + 0.30 * (100 - policy)),
        ("Financials", "커브 정상화와 경기 기대 민감", 0.45 * growth + 0.30 * policy + 0.25 * risk),
        ("Energy", "원유와 인플레 압력 민감", 0.45 * inflation + 0.25 * growth + 0.30 * risk),
        ("Materials", "구리/글로벌 제조 사이클 민감", 0.45 * growth + 0.35 * inflation + 0.20 * risk),
        ("Health Care", "방어 성장", 0.25 * growth + 0.25 * risk + 0.50 * (100 - policy)),
        ("Consumer Staples", "방어/저변동", 0.20 * growth + 0.20 * risk + 0.60 * (100 - inflation)),
        ("Utilities / REITs", "금리 하락 수혜", 0.20 * growth + 0.20 * risk + 0.60 * (100 - policy)),
    ]
    out = pd.DataFrame(rows, columns=["섹터", "민감도", "선호 점수"])
    out["선호 점수"] = out["선호 점수"].astype(float).clip(0.0, 100.0)
    out["의견"] = pd.cut(
        out["선호 점수"],
        bins=[-0.1, 40, 60, 100],
        labels=["비중축소 후보", "중립", "비중확대 후보"],
    ).astype(str)
    return out.sort_values("선호 점수", ascending=False).reset_index(drop=True)


def build_macro_dashboard(
    *,
    start_date: str | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> MacroDashboard:
    start = start_date or DEFAULT_START_DATE
    lookback = max(int(lookback_days), 126)
    tail_n = max(lookback + 60, 260)

    yield_ids = ["DGS2", "DGS10", "DGS30"]
    local_yields, local_yield_source = load_local_yield_df(yield_ids, start=start)
    if local_yields is not None and not local_yields.empty:
        yields = local_yields
        yield_source = local_yield_source or "local_csv"
    else:
        yields = make_fallback_yield_df(yield_ids, start=start)
        yield_source = "fallback"
    yields = yields.tail(tail_n).copy()

    components, components_source = load_sp500_components(max_symbols=120)
    symbols = components["Symbol"].astype(str).head(120).tolist()
    prices, price_source = fetch_sp500_close_prices(symbols, start)
    prices = prices.tail(tail_n).copy()
    spx = _cap_weighted_series(prices, symbols)

    dxy, dxy_source = _load_local_or_fallback(
        "DXY",
        env_name="DXY_CSV_PATH",
        filenames=["data/dxy.csv", "data/DXY.csv"],
        start=start,
        base=100.0,
        drift=0.00005,
        vol=0.004,
        seed=7,
    )
    wti, wti_source = _load_local_or_fallback(
        "WTI Crude",
        env_name="WTI_CSV_PATH",
        filenames=["data/wti.csv", "data/oil.csv", "data/crude_oil.csv"],
        start=start,
        base=78.0,
        drift=0.00004,
        vol=0.018,
        seed=21,
    )
    gold, gold_source = _load_local_or_fallback(
        "Gold",
        env_name="GOLD_CSV_PATH",
        filenames=["data/gold.csv", "data/xau.csv"],
        start=start,
        base=2050.0,
        drift=0.00012,
        vol=0.009,
        seed=22,
    )
    silver, silver_source = _load_local_or_fallback(
        "Silver",
        env_name="SILVER_CSV_PATH",
        filenames=["data/silver.csv", "data/xag.csv"],
        start=start,
        base=24.0,
        drift=0.00010,
        vol=0.014,
        seed=23,
    )
    copper, copper_source = _load_local_or_fallback(
        "Copper",
        env_name="COPPER_CSV_PATH",
        filenames=["data/copper.csv", "data/hg.csv"],
        start=start,
        base=4.0,
        drift=0.00008,
        vol=0.013,
        seed=24,
    )

    dxy = dxy.tail(tail_n)
    commodity_series = pd.concat([wti, gold, silver, copper], axis=1).tail(tail_n).dropna(how="all")
    dgs2 = yields["DGS2"] if "DGS2" in yields else pd.Series(dtype=float)
    dgs10 = yields["DGS10"] if "DGS10" in yields else pd.Series(dtype=float)
    dgs30 = yields["DGS30"] if "DGS30" in yields else pd.Series(dtype=float)
    curve_10y_2y = (dgs10 - dgs2).dropna().rename("10Y-2Y")
    curve_30y_10y = (dgs30 - dgs10).dropna().rename("30Y-10Y")

    latest_dates = [idx.max() for idx in [yields.index, prices.index, dxy.index, spx.index, commodity_series.index] if len(idx) > 0]
    as_of = max(latest_dates).strftime("%Y-%m-%d") if latest_dates else "-"

    risk_return_20d = _return_pct(spx, 20)
    risk_return_60d = _return_pct(spx, 60)
    spx_daily = spx.pct_change(fill_method=None).dropna()
    realized_vol_20d = float(spx_daily.tail(20).std(ddof=1) * np.sqrt(252) * 100.0) if len(spx_daily) >= 20 else np.nan
    rolling_vol = (spx_daily.rolling(20).std(ddof=1) * np.sqrt(252) * 100.0).dropna().rename("20D Ann Vol")
    drawdown = ((spx / spx.cummax() - 1.0) * 100.0).dropna().rename("Drawdown")

    commodity_basket = commodity_series.pct_change(fill_method=None).mean(axis=1).dropna()
    growth_score = np.nanmean(
        [
            _score_from_z(_zscore(spx_daily.rolling(20).sum().dropna())),
            _score_from_z(_zscore(curve_10y_2y)),
            _score_from_z(_zscore(commodity_series["Copper"].pct_change(fill_method=None).rolling(20).sum().dropna())),
        ]
    )
    inflation_score = np.nanmean(
        [
            _score_from_z(_zscore(dgs10)),
            _score_from_z(_zscore(dxy)),
            _score_from_z(_zscore(commodity_basket.rolling(20).sum().dropna())),
        ]
    )
    policy_score = np.nanmean([_score_from_z(_zscore(dgs2)), _score_from_z(_zscore(dgs10))])
    vol_penalty = 50.0 if not np.isfinite(realized_vol_20d) else float(np.clip(100.0 - (realized_vol_20d - 8.0) * 3.0, 0.0, 100.0))
    risk_score = np.nanmean([_score_from_z(_zscore(spx)), _score_from_z(_zscore(drawdown)), vol_penalty])
    recession_score = np.nanmean([_score_from_z(_zscore(curve_10y_2y), inverse=True), _score_from_z(_zscore(spx), inverse=True)])
    liquidity_score = np.nanmean([_score_from_z(_zscore(dgs10), inverse=True), _score_from_z(_zscore(dxy), inverse=True), _score_from_z(_zscore(spx))])

    scores_map = {
        "성장 모멘텀": growth_score,
        "인플레/원자재 압력": inflation_score,
        "정책 긴축도": policy_score,
        "위험선호": risk_score,
        "침체 리스크": recession_score,
        "유동성": liquidity_score,
    }
    regime_label, risk_level, equity_bias = _simple_regime(growth_score, inflation_score, policy_score, risk_score)

    summary = pd.DataFrame(
        [
            {"항목": "거시 국면", "값": regime_label, "해석": "성장, 원자재, 금리, 정책, 위험선호 점수를 조합한 현재 환경입니다."},
            {"항목": "리스크 레벨", "값": risk_level, "해석": "시장 변동성과 성장/침체 신호를 함께 본 위험 단계입니다."},
            {"항목": "주식 비중 의견", "값": equity_bias, "해석": "포트폴리오 의사결정용 정성 가이드입니다."},
            {"항목": "기준일", "값": as_of, "해석": "로컬 데이터에서 확인한 최신 관측일입니다."},
        ]
    )
    scores = pd.DataFrame([{"점수": key, "값": float(np.clip(value, 0, 100)) if np.isfinite(value) else np.nan} for key, value in scores_map.items()])
    indicators = pd.DataFrame(
        [
            {"지표": "S&P 500 20D 수익률", "현재": risk_return_20d, "단위": "%", "해석": "단기 위험자산 방향성"},
            {"지표": "S&P 500 60D 수익률", "현재": risk_return_60d, "단위": "%", "해석": "중기 위험자산 방향성"},
            {"지표": "20D 연율 변동성", "현재": realized_vol_20d, "단위": "연 %", "해석": "최근 시장 변동성"},
            {"지표": "현재 낙폭", "현재": _latest_value(drawdown), "단위": "%", "해석": "고점 대비 조정 폭"},
            {"지표": "10Y-2Y 스프레드", "현재": _latest_value(curve_10y_2y), "단위": "%p", "해석": "경기 사이클/침체 신호"},
            {"지표": "원자재 바스켓 60D 수익률", "현재": _return_pct((1.0 + commodity_basket).cumprod(), 60), "단위": "%", "해석": "물가와 경기민감 수요 압력"},
            {"지표": "DXY 60D 변화율", "현재": _return_pct(dxy, 60), "단위": "%", "해석": "달러 강세와 글로벌 유동성"},
        ]
    )
    rates = pd.DataFrame(
        [
            {"구간": "2Y", "현재 금리": _latest_value(dgs2), "20D 변화": _change(dgs2, 20), "60D 변화": _change(dgs2, 60), "해석": "정책금리 기대에 민감"},
            {"구간": "10Y", "현재 금리": _latest_value(dgs10), "20D 변화": _change(dgs10, 20), "60D 변화": _change(dgs10, 60), "해석": "성장/물가 기대와 할인율"},
            {"구간": "30Y", "현재 금리": _latest_value(dgs30), "20D 변화": _change(dgs30, 20), "60D 변화": _change(dgs30, 60), "해석": "장기 성장과 기간 프리미엄"},
            {"구간": "10Y-2Y", "현재 금리": _latest_value(curve_10y_2y), "20D 변화": _change(curve_10y_2y, 20), "60D 변화": _change(curve_10y_2y, 60), "해석": "커브 정상화/역전"},
        ]
    )
    risk_assets = pd.DataFrame(
        [
            {"자산": "S&P 500 Proxy", "20D 수익률": risk_return_20d, "60D 수익률": risk_return_60d, "20D 연율 변동성": realized_vol_20d, "현재 낙폭": _latest_value(drawdown)},
            {"자산": "DXY", "20D 수익률": _return_pct(dxy, 20), "60D 수익률": _return_pct(dxy, 60), "20D 연율 변동성": float(dxy.pct_change(fill_method=None).dropna().tail(20).std(ddof=1) * np.sqrt(252) * 100.0), "현재 낙폭": _latest_value((dxy / dxy.cummax() - 1.0) * 100.0)},
        ]
    )
    commodity_rows = [
        ("WTI Crude", commodity_series["WTI Crude"], "에너지 비용과 인플레 기대에 직접적인 영향을 줍니다."),
        ("Gold", commodity_series["Gold"], "실질금리, 달러, 안전자산 수요를 반영합니다."),
        ("Silver", commodity_series["Silver"], "귀금속과 산업금속 성격을 함께 가집니다."),
        ("Copper", commodity_series["Copper"], "글로벌 제조업과 중국/인프라 수요에 민감합니다."),
    ]
    dollar_commodities = pd.DataFrame(
        [
            {"지표": "DXY", "현재": _latest_value(dxy), "20D 변화율": _return_pct(dxy, 20), "60D 변화율": _return_pct(dxy, 60), "20D 연율 변동성": float(dxy.pct_change(fill_method=None).dropna().tail(20).std(ddof=1) * np.sqrt(252) * 100.0), "해석": "달러 강세는 해외매출, 원자재, 글로벌 유동성에 영향을 줍니다."},
            *[
                {
                    "지표": name,
                    "현재": _latest_value(series),
                    "20D 변화율": _return_pct(series, 20),
                    "60D 변화율": _return_pct(series, 60),
                    "20D 연율 변동성": float(series.pct_change(fill_method=None).dropna().tail(20).std(ddof=1) * np.sqrt(252) * 100.0),
                    "해석": note,
                }
                for name, series, note in commodity_rows
            ],
        ]
    )
    sector_bias = _sector_bias_table(growth_score, inflation_score, policy_score, risk_score)
    sources = pd.DataFrame(
        [
            {"데이터": "S&P 500 구성", "출처": components_source},
            {"데이터": "S&P 500 가격", "출처": price_source},
            {"데이터": "미국 금리", "출처": yield_source},
            {"데이터": "DXY", "출처": dxy_source},
            {"데이터": "WTI Crude", "출처": wti_source},
            {"데이터": "Gold", "출처": gold_source},
            {"데이터": "Silver", "출처": silver_source},
            {"데이터": "Copper", "출처": copper_source},
        ]
    )
    market_series = pd.concat([spx.rename("S&P 500"), dxy.rename("DXY"), commodity_series["Gold"].rename("Gold")], axis=1).dropna(how="all")
    rate_series = pd.concat([dgs2.rename("2Y"), dgs10.rename("10Y"), dgs30.rename("30Y"), curve_10y_2y, curve_30y_10y], axis=1).dropna(how="all")
    risk_series = pd.concat([spx.rename("S&P 500"), drawdown, rolling_vol], axis=1).dropna(how="all")

    return MacroDashboard(
        as_of_date=as_of,
        regime_label=regime_label,
        risk_level=risk_level,
        equity_bias=equity_bias,
        summary=summary,
        scores=scores,
        indicators=indicators,
        rates=rates,
        risk_assets=risk_assets,
        dollar_commodities=dollar_commodities,
        sector_playbook=sector_bias,
        sources=sources,
        market_series=market_series,
        rate_series=rate_series,
        risk_series=risk_series,
        commodity_series=commodity_series,
    )
