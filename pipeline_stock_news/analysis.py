from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
import os
import re

import numpy as np
import pandas as pd

from pipeline_common.shared_sp500_prices_sql import shared_prices_sqlite_path

DEFAULT_LOOKBACK_DAYS = 45
DEFAULT_EVENT_KEYWORDS = "earnings"
DEFAULT_EVENT_HORIZON_DAYS = 5
DEFAULT_DIVERGENCE_TOP_N = 20
DEFAULT_TOPIC_COUNT = 5
DEFAULT_EXPECTATION_RESET_TOP_N = 20
DEFAULT_VOLATILITY_TOP_N = 20
DEFAULT_VOLATILITY_BASELINE_DAYS = 20
DEFAULT_VOLATILITY_POST_DAYS = 5
DEFAULT_EXPECTATION_SENTIMENT_THRESHOLD = 1.5
DEFAULT_EXPECTATION_WEAK_RETURN_PCT = 1.0
COMPONENTS_CSV_CANDIDATES = (
    Path("data/sp500_components_full.csv"),
    Path("data/sp500_components.csv"),
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _running_on_render() -> bool:
    return bool(os.getenv("RENDER") or os.getenv("RENDER_SERVICE_ID"))


def _news_light_mode() -> bool:
    return _env_bool("KEUMJM_NEWS_LIGHT_MODE", False)


def _max_news_articles() -> int:
    return max(_env_int("KEUMJM_NEWS_MAX_ARTICLES", 300 if _news_light_mode() else 0), 0)

POSITIVE_TITLE_WORDS = {
    "approval",
    "approved",
    "beat",
    "beats",
    "bullish",
    "contract",
    "expands",
    "growth",
    "launch",
    "outperform",
    "partnership",
    "profit",
    "record",
    "strong",
    "surge",
    "upgrade",
    "upgrades",
    "wins",
}
NEGATIVE_TITLE_WORDS = {
    "bankruptcy",
    "cuts",
    "cut",
    "decline",
    "delay",
    "downgrade",
    "downgrades",
    "falls",
    "fraud",
    "investigation",
    "lawsuit",
    "layoff",
    "layoffs",
    "miss",
    "misses",
    "probe",
    "recall",
    "resign",
    "resigns",
    "risk",
    "warning",
    "weak",
}
TOPIC_STOP_WORDS = {
    "stock",
    "stocks",
    "share",
    "shares",
    "market",
    "markets",
    "news",
    "company",
    "companies",
    "says",
    "say",
    "amid",
    "today",
    "analyst",
    "analysts",
    "price",
    "prices",
    "rating",
    "earnings",
    "quarter",
    "report",
    "reports",
    "reporting",
    "yahoo",
    "finance",
    "marketbeat",
    "benzinga",
    "zacks",
    "motley",
    "fool",
    "seeking",
    "alpha",
    "nasdaq",
    "investor",
    "place",
    "watch",
    "update",
    "updates",
    "inc",
    "corp",
    "com",
}


@dataclass(frozen=True)
class EventStudyResult:
    keywords: list[str]
    article_count: int
    matched_ticker_count: int
    summary: pd.DataFrame
    events: pd.DataFrame
    window_start: str
    window_end: str


@dataclass(frozen=True)
class DivergenceResult:
    alerts: pd.DataFrame
    window_start: str
    window_end: str


@dataclass(frozen=True)
class SectorSpilloverResult:
    summary: pd.DataFrame
    events: pd.DataFrame
    window_start: str
    window_end: str


@dataclass(frozen=True)
class ExpectationResetResult:
    candidates: pd.DataFrame
    window_start: str
    window_end: str


@dataclass(frozen=True)
class VolatilityRegimeResult:
    summary: pd.DataFrame
    events: pd.DataFrame
    window_start: str
    window_end: str


@dataclass(frozen=True)
class TopicModelResult:
    topics: pd.DataFrame
    word_cloud: pd.DataFrame
    source_articles: pd.DataFrame
    window_start: str
    window_end: str


@dataclass(frozen=True)
class NewsOverviewResult:
    article_count: int
    unique_ticker_count: int
    unique_source_count: int
    latest_publish_at: str | None
    daily_counts: pd.DataFrame
    top_tickers: pd.DataFrame
    top_sectors: pd.DataFrame
    top_sources: pd.DataFrame
    recent_articles: pd.DataFrame
    window_start: str
    window_end: str


@dataclass(frozen=True)
class StockNewsDashboard:
    applied_keywords: list[str]
    applied_ticker: str | None
    ticker_sector: str | None
    window_start: str
    window_end: str
    computed_sections: frozenset[str]
    overview: NewsOverviewResult
    event_study: EventStudyResult
    sector_spillover: SectorSpilloverResult
    divergence: DivergenceResult
    expectation_reset: ExpectationResetResult
    volatility_regime: VolatilityRegimeResult
    topics: TopicModelResult
    recommendations: list[tuple[str, str]]


DASHBOARD_SECTION_OVERVIEW = "overview"
DASHBOARD_SECTION_EVENT_STUDY = "event_study"
DASHBOARD_SECTION_SECTOR_SPILLOVER = "sector_spillover"
DASHBOARD_SECTION_DIVERGENCE = "divergence"
DASHBOARD_SECTION_EXPECTATION_RESET = "expectation_reset"
DASHBOARD_SECTION_VOLATILITY_REGIME = "volatility_regime"
DASHBOARD_SECTION_TOPICS = "topics"
DASHBOARD_ALL_SECTIONS = frozenset(
    {
        DASHBOARD_SECTION_OVERVIEW,
        DASHBOARD_SECTION_EVENT_STUDY,
        DASHBOARD_SECTION_SECTOR_SPILLOVER,
        DASHBOARD_SECTION_DIVERGENCE,
        DASHBOARD_SECTION_EXPECTATION_RESET,
        DASHBOARD_SECTION_VOLATILITY_REGIME,
        DASHBOARD_SECTION_TOPICS,
    }
)


def _db_path(db_path: Path | str | None = None) -> Path:
    return Path(db_path) if db_path is not None else shared_prices_sqlite_path()


def _connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    return sqlite3.connect(_db_path(db_path))


def _split_keywords(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        parts = [str(item).strip().lower() for item in value]
    else:
        parts = [part.strip().lower() for part in str(value).replace(";", ",").split(",")]
    return [part for part in dict.fromkeys(parts) if part]


@lru_cache(maxsize=1)
def _cached_sector_map() -> dict[str, str]:
    for path in COMPONENTS_CSV_CANDIDATES:
        if not path.exists() or not path.is_file():
            continue
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if frame.empty:
            continue
        cols = {str(col).strip().lower(): col for col in frame.columns}
        symbol_col = cols.get("symbol")
        sector_col = cols.get("sector")
        if symbol_col is None or sector_col is None:
            continue
        out = (
            frame[[symbol_col, sector_col]]
            .dropna(subset=[symbol_col])
            .assign(
                Symbol=lambda df: df[symbol_col].astype(str).str.strip().str.upper(),
                Sector=lambda df: df[sector_col].astype(str).str.strip(),
            )
        )
        out["Sector"] = out["Sector"].replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
        return dict(zip(out["Symbol"], out["Sector"]))
    return {}


def _sector_map() -> dict[str, str]:
    return dict(_cached_sector_map())


def _ticker_sector(ticker: str | None) -> str | None:
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        return None
    return _sector_map().get(symbol)


def _resolve_window(
    conn: sqlite3.Connection,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if end_date:
        end_ts = pd.Timestamp(end_date).normalize()
    else:
        row = conn.execute("SELECT MAX(publish_date) FROM news_articles").fetchone()
        if row and row[0]:
            end_ts = pd.Timestamp(str(row[0])).normalize()
        else:
            end_ts = pd.Timestamp.today().normalize()
    if start_date:
        start_ts = pd.Timestamp(start_date).normalize()
    else:
        start_ts = end_ts - pd.Timedelta(days=max(int(lookback_days) - 1, 0))
    return start_ts, end_ts


def _latest_price_date(conn: sqlite3.Connection) -> pd.Timestamp | None:
    row = conn.execute("SELECT MAX(date) FROM prices").fetchone()
    if not row or not row[0]:
        return None
    return pd.Timestamp(str(row[0])).normalize()


def _max_reference_date_for_forward(conn: sqlite3.Connection, forward_days: int) -> pd.Timestamp | None:
    latest = _latest_price_date(conn)
    if latest is None:
        return None
    return latest - pd.offsets.BDay(max(int(forward_days), 1))


def _load_market_context(
    conn: sqlite3.Connection,
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    ticker: str | None = None,
    keywords: list[str] | None = None,
    max_reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    query = """
        SELECT
            id,
            ticker,
            publish_date,
            publish_day,
            title,
            link,
            source,
            sentiment_score,
            analysis_status,
            reference_price_date,
            matched_on_publish_day,
            open,
            high,
            low,
            close,
            adj_close,
            volume,
            market_cap
        FROM news_articles_market_context
        WHERE date(publish_date) >= ?
          AND date(publish_date) <= ?
    """
    params: list[object] = [start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")]
    ticker_clean = str(ticker or "").strip().upper()
    if ticker_clean:
        query += " AND ticker = ?"
        params.append(ticker_clean)
    if max_reference_date is not None:
        query += " AND reference_price_date IS NOT NULL AND date(reference_price_date) <= ?"
        params.append(pd.Timestamp(max_reference_date).strftime("%Y-%m-%d"))
    kw_list = keywords or []
    if kw_list:
        query += " AND (" + " OR ".join("LOWER(title) LIKE ?" for _ in kw_list) + ")"
        params.extend([f"%{keyword}%" for keyword in kw_list])
    query += " ORDER BY publish_date DESC, id DESC"
    max_rows = _max_news_articles()
    if max_rows > 0:
        query += " LIMIT ?"
        params.append(max_rows)
    frame = pd.read_sql_query(query, conn, params=params)
    if frame.empty:
        return frame
    frame["publish_date"] = pd.to_datetime(frame["publish_date"], errors="coerce")
    frame["publish_day"] = pd.to_datetime(frame["publish_day"], errors="coerce")
    frame["reference_price_date"] = pd.to_datetime(frame["reference_price_date"], errors="coerce")
    for col in ["sentiment_score", "open", "high", "low", "close", "adj_close", "volume", "market_cap"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["matched_on_publish_day"] = pd.to_numeric(frame["matched_on_publish_day"], errors="coerce").fillna(0).astype(int)
    return frame


def _load_prices(
    conn: sqlite3.Connection,
    *,
    tickers: list[str],
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    symbols = [str(symbol).strip().upper() for symbol in tickers if str(symbol).strip()]
    if not symbols:
        return {}
    placeholders = ",".join("?" for _ in symbols)
    query = f"""
        SELECT symbol, date, close, adj_close, volume
        FROM prices
        WHERE symbol IN ({placeholders})
          AND date >= ?
          AND date <= ?
        ORDER BY symbol ASC, date ASC
    """
    params: list[object] = [*symbols, min_date.strftime("%Y-%m-%d"), max_date.strftime("%Y-%m-%d")]
    frame = pd.read_sql_query(query, conn, params=params)
    if frame.empty:
        return {}
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["adj_close"] = pd.to_numeric(frame["adj_close"], errors="coerce")
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
    out: dict[str, pd.DataFrame] = {}
    for symbol, sub in frame.groupby("symbol", sort=False):
        clean = sub.dropna(subset=["date", "close"]).copy()
        if clean.empty:
            continue
        clean = clean.sort_values("date").set_index("date")
        out[str(symbol)] = clean
    return out


def _forward_return_at_day(price_frame: pd.DataFrame | None, reference_price_date: pd.Timestamp | None, day: int) -> float | None:
    forward = _forward_returns(price_frame, reference_price_date, max(int(day), 1))
    if forward is None:
        return None
    value = forward.get(f"day_{int(day)}_return")
    if value is None or pd.isna(value):
        return None
    return float(value)


def _post_event_volatility_ratio(
    price_frame: pd.DataFrame | None,
    reference_price_date: pd.Timestamp | None,
    *,
    baseline_days: int = DEFAULT_VOLATILITY_BASELINE_DAYS,
    post_days: int = DEFAULT_VOLATILITY_POST_DAYS,
) -> dict[str, float] | None:
    if price_frame is None or price_frame.empty or reference_price_date is None or pd.isna(reference_price_date):
        return None
    ref = pd.Timestamp(reference_price_date).normalize()
    if ref not in price_frame.index:
        return None
    position = price_frame.index.get_loc(ref)
    if isinstance(position, slice):
        position = position.start
    if not isinstance(position, int):
        return None
    baseline_prices = price_frame.iloc[max(0, position - max(int(baseline_days), 1)) : position + 1]["close"]
    post_prices = price_frame.iloc[position : position + 1 + max(int(post_days), 1)]["close"]
    baseline = baseline_prices.pct_change().dropna()
    post = post_prices.pct_change().dropna()
    if len(baseline) < 2 or len(post) < 2:
        return None
    baseline_vol = float(baseline.std(ddof=1))
    post_vol = float(post.std(ddof=1))
    if baseline_vol <= 0:
        return None
    return {
        "baseline_vol_pct": baseline_vol * 100.0,
        "post_vol_pct": post_vol * 100.0,
        "volatility_ratio": post_vol / baseline_vol,
    }


def _forward_returns(price_frame: pd.DataFrame | None, reference_price_date: pd.Timestamp | None, horizon_days: int) -> dict[str, object] | None:
    if price_frame is None or price_frame.empty or reference_price_date is None or pd.isna(reference_price_date):
        return None
    ref = pd.Timestamp(reference_price_date).normalize()
    if ref not in price_frame.index:
        return None
    window = price_frame.loc[price_frame.index >= ref].head(max(int(horizon_days), 1) + 1).copy()
    if len(window) < 2:
        return None
    base_close = float(window.iloc[0]["close"])
    result: dict[str, object] = {
        "reference_price_date": ref.strftime("%Y-%m-%d"),
        "base_close": base_close,
    }
    for day in range(1, int(horizon_days) + 1):
        if len(window) > day:
            close_value = float(window.iloc[day]["close"])
            date_value = pd.Timestamp(window.index[day]).strftime("%Y-%m-%d")
            result[f"day_{day}_price_date"] = date_value
            result[f"day_{day}_return"] = (close_value / base_close) - 1.0
        else:
            result[f"day_{day}_price_date"] = None
            result[f"day_{day}_return"] = np.nan
    return result


def _precompute_forward_return_frames(
    price_map: dict[str, pd.DataFrame],
    horizon_days: int,
) -> dict[str, pd.DataFrame]:
    max_day = max(int(horizon_days), 1)
    out: dict[str, pd.DataFrame] = {}
    for symbol, frame in price_map.items():
        if frame is None or frame.empty or "close" not in frame.columns:
            continue
        close = pd.to_numeric(frame["close"], errors="coerce")
        if close.empty:
            continue
        forward = pd.DataFrame(index=frame.index.copy())
        for day in range(1, max_day + 1):
            forward[f"day_{day}_return"] = (close.shift(-day) / close) - 1.0
        out[str(symbol)] = forward
    return out


def heuristic_title_sentiment(title: str) -> float:
    tokens = set(re.findall(r"[a-zA-Z][a-zA-Z0-9'-]+", str(title or "").lower()))
    positive = sum(1 for token in tokens if token in POSITIVE_TITLE_WORDS)
    negative = sum(1 for token in tokens if token in NEGATIVE_TITLE_WORDS)
    return float(positive - negative)


def run_event_study(
    *,
    keywords: str | list[str] | tuple[str, ...] | None = DEFAULT_EVENT_KEYWORDS,
    ticker: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    horizon_days: int = DEFAULT_EVENT_HORIZON_DAYS,
    db_path: Path | str | None = None,
    conn: sqlite3.Connection | None = None,
) -> EventStudyResult:
    keyword_list = _split_keywords(keywords)
    _conn = conn if conn is not None else _connect(db_path)
    try:
        start_ts, end_ts = _resolve_window(_conn, start_date=start_date, end_date=end_date, lookback_days=lookback_days)
        max_ref_date = _max_reference_date_for_forward(_conn, horizon_days)
        news = _load_market_context(
            _conn,
            start_date=start_ts,
            end_date=end_ts,
            ticker=ticker,
            keywords=keyword_list,
            max_reference_date=max_ref_date,
        )
        if news.empty:
            empty = pd.DataFrame()
            return EventStudyResult(keyword_list, 0, 0, empty, empty, start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d"))
        refs = news["reference_price_date"].dropna()
        if refs.empty:
            empty = pd.DataFrame()
            return EventStudyResult(keyword_list, len(news), int(news["ticker"].nunique()), empty, empty, start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d"))
        price_map = _load_prices(
            _conn,
            tickers=sorted(news["ticker"].dropna().astype(str).unique().tolist()),
            min_date=refs.min(),
            max_date=refs.max() + pd.offsets.BDay(max(int(horizon_days), 1) + 2),
        )
    finally:
        if conn is None:
            _conn.close()

    event_rows: list[dict[str, object]] = []
    for row in news.itertuples(index=False):
        forward = _forward_returns(price_map.get(str(row.ticker)), getattr(row, "reference_price_date"), horizon_days)
        if forward is None:
            continue
        event_row = {
            "id": int(row.id),
            "ticker": str(row.ticker),
            "publish_date": pd.Timestamp(row.publish_date).strftime("%Y-%m-%d %H:%M:%S"),
            "title": str(row.title),
            "source": str(row.source),
            "analysis_status": str(row.analysis_status),
            "reference_price_date": forward["reference_price_date"],
            "base_close": float(forward["base_close"]),
        }
        for day in range(1, int(horizon_days) + 1):
            event_row[f"day_{day}_price_date"] = forward[f"day_{day}_price_date"]
            event_row[f"day_{day}_return"] = forward[f"day_{day}_return"]
        event_rows.append(event_row)
    events = pd.DataFrame(event_rows)

    # 요약 통계를 위한 전체 매치 수 저장
    total_article_count = int(len(events))

    # [수정] 최신 뉴스부터 역순(최신순)으로 정렬하여 반환
    if not events.empty:
        events_for_display = events.sort_values('publish_date', ascending=False).reset_index(drop=True)
    else:
        events_for_display = events

    if events.empty:
        empty = pd.DataFrame()
        return EventStudyResult(keyword_list, len(news), int(news["ticker"].nunique()), empty, events_for_display, start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d"))

    summary_rows: list[dict[str, object]] = []
    for day in range(1, int(horizon_days) + 1):
        series = pd.to_numeric(events[f"day_{day}_return"], errors="coerce").dropna()
        if series.empty:
            continue
        std = float(series.std(ddof=1)) if len(series) > 1 else np.nan
        t_stat = (float(series.mean()) / (std / np.sqrt(len(series)))) if len(series) > 1 and std not in (0.0, np.nan) and not np.isnan(std) else np.nan
        summary_rows.append(
            {
                "day": day,
                "sample_size": int(len(series)),
                "mean_return_pct": float(series.mean() * 100.0),
                "median_return_pct": float(series.median() * 100.0),
                "positive_ratio_pct": float((series > 0).mean() * 100.0),
                "t_stat": float(t_stat) if not np.isnan(t_stat) else np.nan,
            }
        )
    summary = pd.DataFrame(summary_rows)
    return EventStudyResult(
        keywords=keyword_list,
        article_count=total_article_count,
        matched_ticker_count=int(events["ticker"].nunique()),
        summary=summary,
        events=events_for_display,
        window_start=start_ts.strftime("%Y-%m-%d"),
        window_end=end_ts.strftime("%Y-%m-%d"),
    )


def run_divergence_scan(
    *,
    ticker: str | None = None,
    keywords: str | list[str] | tuple[str, ...] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    min_abs_sentiment: float = 1.0,
    min_abs_return: float = 2.0,
    top_n: int = DEFAULT_DIVERGENCE_TOP_N,
    db_path: Path | str | None = None,
    conn: sqlite3.Connection | None = None,
) -> DivergenceResult:
    keyword_list = _split_keywords(keywords)
    _conn = conn if conn is not None else _connect(db_path)
    try:
        start_ts, end_ts = _resolve_window(_conn, start_date=start_date, end_date=end_date, lookback_days=lookback_days)
        max_ref_date = _max_reference_date_for_forward(_conn, 5)
        news = _load_market_context(
            _conn,
            start_date=start_ts,
            end_date=end_ts,
            ticker=ticker,
            keywords=keyword_list,
            max_reference_date=max_ref_date,
        )
        if news.empty:
            return DivergenceResult(pd.DataFrame(), start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d"))
        refs = news["reference_price_date"].dropna()
        if refs.empty:
            return DivergenceResult(pd.DataFrame(), start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d"))
        price_map = _load_prices(
            _conn,
            tickers=sorted(news["ticker"].dropna().astype(str).unique().tolist()),
            min_date=refs.min(),
            max_date=refs.max() + pd.offsets.BDay(7),
        )
    finally:
        if conn is None:
            _conn.close()

    rows: list[dict[str, object]] = []
    min_return_decimal = float(min_abs_return) / 100.0
    for row in news.itertuples(index=False):
        forward = _forward_returns(price_map.get(str(row.ticker)), getattr(row, "reference_price_date"), 5)
        if forward is None:
            continue
        sentiment = float(row.sentiment_score) if not pd.isna(row.sentiment_score) else heuristic_title_sentiment(str(row.title))
        if abs(sentiment) < float(min_abs_sentiment):
            continue
        returns = {
            1: float(forward["day_1_return"]) if not pd.isna(forward["day_1_return"]) else np.nan,
            3: float(forward["day_3_return"]) if not pd.isna(forward["day_3_return"]) else np.nan,
            5: float(forward["day_5_return"]) if not pd.isna(forward["day_5_return"]) else np.nan,
        }
        if sentiment > 0:
            opposing = {day: value for day, value in returns.items() if not np.isnan(value) and value <= -min_return_decimal}
        else:
            opposing = {day: value for day, value in returns.items() if not np.isnan(value) and value >= min_return_decimal}
        if not opposing:
            continue
        chosen_day, chosen_return = max(opposing.items(), key=lambda item: abs(item[1]))
        rows.append(
            {
                "id": int(row.id),
                "ticker": str(row.ticker),
                "publish_date": pd.Timestamp(row.publish_date).strftime("%Y-%m-%d %H:%M:%S"),
                "title": str(row.title),
                "source": str(row.source),
                "analysis_status": str(row.analysis_status),
                "effective_sentiment": sentiment,
                "reference_price_date": forward["reference_price_date"],
                "day_1_return_pct": returns[1] * 100.0 if not np.isnan(returns[1]) else np.nan,
                "day_3_return_pct": returns[3] * 100.0 if not np.isnan(returns[3]) else np.nan,
                "day_5_return_pct": returns[5] * 100.0 if not np.isnan(returns[5]) else np.nan,
                "divergence_horizon_days": int(chosen_day),
                "divergence_return_pct": chosen_return * 100.0,
                "divergence_score": abs(sentiment) * abs(chosen_return) * 100.0,
            }
        )
    alerts = pd.DataFrame(rows)
    if not alerts.empty:
        alerts = alerts.sort_values(["divergence_score", "publish_date"], ascending=[False, False]).head(max(int(top_n), 1)).reset_index(drop=True)
    return DivergenceResult(alerts=alerts, window_start=start_ts.strftime("%Y-%m-%d"), window_end=end_ts.strftime("%Y-%m-%d"))


def run_topic_model(
    *,
    ticker: str | None = None,
    keywords: str | list[str] | tuple[str, ...] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    topic_count: int = DEFAULT_TOPIC_COUNT,
    db_path: Path | str | None = None,
    conn: sqlite3.Connection | None = None,
) -> TopicModelResult:
    keyword_list = _split_keywords(keywords)
    _conn = conn if conn is not None else _connect(db_path)
    try:
        start_ts, end_ts = _resolve_window(_conn, start_date=start_date, end_date=end_date, lookback_days=lookback_days)
        news = _load_market_context(_conn, start_date=start_ts, end_date=end_ts, ticker=ticker, keywords=keyword_list)
    finally:
        if conn is None:
            _conn.close()

    if news.empty:
        empty = pd.DataFrame()
        return TopicModelResult(empty, empty, empty, start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d"))
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer

    titles = news["title"].fillna("").astype(str).tolist()
    vectorizer = CountVectorizer(
        stop_words=sorted(set(CountVectorizer(stop_words="english").get_stop_words() or set()).union(TOPIC_STOP_WORDS)),
        max_features=250 if _news_light_mode() else 600,
        min_df=2,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9&.-]{1,}\b",
    )
    try:
        matrix = vectorizer.fit_transform(titles)
    except ValueError:
        empty = pd.DataFrame()
        return TopicModelResult(empty, empty, news[["ticker", "publish_date", "title"]].copy(), start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d"))
    feature_names = np.array(vectorizer.get_feature_names_out())
    if matrix.shape[0] < 5 or matrix.shape[1] < 5:
        counts = np.asarray(matrix.sum(axis=0)).ravel()
        order = np.argsort(counts)[::-1][:30]
        word_cloud = pd.DataFrame({"term": feature_names[order], "weight": counts[order]})
        return TopicModelResult(pd.DataFrame(), word_cloud, news[["ticker", "publish_date", "title"]].copy(), start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d"))
    effective_topic_count = max(2, min(int(topic_count), matrix.shape[0] // 4 if matrix.shape[0] >= 8 else 2, matrix.shape[1]))
    lda = LatentDirichletAllocation(
        n_components=effective_topic_count,
        random_state=42,
        learning_method="online" if _news_light_mode() else "batch",
        max_iter=5 if _news_light_mode() else 10,
    )
    doc_topic = lda.fit_transform(matrix)
    topic_weights = doc_topic.mean(axis=0)
    dominant_topic = doc_topic.argmax(axis=1)
    topic_rows: list[dict[str, object]] = []
    weighted_terms = np.dot(topic_weights, lda.components_)
    weight_order = np.argsort(weighted_terms)[::-1][:40]
    word_cloud = pd.DataFrame(
        {
            "term": feature_names[weight_order],
            "weight": weighted_terms[weight_order],
        }
    )
    source_articles = news[["ticker", "publish_date", "title"]].copy()
    source_articles["dominant_topic"] = dominant_topic
    for topic_idx in range(effective_topic_count):
        term_order = np.argsort(lda.components_[topic_idx])[::-1][:8]
        topic_terms = [str(feature_names[i]) for i in term_order]
        topic_docs = source_articles[source_articles["dominant_topic"] == topic_idx].copy()
        sample_headline = str(topic_docs.iloc[0]["title"]) if not topic_docs.empty else ""
        topic_rows.append(
            {
                "topic_id": int(topic_idx),
                "topic_weight_pct": float(topic_weights[topic_idx] * 100.0),
                "headline_count": int(len(topic_docs)),
                "top_terms": ", ".join(topic_terms),
                "sample_headline": sample_headline,
            }
        )
    topics = pd.DataFrame(topic_rows).sort_values("topic_weight_pct", ascending=False).reset_index(drop=True)
    return TopicModelResult(topics=topics, word_cloud=word_cloud, source_articles=source_articles, window_start=start_ts.strftime("%Y-%m-%d"), window_end=end_ts.strftime("%Y-%m-%d"))


def run_sector_spillover_monitor(
    *,
    keywords: str | list[str] | tuple[str, ...] | None = DEFAULT_EVENT_KEYWORDS,
    ticker: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    horizon_days: int = DEFAULT_EVENT_HORIZON_DAYS,
    db_path: Path | str | None = None,
    conn: sqlite3.Connection | None = None,
) -> SectorSpilloverResult:
    keyword_list = _split_keywords(keywords)
    sector_map = _sector_map()
    sector_to_symbols: dict[str, list[str]] = {}
    for symbol, sector in sector_map.items():
        sector_to_symbols.setdefault(str(sector), []).append(str(symbol))
    _conn = conn if conn is not None else _connect(db_path)
    try:
        start_ts, end_ts = _resolve_window(_conn, start_date=start_date, end_date=end_date, lookback_days=lookback_days)
        max_ref_date = _max_reference_date_for_forward(_conn, horizon_days)
        news = _load_market_context(
            _conn,
            start_date=start_ts,
            end_date=end_ts,
            ticker=ticker,
            keywords=keyword_list,
            max_reference_date=max_ref_date,
        )
        if news.empty:
            empty = pd.DataFrame()
            return SectorSpilloverResult(empty, empty, start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d"))
        refs = news["reference_price_date"].dropna()
        if refs.empty:
            empty = pd.DataFrame()
            return SectorSpilloverResult(empty, empty, start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d"))
        news = news.copy()
        news["sector"] = news["ticker"].map(sector_map).fillna("Unknown")
        peer_symbols = sorted(
            {
                peer
                for sector in news["sector"].astype(str).unique().tolist()
                for peer in sector_to_symbols.get(str(sector), [])
            }
        )
        if not peer_symbols:
            empty = pd.DataFrame()
            return SectorSpilloverResult(empty, empty, start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d"))
        price_map = _load_prices(
            _conn,
            tickers=peer_symbols,
            min_date=refs.min(),
            max_date=refs.max() + pd.offsets.BDay(max(int(horizon_days), 1) + 2),
        )
    finally:
        if conn is None:
            _conn.close()

    max_day = max(int(horizon_days), 1)
    day_numbers = list(range(1, max_day + 1))
    forward_return_columns = [f"day_{day}_return" for day in day_numbers]
    forward_return_frames = _precompute_forward_return_frames(price_map, max_day)
    sector_price_symbols = {
        sector: tuple(symbol for symbol in symbols if symbol in forward_return_frames)
        for sector, symbols in sector_to_symbols.items()
    }
    peer_forward_cache: dict[tuple[str, pd.Timestamp], tuple[float | None, ...]] = {}
    missing_forward = object()

    event_rows: list[dict[str, object]] = []
    for row in news.itertuples(index=False):
        sector = str(getattr(row, "sector", "Unknown") or "Unknown")
        source_ticker = str(row.ticker)
        sector_symbols = sector_price_symbols.get(sector, ())
        if not sector_symbols:
            continue
        reference_price_date = pd.Timestamp(getattr(row, "reference_price_date")).normalize()
        peer_sums = [0.0] * max_day
        peer_counts = [0] * max_day
        for peer in sector_symbols:
            if peer == source_ticker:
                continue
            cache_key = (peer, reference_price_date)
            peer_returns = peer_forward_cache.get(cache_key, missing_forward)
            if peer_returns is missing_forward:
                forward_frame = forward_return_frames.get(peer)
                if forward_frame is None or reference_price_date not in forward_frame.index:
                    peer_returns = tuple([None] * max_day)
                else:
                    values = forward_frame.loc[reference_price_date, forward_return_columns]
                    if isinstance(values, pd.DataFrame):
                        values = values.iloc[0]
                    peer_returns = tuple(
                        None if pd.isna(values[col]) else float(values[col])
                        for col in forward_return_columns
                    )
                peer_forward_cache[cache_key] = peer_returns
            for idx, value in enumerate(peer_returns):
                if value is not None:
                    peer_sums[idx] += value
                    peer_counts[idx] += 1
        if not any(peer_counts):
            continue
        event_row: dict[str, object] = {
            "source_ticker": source_ticker,
            "sector": sector,
            "publish_date": pd.Timestamp(row.publish_date).strftime("%Y-%m-%d %H:%M:%S"),
            "peer_count": int(len(sector_symbols) - (1 if source_ticker in sector_symbols else 0)),
            "title": str(row.title),
        }
        for idx, day in enumerate(day_numbers):
            event_row[f"peer_day_{day}_return_pct"] = (
                float((peer_sums[idx] / peer_counts[idx]) * 100.0)
                if peer_counts[idx]
                else np.nan
            )
        event_rows.append(event_row)
    events = pd.DataFrame(event_rows)
    if events.empty:
        empty = pd.DataFrame()
        return SectorSpilloverResult(empty, events, start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d"))
    agg_map = {"source_ticker": "count", "peer_count": "mean"}
    rename_map = {"source_ticker": "event_count", "peer_count": "avg_peer_count"}
    for day in range(1, int(horizon_days) + 1):
        col = f"peer_day_{day}_return_pct"
        agg_map[col] = "mean"
        rename_map[col] = col
    summary = (
        events.groupby("sector", dropna=False)
        .agg(agg_map)
        .rename(columns=rename_map)
        .reset_index()
        .sort_values([f"peer_day_{max(int(horizon_days), 1)}_return_pct", "event_count"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return SectorSpilloverResult(summary=summary, events=events, window_start=start_ts.strftime("%Y-%m-%d"), window_end=end_ts.strftime("%Y-%m-%d"))


def run_expectation_reset_tracker(
    *,
    ticker: str | None = None,
    keywords: str | list[str] | tuple[str, ...] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    sentiment_threshold: float = DEFAULT_EXPECTATION_SENTIMENT_THRESHOLD,
    weak_return_pct: float = DEFAULT_EXPECTATION_WEAK_RETURN_PCT,
    top_n: int = DEFAULT_EXPECTATION_RESET_TOP_N,
    db_path: Path | str | None = None,
    conn: sqlite3.Connection | None = None,
) -> ExpectationResetResult:
    keyword_list = _split_keywords(keywords)
    _conn = conn if conn is not None else _connect(db_path)
    try:
        start_ts, end_ts = _resolve_window(_conn, start_date=start_date, end_date=end_date, lookback_days=lookback_days)
        max_ref_date = _max_reference_date_for_forward(_conn, 5)
        news = _load_market_context(
            _conn,
            start_date=start_ts,
            end_date=end_ts,
            ticker=ticker,
            keywords=keyword_list,
            max_reference_date=max_ref_date,
        )
        if news.empty:
            return ExpectationResetResult(pd.DataFrame(), start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d"))
        refs = news["reference_price_date"].dropna()
        if refs.empty:
            return ExpectationResetResult(pd.DataFrame(), start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d"))
        price_map = _load_prices(
            _conn,
            tickers=sorted(news["ticker"].dropna().astype(str).unique().tolist()),
            min_date=refs.min(),
            max_date=refs.max() + pd.offsets.BDay(7),
        )
    finally:
        if conn is None:
            _conn.close()

    rows: list[dict[str, object]] = []
    for row in news.itertuples(index=False):
        forward = _forward_returns(price_map.get(str(row.ticker)), getattr(row, "reference_price_date"), 5)
        if forward is None:
            continue
        sentiment = float(row.sentiment_score) if not pd.isna(row.sentiment_score) else heuristic_title_sentiment(str(row.title))
        day_5_return = forward.get("day_5_return")
        day_5_return_pct = float(day_5_return) * 100.0 if day_5_return is not None and not pd.isna(day_5_return) else np.nan
        if np.isnan(day_5_return_pct):
            continue
        thesis = ""
        reset_score = np.nan
        if sentiment >= float(sentiment_threshold) and day_5_return_pct <= float(weak_return_pct):
            thesis = "positive_news_no_follow_through"
            reset_score = float(sentiment) * max(float(weak_return_pct) - day_5_return_pct, 0.1)
        elif sentiment <= -float(sentiment_threshold) and day_5_return_pct >= -float(weak_return_pct):
            thesis = "negative_news_absorbed"
            reset_score = abs(float(sentiment)) * max(day_5_return_pct + float(weak_return_pct), 0.1)
        if not thesis:
            continue
        rows.append(
            {
                "ticker": str(row.ticker),
                "publish_date": pd.Timestamp(row.publish_date).strftime("%Y-%m-%d %H:%M:%S"),
                "title": str(row.title),
                "source": str(row.source),
                "effective_sentiment": sentiment,
                "day_1_return_pct": float(forward["day_1_return"]) * 100.0 if not pd.isna(forward["day_1_return"]) else np.nan,
                "day_3_return_pct": float(forward["day_3_return"]) * 100.0 if not pd.isna(forward["day_3_return"]) else np.nan,
                "day_5_return_pct": day_5_return_pct,
                "reset_type": thesis,
                "reset_score": float(reset_score),
            }
        )
    candidates = pd.DataFrame(rows)
    if not candidates.empty:
        candidates = candidates.sort_values(["reset_score", "publish_date"], ascending=[False, False]).head(max(int(top_n), 1)).reset_index(drop=True)
    return ExpectationResetResult(candidates=candidates, window_start=start_ts.strftime("%Y-%m-%d"), window_end=end_ts.strftime("%Y-%m-%d"))


def run_volatility_regime_after_news(
    *,
    ticker: str | None = None,
    keywords: str | list[str] | tuple[str, ...] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    baseline_days: int = DEFAULT_VOLATILITY_BASELINE_DAYS,
    post_days: int = DEFAULT_VOLATILITY_POST_DAYS,
    top_n: int = DEFAULT_VOLATILITY_TOP_N,
    db_path: Path | str | None = None,
    conn: sqlite3.Connection | None = None,
) -> VolatilityRegimeResult:
    keyword_list = _split_keywords(keywords)
    _conn = conn if conn is not None else _connect(db_path)
    try:
        start_ts, end_ts = _resolve_window(_conn, start_date=start_date, end_date=end_date, lookback_days=lookback_days)
        max_ref_date = _max_reference_date_for_forward(_conn, post_days)
        news = _load_market_context(
            _conn,
            start_date=start_ts,
            end_date=end_ts,
            ticker=ticker,
            keywords=keyword_list,
            max_reference_date=max_ref_date,
        )
        if news.empty:
            empty = pd.DataFrame()
            return VolatilityRegimeResult(empty, empty, start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d"))
        refs = news["reference_price_date"].dropna()
        if refs.empty:
            empty = pd.DataFrame()
            return VolatilityRegimeResult(empty, empty, start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d"))
        price_map = _load_prices(
            _conn,
            tickers=sorted(news["ticker"].dropna().astype(str).unique().tolist()),
            min_date=refs.min() - pd.offsets.BDay(max(int(baseline_days), 2) + 2),
            max_date=refs.max() + pd.offsets.BDay(max(int(post_days), 2) + 2),
        )
    finally:
        if conn is None:
            _conn.close()

    rows: list[dict[str, object]] = []
    for row in news.itertuples(index=False):
        regime = _post_event_volatility_ratio(
            price_map.get(str(row.ticker)),
            getattr(row, "reference_price_date"),
            baseline_days=baseline_days,
            post_days=post_days,
        )
        if regime is None:
            continue
        rows.append(
            {
                "ticker": str(row.ticker),
                "publish_date": pd.Timestamp(row.publish_date).strftime("%Y-%m-%d %H:%M:%S"),
                "title": str(row.title),
                "source": str(row.source),
                "baseline_vol_pct": float(regime["baseline_vol_pct"]),
                "post_vol_pct": float(regime["post_vol_pct"]),
                "volatility_ratio": float(regime["volatility_ratio"]),
            }
        )
    events = pd.DataFrame(rows)
    if events.empty:
        empty = pd.DataFrame()
        return VolatilityRegimeResult(empty, events, start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d"))
    summary = (
        events.groupby("ticker", dropna=False)
        .agg(
            event_count=("ticker", "count"),
            avg_baseline_vol_pct=("baseline_vol_pct", "mean"),
            avg_post_vol_pct=("post_vol_pct", "mean"),
            avg_volatility_ratio=("volatility_ratio", "mean"),
        )
        .reset_index()
        .sort_values(["avg_volatility_ratio", "event_count"], ascending=[False, False])
        .head(max(int(top_n), 1))
        .reset_index(drop=True)
    )
    events = events.sort_values(["volatility_ratio", "publish_date"], ascending=[False, False]).head(max(int(top_n), 1)).reset_index(drop=True)
    return VolatilityRegimeResult(summary=summary, events=events, window_start=start_ts.strftime("%Y-%m-%d"), window_end=end_ts.strftime("%Y-%m-%d"))


def recommended_capabilities() -> list[tuple[str, str]]:
    return [
        ("Source reliability calibration", "출처별로 뉴스 이후 성과를 누적해 신호 가중치를 조정합니다."),
        ("Intraday event clock", "장전, 장중, 장후 뉴스가 다음 거래 흐름에 남기는 차이를 분해합니다."),
        ("Analyst drift monitor", "같은 종목에서 긍정 헤드라인이 이어질 때 가격 반응이 둔화되는 시점을 잡습니다."),
    ]


def build_news_overview(
    *,
    keywords: str | list[str] | tuple[str, ...] | None = DEFAULT_EVENT_KEYWORDS,
    ticker: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    db_path: Path | str | None = None,
    conn: sqlite3.Connection | None = None,
) -> NewsOverviewResult:
    keyword_list = _split_keywords(keywords)
    sector_map = _sector_map()
    _conn = conn if conn is not None else _connect(db_path)
    try:
        start_ts, end_ts = _resolve_window(_conn, start_date=start_date, end_date=end_date, lookback_days=lookback_days)
        news = _load_market_context(_conn, start_date=start_ts, end_date=end_ts, ticker=ticker, keywords=keyword_list)
    finally:
        if conn is None:
            _conn.close()

    window_start = start_ts.strftime("%Y-%m-%d")
    window_end = end_ts.strftime("%Y-%m-%d")
    if news.empty:
        empty = pd.DataFrame()
        return NewsOverviewResult(0, 0, 0, None, empty, empty, empty, empty, empty, window_start, window_end)

    overview = news.copy()
    overview["sector"] = overview["ticker"].map(sector_map).fillna("Unknown")
    overview["publish_day"] = pd.to_datetime(overview["publish_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    latest_publish_at = None
    latest_publish = overview["publish_date"].dropna()
    if not latest_publish.empty:
        latest_publish_at = pd.Timestamp(latest_publish.max()).strftime("%Y-%m-%d %H:%M:%S")

    daily_counts = (
        overview.groupby("publish_day", dropna=False)
        .agg(article_count=("id", "count"))
        .reset_index()
        .rename(columns={"publish_day": "date"})
        .sort_values("date", ascending=False)
        .reset_index(drop=True)
    )
    top_tickers = (
        overview.groupby("ticker", dropna=False)
        .agg(article_count=("id", "count"), latest_publish_date=("publish_date", "max"))
        .reset_index()
        .sort_values(["article_count", "latest_publish_date", "ticker"], ascending=[False, False, True])
        .head(10)
        .reset_index(drop=True)
    )
    if not top_tickers.empty:
        top_tickers["latest_publish_date"] = pd.to_datetime(top_tickers["latest_publish_date"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    top_sectors = (
        overview.groupby("sector", dropna=False)
        .agg(article_count=("id", "count"), ticker_count=("ticker", "nunique"))
        .reset_index()
        .sort_values(["article_count", "ticker_count", "sector"], ascending=[False, False, True])
        .head(10)
        .reset_index(drop=True)
    )
    top_sources = (
        overview.groupby("source", dropna=False)
        .agg(article_count=("id", "count"), ticker_count=("ticker", "nunique"))
        .reset_index()
        .sort_values(["article_count", "ticker_count", "source"], ascending=[False, False, True])
        .head(10)
        .reset_index(drop=True)
    )
    recent_articles = (
        overview[["ticker", "sector", "publish_date", "source", "title"]]
        .sort_values(["publish_date", "ticker"], ascending=[False, True])
        .head(20)
        .copy()
        .reset_index(drop=True)
    )
    if not recent_articles.empty:
        recent_articles["publish_date"] = pd.to_datetime(recent_articles["publish_date"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    return NewsOverviewResult(
        article_count=int(len(overview)),
        unique_ticker_count=int(overview["ticker"].dropna().astype(str).nunique()),
        unique_source_count=int(overview["source"].dropna().astype(str).nunique()),
        latest_publish_at=latest_publish_at,
        daily_counts=daily_counts,
        top_tickers=top_tickers,
        top_sectors=top_sectors,
        top_sources=top_sources,
        recent_articles=recent_articles,
        window_start=window_start,
        window_end=window_end,
    )


def _normalize_dashboard_sections(
    sections: list[str] | tuple[str, ...] | set[str] | frozenset[str] | None,
) -> frozenset[str]:
    if sections is None:
        return DASHBOARD_ALL_SECTIONS
    return frozenset(str(section).strip() for section in sections if str(section).strip() in DASHBOARD_ALL_SECTIONS)


def _empty_event_study_result(keyword_list: list[str], window_start: str, window_end: str) -> EventStudyResult:
    empty = pd.DataFrame()
    return EventStudyResult(keyword_list, 0, 0, empty, empty, window_start, window_end)


def _empty_overview_result(window_start: str, window_end: str) -> NewsOverviewResult:
    empty = pd.DataFrame()
    return NewsOverviewResult(0, 0, 0, None, empty, empty, empty, empty, empty, window_start, window_end)


def _empty_divergence_result(window_start: str, window_end: str) -> DivergenceResult:
    return DivergenceResult(pd.DataFrame(), window_start, window_end)


def _empty_sector_spillover_result(window_start: str, window_end: str) -> SectorSpilloverResult:
    empty = pd.DataFrame()
    return SectorSpilloverResult(empty, empty, window_start, window_end)


def _empty_expectation_reset_result(window_start: str, window_end: str) -> ExpectationResetResult:
    return ExpectationResetResult(pd.DataFrame(), window_start, window_end)


def _empty_volatility_regime_result(window_start: str, window_end: str) -> VolatilityRegimeResult:
    empty = pd.DataFrame()
    return VolatilityRegimeResult(empty, empty, window_start, window_end)


def _empty_topic_model_result(window_start: str, window_end: str) -> TopicModelResult:
    empty = pd.DataFrame()
    return TopicModelResult(empty, empty, empty, window_start, window_end)


def build_stock_news_dashboard(
    *,
    event_keywords: str | list[str] | tuple[str, ...] | None = DEFAULT_EVENT_KEYWORDS,
    ticker: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    horizon_days: int = DEFAULT_EVENT_HORIZON_DAYS,
    divergence_top_n: int = DEFAULT_DIVERGENCE_TOP_N,
    topic_count: int = DEFAULT_TOPIC_COUNT,
    db_path: Path | str | None = None,
    sections: list[str] | tuple[str, ...] | set[str] | frozenset[str] | None = None,
) -> StockNewsDashboard:
    keyword_list = _split_keywords(event_keywords)
    ticker_clean = str(ticker or "").strip().upper() or None
    computed_sections = _normalize_dashboard_sections(sections)
    with _connect(db_path) as shared_conn:
        start_ts, end_ts = _resolve_window(
            shared_conn,
            start_date=start_date,
            end_date=end_date,
            lookback_days=lookback_days,
        )
        resolved_start = start_ts.strftime("%Y-%m-%d")
        resolved_end = end_ts.strftime("%Y-%m-%d")
        overview_result = build_news_overview(
            keywords=keyword_list,
            ticker=ticker_clean,
            start_date=resolved_start,
            end_date=resolved_end,
            lookback_days=lookback_days,
            conn=shared_conn,
        ) if DASHBOARD_SECTION_OVERVIEW in computed_sections else _empty_overview_result(resolved_start, resolved_end)
        event_result = run_event_study(
            keywords=keyword_list,
            ticker=ticker_clean,
            start_date=resolved_start,
            end_date=resolved_end,
            lookback_days=lookback_days,
            horizon_days=horizon_days,
            conn=shared_conn,
        ) if DASHBOARD_SECTION_EVENT_STUDY in computed_sections else _empty_event_study_result(keyword_list, resolved_start, resolved_end)
        sector_spillover = run_sector_spillover_monitor(
            keywords=keyword_list,
            ticker=ticker_clean,
            start_date=resolved_start,
            end_date=resolved_end,
            lookback_days=lookback_days,
            horizon_days=horizon_days,
            conn=shared_conn,
        ) if DASHBOARD_SECTION_SECTOR_SPILLOVER in computed_sections else _empty_sector_spillover_result(resolved_start, resolved_end)
        divergence_result = run_divergence_scan(
            ticker=ticker_clean,
            keywords=keyword_list,
            start_date=resolved_start,
            end_date=resolved_end,
            lookback_days=lookback_days,
            top_n=divergence_top_n,
            conn=shared_conn,
        ) if DASHBOARD_SECTION_DIVERGENCE in computed_sections else _empty_divergence_result(resolved_start, resolved_end)
        expectation_reset = run_expectation_reset_tracker(
            ticker=ticker_clean,
            keywords=keyword_list,
            start_date=resolved_start,
            end_date=resolved_end,
            lookback_days=lookback_days,
            top_n=divergence_top_n,
            conn=shared_conn,
        ) if DASHBOARD_SECTION_EXPECTATION_RESET in computed_sections else _empty_expectation_reset_result(resolved_start, resolved_end)
        volatility_regime = run_volatility_regime_after_news(
            ticker=ticker_clean,
            keywords=keyword_list,
            start_date=resolved_start,
            end_date=resolved_end,
            lookback_days=lookback_days,
            top_n=divergence_top_n,
            conn=shared_conn,
        ) if DASHBOARD_SECTION_VOLATILITY_REGIME in computed_sections else _empty_volatility_regime_result(resolved_start, resolved_end)
        topic_result = run_topic_model(
            ticker=ticker_clean,
            keywords=keyword_list,
            start_date=resolved_start,
            end_date=resolved_end,
            lookback_days=lookback_days,
            topic_count=topic_count,
            conn=shared_conn,
        ) if DASHBOARD_SECTION_TOPICS in computed_sections else _empty_topic_model_result(resolved_start, resolved_end)

    return StockNewsDashboard(
        applied_keywords=keyword_list,
        applied_ticker=ticker_clean,
        ticker_sector=_ticker_sector(ticker_clean),
        window_start=resolved_start,
        window_end=resolved_end,
        computed_sections=computed_sections,
        overview=overview_result,
        event_study=event_result,
        sector_spillover=sector_spillover,
        divergence=divergence_result,
        expectation_reset=expectation_reset,
        volatility_regime=volatility_regime,
        topics=topic_result,
        recommendations=recommended_capabilities(),
    )
