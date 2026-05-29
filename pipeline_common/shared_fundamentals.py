from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .shared_sp500_prices_sql import (
    load_shared_close_prices_for_symbols,
    load_shared_market_caps_for_symbols,
    load_shared_quarterly_fundamentals_for_symbols,
    shared_prices_sqlite_path,
)


def _normalize_symbol(value: object) -> str:
    return str(value or "").strip().upper()


def _safe_positive(value: object) -> float | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    if not np.isfinite(numeric) or numeric <= 0.0:
        return None
    return numeric


def _last_value_on_or_before(series: pd.Series, as_of_ts: pd.Timestamp) -> tuple[pd.Timestamp, float] | tuple[None, None]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None, None
    clean.index = pd.to_datetime(clean.index, errors="coerce")
    clean = clean[~clean.index.isna()].sort_index()
    clean = clean[clean.index <= as_of_ts]
    if clean.empty:
        return None, None
    return pd.Timestamp(clean.index[-1]).normalize(), float(clean.iloc[-1])


def _recent_quarter_values(rows: pd.DataFrame, column: str, *, count: int, source_is_sec: bool) -> list[float]:
    if rows.empty or column not in rows.columns:
        return []

    ordered = rows[["fiscal_date", column]].copy()
    ordered["fiscal_date"] = pd.to_datetime(ordered["fiscal_date"], errors="coerce")
    ordered[column] = pd.to_numeric(ordered[column], errors="coerce")
    ordered = ordered.dropna(subset=["fiscal_date", column]).sort_values("fiscal_date")
    if ordered.empty:
        return []

    if not source_is_sec:
        return [float(value) for value in ordered[column].tail(max(int(count), 1)).tolist()][::-1]

    quarter_values: list[tuple[pd.Timestamp, float]] = []
    previous_value: float | None = None
    previous_date: pd.Timestamp | None = None
    for item in ordered.itertuples(index=False):
        fiscal_date = pd.Timestamp(item.fiscal_date)
        current = float(getattr(item, column))
        if previous_value is None or previous_date is None:
            quarter_value = current
        else:
            gap_days = abs((fiscal_date - previous_date).days)
            quarter_value = current - previous_value if current >= previous_value and gap_days <= 130 else current
        quarter_values.append((fiscal_date, quarter_value))
        previous_value = current
        previous_date = fiscal_date

    return [float(value) for _, value in quarter_values[-max(int(count), 1):]][::-1]


def derive_shared_fundamental_metrics(
    symbols: list[str],
    *,
    as_of_date: str | pd.Timestamp | None = None,
    limit_per_symbol: int = 4,
    shared_db_root: Path | str | None = None,
    db_path: Path | str | None = None,
) -> tuple[pd.DataFrame, str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        clean = _normalize_symbol(symbol)
        if not clean or clean in seen:
            continue
        seen.add(clean)
        normalized.append(clean)
    if not normalized:
        return pd.DataFrame(), "not_available"

    target = Path(db_path) if db_path is not None else shared_prices_sqlite_path(shared_db_root)
    load_limit = max(int(limit_per_symbol) * 2, 8)
    quarterly, quarterly_source = load_shared_quarterly_fundamentals_for_symbols(
        normalized,
        limit_per_symbol=load_limit,
        shared_db_root=shared_db_root,
        db_path=target,
    )
    if quarterly is None or quarterly.empty:
        return pd.DataFrame(), "not_available"

    as_of_ts = pd.Timestamp(as_of_date).normalize() if as_of_date is not None else pd.Timestamp.today().normalize()
    price_start = (as_of_ts - pd.Timedelta(days=420)).strftime("%Y-%m-%d")
    price_end = as_of_ts.strftime("%Y-%m-%d")
    close_history, price_source = load_shared_close_prices_for_symbols(
        normalized,
        start_date=price_start,
        end_date=price_end,
        shared_db_root=shared_db_root,
        db_path=target,
    )
    market_caps, market_cap_source = load_shared_market_caps_for_symbols(
        normalized,
        start_date=price_start,
        end_date=price_end,
        shared_db_root=shared_db_root,
        db_path=target,
    )

    close_history = close_history if close_history is not None else pd.DataFrame()
    market_caps = market_caps if market_caps is not None else pd.DataFrame()

    rows: list[dict[str, object]] = []
    for symbol, sub in quarterly.groupby("symbol", sort=True):
        ordered = sub.copy()
        ordered["fiscal_date"] = pd.to_datetime(ordered["fiscal_date"], errors="coerce")
        ordered = ordered.dropna(subset=["fiscal_date"]).sort_values("fiscal_date", ascending=False).reset_index(drop=True)
        if ordered.empty:
            continue

        source_is_sec = (
            "source" in ordered.columns
            and ordered["source"].astype(str).str.lower().str.startswith("sec:").any()
        )
        ttm_count = max(int(limit_per_symbol), 4)
        ttm_rows = ordered.head(ttm_count).copy()
        latest = ordered.iloc[0]

        net_income_values = _recent_quarter_values(ordered, "net_income", count=ttm_count, source_is_sec=source_is_sec)
        eps_values = _recent_quarter_values(ordered, "diluted_eps", count=ttm_count, source_is_sec=source_is_sec)
        ttm_net_income = float(sum(net_income_values)) if net_income_values else np.nan
        ttm_eps = float(sum(eps_values)) if eps_values else np.nan
        equity_values = pd.to_numeric(ttm_rows["stockholders_equity"], errors="coerce").dropna()
        latest_equity = pd.to_numeric(pd.Series([latest.get("stockholders_equity")]), errors="coerce").dropna()
        latest_equity_value = float(latest_equity.iloc[0]) if not latest_equity.empty else np.nan
        average_equity = float(equity_values.mean()) if not equity_values.empty else np.nan

        latest_net_income = pd.to_numeric(pd.Series([latest.get("net_income")]), errors="coerce").dropna()
        latest_eps = pd.to_numeric(pd.Series([latest.get("diluted_eps")]), errors="coerce").dropna()
        latest_net_income_value = float(latest_net_income.iloc[0]) if not latest_net_income.empty else np.nan
        latest_eps_value = float(latest_eps.iloc[0]) if not latest_eps.empty else np.nan

        price_date = None
        latest_price = np.nan
        if symbol in close_history.columns:
            price_date, latest_price_value = _last_value_on_or_before(close_history[symbol], as_of_ts)
            latest_price = latest_price_value if latest_price_value is not None else np.nan

        market_cap_date = None
        latest_market_cap = np.nan
        if symbol in market_caps.columns:
            market_cap_date, market_cap_value = _last_value_on_or_before(market_caps[symbol], as_of_ts)
            latest_market_cap = market_cap_value if market_cap_value is not None else np.nan

        implied_shares = None
        if np.isfinite(latest_net_income_value) and np.isfinite(latest_eps_value) and latest_eps_value != 0.0:
            implied_shares = abs(float(latest_net_income_value) / float(latest_eps_value))
        elif np.isfinite(ttm_net_income) and np.isfinite(ttm_eps) and float(ttm_eps) != 0.0:
            implied_shares = abs(float(ttm_net_income) / float(ttm_eps))

        if (not np.isfinite(latest_market_cap)) and np.isfinite(latest_price) and implied_shares is not None:
            latest_market_cap = float(latest_price) * float(implied_shares)

        positive_ttm_eps = _safe_positive(ttm_eps)
        positive_latest_equity = _safe_positive(latest_equity_value)
        positive_average_equity = _safe_positive(average_equity)
        positive_market_cap = _safe_positive(latest_market_cap)
        positive_latest_price = _safe_positive(latest_price)

        roe = (
            float(ttm_net_income) / float(positive_average_equity)
            if np.isfinite(ttm_net_income) and positive_average_equity is not None
            else np.nan
        )
        per = (
            float(positive_latest_price) / float(positive_ttm_eps)
            if positive_latest_price is not None and positive_ttm_eps is not None
            else np.nan
        )
        pbr = (
            float(positive_market_cap) / float(positive_latest_equity)
            if positive_market_cap is not None and positive_latest_equity is not None
            else np.nan
        )

        latest_debt = pd.to_numeric(pd.Series([latest.get("total_debt")]), errors="coerce").dropna()
        current_assets = pd.to_numeric(pd.Series([latest.get("current_assets")]), errors="coerce").dropna()
        current_liabilities = pd.to_numeric(pd.Series([latest.get("current_liabilities")]), errors="coerce").dropna()
        latest_debt_value = float(latest_debt.iloc[0]) if not latest_debt.empty else np.nan
        current_assets_value = float(current_assets.iloc[0]) if not current_assets.empty else np.nan
        current_liabilities_value = float(current_liabilities.iloc[0]) if not current_liabilities.empty else np.nan

        debt_to_equity = (
            (float(latest_debt_value) / float(positive_latest_equity)) * 100.0
            if np.isfinite(latest_debt_value) and positive_latest_equity is not None
            else np.nan
        )
        current_ratio = (
            float(current_assets_value) / float(current_liabilities_value)
            if np.isfinite(current_assets_value) and _safe_positive(current_liabilities_value) is not None
            else np.nan
        )

        year_high = np.nan
        year_low = np.nan
        if symbol in close_history.columns:
            window = pd.to_numeric(close_history[symbol], errors="coerce").dropna()
            window.index = pd.to_datetime(window.index, errors="coerce")
            window = window[(~window.index.isna()) & (window.index <= as_of_ts)]
            window = window[window.index >= (as_of_ts - pd.Timedelta(days=365))]
            if not window.empty:
                year_high = float(window.max())
                year_low = float(window.min())

        rows.append(
            {
                "symbol": symbol,
                "as_of_date": as_of_ts.strftime("%Y-%m-%d"),
                "price_date": price_date.strftime("%Y-%m-%d") if price_date is not None else None,
                "market_cap_date": market_cap_date.strftime("%Y-%m-%d") if market_cap_date is not None else None,
                "latest_fiscal_date": pd.Timestamp(latest["fiscal_date"]).strftime("%Y-%m-%d"),
                "statement_count": int(len(ordered.index)),
                "latest_price": latest_price if np.isfinite(latest_price) else np.nan,
                "market_cap": latest_market_cap if np.isfinite(latest_market_cap) else np.nan,
                "ttm_net_income": float(ttm_net_income) if np.isfinite(ttm_net_income) else np.nan,
                "ttm_eps": float(ttm_eps) if np.isfinite(ttm_eps) else np.nan,
                "roe": roe if np.isfinite(roe) else np.nan,
                "per": per if np.isfinite(per) else np.nan,
                "pbr": pbr if np.isfinite(pbr) else np.nan,
                "latest_equity": latest_equity_value if np.isfinite(latest_equity_value) else np.nan,
                "average_equity": average_equity if np.isfinite(average_equity) else np.nan,
                "latest_debt": latest_debt_value if np.isfinite(latest_debt_value) else np.nan,
                "current_assets": current_assets_value if np.isfinite(current_assets_value) else np.nan,
                "current_liabilities": current_liabilities_value if np.isfinite(current_liabilities_value) else np.nan,
                "debt_to_equity": debt_to_equity if np.isfinite(debt_to_equity) else np.nan,
                "current_ratio": current_ratio if np.isfinite(current_ratio) else np.nan,
                "year_high": year_high if np.isfinite(year_high) else np.nan,
                "year_low": year_low if np.isfinite(year_low) else np.nan,
                "source": quarterly_source or f"sqlite:{target.as_posix()}",
            }
        )

    if not rows:
        return pd.DataFrame(), "not_available"

    frame = pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)
    source_parts = [part for part in [quarterly_source, price_source, market_cap_source] if part]
    source = " | ".join(dict.fromkeys(source_parts)) if source_parts else f"sqlite:{target.as_posix()}"
    return frame, source
