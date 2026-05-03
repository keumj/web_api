from __future__ import annotations

import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from bs4 import BeautifulSoup

from pipeline_portfolio import web_gui as portfolio_web
from pipeline_portfolio.analysis import (
    DEFAULT_CASH_BUFFER_PCT,
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_MAX_POSITION_PCT,
    DEFAULT_OPTIMIZATION_UNIVERSE_SIZE,
    DEFAULT_SECTOR_CAP_PCT,
    add_trade,
    analyze_virtual_trade,
    build_portfolio_dashboard,
    build_portfolio_optimization,
    delete_trade,
    _get_db_max_date,
)

from app.web import add_start_page_link
from app.services.dataframe import frame_records
from app.services.auth_service import AuthUser, portfolio_db_for_user
from app.services import db_service


DEFAULT_START_DATE = "2025-12-31"
HISTORICAL_PAGES = {"virtual-trade", "optimization"}


def _remove_refresh_links(soup: BeautifulSoup) -> None:
    for link in soup.find_all("a", href=True):
        href = str(link.get("href") or "")
        if href == "/refresh" or href.startswith("/refresh?") or href.startswith("/refresh/"):
            link.decompose()


@dataclass
class PortfolioRange:
    lookback_days: int
    start_date: str
    end_date: str


def _prepare_portfolio_html(page: str, html: str, *, user: AuthUser | None = None) -> str:
    html = add_start_page_link(html)
    soup: BeautifulSoup | None = None
    if user is not None:
        soup = BeautifulSoup(html, "html.parser")
        wrap = soup.select_one(".wrap")
        if wrap is not None and wrap.find(attrs={"data-user-bar": "1"}) is None:
            bar = soup.new_tag("div")
            bar["class"] = "notice ok"
            bar["style"] = "display:flex; justify-content:space-between; gap:12px; align-items:center; flex-wrap:wrap;"
            bar["data-user-bar"] = "1"
            label = soup.new_tag("span")
            role = "관리자" if user.is_admin else "일반"
            label.string = f"로그인 사용자: {user.username} ({role})"
            actions = soup.new_tag("span")
            actions["style"] = "display:flex; gap:10px; align-items:center; flex-wrap:wrap;"
            if user.is_admin:
                admin_link = soup.new_tag("a", href="/admin/users")
                admin_link.string = "사용자 관리"
                actions.append(admin_link)
            link = soup.new_tag("a", href="/logout")
            link.string = "로그아웃"
            actions.append(link)
            bar.append(label)
            bar.append(actions)
            wrap.insert(1, bar)
    if soup is None:
        soup = BeautifulSoup(html, "html.parser")
    _remove_refresh_links(soup)
    if soup is not None:
        html = str(soup)
    if page not in HISTORICAL_PAGES:
        return html

    latest_db_date = _latest_db_date()
    soup = BeautifulSoup(html, "html.parser")
    for form in soup.find_all("form", attrs={"method": "get"}):
        action = str(form.get("action", ""))
        if action not in {"/virtual-trade", "/optimization"}:
            continue
        start_field = form.find("input", attrs={"name": "start_date"})
        if start_field is None or str(start_field.get("type", "")).lower() == "hidden":
            continue
        for field_name in ("start_date", "end_date"):
            field = form.find("input", attrs={"name": field_name})
            wrapper = field.find_parent("div") if field is not None else None
            if wrapper is not None:
                wrapper.decompose()
        button = form.find("button")
        if button is not None:
            button.string = "최신 DB 기준으로 분석 실행"
        card = form.find_parent("div", class_="card")
        if card is not None and card.find(attrs={"data-historical-note": "1"}) is None:
            for old_note in card.find_all("div", class_="small"):
                old_note.decompose()
            note = soup.new_tag("div")
            note["class"] = "small"
            note["style"] = "margin-top:8px;"
            note["data-historical-note"] = "1"
            note.string = f"최신 DB 날짜는 {latest_db_date}이며, 동 일자를 기준으로 분석합니다."
            card.append(note)
    return str(soup)


def _portfolio_db(user: AuthUser | None) -> str | None:
    if user is not None and db_service.using_remote_app_db():
        return None
    return str(portfolio_db_for_user(user)) if user is not None else None


def _portfolio_user_id(user: AuthUser | None) -> str | None:
    return user.id if user is not None and db_service.using_remote_app_db() else None


def _require_user_for_remote_db(user: AuthUser | None) -> None:
    if db_service.using_remote_app_db() and user is None:
        raise PermissionError("Turso portfolio storage requires an authenticated user.")


def _latest_db_date() -> str:
    return _get_db_max_date().strftime("%Y-%m-%d")


def _latest_db_range(lookback_days: int | None = None) -> PortfolioRange:
    return resolve_range(DEFAULT_START_DATE, _latest_db_date(), lookback_days)


def resolve_range(
    start_date: str | None = None,
    end_date: str | None = None,
    lookback_days: int | None = None,
) -> PortfolioRange:
    lookback = max(int(lookback_days or DEFAULT_LOOKBACK_DAYS), 21)
    today = datetime.now().date()
    end = end_date or today.isoformat()
    start = start_date or DEFAULT_START_DATE
    if start > end:
        start = DEFAULT_START_DATE
    return PortfolioRange(lookback_days=lookback, start_date=start, end_date=end)


def dashboard_payload(
    *,
    user: AuthUser | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, object]:
    _require_user_for_remote_db(user)
    date_range = resolve_range(start_date, end_date, lookback_days)
    dashboard = build_portfolio_dashboard(
        portfolio_db=_portfolio_db(user),
        portfolio_user_id=_portfolio_user_id(user),
        lookback_days=date_range.lookback_days,
        start_date=date_range.start_date,
        end_date=date_range.end_date,
    )
    return {
        "as_of_date": dashboard.as_of_date,
        "range": date_range.__dict__,
        "summary": frame_records(dashboard.portfolio_summary, max_rows=5),
        "positions": frame_records(dashboard.positions, max_rows=100),
        "holdings_performance": frame_records(dashboard.holdings_performance, max_rows=100),
        "attribution": frame_records(dashboard.attribution, max_rows=100),
        "risk_summary": frame_records(dashboard.risk_summary, max_rows=20),
        "scoring": frame_records(dashboard.scoring, max_rows=100),
        "diagnostics": dashboard.diagnostics,
    }


def render_page(
    page: str,
    *,
    user: AuthUser | None = None,
    run: bool = False,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    start_date: str | None = None,
    end_date: str | None = None,
    universe_size: int = DEFAULT_OPTIMIZATION_UNIVERSE_SIZE,
    sector_cap_pct: float = DEFAULT_SECTOR_CAP_PCT,
    max_position_pct: float = DEFAULT_MAX_POSITION_PCT,
    cash_buffer_pct: float = DEFAULT_CASH_BUFFER_PCT,
    message: str | None = None,
    error: str | None = None,
) -> str:
    _require_user_for_remote_db(user)
    date_range = resolve_range(start_date, end_date, lookback_days)
    if page in HISTORICAL_PAGES:
        date_range = _latest_db_range(lookback_days)
    dashboard = None
    page_error = error
    if run:
        try:
            dashboard = build_portfolio_dashboard(
                portfolio_db=_portfolio_db(user),
                portfolio_user_id=_portfolio_user_id(user),
                lookback_days=date_range.lookback_days,
                start_date=date_range.start_date,
                end_date=date_range.end_date,
            )
        except Exception as exc:
            page_error = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc(limit=3)}"
    ctx = portfolio_web._PageContext(
        dashboard=dashboard,
        lookback_days=date_range.lookback_days,
        start_date=date_range.start_date,
        end_date=date_range.end_date,
        message=message,
        error=page_error,
    )
    renderers: dict[str, Callable[[portfolio_web._PageContext], str]] = {
        "data-entry": portfolio_web._data_entry_page,
        "overview": portfolio_web._overview_page,
        "attribution": portfolio_web._attribution_page,
        "risk": portfolio_web._risk_page,
        "scoring": portfolio_web._scoring_page,
        "virtual-trade": portfolio_web._virtual_trade_page,
    }
    if page == "optimization":
        optimization = None
        if run and page_error is None:
            try:
                optimization = build_portfolio_optimization(
                    portfolio_db=_portfolio_db(user),
                    portfolio_user_id=_portfolio_user_id(user),
                    lookback_days=date_range.lookback_days,
                    start_date=date_range.start_date,
                    end_date=date_range.end_date,
                    universe_size=universe_size,
                    sector_cap_pct=sector_cap_pct,
                    max_position_pct=max_position_pct,
                    cash_buffer_pct=cash_buffer_pct,
                )
            except Exception as exc:
                page_error = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc(limit=3)}"
        ctx.error = page_error
        ctx.optimization = optimization
        ctx.optimization_params = {
            "universe_size": universe_size,
            "sector_cap_pct": sector_cap_pct,
            "max_position_pct": max_position_pct,
            "cash_buffer_pct": cash_buffer_pct,
        }
        return _prepare_portfolio_html(page, portfolio_web._optimization_page(ctx), user=user)
    return _prepare_portfolio_html(page, renderers.get(page, portfolio_web._overview_page)(ctx), user=user)


def create_trade(form: dict[str, str], *, user: AuthUser | None = None) -> None:
    _require_user_for_remote_db(user)
    add_trade(
        trade_date=form.get("trade_date", ""),
        ticker=form.get("ticker", ""),
        side=form.get("side", ""),
        quantity=float(form.get("quantity", "0") or 0),
        price=float(form.get("price", "0") or 0),
        fees=float(form.get("fees", "0") or 0),
        notes=form.get("notes", ""),
        db_path=_portfolio_db(user),
        user_id=_portfolio_user_id(user),
    )


def remove_trade(trade_id: int, *, user: AuthUser | None = None) -> None:
    _require_user_for_remote_db(user)
    delete_trade(int(trade_id), db_path=_portfolio_db(user), user_id=_portfolio_user_id(user))


def virtual_trade_payload(form: dict[str, str], *, user: AuthUser | None = None) -> dict[str, object]:
    _require_user_for_remote_db(user)
    date_range = _latest_db_range(int(form.get("lookback_days", DEFAULT_LOOKBACK_DAYS) or DEFAULT_LOOKBACK_DAYS))
    result = analyze_virtual_trade(
        portfolio_db=_portfolio_db(user),
        portfolio_user_id=_portfolio_user_id(user),
        ticker=form.get("ticker", ""),
        side=form.get("side", ""),
        quantity=float(form.get("quantity", "0") or 0),
        price=float(form["price"]) if str(form.get("price", "")).strip() else None,
        fees=float(form.get("fees", "0") or 0),
        lookback_days=date_range.lookback_days,
        start_date=date_range.start_date,
        end_date=date_range.end_date,
        forecast_horizon_days=int(form.get("forecast_horizon_days", "10") or 10),
    )
    return {
        "input_summary": frame_records(result.input_summary),
        "before_summary": frame_records(result.before_summary),
        "after_summary": frame_records(result.after_summary),
        "position_changes": frame_records(result.position_changes, max_rows=100),
        "risk_changes": frame_records(result.risk_changes),
        "diagnostics": result.diagnostics,
    }


def render_virtual_trade(form: dict[str, str], *, user: AuthUser | None = None) -> str:
    _require_user_for_remote_db(user)
    date_range = resolve_range(
        form.get("start_date"),
        form.get("end_date"),
        int(form.get("lookback_days", DEFAULT_LOOKBACK_DAYS) or DEFAULT_LOOKBACK_DAYS),
    )
    date_range = _latest_db_range(date_range.lookback_days)
    dashboard = None
    dashboard_error = None
    try:
        dashboard = build_portfolio_dashboard(
            portfolio_db=_portfolio_db(user),
            portfolio_user_id=_portfolio_user_id(user),
            lookback_days=date_range.lookback_days,
            start_date=date_range.start_date,
            end_date=date_range.end_date,
        )
    except Exception as exc:
        dashboard_error = f"{type(exc).__name__}: {exc}"
    result = analyze_virtual_trade(
        portfolio_db=_portfolio_db(user),
        portfolio_user_id=_portfolio_user_id(user),
        ticker=form.get("ticker", ""),
        side=form.get("side", ""),
        quantity=float(form.get("quantity", "0") or 0),
        price=float(form["price"]) if str(form.get("price", "")).strip() else None,
        fees=float(form.get("fees", "0") or 0),
        lookback_days=date_range.lookback_days,
        start_date=date_range.start_date,
        end_date=date_range.end_date,
        forecast_horizon_days=int(form.get("forecast_horizon_days", "10") or 10),
    )
    html = portfolio_web._virtual_trade_page(
        portfolio_web._PageContext(
            dashboard=dashboard,
            lookback_days=date_range.lookback_days,
            start_date=date_range.start_date,
            end_date=date_range.end_date,
            message="가상 거래 계산이 완료되었습니다.",
            error=dashboard_error,
            virtual_result=result,
        )
    )
    return _prepare_portfolio_html("virtual-trade", html, user=user)
