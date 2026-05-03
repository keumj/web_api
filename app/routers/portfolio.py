from __future__ import annotations

from urllib.parse import quote

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app.form import read_form
from app.services import auth_service, portfolio_service

router = APIRouter()


@router.get("/portfolio/{page}", response_class=HTMLResponse)
def portfolio_page(
    request: Request,
    page: str,
    intent: str | None = None,
    lookback_days: int = portfolio_service.DEFAULT_LOOKBACK_DAYS,
    start_date: str | None = None,
    end_date: str | None = None,
    universe_size: int = portfolio_service.DEFAULT_OPTIMIZATION_UNIVERSE_SIZE,
    sector_cap_pct: float = portfolio_service.DEFAULT_SECTOR_CAP_PCT,
    max_position_pct: float = portfolio_service.DEFAULT_MAX_POSITION_PCT,
    cash_buffer_pct: float = portfolio_service.DEFAULT_CASH_BUFFER_PCT,
    message: str | None = None,
    error: str | None = None,
) -> HTMLResponse:
    return HTMLResponse(
        portfolio_service.render_page(
            page,
            user=auth_service.current_user(request),
            run=str(intent or "").lower() in {"run", "analyze", "refresh"},
            lookback_days=lookback_days,
            start_date=start_date,
            end_date=end_date,
            universe_size=universe_size,
            sector_cap_pct=sector_cap_pct,
            max_position_pct=max_position_pct,
            cash_buffer_pct=cash_buffer_pct,
            message=message,
            error=error,
        )
    )


@router.get("/{legacy_page}", response_class=HTMLResponse)
def portfolio_legacy_page(
    request: Request,
    legacy_page: str,
    intent: str | None = None,
    lookback_days: int = portfolio_service.DEFAULT_LOOKBACK_DAYS,
    start_date: str | None = None,
    end_date: str | None = None,
    universe_size: int = portfolio_service.DEFAULT_OPTIMIZATION_UNIVERSE_SIZE,
    sector_cap_pct: float = portfolio_service.DEFAULT_SECTOR_CAP_PCT,
    max_position_pct: float = portfolio_service.DEFAULT_MAX_POSITION_PCT,
    cash_buffer_pct: float = portfolio_service.DEFAULT_CASH_BUFFER_PCT,
    message: str | None = None,
    error: str | None = None,
) -> HTMLResponse:
    if legacy_page not in {"data-entry", "attribution", "risk", "scoring", "virtual-trade", "optimization"}:
        return HTMLResponse("<h1>Not Found</h1>", status_code=404)
    return HTMLResponse(
        portfolio_service.render_page(
            legacy_page,
            user=auth_service.current_user(request),
            run=str(intent or "").lower() in {"run", "analyze", "refresh"},
            lookback_days=lookback_days,
            start_date=start_date,
            end_date=end_date,
            universe_size=universe_size,
            sector_cap_pct=sector_cap_pct,
            max_position_pct=max_position_pct,
            cash_buffer_pct=cash_buffer_pct,
            message=message,
            error=error,
        )
    )


@router.get("/api/portfolio/dashboard")
def portfolio_dashboard(
    request: Request,
    lookback_days: int = portfolio_service.DEFAULT_LOOKBACK_DAYS,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, object]:
    return portfolio_service.dashboard_payload(
        user=auth_service.current_user(request),
        lookback_days=lookback_days,
        start_date=start_date,
        end_date=end_date,
    )


@router.post("/run_add_trade")
async def run_add_trade(request: Request) -> RedirectResponse:
    form = await read_form(request)
    try:
        portfolio_service.create_trade(form, user=auth_service.current_user(request))
    except Exception as exc:
        return RedirectResponse(
            f"/portfolio/data-entry?intent=run&error={quote(type(exc).__name__ + ': ' + str(exc), safe='')}",
            status_code=303,
        )
    return RedirectResponse("/portfolio/data-entry?intent=run&message=trade_saved", status_code=303)


@router.post("/run_delete_trade")
async def run_delete_trade(request: Request) -> RedirectResponse:
    form = await read_form(request)
    try:
        portfolio_service.remove_trade(int(form.get("trade_id", "0") or 0), user=auth_service.current_user(request))
    except Exception as exc:
        return RedirectResponse(
            f"/portfolio/data-entry?intent=run&error={quote(type(exc).__name__ + ': ' + str(exc), safe='')}",
            status_code=303,
        )
    return RedirectResponse("/portfolio/data-entry?intent=run&message=trade_deleted", status_code=303)


@router.post("/run_virtual_trade", response_class=HTMLResponse)
async def run_virtual_trade(request: Request) -> HTMLResponse:
    form = await read_form(request)
    return HTMLResponse(portfolio_service.render_virtual_trade(form, user=auth_service.current_user(request)))
