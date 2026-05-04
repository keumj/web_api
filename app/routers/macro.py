from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from app.services import macro_service

router = APIRouter(prefix="/macro")


@router.get("/{page}", response_class=HTMLResponse)
def macro_page(
    page: str,
    start_date: str | None = None,
    lookback_days: int = 504,
) -> HTMLResponse:
    return HTMLResponse(macro_service.render(page, start_date=start_date, lookback_days=lookback_days))
