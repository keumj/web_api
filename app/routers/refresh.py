from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from app.form import read_form
from app.services import auth_service, portfolio_service, refresh_service

router = APIRouter()


@router.get("/refresh", response_class=HTMLResponse)
def refresh_page(
    request: Request,
    lookback_days: int = portfolio_service.DEFAULT_LOOKBACK_DAYS,
    start_date: str | None = None,
    end_date: str | None = None,
) -> HTMLResponse:
    user = auth_service.current_user(request)
    return HTMLResponse(refresh_service.render_original_refresh_page(
        lookback_days=lookback_days,
        start_date=start_date,
        end_date=end_date,
        admin=bool(user and user.is_admin),
    ))


@router.get("/refresh_status")
def refresh_status() -> dict[str, object]:
    return refresh_service.original_status_payload()


@router.post("/run_refresh")
async def run_refresh(request: Request) -> Response:
    form = await read_form(request)
    result = refresh_service.start_original_job(str(form.get("job_id", "")))
    wants_json = (
        request.headers.get("x-requested-with", "").lower() == "fetch"
        or "application/json" in request.headers.get("accept", "").lower()
    )
    if wants_json:
        status_code = 200 if result.get("ok") else 400
        return JSONResponse(result, status_code=status_code)
    return HTMLResponse(refresh_service.render_original_refresh_page(
        lookback_days=int(form.get("lookback_days", portfolio_service.DEFAULT_LOOKBACK_DAYS) or portfolio_service.DEFAULT_LOOKBACK_DAYS),
        start_date=form.get("start_date") or None,
        end_date=form.get("end_date") or None,
        admin=bool((user := auth_service.current_user(request)) and user.is_admin),
    ))


@router.get("/api/refresh/jobs")
def refresh_jobs() -> dict[str, object]:
    return {"jobs": refresh_service.list_jobs()}


@router.post("/api/refresh/jobs/{job_id}/run")
def start_refresh_job(job_id: str) -> dict[str, object]:
    return refresh_service.start_original_job(job_id)


@router.post("/refresh/{job_id}/run")
def start_refresh_job_from_page(job_id: str) -> RedirectResponse:
    refresh_service.start_original_job(job_id)
    return RedirectResponse("/refresh", status_code=303)
