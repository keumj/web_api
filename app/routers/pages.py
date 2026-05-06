from __future__ import annotations

import html

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app.routers.auth import _safe_next, auth_panel
from app.services import auth_service, refresh_state
from app.settings import settings
from app.web import shell

router = APIRouter()


def _refresh_notice_card(*, visible: bool) -> str:
    if not visible:
        return ""

    state = refresh_state.notice_state()
    notice = state.get("notice") if isinstance(state.get("notice"), dict) else {}
    severity = str(notice.get("severity") or "warning")
    tone, border = {
        "success": ("#ecfdf5", "#bbf7d0"),
        "warning": ("#fff7ed", "#fed7aa"),
        "error": ("#fef2f2", "#fecaca"),
    }.get(severity, ("#fff7ed", "#fed7aa"))
    headline = str(notice.get("headline") or "데이터 갱신 상태를 확인해야 합니다.")

    git = state.get("git") if isinstance(state.get("git"), dict) else {}
    git_lines = [str(line) for line in git.get("lines", [])] if isinstance(git.get("lines"), list) else []
    git_html = "".join(f"<li>{html.escape(line)}</li>" for line in git_lines) or "<li>변경 없음</li>"

    scheduler_lines = (
        [str(line) for line in state.get("scheduler_log_tail", [])]
        if isinstance(state.get("scheduler_log_tail"), list)
        else []
    )
    scheduler_html = (
        "".join(f"<li>{html.escape(line)}</li>" for line in scheduler_lines)
        or "<li>아직 스케줄러 실행 로그가 없습니다.</li>"
    )

    error_parts = [str(value) for value in (git.get("error"), state.get("state_read_error"), state.get("last_error")) if value]
    error_html = "".join(f"<div><strong>오류</strong> {html.escape(part)}</div>" for part in error_parts)

    data = state.get("data") if isinstance(state.get("data"), dict) else {}
    prices = data.get("prices") if isinstance(data.get("prices"), dict) else {}
    quarterly = data.get("quarterly") if isinstance(data.get("quarterly"), dict) else {}
    news = data.get("news") if isinstance(data.get("news"), dict) else {}
    macro = data.get("macro") if isinstance(data.get("macro"), dict) else {}
    data_html = "".join(
        f"<li>{html.escape(label)}: {html.escape(value)}</li>"
        for label, value in (
            ("가격", f"latest={prices.get('latest_date') or '-'} rows={prices.get('rows') or '-'}"),
            ("분기 재무", f"latest={quarterly.get('latest_fiscal_date') or '-'} rows={quarterly.get('rows') or '-'}"),
            ("뉴스", f"latest={news.get('latest_publish_date') or '-'} rows={news.get('rows') or '-'}"),
            ("매크로", f"latest={macro.get('latest_date') or '-'} rows={macro.get('rows') or '-'}"),
        )
    )
    exit_code = state.get("last_exit_code") if state.get("last_exit_code") is not None else "-"
    run_meta = (
        f"상태={state.get('status') or 'unknown'} / "
        f"출처={state.get('source') or '-'} / "
        f"시작={state.get('last_started_at') or '-'} / "
        f"종료={state.get('last_finished_at') or '-'} / "
        f"exit={exit_code}"
    )

    return f"""
      <div class="service-card" style="background:{tone}; border-color:{border};">
        <h3 style="margin:0 0 6px;">데이터 갱신 알림</h3>
        <p class="service-muted" style="margin:0 0 10px;">
          로컬 갱신 작업의 시작/종료 상태, SQLite 변경 여부, 최신 데이터 날짜를 구조화된 상태 파일 기준으로 확인합니다.
        </p>
        <div style="display:grid; gap:8px; font-size:13px;">
          <div><strong>{html.escape(headline)}</strong></div>
          <div class="service-muted">{html.escape(run_meta)}</div>
          {error_html}
          <div>
            <div class="service-muted">SQLite Git 상태</div>
            <ul style="margin:4px 0 0 18px; padding:0;">{git_html}</ul>
          </div>
          <div>
            <div class="service-muted">최신 데이터</div>
            <ul style="margin:4px 0 0 18px; padding:0;">{data_html}</ul>
          </div>
          <div>
            <div class="service-muted">최근 스케줄러 로그</div>
            <ul style="margin:4px 0 0 18px; padding:0;">{scheduler_html}</ul>
          </div>
        </div>
      </div>
    """


@router.get("/", response_class=HTMLResponse)
def index(request: Request, next: str | None = None, auth_error: str | None = None) -> HTMLResponse:
    next_url = _safe_next(next)
    error = "사용자명 또는 비밀번호가 올바르지 않습니다." if auth_error == "login" else str(auth_error or "")
    user = auth_service.current_user(request)
    can_see_refresh_notice = (not settings.auth_enabled) or bool(user and user.is_admin)
    refresh_notice = _refresh_notice_card(visible=can_see_refresh_notice)
    admin_card = (
        """
        <a class="service-card" href="/admin/users">
          <h3>사용자 관리</h3>
          <p>계정 생성, 관리자 권한, 사용자 비활성화, 비밀번호 초기화를 관리합니다.</p>
        </a>
        """
        if user is not None and user.is_admin
        else ""
    )
    macro_card = (
        """
        <a class="service-card" href="/macro/overview">
          <h3>거시분석</h3>
          <p>금리, 달러, 위험신호, 팩터 프레임워크를 종목/뉴스 분석의 배경 환경으로 제공합니다.</p>
        </a>
        """
        if settings.enable_macro
        else ""
    )
    body = f"""
    <div class="service-stack">
      <div class="service-card">
        <h1>Keumj 포트폴리오 분석 서비스</h1>
        <p class="service-muted">포트폴리오를 중심으로 종목 분석, 뉴스 분석, 데이터 갱신 기능을 한 포트에서 실행합니다.</p>
      </div>
      {refresh_notice}
      <div class="service-grid">
        <a class="service-card" href="/portfolio/overview">
          <h3>포트폴리오</h3>
          <p>보유 종목, 성과, 위험, 최적화 분석을 실행합니다.</p>
        </a>
        <a class="service-card" href="/stock/financials">
          <h3>종목 분석</h3>
          <p>개별 종목 예측, 재무, 기술적 분석, 의사결정 화면으로 이동합니다.</p>
        </a>
        <a class="service-card" href="/stock-news/overview">
          <h3>뉴스 분석</h3>
          <p>뉴스 기반 이벤트 스터디, 팩터 영향, 토픽, 가격 반응 분석을 실행합니다.</p>
        </a>
        {macro_card}
        {admin_card}
      </div>
      {auth_panel(next_url=next_url, user=user, error=error)}
    </div>
    """
    return HTMLResponse(shell("Keumj Portfolio Lab", body, admin=bool(user and user.is_admin)))


@router.get("/healthz")
def healthz() -> dict[str, object]:
    return {"ok": True, "service": "keumj-single-port"}


@router.get("/external_command_state")
def external_command_state() -> dict[str, object]:
    return {"command_id": 0, "navigate_url": None}


def _redirect_with_query(request: Request, path: str) -> RedirectResponse:
    query = str(request.url.query)
    target = f"{path}?{query}" if query else path
    return RedirectResponse(target)


@router.get("/overview")
def legacy_overview(request: Request) -> RedirectResponse:
    return _redirect_with_query(request, "/portfolio/overview")


@router.get("/portfolio")
def portfolio_home(request: Request) -> RedirectResponse:
    return _redirect_with_query(request, "/portfolio/overview")


@router.get("/stock")
def stock_home(request: Request) -> RedirectResponse:
    return _redirect_with_query(request, "/stock/financials")


@router.get("/stock-news")
def stock_news_home(request: Request) -> RedirectResponse:
    return _redirect_with_query(request, "/stock-news/overview")


@router.get("/macro")
def macro_home(request: Request) -> RedirectResponse:
    if not settings.enable_macro:
        return RedirectResponse("/", status_code=303)
    return _redirect_with_query(request, "/macro/overview")


@router.get("/stock-forecast")
def legacy_stock_forecast(request: Request) -> RedirectResponse:
    return _redirect_with_query(request, "/stock/forecast")


@router.get("/stock-financials")
def legacy_stock_financials(request: Request) -> RedirectResponse:
    return _redirect_with_query(request, "/stock/financials")


@router.get("/stock-technical")
def legacy_stock_technical(request: Request) -> RedirectResponse:
    return _redirect_with_query(request, "/stock/technical")


@router.get("/stock-wfv")
def legacy_stock_walk_forward(request: Request) -> RedirectResponse:
    return _redirect_with_query(request, "/stock/walk-forward")
