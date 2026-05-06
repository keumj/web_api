from __future__ import annotations

import html
import subprocess

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app.routers.auth import _safe_next, auth_panel
from app.services import auth_service
from app.settings import settings
from app.web import shell

router = APIRouter()

SQLITE_REFRESH_TARGETS = (
    "data/sp500_shared_db/sp500_shared_prices.sqlite",
    "data/macro_prices.sqlite",
)


def _latest_scheduler_lines() -> list[str]:
    log_path = settings.project_root / "outputs" / "refresh_local_data_scheduler.log"
    if not log_path.exists():
        return []
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []
    matches = [line.strip() for line in lines if "Scheduled refresh" in line or "SQLite auto sync is disabled" in line]
    return matches[-3:]


def _sqlite_git_status() -> tuple[bool, list[str], str]:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--", *SQLITE_REFRESH_TARGETS],
            cwd=str(settings.project_root),
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=5,
        )
    except Exception as exc:
        return False, [], f"SQLite Git 상태를 확인하지 못했습니다: {type(exc).__name__}"
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        return False, [], f"SQLite Git 상태를 확인하지 못했습니다: {detail or 'git status 실패'}"
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return bool(lines), lines, ""


def _refresh_notice_card(*, admin: bool) -> str:
    if not admin:
        return ""
    has_changes, status_lines, error = _sqlite_git_status()
    scheduler_lines = _latest_scheduler_lines()
    tone = "#fff7ed" if has_changes or error else "#ecfdf5"
    border = "#fed7aa" if has_changes or error else "#bbf7d0"
    headline = "GitHub에 아직 반영되지 않은 SQLite 변경이 있습니다." if has_changes else "현재 추적 중인 SQLite 변경은 없습니다."
    if error:
        headline = "SQLite Git 상태 확인이 필요합니다."
    status_html = "".join(f"<li>{html.escape(line)}</li>" for line in status_lines) or "<li>변경 없음</li>"
    scheduler_html = "".join(f"<li>{html.escape(line)}</li>" for line in scheduler_lines) or "<li>아직 스케줄 실행 로그가 없습니다.</li>"
    error_html = f" {html.escape(error)}" if error else ""
    return f"""
      <div class="service-card" style="background:{tone}; border-color:{border};">
        <h3 style="margin:0 0 6px;">데이터 갱신 알림</h3>
        <p class="service-muted" style="margin:0 0 10px;">
          일일 데이터 갱신 후 GitHub 강제 push는 꺼져 있습니다. 아래 상태를 확인한 뒤 필요한 경우에만 수동으로 커밋/푸시하세요.
        </p>
        <div style="display:grid; gap:8px; font-size:13px;">
          <div><strong>{html.escape(headline)}</strong>{error_html}</div>
          <div>
            <div class="service-muted">SQLite Git 상태</div>
            <ul style="margin:4px 0 0 18px; padding:0;">{status_html}</ul>
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
    if auth_error == "login":
        error = "사용자명 또는 비밀번호가 올바르지 않습니다."
    else:
        error = str(auth_error or "")
    user = auth_service.current_user(request)
    refresh_notice = _refresh_notice_card(admin=bool(user and user.is_admin))
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
          <p>금리, 달러, 위험신호, 섹터 플레이북을 종목/뉴스 분석의 배경 환경으로 제공합니다.</p>
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
          <p>뉴스 기반 이벤트 스터디, 섹터 전이, 토픽, 가격 반응 분석을 실행합니다.</p>
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
