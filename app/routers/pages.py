from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app.routers.auth import _safe_next, auth_panel
from app.services import auth_service
from app.web import shell

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def index(request: Request, next: str | None = None, auth_error: str | None = None) -> HTMLResponse:
    next_url = _safe_next(next)
    if auth_error == "login":
        error = "사용자명 또는 비밀번호가 올바르지 않습니다."
    else:
        error = str(auth_error or "")
    user = auth_service.current_user(request)
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
    body = f"""
    <div class="service-stack">
      <div class="service-card">
        <h1>Keumj 포트폴리오 분석 서비스</h1>
        <p class="service-muted">포트폴리오를 중심으로 종목 분석, 뉴스 분석, 데이터 갱신 기능을 한 포트에서 실행합니다.</p>
      </div>
      <div class="service-grid">
        <a class="service-card" href="/portfolio/overview?intent=run">
          <h3>포트폴리오</h3>
          <p>보유 종목, 성과, 위험, 최적화 분석을 실행합니다.</p>
        </a>
        <a class="service-card" href="/stock/forecast">
          <h3>종목 분석</h3>
          <p>개별 종목 예측, 재무, 기술적 분석, 의사결정 화면으로 이동합니다.</p>
        </a>
        <a class="service-card" href="/stock-news/overview">
          <h3>뉴스 분석</h3>
          <p>뉴스 기반 이벤트, 섹터 전이, 토픽, 가격 반응 분석을 실행합니다.</p>
        </a>
        <a class="service-card" href="/macro/overview">
          <h3>거시분석</h3>
          <p>금리, 달러, 위험선호, 섹터 플레이북을 종목/뉴스 분석의 배경 환경으로 점검합니다.</p>
        </a>
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
    return _redirect_with_query(request, "/stock/forecast")


@router.get("/stock-news")
def stock_news_home(request: Request) -> RedirectResponse:
    return _redirect_with_query(request, "/stock-news/overview")


@router.get("/macro")
def macro_home(request: Request) -> RedirectResponse:
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
