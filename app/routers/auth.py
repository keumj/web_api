from __future__ import annotations

import html
from urllib.parse import quote, unquote

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app.form import read_form
from app.services import auth_service
from app.settings import settings


router = APIRouter()


def _safe_next(value: str | None) -> str:
    target = unquote(str(value or "").strip()) or "/"
    if not target.startswith("/") or target.startswith("//"):
        return "/"
    if target.startswith(("/login", "/register", "/logout")):
        return "/"
    return target


def _auth_page(*, mode: str, next_url: str = "", error: str = "") -> str:
    is_register = mode == "register"
    title = "회원가입" if is_register else "로그인"
    action = "/register" if is_register else "/login"
    alt_href = f"/login?next={html.escape(next_url)}" if is_register else f"/register?next={html.escape(next_url)}"
    alt_text = "이미 계정이 있으면 로그인" if is_register else "새 사용자 등록"
    button = "계정 만들기" if is_register else "로그인"
    error_html = f'<div class="error">{html.escape(error)}</div>' if error else ""
    register_note = ""
    if is_register and not settings.auth_allow_registration and auth_service.user_count() > 0:
        register_note = '<p class="muted">신규 사용자 등록이 비활성화되어 있습니다.</p>'
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title} | Keumj Portfolio Lab</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; min-height: 100vh; display: grid; place-items: center; background: #f5f7fa; color: #1f2937; font-family: "Segoe UI", "Noto Sans KR", sans-serif; }}
    .panel {{ width: min(420px, calc(100vw - 32px)); background: #fff; border: 1px solid #d7e0ea; border-radius: 8px; padding: 22px; }}
    h1 {{ margin: 0 0 8px; font-size: 22px; }}
    p {{ margin: 0 0 16px; color: #667085; font-size: 13px; line-height: 1.5; }}
    label {{ display: block; margin: 12px 0 5px; font-size: 12px; color: #667085; }}
    input {{ width: 100%; border: 1px solid #d7e0ea; border-radius: 8px; padding: 10px; font-size: 15px; }}
    button {{ width: 100%; margin-top: 16px; border: 0; background: #111827; color: #fff; border-radius: 8px; padding: 11px 14px; font-weight: 700; cursor: pointer; }}
    .alt {{ display: block; margin-top: 14px; text-align: center; color: #0f766e; text-decoration: none; font-size: 13px; }}
    .error {{ margin: 12px 0; padding: 10px; border: 1px solid #efadad; background: #fff2f2; color: #a12626; border-radius: 8px; font-size: 13px; }}
    .muted {{ color: #667085; }}
  </style>
</head>
<body>
  <main class="panel">
    <h1>{title}</h1>
    <p>로그인한 사용자별로 거래 DB가 분리됩니다.</p>
    {error_html}
    {register_note}
    <form method="post" action="{action}">
      <input type="hidden" name="next" value="{html.escape(next_url)}" />
      <label for="username">사용자명</label>
      <input id="username" name="username" autocomplete="username" required />
      <label for="password">비밀번호</label>
      <input id="password" name="password" type="password" autocomplete="{"new-password" if is_register else "current-password"}" required />
      <button type="submit">{button}</button>
    </form>
    <a class="alt" href="{alt_href}">{alt_text}</a>
  </main>
</body>
</html>"""


def auth_panel(*, next_url: str, user: auth_service.AuthUser | None = None, error: str = "") -> str:
    if not settings.auth_enabled:
        return ""
    if user is not None:
        admin_link = (
            '<a class="service-button secondary" style="text-decoration:none;" href="/admin/users">사용자 관리</a>'
            if user.is_admin
            else ""
        )
        return f"""
        <section id="login" class="service-card">
          <h2 style="margin:0 0 8px;">로그인</h2>
          <p class="service-muted" style="margin:0 0 12px;">현재 <strong>{html.escape(user.username)}</strong> 계정으로 로그인되어 있습니다.</p>
          <div class="service-actions">
            <a class="service-button" style="text-decoration:none;" href="/portfolio/overview?intent=run">내 포트폴리오 열기</a>
            {admin_link}
            <a class="service-button secondary" style="text-decoration:none;" href="/logout">로그아웃</a>
          </div>
        </section>
        """

    error_html = f'<div class="service-error" style="margin-bottom:10px;">{html.escape(error)}</div>' if error else ""
    register_disabled = (not settings.auth_allow_registration and auth_service.user_count() > 0)
    register_html = (
        '<p class="service-muted" style="margin:0;">신규 사용자 등록이 비활성화되어 있습니다.</p>'
        if register_disabled
        else f"""
          <form method="post" action="/register">
            <input type="hidden" name="next" value="{html.escape(next_url)}" />
            <label>사용자명</label>
            <input name="username" autocomplete="username" required />
            <label>비밀번호</label>
            <input name="password" type="password" autocomplete="new-password" required />
            <button class="service-button secondary" type="submit">계정 만들기</button>
          </form>
        """
    )
    return f"""
    <section id="login" class="service-card">
      <h2 style="margin:0 0 8px;">로그인</h2>
      <p class="service-muted" style="margin:0 0 12px;">로그인한 사용자별로 거래 DB가 분리됩니다.</p>
      {error_html}
      <div class="service-login-grid">
        <form method="post" action="/login">
          <h3>기존 계정</h3>
          <input type="hidden" name="next" value="{html.escape(next_url)}" />
          <label>사용자명</label>
          <input name="username" autocomplete="username" required />
          <label>비밀번호</label>
          <input name="password" type="password" autocomplete="current-password" required />
          <button class="service-button" type="submit">로그인</button>
        </form>
        <div>
          <h3>새 계정</h3>
          {register_html}
        </div>
      </div>
    </section>
    """


def _set_session_cookie(response: RedirectResponse, user: auth_service.AuthUser) -> None:
    max_age = max(int(settings.auth_session_days), 1) * 86_400
    response.set_cookie(
        settings.auth_cookie_name,
        auth_service.make_session_token(user),
        max_age=max_age,
        httponly=True,
        secure=settings.auth_cookie_secure,
        samesite="lax",
    )


@router.get("/login", response_class=HTMLResponse)
def login_page(next: str | None = None) -> RedirectResponse:
    return RedirectResponse(f"/?next={quote(_safe_next(next), safe='')}#login", status_code=303)


@router.post("/login")
async def login(request: Request):
    form = await read_form(request)
    next_url = _safe_next(form.get("next"))
    user = auth_service.authenticate(form.get("username", ""), form.get("password", ""))
    if user is None:
        return RedirectResponse(f"/?next={quote(next_url, safe='')}&auth_error=login#login", status_code=303)
    response = RedirectResponse(next_url, status_code=303)
    _set_session_cookie(response, user)
    return response


@router.get("/register", response_class=HTMLResponse)
def register_page(next: str | None = None) -> RedirectResponse:
    return RedirectResponse(f"/?next={quote(_safe_next(next), safe='')}#login", status_code=303)


@router.post("/register")
async def register(request: Request):
    form = await read_form(request)
    next_url = _safe_next(form.get("next"))
    try:
        user = auth_service.create_user(form.get("username", ""), form.get("password", ""))
    except ValueError as exc:
        return RedirectResponse(f"/?next={quote(next_url, safe='')}&auth_error={quote(str(exc), safe='')}#login", status_code=303)
    response = RedirectResponse(next_url, status_code=303)
    _set_session_cookie(response, user)
    return response


@router.get("/logout")
def logout() -> RedirectResponse:
    response = RedirectResponse("/#login", status_code=303)
    response.delete_cookie(settings.auth_cookie_name)
    return response
