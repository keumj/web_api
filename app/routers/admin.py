from __future__ import annotations

import html
from urllib.parse import quote

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse

from app.form import read_form
from app.services import auth_service, db_service
from app.web import shell


router = APIRouter(prefix="/admin")


def _require_admin(request: Request) -> auth_service.AuthUser | None:
    user = auth_service.current_user(request)
    if user is None or not user.is_admin:
        return None
    return user


def _forbidden() -> PlainTextResponse:
    return PlainTextResponse("관리자 계정으로만 접근할 수 있습니다.", status_code=403)


def _redirect(*, message: str = "", error: str = "") -> RedirectResponse:
    params: list[str] = []
    if message:
        params.append(f"message={quote(message, safe='')}")
    if error:
        params.append(f"error={quote(error, safe='')}")
    query = "?" + "&".join(params) if params else ""
    return RedirectResponse(f"/admin/users{query}", status_code=303)


def _delete_user_account(user_id: str) -> str:
    clean_user_id = str(user_id or "")
    auth_service.ensure_auth_db()
    with db_service.app_db_connection() as conn:
        row = conn.execute(
            "SELECT id, username, is_admin FROM users WHERE id = ?",
            (clean_user_id,),
        ).fetchone()
        if not row:
            raise ValueError("사용자를 찾을 수 없습니다.")
        if int(row[2] or 0) and auth_service._admin_count(conn) <= 1:
            raise ValueError("마지막 활성 관리자 계정은 삭제할 수 없습니다.")

        if db_service.using_remote_app_db():
            try:
                conn.execute("DELETE FROM portfolio_trades WHERE user_id = ?", (clean_user_id,))
            except Exception as exc:
                if "portfolio_trades" not in str(exc):
                    raise
        try:
            conn.execute("DELETE FROM portfolio_latest_snapshots WHERE user_id = ?", (clean_user_id,))
        except Exception as exc:
            if "portfolio_latest_snapshots" not in str(exc):
                raise

        cur = conn.execute("DELETE FROM users WHERE id = ?", (clean_user_id,))
        conn.commit()
        if int(cur.rowcount or 0) <= 0:
            raise ValueError("사용자를 찾을 수 없습니다.")
        deleted_username = str(row[1])

    if not db_service.using_remote_app_db():
        portfolio_db = auth_service.portfolio_db_for_user(clean_user_id)
        if portfolio_db.exists():
            portfolio_db.unlink()

    return deleted_username


def _user_rows(users: list[dict[str, object]]) -> str:
    rows: list[str] = []
    for item in users:
        user_id = str(item["id"])
        username = html.escape(str(item["username"]))
        is_admin = bool(item["is_admin"])
        is_active = bool(item["is_active"])
        admin_badge = "관리자" if is_admin else "일반"
        active_badge = "활성" if is_active else "비활성"
        created_at = html.escape(str(item.get("created_at") or "-"))
        last_login_at = html.escape(str(item.get("last_login_at") or "-"))
        portfolio_db = html.escape(str(item.get("portfolio_db") or "-"))
        next_active = "0" if is_active else "1"
        next_active_label = "비활성화" if is_active else "활성화"
        next_admin = "0" if is_admin else "1"
        next_admin_label = "권한 해제" if is_admin else "관리자 지정"
        rows.append(
            f"""
            <tr>
              <td><strong>{username}</strong></td>
              <td>{admin_badge}</td>
              <td>{active_badge}</td>
              <td>{created_at}</td>
              <td>{last_login_at}</td>
              <td><code>{portfolio_db}</code></td>
              <td>
                <div class="service-actions">
                  <form method="post" action="/admin/users/{html.escape(user_id)}/active">
                    <input type="hidden" name="is_active" value="{next_active}" />
                    <button class="service-button secondary" type="submit">{next_active_label}</button>
                  </form>
                  <form method="post" action="/admin/users/{html.escape(user_id)}/admin">
                    <input type="hidden" name="is_admin" value="{next_admin}" />
                    <button class="service-button secondary" type="submit">{next_admin_label}</button>
                  </form>
                  <form method="post" action="/admin/users/{html.escape(user_id)}/password">
                    <input name="password" type="password" placeholder="새 비밀번호" minlength="8" required />
                    <button class="service-button" type="submit">초기화</button>
                  </form>
                  <form method="post" action="/admin/users/{html.escape(user_id)}/delete" onsubmit="return confirm('이 계정과 계정별 포트폴리오 데이터를 삭제할까요?');">
                    <button class="service-button" type="submit" style="background:var(--danger);">삭제</button>
                  </form>
                </div>
              </td>
            </tr>
            """
        )
    return "\n".join(rows)


@router.get("/users", response_class=HTMLResponse)
def users_page(request: Request, message: str = "", error: str = ""):
    if _require_admin(request) is None:
        return _forbidden()
    message_html = f'<div class="service-card" style="border-color:#9ed6cb;">{html.escape(message)}</div>' if message else ""
    error_html = f'<div class="service-error service-card">{html.escape(error)}</div>' if error else ""
    rows = _user_rows(auth_service.list_users())
    body = f"""
    <div class="service-stack">
      <section class="service-card">
        <h1 style="margin:0 0 8px;">사용자 관리</h1>
        <p class="service-muted" style="margin:0;">첫 가입자는 자동으로 관리자 권한을 받습니다. 이후 계정 생성과 권한 변경은 이 화면에서 처리합니다.</p>
      </section>
      {message_html}
      {error_html}
      <section class="service-card">
        <h2 style="margin:0 0 12px;">계정 만들기</h2>
        <form class="service-actions" method="post" action="/admin/users/create">
          <input name="username" placeholder="사용자명" autocomplete="username" required />
          <input name="password" type="password" placeholder="비밀번호" autocomplete="new-password" minlength="8" required />
          <label style="display:flex;align-items:center;gap:6px;color:var(--muted);font-size:13px;">
            <input name="is_admin" type="checkbox" value="1" style="width:auto;" />
            관리자
          </label>
          <button class="service-button" type="submit">생성</button>
        </form>
      </section>
      <section class="service-card">
        <h2 style="margin:0 0 12px;">계정 목록</h2>
        <div class="service-table-wrap">
          <table class="service-table">
            <thead>
              <tr>
                <th>사용자명</th>
                <th>권한</th>
                <th>상태</th>
                <th>생성일</th>
                <th>최근 로그인</th>
                <th>포트폴리오 DB</th>
                <th>작업</th>
              </tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
        </div>
      </section>
    </div>
    """
    return HTMLResponse(shell("사용자 관리", body, active="admin", admin=True))


@router.post("/users/create")
async def create_user(request: Request):
    if _require_admin(request) is None:
        return _forbidden()
    form = await read_form(request)
    try:
        user = auth_service.create_user(
            form.get("username", ""),
            form.get("password", ""),
            is_admin=form.get("is_admin") == "1",
            require_registration_open=False,
        )
    except ValueError as exc:
        return _redirect(error=str(exc))
    role = "관리자" if user.is_admin else "일반"
    return _redirect(message=f"{user.username} 계정을 {role} 권한으로 만들었습니다.")


@router.post("/users/{user_id}/active")
async def set_active(user_id: str, request: Request):
    if _require_admin(request) is None:
        return _forbidden()
    form = await read_form(request)
    try:
        auth_service.set_user_active(user_id, form.get("is_active") == "1")
    except ValueError as exc:
        return _redirect(error=str(exc))
    return _redirect(message="계정 상태를 변경했습니다.")


@router.post("/users/{user_id}/admin")
async def set_admin(user_id: str, request: Request):
    if _require_admin(request) is None:
        return _forbidden()
    form = await read_form(request)
    try:
        auth_service.set_user_admin(user_id, form.get("is_admin") == "1")
    except ValueError as exc:
        return _redirect(error=str(exc))
    return _redirect(message="관리자 권한을 변경했습니다.")


@router.post("/users/{user_id}/password")
async def reset_password(user_id: str, request: Request):
    if _require_admin(request) is None:
        return _forbidden()
    form = await read_form(request)
    try:
        auth_service.reset_password(user_id, form.get("password", ""))
    except ValueError as exc:
        return _redirect(error=str(exc))
    return _redirect(message="비밀번호를 초기화했습니다.")


@router.post("/users/{user_id}/delete")
async def delete_user(user_id: str, request: Request):
    admin_user = _require_admin(request)
    if admin_user is None:
        return _forbidden()
    if str(admin_user.id) == str(user_id):
        return _redirect(error="현재 로그인한 관리자 계정은 바로 삭제할 수 없습니다. 다른 관리자 계정으로 로그인한 뒤 삭제해 주세요.")
    try:
        username = _delete_user_account(user_id)
    except ValueError as exc:
        return _redirect(error=str(exc))
    return _redirect(message=f"{username} 계정을 삭제했습니다.")
