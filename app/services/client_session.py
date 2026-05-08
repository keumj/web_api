from __future__ import annotations

import re
import secrets
from dataclasses import dataclass

from fastapi import Request
from fastapi.responses import Response

from app.services import auth_service
from app.settings import settings


COOKIE_NAME = "keumjm_client_session"
SESSION_DAYS = 30
SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{16,80}$")


@dataclass(frozen=True)
class ClientSession:
    state_key: str
    cookie_value: str | None = None


def resolve(request: Request) -> ClientSession:
    user = auth_service.current_user(request)
    if user is not None:
        return ClientSession(state_key=f"user:{user.id}")

    cookie_value = str(request.cookies.get(COOKIE_NAME, "") or "").strip()
    if SESSION_RE.match(cookie_value):
        return ClientSession(state_key=f"anon:{cookie_value}")

    cookie_value = secrets.token_urlsafe(24)
    return ClientSession(state_key=f"anon:{cookie_value}", cookie_value=cookie_value)


def attach_cookie(response: Response, session: ClientSession) -> None:
    if not session.cookie_value:
        return
    response.set_cookie(
        COOKIE_NAME,
        session.cookie_value,
        max_age=SESSION_DAYS * 86_400,
        httponly=True,
        secure=settings.auth_cookie_secure,
        samesite="lax",
    )
