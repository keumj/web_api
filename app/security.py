from __future__ import annotations

from ipaddress import ip_address
from urllib.parse import quote

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import PlainTextResponse, RedirectResponse

from app.services import auth_service
from app.settings import settings


class LanAccessMiddleware(BaseHTTPMiddleware):
    """Allow all traffic in internet mode, but restrict LAN mode to local/private IPs."""

    async def dispatch(self, request: Request, call_next):
        if is_request_allowed(request):
            return await call_next(request)
        return PlainTextResponse("Forbidden: this service is limited to the local network.", status_code=403)


class AuthMiddleware(BaseHTTPMiddleware):
    """Require a signed login session for browser/API access."""

    ADMIN_ONLY_PATHS = (
        "/docs",
        "/redoc",
        "/openapi.json",
        "/refresh",
        "/refresh_status",
        "/run_refresh",
        "/api/refresh",
    )
    EXEMPT_PREFIXES = (
        "/",
        "/login",
        "/register",
        "/logout",
        "/healthz",
        "/favicon.ico",
    )

    async def dispatch(self, request: Request, call_next):
        if not settings.auth_enabled or self._is_exempt(request.url.path):
            return await call_next(request)

        user = auth_service.verify_session_token(request.cookies.get(settings.auth_cookie_name, ""))
        if user is not None:
            if self._is_admin_only_path(request.url.path) and not user.is_admin:
                return PlainTextResponse("Forbidden: admin only.", status_code=403)
            request.state.user = user
            return await call_next(request)

        if self._wants_html(request):
            target = quote(str(request.url.path) + (f"?{request.url.query}" if request.url.query else ""), safe="")
            return RedirectResponse(f"/?next={target}#login", status_code=303)
        return PlainTextResponse("Unauthorized", status_code=401)

    @classmethod
    def _is_exempt(cls, path: str) -> bool:
        return any(path == prefix or path.startswith(prefix + "/") for prefix in cls.EXEMPT_PREFIXES)

    @classmethod
    def _is_admin_only_path(cls, path: str) -> bool:
        return any(path == prefix or path.startswith(prefix + "/") for prefix in cls.ADMIN_ONLY_PATHS)

    @staticmethod
    def _wants_html(request: Request) -> bool:
        accept = request.headers.get("accept", "")
        return "text/html" in accept or "*/*" in accept or not accept


def is_request_allowed(request: Request) -> bool:
    if settings.access_mode in {"internet", "public"}:
        return True

    host = request.client.host if request.client else ""
    if not host:
        return True

    try:
        client_ip = ip_address(host)
    except ValueError:
        return False

    if client_ip.version == 6 and client_ip.ipv4_mapped is not None:
        client_ip = client_ip.ipv4_mapped

    if client_ip.is_loopback or client_ip.is_private or client_ip.is_link_local:
        return True

    for network in settings.parsed_allowed_networks():
        if client_ip in network:
            return True

    return False
