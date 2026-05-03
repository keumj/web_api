from __future__ import annotations

from fastapi import FastAPI
from starlette.middleware.gzip import GZipMiddleware

from app.routers import admin, auth, pages, portfolio, stock, stock_news
from app.security import AuthMiddleware, LanAccessMiddleware
from app.settings import settings


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Portfolio-first single-port service with stock and stock-news submodules.",
    docs_url="/docs" if settings.enable_docs else None,
    redoc_url="/redoc" if settings.enable_docs else None,
    openapi_url="/openapi.json" if settings.enable_docs else None,
)

app.add_middleware(
    GZipMiddleware,
    minimum_size=settings.gzip_minimum_size,
    compresslevel=settings.gzip_compresslevel,
)
app.add_middleware(AuthMiddleware)
app.add_middleware(LanAccessMiddleware)

app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(pages.router)
app.include_router(stock.router)
app.include_router(stock_news.router)
app.include_router(portfolio.router)
