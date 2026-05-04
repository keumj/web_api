from __future__ import annotations

import html
import json
import threading
import traceback
import os
import re
import socket
import sqlite3
import subprocess
import sys
import time
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.parse import parse_qs, quote, urlencode, urlparse
from urllib.request import urlopen
from uuid import uuid4

import numpy as np
import pandas as pd

import pipeline_stock.web_gui as stock_web_gui
import pipeline_stock_news.web_gui as news_web_gui

from .analysis import (
    DEFAULT_CASH_BUFFER_PCT,
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_MAX_POSITION_PCT,
    DEFAULT_OPTIMIZATION_UNIVERSE_SIZE,
    DEFAULT_SECTOR_CAP_PCT,
    DEFAULT_VIRTUAL_FORECAST_HORIZON_DAYS,
    OptimizationResult,
    PortfolioDashboard,
    VirtualTradeResult,
    add_trade,
    analyze_virtual_trade,
    build_portfolio_dashboard,
    build_portfolio_optimization,
    delete_trade,
    DEFAULT_LOOKBACK_DAYS as PORTFOLIO_DEFAULT_LOOKBACK_DAYS,
    DEFAULT_OPTIMIZATION_UNIVERSE_SIZE as PORTFOLIO_DEFAULT_OPTIMIZATION_UNIVERSE_SIZE,
    DEFAULT_VIRTUAL_FORECAST_HORIZON_DAYS as PORTFOLIO_DEFAULT_VIRTUAL_FORECAST_HORIZON_DAYS,
)


@dataclass
class _PageContext:
    dashboard: PortfolioDashboard | None
    lookback_days: int
    start_date: str
    end_date: str
    message: str | None = None
    error: str | None = None
    virtual_result: VirtualTradeResult | None = None
    stock_forecast_ctx: stock_web_gui._RunContext | None = None
    stock_financials_ctx: stock_web_gui._FinancialContext | None = None
    stock_technical_ctx: stock_web_gui.ta_web_gui._RunContext | None = None
    stock_wfv_ctx: stock_web_gui._WalkForwardContext | None = None
    news_overview_ctx: news_web_gui.StockNewsDashboard | None = None
    news_event_ctx: news_web_gui.StockNewsDashboard | None = None
    news_topics_ctx: news_web_gui.StockNewsDashboard | None = None
    optimization: OptimizationResult | None = None
    optimization_params: dict[str, float | int] | None = None


def _range_query(ctx: _PageContext) -> str:
    return urlencode(
        {
            "lookback_days": int(ctx.lookback_days),
            "start_date": str(ctx.start_date),
            "end_date": str(ctx.end_date),
        }
    )


def _nav(active: str, ctx: _PageContext) -> str:
    items = [
        ("data-entry", "/data-entry", "거래 입력"),
        ("overview", "/overview", "포트폴리오 개요"),
        ("attribution", "/attribution", "성과/Attribution"),
        ("risk", "/risk", "리스크"),
        ("scoring", "/scoring", "통합 스코어"),
        ("virtual-trade", "/virtual-trade", "가상 거래"),
        ("optimization", "/optimization", "최적화"),
        ("refresh", "/refresh", "데이터 갱신"),
    ]
    links = []
    for page_id, href, label in items:
        separator = "&" if "?" in href else "?"
        final_href = f"{href}{separator}{_range_query(ctx)}"
        css = "active" if page_id == active else ""
        links.append(f'<a class="{css}" href="{final_href}">{html.escape(label)}</a>')
    return '<div class="nav">' + "".join(links) + "</div>"


def _base_css() -> str:
    return """
    :root {
      --bg: #f3f5f7;
      --card: #ffffff;
      --line: #d4dde8;
      --text: #1f2937;
      --muted: #5f6b7a;
      --brand: #111111;
      --accent: #0f4c81;
      --accent-light: #eef4fb;
      --ok-bg: #e8f7ee;
      --ok-line: #99d5af;
      --err-bg: #fff2f2;
      --err-line: #efadad;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); font-family: "Segoe UI", "Noto Sans KR", sans-serif; }
    .wrap { width: 100%; max-width: 1460px; margin: 0 auto; padding: 20px; }
    .page-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; margin-bottom: 10px; }
    .page-head h1 { margin: 0; font-size: 24px; }
    .page-credit { color: var(--muted); font-size: 11px; white-space: nowrap; padding-top: 4px; }
    .sub { color: var(--muted); margin-bottom: 14px; line-height: 1.5; }
    .nav { display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap; }
    .nav a { text-decoration: none; color: #111; border: 1px solid #111; background: #fff; border-radius: 999px; padding: 7px 12px; font-size: 13px; }
    .nav a:hover { background: #eee; }
    .nav a.active { background: #111; color: #fff; border-color: #111; }
    .card { min-width: 0; background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }
    .stack { display: grid; gap: 12px; }
    .grid-2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    .grid-3 { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
    .metrics { display: grid; grid-template-columns: repeat(4, minmax(180px, 1fr)); gap: 10px; margin-top: 12px; }
    .metric { min-width: 0; background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 10px; }
    .metric span { display: block; font-size: 12px; color: var(--muted); }
    .metric strong { display: block; margin-top: 5px; font-size: 18px; line-height: 1.3; }
    .chart-img { width: 100%; height: auto; border-radius: 8px; margin-top: 10px; border: 1px solid var(--line); }
    .form-grid { display: grid; grid-template-columns: repeat(6, minmax(120px, 1fr)); gap: 10px 12px; }
    .form-grid.form-virtual { grid-template-columns: repeat(7, minmax(110px, 1fr)); }
    .form-grid label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 4px; }
    .form-grid input, .form-grid select {
      width: 100%; box-sizing: border-box; padding: 8px; border: 1px solid var(--line); border-radius: 6px;
    }
    .toolbar { display: flex; gap: 10px; align-items: end; flex-wrap: wrap; margin-top: 10px; }
    button { background: #111; border: 0; color: #fff; padding: 10px 16px; border-radius: 8px; cursor: pointer; font-weight: 600; }
    .notice { margin-top: 10px; border-radius: 8px; padding: 10px; }
    .notice a { color: var(--accent); text-decoration: underline; }
    .notice.ok { background: var(--ok-bg); border: 1px solid var(--ok-line); }
    .notice.err { background: var(--err-bg); border: 1px solid var(--err-line); }
    .table-wrap { width: 100%; max-width: 100%; min-width: 0; overflow-x: auto; -webkit-overflow-scrolling: touch; }
    .data-table { width: max-content; min-width: 100%; border-collapse: collapse; font-size: 12px; }
    .data-table th, .data-table td { border: 1px solid var(--line); padding: 6px; text-align: left; vertical-align: top; white-space: nowrap; }
    .small { font-size: 12px; color: var(--muted); }
    .split-grid { margin-top: 12px; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
    .pane { min-width: 0; background: #fff; border: 1px solid var(--line); border-radius: 10px; padding: 12px; }
    .pane h3 { margin: 0 0 8px 0; font-size: 14px; }
    .line-list { height: 480px; overflow: auto; border: 1px solid var(--line); border-radius: 8px; background: #fff; padding: 8px; }
    .line { font-family: Consolas, "Courier New", monospace; font-size: 12px; line-height: 1.45; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; border-bottom: 1px solid #eef2f7; padding: 3px 2px; }
    .line:last-child { border-bottom: 0; }
    .hint { color: var(--muted); font-size: 13px; line-height: 1.5; }
    .muted { color: var(--muted); }
    .db-note { font-size: 12px; color: var(--muted); margin-top: 6px; }
    .refresh-card { border-left: 5px solid var(--accent); }
    .refresh-latest-inline {
      min-width: 280px; padding: 8px 10px; border-radius: 8px;
      background: #eef4fb; border: 1px solid #c7d9ee; color: #24425f; font-size: 12px; line-height: 1.45;
    }
    .latest-inline strong { display: block; margin-bottom: 3px; color: var(--text); }
    .refresh-card-head {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(240px, 320px);
      gap: 12px;
      align-items: start;
      margin-bottom: 10px;
    }
    .refresh-card-title {
      margin: 0 0 6px 0;
      font-size: 22px;
      line-height: 1.2;
    }
    .refresh-card-desc {
      margin: 0;
      max-width: 760px;
    }
    .refresh-action-row {
      display: grid;
      grid-template-columns: auto minmax(0, 1fr);
      gap: 10px;
      align-items: center;
      margin-top: 0;
      padding: 8px 12px;
      border: 1px solid #d7e2ec;
      border-radius: 10px;
      background: #f9fbfd;
    }
    .refresh-action-row button {
      min-width: 148px;
    }
    .refresh-meta {
      font-size: 12px;
      color: var(--muted);
      line-height: 1.5;
    }
    .refresh-hero {
      margin-bottom: 14px;
      border-left: 5px solid var(--accent);
      background:
        radial-gradient(circle at top right, rgba(15, 76, 129, 0.08), transparent 26%),
        linear-gradient(180deg, #ffffff 0%, #f6fbff 100%);
    }
    .refresh-hero-grid {
      display: grid;
      grid-template-columns: minmax(0, 1.8fr) minmax(260px, 1fr);
      gap: 14px;
      align-items: start;
    }
    .refresh-hero h2 { margin: 0 0 8px 0; font-size: 22px; }
    .refresh-summary {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: rgba(255, 255, 255, 0.92);
      padding: 12px;
    }
    .refresh-summary-title {
      margin: 0 0 8px 0;
      font-size: 12px;
      font-weight: 700;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }
    .refresh-summary-list {
      display: grid;
      gap: 8px;
    }
    .refresh-summary-item {
      border: 1px solid #d9e5f1;
      border-radius: 8px;
      background: #fbfdff;
      padding: 9px 10px;
      font-size: 13px;
      line-height: 1.45;
    }
    .refresh-summary-item strong {
      display: block;
      margin-bottom: 2px;
      color: var(--text);
      font-size: 12px;
    }
    .refresh-split { grid-template-columns: repeat(2, minmax(280px, 1fr)); }
    .refresh-pane-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 8px;
    }
    .refresh-pane-title h3 {
      margin: 0;
      font-size: 14px;
    }
    .refresh-pane-title span {
      font-size: 11px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }
    .chart-card {
      border: 1px solid var(--line); border-radius: 8px; padding: 14px;
      letter-spacing: 0.03em;
    }
    .refresh-log-list { height: 360px; }
    .refresh-update-list { height: 360px; }
    @media (max-width: 1100px) {
      .form-grid, .form-grid.form-virtual { grid-template-columns: repeat(2, minmax(140px, 1fr)); }
      .metrics { grid-template-columns: repeat(2, minmax(160px, 1fr)); }
      .grid-2, .grid-3, .split-grid { grid-template-columns: 1fr; }
      .refresh-card-head { grid-template-columns: 1fr; }
      .refresh-action-row { grid-template-columns: 1fr; }
      .refresh-action-row button { width: 100%; }
      .refresh-hero-grid { grid-template-columns: 1fr; }
      .latest-inline { min-width: 0; width: 100%; }
      .refresh-split { grid-template-columns: 1fr; }
      .refresh-log-list, .refresh-update-list { height: 280px; }
    }
    """

def _page_head(title: str) -> str:
    return (
        '<div class="page-head">'
        f"<h1>{html.escape(title)}</h1>"
        '<div class="page-credit">Keumj 제작</div>'
        "</div>"
    )

def _safe_table(frame: pd.DataFrame, *, max_rows: int = 200, escape: bool = True) -> str:
    if frame is None or frame.empty:
        return "<p class='hint'>데이터가 없습니다.</p>"
    show = frame.head(max_rows).copy()
    return f'<div class="table-wrap">{show.to_html(index=False, border=0, classes="data-table", escape=escape)}</div>'


def _trade_history_table(trades: pd.DataFrame, ctx: _PageContext) -> str:
    if trades is None or trades.empty:
        return "<p class='hint'>데이터가 없습니다.</p>"
    show = trades.sort_values(["trade_date", "id"], ascending=[False, False]).reset_index(drop=True).copy()
    if pd.api.types.is_datetime64_any_dtype(show["trade_date"]):
        show["trade_date"] = pd.to_datetime(show["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    delete_forms: list[str] = []
    for row in show.itertuples(index=False):
        trade_id = int(getattr(row, "id"))
        delete_forms.append(
            "".join(
                [
                    '<form method="post" action="/run_delete_trade" onsubmit="return confirm(\'선택한 거래를 삭제할까요?\');">',
                    f'<input type="hidden" name="trade_id" value="{trade_id}">',
                    f'<input type="hidden" name="lookback_days" value="{int(ctx.lookback_days)}">',
                    f'<input type="hidden" name="start_date" value="{html.escape(ctx.start_date)}">',
                    f'<input type="hidden" name="end_date" value="{html.escape(ctx.end_date)}">',
                    '<button type="submit" style="padding:6px 10px; border-radius:6px; background:#8f1d1d; font-size:12px;">삭제</button>',
                    "</form>",
                ]
            )
        )
    show["액션"] = delete_forms
    ordered_cols = [
        "id",
        "trade_date",
        "ticker",
        "side",
        "quantity",
        "price",
        "fees",
        "notes",
        "액션",
    ]
    present_cols = [col for col in ordered_cols if col in show.columns]
    return _safe_table(show[present_cols], max_rows=200, escape=False)


def _fmt(value: object, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "-"
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(numeric):
        return "-"
    return f"{numeric:,.{digits}f}{suffix}"


def _summary_metrics(dashboard: PortfolioDashboard | None) -> str:
    if dashboard is None or dashboard.portfolio_summary.empty:
        return "<p class='hint'>포트폴리오 요약 데이터가 없습니다.</p>"
    row = dashboard.portfolio_summary.iloc[0]
    items = [
        ("기준일", row.get("as_of_date")),
        ("선택 시작", row.get("selected_start_date")),
        ("선택 종료", row.get("selected_end_date")),
        ("보유 종목 수", _fmt(row.get("holding_count"), 0)),
        ("시장가치", _fmt(row.get("market_value"))),
        ("미실현 손익", _fmt(row.get("unrealized_pnl"))),
        ("총 수익률", _fmt(row.get("total_return_pct"), suffix="%")),
        ("선택 기간 수익률", _fmt(row.get("portfolio_return_selected_pct"), suffix="%")),
        ("WTD", _fmt(row.get("portfolio_return_wtd_pct"), suffix="%")),
        ("MTD", _fmt(row.get("portfolio_return_mtd_pct"), suffix="%")),
        ("YTD", _fmt(row.get("portfolio_return_ytd_pct"), suffix="%")),
        ("20일 수익률", _fmt(row.get("portfolio_return_20d_pct"), suffix="%")),
        ("60일 수익률", _fmt(row.get("portfolio_return_60d_pct"), suffix="%")),
        ("연율화 변동성", _fmt(row.get("portfolio_vol_annual_pct"), suffix="%")),
        ("SP500 베타", _fmt(row.get("benchmark_beta"), 3)),
    ]
    html_parts = []
    for label, value in items:
        html_parts.append(f'<div class="metric"><span>{html.escape(str(label))}</span><strong>{html.escape(str(value))}</strong></div>')
    return '<div class="metrics">' + "".join(html_parts) + "</div>"


def _date_range_form(active: str, ctx: _PageContext) -> str:
    action_map = {
        "data-entry": "/data-entry",
        "overview": "/overview",
        "attribution": "/attribution",
        "risk": "/risk",
        "scoring": "/scoring",
        "virtual-trade": "/virtual-trade",
        "optimization": "/optimization",
        "stock-forecast": "/stock-forecast",
        "stock-financials": "/stock-financials",
        "stock-technical": "/stock-technical",
        "stock-wfv": "/stock-wfv",
    }
    action = action_map.get(active, "/overview")
    run_notice = ""
    if ctx.dashboard is None and active != "refresh":
        run_notice = '<div class="small" style="margin-top:8px;">페이지는 먼저 열리고, 아래 버튼을 눌렀을 때 현재 조건으로 분석을 실행합니다.</div>'
    return f"""
    <div class="card">
      <form method="get" action="{html.escape(action)}">
        <div class="toolbar">
          <div>
            <label class="muted">시작일</label>
            <input type="date" name="start_date" value="{html.escape(ctx.start_date)}">
          </div>
          <div>
            <label class="muted">종료일</label>
            <input type="date" name="end_date" value="{html.escape(ctx.end_date)}">
          </div>
          <input type="hidden" name="lookback_days" value="{int(ctx.lookback_days)}">
          <input type="hidden" name="intent" value="run">
          <button type="submit">분석 실행</button>
        </div>
        {run_notice}
      </form>
    </div>
    """


def _layout(title: str, subtitle: str, active: str, ctx: _PageContext, body: str, *, show_nav: bool = True) -> str:
    nav_html = _nav(active, ctx) if show_nav else ""
    return f"""
    <!doctype html>
    <html lang="ko">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>{html.escape(title)}</title>
      <style>{_base_css()}</style>
    </head>
    <body>
      <div class="wrap">
        {_page_head("Portfolio Lab | S&P500")}
        {nav_html}
        <div class="sub">{html.escape(subtitle)}</div>
        {body}
      </div>
    </body>
    </html>
    """


def _message_block(ctx: _PageContext) -> str:
    parts: list[str] = []
    if ctx.message:
        parts.append(f'<div class="notice ok">{html.escape(ctx.message)}</div>')
    if ctx.error:
        parts.append(f'<div class="notice err">{html.escape(ctx.error)}</div>')
    
    # 포트폴리오 분석 결과의 현금 경고 표시
    if ctx.dashboard and ctx.dashboard.diagnostics.get("cash_warning"):
        parts.append(f'<div class="notice err">{html.escape(ctx.dashboard.diagnostics["cash_warning"])}</div>')
    # 가상 거래 결과의 현금 경고 표시
    if ctx.virtual_result and ctx.virtual_result.diagnostics.get("cash_warning"):
        parts.append(f'<div class="notice err">{html.escape(ctx.virtual_result.diagnostics["cash_warning"])}</div>')
        
    return "".join(parts)


def _data_entry_page(ctx: _PageContext) -> str:
    dashboard = ctx.dashboard
    trades = dashboard.trades if dashboard is not None else pd.DataFrame()
    positions = dashboard.positions if dashboard is not None else pd.DataFrame()
    body = f"""
    {_message_block(ctx)}
    {_date_range_form("data-entry", ctx)}
    <div class="card" style="margin-bottom: 12px; border-left: 5px solid var(--accent);">
      <h2 style="margin-top:0;">현금 입출금 관리</h2>
      <form method="post" action="/run_add_trade">
        <input type="hidden" name="lookback_days" value="{int(ctx.lookback_days)}">
        <input type="hidden" name="start_date" value="{html.escape(ctx.start_date)}">
        <input type="hidden" name="end_date" value="{html.escape(ctx.end_date)}">
        <input type="hidden" name="ticker" value="CASH">
        <input type="hidden" name="price" value="1">
        <input type="hidden" name="fees" value="0">
        <div class="form-grid" style="grid-template-columns: repeat(3, 1fr);">
          <div>
            <label>날짜</label>
            <input type="date" name="trade_date" required value="{pd.Timestamp.today().strftime('%Y-%m-%d')}">
          </div>
          <div>
            <label>유형</label>
            <select name="side">
              <option value="BUY">현금 입금 (Deposit)</option>
              <option value="SELL">현금 출금 (Withdrawal)</option>
            </select>
          </div>
          <div><label>금액 ($)</label><input type="number" min="0.01" step="0.01" name="quantity" required placeholder="0.00"></div>
        </div>
        <div class="toolbar">
          <div style="flex:1;"><input type="text" name="notes" placeholder="입출금 사유 (예: 초기 자본, 추가 입금 등)"></div>
          <button type="submit" style="background: var(--accent);">현금 내역 저장</button>
        </div>
      </form>
    </div>
    <div class="card">
      <h2 style="margin-top:0;">거래 입력</h2>
      <form method="post" action="/run_add_trade">
        <input type="hidden" name="lookback_days" value="{int(ctx.lookback_days)}">
        <input type="hidden" name="start_date" value="{html.escape(ctx.start_date)}">
        <input type="hidden" name="end_date" value="{html.escape(ctx.end_date)}">
        <div class="form-grid">
          <div><label>거래일</label><input type="date" name="trade_date" required></div>
          <div><label>티커</label><input type="text" name="ticker" placeholder="AAPL" required></div>
          <div><label>매수/매도</label><select name="side"><option value="BUY">BUY</option><option value="SELL">SELL</option></select></div>
          <div><label>수량</label><input type="number" min="0.0001" step="0.0001" name="quantity" required></div>
          <div><label>가격</label><input type="number" min="0.0001" step="0.0001" name="price" required></div>
          <div><label>수수료</label><input type="number" min="0" step="0.0001" name="fees" value="0"></div>
        </div>
        <div class="toolbar">
          <div style="flex:1; min-width: 220px;">
            <label class="muted">메모</label>
            <input type="text" name="notes" placeholder="예: rebalance">
          </div>
          <button type="submit">거래 저장</button>
        </div>
      </form>
      <div class="db-note">포트폴리오 거래 DB에 누적 저장되고, 가격/뉴스는 shared DB를 참조합니다.</div>
    </div>
    <div class="grid-2">
      <div class="card">
        <h3>현재 포지션</h3>
        {_safe_table(positions)}
      </div>
      <div class="card">
        <h3>최근 거래</h3>
        {_trade_history_table(trades, ctx)}
      </div>
    </div>
    """
    return _layout("Portfolio Lab | 거래 입력", "거래 입력 아울렛과 거래 내역 저장 DB", "data-entry", ctx, body)


def _overview_page(ctx: _PageContext) -> str:
    dashboard = ctx.dashboard
    # 중첩 f-string을 피하기 위해 차트 HTML을 미리 생성
    chart1_html = "<p class='hint'>포트폴리오 배분 차트를 불러올 수 없습니다.</p>"
    if dashboard and dashboard.sector_allocation_chart:
        chart1_html = f'<img src="data:image/png;base64,{dashboard.sector_allocation_chart}" class="chart-img" style="max-width:600px; margin: 0 auto; display:block;" alt="Portfolio Sector Allocation" />'

    chart2_html = "<p class='hint'>벤치마크 배분 차트를 불러올 수 없습니다.</p>"
    if dashboard and dashboard.benchmark_sector_allocation_chart:
        chart2_html = f'<img src="data:image/png;base64,{dashboard.benchmark_sector_allocation_chart}" class="chart-img" style="max-width:600px; margin: 0 auto; display:block;" />'

    body = f"""
    {_message_block(ctx)}
    {_date_range_form("overview", ctx)}
    {_summary_metrics(dashboard)}
    <div class="card" style="margin-top: 12px;">
      <h3 style="margin-top:0;">섹터 배분 비교 (Portfolio vs S&P500)</h3>
      <div class="grid-2">
        <div class="chart-card">
          {chart1_html}
        </div>
        <div class="chart-card">
          {chart2_html}
        </div>
      </div>
    </div>
    <div class="grid-2">
      <div class="card">
        <h3 style="margin-top:0;">보유 종목 퍼포먼스</h3>
        {_safe_table(dashboard.holdings_performance if dashboard else pd.DataFrame())}
      </div>
      <div class="card">
        <h3 style="margin-top:0;">현재 포지션</h3>
        {_safe_table(dashboard.positions if dashboard else pd.DataFrame())}
      </div>
    </div>
    <div class="card">
      <h3>분석 진단</h3>
      {_safe_table(pd.DataFrame([dashboard.diagnostics]) if dashboard else pd.DataFrame())}
    </div>
    """
    return _layout("Portfolio Lab | 포트폴리오 개요", "보유 종목별 퍼포먼스와 포트폴리오 상태를 한눈에 봅니다.", "overview", ctx, body)


def _attribution_page(ctx: _PageContext) -> str:
    dashboard = ctx.dashboard
    chart_cum_html = "<p class='hint'>차트 데이터를 불러올 수 없습니다.</p>"
    if dashboard and dashboard.cumulative_chart:
        chart_cum_html = f'<img src="data:image/png;base64,{dashboard.cumulative_chart}" class="chart-img" alt="Cumulative Return Chart" />'

    chart_sec_html = "<p class='hint'>차트 데이터를 불러올 수 없습니다.</p>"
    if dashboard and dashboard.sector_contribution_chart:
        chart_sec_html = f'<img src="data:image/png;base64,{dashboard.sector_contribution_chart}" class="chart-img" alt="Sector Contribution Chart" />'

    chart_style_html = "<p class='hint'>차트 데이터를 불러올 수 없습니다.</p>"
    if dashboard and dashboard.style_exposure_chart:
        chart_style_html = f'<img src="data:image/png;base64,{dashboard.style_exposure_chart}" class="chart-img" alt="Style Exposure Chart" />'

    body = f"""
    {_message_block(ctx)}
    {_date_range_form("attribution", ctx)}
    <div class="card" style="margin-bottom: 12px;">
      <h3 style="margin-top:0;">누적 수익률 추이 (vs S&P500)</h3>
      {f'<img src="data:image/png;base64,{dashboard.cumulative_chart}" class="chart-img" alt="Cumulative Return Chart" />' if dashboard and dashboard.cumulative_chart else "<p class='hint'>차트 데이터를 불러올 수 없습니다.</p>"}
    </div>
    <div class="grid-2" style="margin-bottom: 12px;">
       <div class="card">
         <h3 style="margin-top:0;">섹터별 성과 기여도 (Attribution)</h3>
         {f'<img src="data:image/png;base64,{dashboard.sector_contribution_chart}" class="chart-img" alt="Sector Contribution Chart" />' if dashboard and dashboard.sector_contribution_chart else "<p class='hint'>차트 데이터를 불러올 수 없습니다.</p>"}
       </div>
       <div class="card">
         <h3 style="margin-top:0;">스타일 노출 (Style Exposure)</h3>
         {f'<img src="data:image/png;base64,{dashboard.style_exposure_chart}" class="chart-img" alt="Style Exposure Chart" />' if dashboard and dashboard.style_exposure_chart else "<p class='hint'>차트 데이터를 불러올 수 없습니다.</p>"}
       </div>
    </div>
    <div class="card">
      <h3>보유 종목 절대 성과</h3>
      {_safe_table(dashboard.holdings_performance if dashboard else pd.DataFrame())}
    </div>
    <div class="grid-2">
      <div class="card">
        <h3 style="margin-top:0;">포트폴리오 절대 성과 요약</h3>
        {_safe_table(dashboard.portfolio_summary if dashboard else pd.DataFrame())}
      </div>
      <div class="card">
        <h3 style="margin-top:0;">스타일 노출</h3>
        {_safe_table(dashboard.style_exposure if dashboard else pd.DataFrame())}
      </div>
    </div>
    <div class="card" style="margin-top:12px;">
      <h3>SP500 대비 종목별 상대 기여</h3>
      {_safe_table(dashboard.stock_attribution if dashboard else pd.DataFrame())}
    </div>
    <div class="grid-2">
      <div class="card">
        <h3>SP500 대비 섹터 Attribution</h3>
        {_safe_table(dashboard.attribution if dashboard else pd.DataFrame())}
      </div>
      <div class="card">
        <h3>스타일 Attribution</h3>
        {_safe_table(dashboard.style_attribution if dashboard else pd.DataFrame())}
      </div>
    </div>
    """
    return _layout("Portfolio Lab | 성과 Attribution", "보유 종목 절대 성과를 먼저 보고, 그 다음 SP500 대비 상대성과를 이어서 봅니다. WTD, MTD, YTD는 선택 기간과 별개로 항상 함께 계산됩니다.", "attribution", ctx, body)


def _risk_page(ctx: _PageContext) -> str:
    dashboard = ctx.dashboard
    chart_risk_html = "<p class='hint'>차트 데이터를 불러올 수 없습니다.</p>"
    if dashboard and dashboard.risk_contribution_chart:
        chart_risk_html = f'<img src="data:image/png;base64,{dashboard.risk_contribution_chart}" class="chart-img" alt="Risk Contribution Chart" />'

    chart_active_html = "<p class='hint'>차트 데이터를 불러올 수 없습니다.</p>"
    if dashboard and dashboard.active_risk_contribution_chart:
        chart_active_html = f'<img src="data:image/png;base64,{dashboard.active_risk_contribution_chart}" class="chart-img" alt="Active Risk Contribution Chart" />'

    body = f"""
    {_message_block(ctx)}
    {_date_range_form("risk", ctx)}
    <div class="grid-2">
      <div class="card" style="margin-top:12px;">
        <h3 style="margin-top:0;">절대 리스크</h3>
        {_safe_table(dashboard.risk_summary if dashboard else pd.DataFrame())}
      </div>
      <div class="card" style="margin-top:12px;">
        <h3 style="margin-top:0;">SP500 대비 상대 리스크</h3>
        {_safe_table(dashboard.relative_risk_summary if dashboard else pd.DataFrame())}
      </div>
      <div class="card">
        <h3 style="margin-top:0;">종목별 절대 리스크 기여 (%)</h3>
        {f'<img src="data:image/png;base64,{dashboard.risk_contribution_chart}" class="chart-img" alt="Risk Contribution Chart" />' if dashboard and dashboard.risk_contribution_chart else "<p class='hint'>차트 데이터를 불러올 수 없습니다.</p>"}
        <div style="margin-top:10px;">{_safe_table(dashboard.risk_contribution if dashboard else pd.DataFrame(), max_rows=10)}</div>
      </div>
      <div class="card">
        <h3 style="margin-top:0;">종목별 Active Risk 기여 (%)</h3>
        {f'<img src="data:image/png;base64,{dashboard.active_risk_contribution_chart}" class="chart-img" alt="Active Risk Contribution Chart" />' if dashboard and dashboard.active_risk_contribution_chart else "<p class='hint'>차트 데이터를 불러올 수 없습니다.</p>"}
        <div style="margin-top:10px;">{_safe_table(dashboard.active_risk_contribution if dashboard else pd.DataFrame(), max_rows=10)}</div>
      </div>
      <div class="card">
        <h3>스타일 노출</h3>
        {_safe_table(dashboard.style_exposure if dashboard else pd.DataFrame())}
      </div>
      <div class="card">
        <h3 style="margin-top:0;">팩터 분해</h3>
        {_safe_table(dashboard.factor_risk if dashboard else pd.DataFrame())}
      </div>
    </div>
    """
    return _layout("Portfolio Lab | 리스크", "절대 리스크와 함께 SP500 대비 tracking error, active risk, 스타일 노출을 함께 봅니다.", "risk", ctx, body)


def _scoring_page(ctx: _PageContext) -> str:
    dashboard = ctx.dashboard
    scoring = dashboard.scoring if dashboard else pd.DataFrame()
    note = ""
    if dashboard and dashboard.diagnostics.get("financial_metric_source") == "not_available":
        note = "<p class='hint'>ROE는 shared DB의 최근 4분기 재무 이력에서 계산하고, PER/PBR은 같은 재무 이력과 주가 DB를 결합해 분석 시점에 재계산합니다. 아직 fundamentals_quarterly 데이터가 없어서 현재는 가격/기술/뉴스 신호 중심으로 통합 점수가 계산됩니다.</p>"

    # 통합스코어 페이지의 종목 링크는 Pipeline Stock으로만 연결합니다.
    # Pipeline Stock News 쪽으로는 연결하지 않습니다. (이제는 통합되므로 내부 경로로 변경)
    def _with_stock_lab_links(df_in: pd.DataFrame | None) -> pd.DataFrame:
        if df_in is None or df_in.empty or 'ticker' not in df_in.columns:
            return df_in if df_in is not None else pd.DataFrame()
        df = df_in.copy()
        def _link(t: str) -> str:
            if t == "CASH": return t
            # 로컬 중계 경로를 통해 Stock Lab 자동 실행 후 리다이렉트
            safe_ticker = html.escape(str(t).strip().upper())
            url = f"/stock-forecast?ticker={safe_ticker}&intent=run" # 내부 경로로 변경
            return (
                f'<a href="{url}" '
                f'onclick="return triggerStockLabSync(\'{safe_ticker}\', this.href);" '
                f'style="color:var(--accent); font-weight:bold;">{html.escape(safe_ticker)}</a>'
            )
        df['ticker'] = df['ticker'].apply(_link)
        return df

    best_linked = _with_stock_lab_links(dashboard.best_scoring_stocks if dashboard else None)
    worst_linked = _with_stock_lab_links(dashboard.worst_scoring_stocks if dashboard else None)
    recs_linked = _with_stock_lab_links(dashboard.top_recommendations if dashboard else None)
    details_linked = _with_stock_lab_links(scoring)

    

    # 중첩 f-string 및 잘못된 주석을 피하기 위해 미리 변수 처리
    commentary_html = ""
    if dashboard and dashboard.scoring_commentary:
        commentary_html = f'<p class="hint">{html.escape(dashboard.scoring_commentary)}</p>'
    
    chart_score_html = "<p class='hint'>통합 스코어 차트를 불러올 수 없습니다.</p>"
    if dashboard and dashboard.integrated_score_chart:
        chart_score_html = f'<img src="data:image/png;base64,{dashboard.integrated_score_chart}" class="chart-img" style="max-width: 800px; margin: 0 auto; display: block;" />'

    body = f"""
    {_message_block(ctx)}
    {_date_range_form("scoring", ctx)}
    <div class="card">
      <h2 style="margin-top:0;">통합 스코어링 엔진</h2>
      {note}
      {commentary_html}
      {chart_score_html}
    </div>
    <div class="grid-2" style="margin-top: 12px;">
      <div class="card">
        <h3>상위 10개 종목</h3>
        {_safe_table(best_linked, max_rows=10, escape=False)}
      </div>
      <div class="card">
        <h3>하위 10개 종목</h3>
        {_safe_table(worst_linked, max_rows=10, escape=False)}
      </div>
    </div>
    <div class="card" style="margin-top: 12px; border-left: 5px solid var(--ok-line);">
      <h2>신규 추천 종목 (미보유 S&P500 상위 10)</h2>
      <p class="hint">현재 포트폴리오에 보유하지 않은 S&P500 종목 중 통합 스코어가 가장 높은 종목들입니다. 새로운 투자 기회 발굴에 참고하십시오.</p>
      {_safe_table(recs_linked, escape=False)}
    </div>
    <div class="card" style="margin-top: 12px;">
      <h2>전체 스코어링 상세 내역</h2>
      {_safe_table(details_linked, escape=False)}
    </div>
    <script>
      function triggerStockLabSync(ticker, targetPath) {{
        // Stock Lab 페이지로 이동하면서 티커를 전달
        const url = targetPath + "?ticker=" + encodeURIComponent(ticker) + "&intent=run";
        window.location.href = url;
        return true; // 링크 클릭 시 페이지 이동을 막지 않음
      }}
    </script>
    """
    return _layout("Portfolio Lab | 통합 스코어", "기술적 신호와 뉴스 신호, 재무 지표 슬롯을 묶어 종목별 우선순위를 봅니다.", "scoring", ctx, body)


def _virtual_trade_page(ctx: _PageContext) -> str:
    result = ctx.virtual_result
    body = f"""
    {_message_block(ctx)}
    {_date_range_form("virtual-trade", ctx)}
    <div class="card">
      <h2 style="margin-top:0;">가상 거래</h2>
      <form method="post" action="/run_virtual_trade">
        <input type="hidden" name="lookback_days" value="{int(ctx.lookback_days)}">
        <input type="hidden" name="start_date" value="{html.escape(ctx.start_date)}">
        <input type="hidden" name="end_date" value="{html.escape(ctx.end_date)}">
        <div class="form-grid form-virtual">
          <div><label>티커</label><input type="text" name="ticker" placeholder="AAPL" required></div>
          <div><label>매수/매도</label><select name="side"><option value="BUY">BUY</option><option value="SELL">SELL</option></select></div>
          <div><label>수량</label><input type="number" min="0.0001" step="0.0001" name="quantity" required></div>
          <div><label>가격(비우면 최신가)</label><input type="number" min="0.0001" step="0.0001" name="price" placeholder="자동"></div>
          <div><label>수수료</label><input type="number" min="0" step="0.0001" name="fees" value="0"></div>
          <div><label>예측 기간</label><input type="number" min="1" step="1" name="forecast_horizon_days" value="{DEFAULT_VIRTUAL_FORECAST_HORIZON_DAYS}"></div>
          <div><button type="submit">가상 거래 계산</button></div>
        </div>
      </form>
      <div class="db-note">10일 예측은 shared DB 가격 이력 기반의 로컬 프록시 모델로 계산합니다.</div>
    </div>
    <div class="grid-2">
      <div class="card">
        <h3>입력 요약</h3>
        {_safe_table(result.input_summary if result else pd.DataFrame())}
      </div>
      <div class="card">
        <h3>포지션 변화</h3>
        {_safe_table(result.position_changes if result else pd.DataFrame())}
      </div>
    </div>
    <div class="grid-2">
      <div class="card">
        <h3>거래 전</h3>
        {_safe_table(result.before_summary if result else pd.DataFrame())}
      </div>
      <div class="card">
        <h3>거래 후</h3>
        {_safe_table(result.after_summary if result else pd.DataFrame())}
      </div>
    </div>
    <div class="card">
      <h3>리스크 변화</h3>
      {_safe_table(result.risk_changes if result else pd.DataFrame())}
    </div>
    """
    return _layout("Portfolio Lab | 가상 거래", "가상 매수/매도 입력 시 성과와 리스크가 어떻게 바뀌는지 봅니다.", "virtual-trade", ctx, body)


def _optimization_page(ctx: _PageContext) -> str:
    optimization = ctx.optimization
    params = ctx.optimization_params or {
        "universe_size": DEFAULT_OPTIMIZATION_UNIVERSE_SIZE,
        "sector_cap_pct": DEFAULT_SECTOR_CAP_PCT,
        "max_position_pct": DEFAULT_MAX_POSITION_PCT,
        "cash_buffer_pct": DEFAULT_CASH_BUFFER_PCT,
    }
    
    # 최적화 차트 HTML 미리 준비
    rep_chart = f'<img src="data:image/png;base64,{optimization.replication_chart}" class="chart-img" />' if optimization and optimization.replication_chart else ""
    agg_chart = f'<img src="data:image/png;base64,{optimization.aggressive_chart}" class="chart-img" />' if optimization and optimization.aggressive_chart else ""
    def_chart = f'<img src="data:image/png;base64,{optimization.defensive_chart}" class="chart-img" />' if optimization and optimization.defensive_chart else ""

    body = f"""
    {_message_block(ctx)}
    {_date_range_form("optimization", ctx)}
    <div class="card">
      <h2 style="margin-top:0;">포트폴리오 최적화</h2>
      <p class="hint">S&P500 복제, 공격형, 방어형 포트폴리오를 현재 보유 종목과 비교하면서 제약조건을 반영해 재구성합니다.</p>
      <form method="get" action="/optimization">
        <div class="form-grid">
          <input type="hidden" name="intent" value="run">
          <input type="hidden" name="lookback_days" value="{int(ctx.lookback_days)}">
          <input type="hidden" name="start_date" value="{html.escape(ctx.start_date)}">
          <input type="hidden" name="end_date" value="{html.escape(ctx.end_date)}">
          <div><label>최적화 유니버스</label><input type="number" name="universe_size" min="20" step="1" value="{int(params['universe_size'])}"></div>
          <div><label>섹터 상한(%)</label><input type="number" name="sector_cap_pct" min="1" step="0.1" value="{float(params['sector_cap_pct']):.1f}"></div>
          <div><label>종목 최대비중(%)</label><input type="number" name="max_position_pct" min="1" step="0.1" value="{float(params['max_position_pct']):.1f}"></div>
          <div><label>현금비중(%)</label><input type="number" name="cash_buffer_pct" min="0" max="95" step="0.1" value="{float(params['cash_buffer_pct']):.1f}"></div>
          <div><button type="submit">최적화 실행</button></div>
        </div>
      </form>
    </div>
    <div class="grid-3">
      <div class="card">
        <h3>S&P500 복제</h3>
        <p class="hint" style="margin-bottom: 12px;">
          <strong>목적:</strong> 시장 평균 수익률 추종 및 추적 오차 최소화<br>
          <strong>방법:</strong> 유니버스 내에서 섹터 비중과 시가총액 가중치를 S&P 500 지수와 유사하게 매칭하여 재구성합니다.<br>
          <strong>제언:</strong> 포트폴리오의 중심(Core)을 잡을 때 활용하며, 지수 대비 소외되지 않는 안정적인 운영을 원할 때 적합합니다.
        </p>
        {rep_chart}
        {_safe_table(_format_opt_df(optimization.replication) if optimization and optimization.replication is not None else pd.DataFrame())}
        {_render_strategy_impact(optimization.impact_summary, "복제") if optimization and optimization.impact_summary is not None else ""}
      </div>
      <div class="card">
        <h3>공격형</h3>
        <p class="hint" style="margin-bottom: 12px;">
          <strong>목적:</strong> 시장 대비 초과 수익(Alpha) 극대화<br>
          <strong>방법:</strong> 통합 스코어(기술적 신호, 뉴스, 재무 지표)가 높은 주도주에 가중치를 집중적으로 할당합니다.<br>
          <strong>제언:</strong> 상승 모멘텀이 강한 종목을 선점하고자 할 때 활용하십시오. 변동성이 높을 수 있으므로 분할 매수 관점으로 접근하는 것이 유리합니다.
        </p>
        {agg_chart}
        {_safe_table(_format_opt_df(optimization.aggressive) if optimization and optimization.aggressive is not None else pd.DataFrame())}
        {_render_strategy_impact(optimization.impact_summary, "공격") if optimization else ""}
      </div>
      <div class="card">
        <h3>방어형</h3>
        <p class="hint" style="margin-bottom: 12px;">
          <strong>목적:</strong> 변동성 관리 및 하락장 방어력 강화<br>
          <strong>방법:</strong> 저변동성 종목과 재무 건전성이 우수한 종목을 중심으로 리스크 기여도를 낮추도록 최적화합니다.<br>
          <strong>제언:</strong> 시장의 불확실성이 크거나 자산 보호가 우선인 국면에서 비중을 늘리십시오. 장기적인 변동성 대비 수익성을 높이는 데 도움을 줍니다.
        </p>
        {def_chart}
        {_safe_table(_format_opt_df(optimization.defensive) if optimization and optimization.defensive is not None else pd.DataFrame())}
        {_render_strategy_impact(optimization.impact_summary, "방어") if optimization else ""}
      </div>
    </div>
    <div class="card">
      <h3 style="margin-top:0;">최적화 진단</h3>
      {_safe_table(optimization.diagnostics if optimization else pd.DataFrame())}
    </div>
    """
    return _layout("Portfolio Lab | 최적화", "포트폴리오를 통한 SP500 복제와 성향별 구성 추천", "optimization", ctx, body)

_REFRESH_JOB_DEFS: list[dict[str, str]] = [
    { # Portfolio's own stock refresh
        "job_id": "stock",
        "label": "SP500 가격/시총",
        "description": "S&P500 가격 패널, 시가총액 CSV, shared SQLite 가격 테이블을 갱신합니다.",
        "module": "pipeline_common.refresh_sp500_shared_prices",
        "button_label": "가격/시총 갱신",
    },
    {
        # Portfolio's own quarterly refresh
        "job_id": "quarterly",
        "label": "분기 재무",
        "description": "shared SQLite의 fundamentals_quarterly 테이블을 최신 분기 기준으로 채웁니다.",
        "module": "pipeline_common.refresh_shared_quarterly_fundamentals",
        "button_label": "분기 재무 갱신",
    },
    {
        # Portfolio's own news refresh
        "job_id": "news",
        "label": "뉴스",
        "description": "S&P500 구성 종목 뉴스 기사와 뉴스 분석 대기열을 shared SQLite에 적재합니다.",
        "module": "pipeline_common.refresh_sp500_news",
        "button_label": "뉴스 갱신",
    },
]

# Add refresh jobs from pipeline_stock.web_gui
_REFRESH_JOB_DEFS.extend([
    {
        "job_id": "stock_lab_stock",
        "label": "Stock Lab: SP500 가격/시총",
        "description": "Stock Analysis Lab에서 사용하는 S&P500 가격 및 시가총액 데이터를 갱신합니다.",
        "module": "pipeline_stock.cli", # This module has the refresh command
        "command_args": ["refresh"],
        "button_label": "Stock Lab 가격 갱신",
        "is_stock_lab_refresh": True,
    },
])



def _refresh_job_defs() -> list[dict[str, str]]:
    return [dict(item) for item in _REFRESH_JOB_DEFS]


def _refresh_job_def(job_id: str) -> dict[str, str] | None:
    for item in _REFRESH_JOB_DEFS:
        if item["job_id"] == job_id:
            return dict(item)
    return None


def _refresh_job_title(job_id: str) -> str:
    item = _refresh_job_def(job_id)
    return item["label"] if item else job_id


def _project_root_dir() -> Path:
    if getattr(sys, "frozen", False):
        exe_root = Path(sys.executable).resolve().parent
        if _has_runtime_data(exe_root):
            return exe_root
        internal_root = exe_root / "_internal"
        if _has_runtime_data(internal_root):
            return internal_root
        return exe_root
    return Path(__file__).resolve().parents[1]


def _has_runtime_data(root: Path) -> bool:
    data_dir = root / "data"
    return (
        (data_dir / "sp500_components_full.csv").is_file()
        and (data_dir / "sp500_shared_db" / "sp500_shared_prices.sqlite").is_file()
    )


def _refresh_subprocess_command(job_id: str) -> list[str]:
    job = _refresh_job_def(job_id)
    if job is None:
        return []

    # pipeline_stock.cli의 refresh 명령을 직접 호출
    if job.get("is_stock_lab_refresh"):
        if getattr(sys, "frozen", False):
            return [sys.executable, "refresh"] # Assuming 'refresh' is a top-level command in the frozen executable
        return [sys.executable, "-u", "-m", "pipeline_stock.cli", "refresh"]

    # pipeline_stock_news.cli의 refresh 명령을 직접 호출
    if job.get("is_news_lab_refresh"):
        if getattr(sys, "frozen", False):
            return [sys.executable, "refresh-news"] # Assuming 'refresh-news' is a top-level command in the frozen executable
        return [sys.executable, "-u", "-m", "pipeline_stock_news.cli", "refresh"]

    if getattr(sys, "frozen", False):
        refresh_command = {
            "stock": "refresh-stock",
            "quarterly": "refresh-quarterly",
            "news": "refresh-news",
        }.get(job_id)
        if refresh_command is None: # Should not happen if job_id is valid
            return []
        return [sys.executable, refresh_command]
    return [sys.executable, "-u", "-m", job["module"]]


def _format_timestamp(ts: float | None) -> str | None:
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def _path_label(path: Path, root_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(root_dir.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve())


def _build_file_item(
    path: Path,
    *,
    root_dir: Path,
    label: str | None = None,
    latest_value: str | None = None,
    rows: int | None = None,
    detail: str | None = None,
) -> dict[str, object]:
    exists = path.exists()
    return {
        "label": label or _path_label(path, root_dir),
        "path": _path_label(path, root_dir),
        "exists": exists,
        "latest_value": latest_value,
        "rows": rows,
        "modified_at": _format_timestamp(path.stat().st_mtime) if exists else None,
        "detail": detail,
    }


def _latest_date_from_csv(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        raw = pd.read_csv(path, nrows=5000)
    except Exception:
        return None
    if raw.empty:
        return None
    cols = {str(col).strip().lower(): col for col in raw.columns}
    candidate_col = cols.get("date") or cols.get("datetime") or raw.columns[0]
    try:
        dates = pd.to_datetime(raw[candidate_col], errors="coerce").dropna()
    except Exception:
        return None
    if dates.empty:
        return None
    return dates.max().strftime("%Y-%m-%d")


def _csv_row_count(path: Path) -> int | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            count = sum(1 for _ in handle)
    except Exception:
        return None
    return max(count - 1, 0)


def _sqlite_fetchone(path: Path, query: str) -> tuple[object, ...] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        with sqlite3.connect(path) as conn:
            return conn.execute(query).fetchone()
    except Exception:
        return None


def _collect_stock_refresh_snapshot(root_dir: Path) -> dict[str, object]:
    data_dir = root_dir / "data"
    metrics_path = data_dir / "sp500_all_metrics_prices.csv"
    market_cap_path = data_dir / "sp500_market_caps.csv"
    sqlite_path = data_dir / "sp500_shared_db" / "sp500_shared_prices.sqlite"
    sqlite_row = _sqlite_fetchone(sqlite_path, "SELECT MAX(date), COUNT(*) FROM prices")
    sqlite_max_date = str(sqlite_row[0]) if sqlite_row and sqlite_row[0] else None
    sqlite_rows = int(sqlite_row[1]) if sqlite_row and sqlite_row[1] is not None else None
    items = [
        _build_file_item(
            metrics_path,
            root_dir=root_dir,
            latest_value=_latest_date_from_csv(metrics_path),
            rows=_csv_row_count(metrics_path),
        ),
        _build_file_item(
            market_cap_path,
            root_dir=root_dir,
            latest_value=_latest_date_from_csv(market_cap_path),
            rows=_csv_row_count(market_cap_path),
        ),
        _build_file_item(
            sqlite_path,
            root_dir=root_dir,
            latest_value=sqlite_max_date,
            rows=sqlite_rows,
            detail="prices 테이블",
        ),
    ]
    summary = (
        f"가격 {items[0]['latest_value'] or '-'} / "
        f"시총 {items[1]['latest_value'] or '-'} / "
        f"DB 최대일 {sqlite_max_date or '-'}"
    )
    return {"summary": summary, "items": items}


def _collect_quarterly_refresh_snapshot(root_dir: Path) -> dict[str, object]:
    sqlite_path = root_dir / "data" / "sp500_shared_db" / "sp500_shared_prices.sqlite"
    row = _sqlite_fetchone(
        sqlite_path,
        "SELECT MAX(fiscal_date), COUNT(*), COUNT(DISTINCT symbol), MAX(updated_at) FROM fundamentals_quarterly",
    )
    max_fiscal = str(row[0]) if row and row[0] else None
    total_rows = int(row[1]) if row and row[1] is not None else None
    symbol_count = int(row[2]) if row and row[2] is not None else None
    updated_at = str(row[3]) if row and row[3] else None
    item = _build_file_item(
        sqlite_path,
        root_dir=root_dir,
        latest_value=max_fiscal,
        rows=total_rows,
        detail=f"symbols={symbol_count or 0}, updated_at={updated_at or '-'}",
    )
    summary = f"최신 fiscal_date {max_fiscal or '-'} / 종목 {symbol_count or 0}개"
    return {"summary": summary, "items": [item]}


def _collect_news_refresh_snapshot(root_dir: Path) -> dict[str, object]:
    sqlite_path = root_dir / "data" / "sp500_shared_db" / "sp500_shared_prices.sqlite"
    row = _sqlite_fetchone(
        sqlite_path,
        "SELECT MAX(publish_date), COUNT(*), COUNT(DISTINCT ticker) FROM news_articles",
    )
    max_publish = str(row[0]) if row and row[0] else None
    total_rows = int(row[1]) if row and row[1] is not None else None
    ticker_count = int(row[2]) if row and row[2] is not None else None
    item = _build_file_item(
        sqlite_path,
        root_dir=root_dir,
        latest_value=max_publish,
        rows=total_rows,
        detail=f"tickers={ticker_count or 0}, table=news_articles",
    )
    summary = f"최신 publish_date {max_publish or '-'} / 기사 {total_rows or 0}건"
    return {"summary": summary, "items": [item]}


def _collect_refresh_snapshot(root_dir: Path, job_id: str) -> dict[str, object]:
    if job_id == "stock":
        return _collect_stock_refresh_snapshot(root_dir)
    if job_id == "quarterly":
        return _collect_quarterly_refresh_snapshot(root_dir)
    if job_id == "news":
        return _collect_news_refresh_snapshot(root_dir)
    return {"summary": "알 수 없는 작업", "items": []}


def _copy_refresh_items(items: list[dict[str, object]] | None) -> list[dict[str, object]]:
    return [dict(item) for item in (items or [])]


def _refresh_page(ctx: _PageContext) -> str:
    job_count = len(_REFRESH_JOB_DEFS)
    cards: list[str] = []
    for job in _refresh_job_defs():
        job_id = job["job_id"]
        cards.append(
            f"""
            <div class="card refresh-card" data-job-id="{html.escape(job_id)}">
              <div class="refresh-card-head">
                <div>
                  <h2 class="refresh-card-title">{html.escape(job["label"])}</h2>
                  <p class="hint refresh-card-desc" style="margin-top:0;">{html.escape(job["description"])}</p>
                </div>
                <div class="latest-inline" id="refresh-latest-{html.escape(job_id)}">
                  <strong>실행 전 최신 현황</strong>
                  <div id="refresh-latest-summary-{html.escape(job_id)}" class="small">최신 현황 확인 중...</div>
                  <div id="refresh-latest-items-{html.escape(job_id)}"></div>
                </div>
              </div>
              <form method="post" action="/run_refresh" class="refresh-run-form">
                <input type="hidden" name="lookback_days" value="{int(ctx.lookback_days)}">
                <input type="hidden" name="start_date" value="{html.escape(ctx.start_date)}">
                <input type="hidden" name="end_date" value="{html.escape(ctx.end_date)}">
                <input type="hidden" name="job_id" value="{html.escape(job_id)}">
                <div class="refresh-action-row">
                  <button type="submit" id="refresh-btn-{html.escape(job_id)}">{html.escape(job["button_label"])}</button>
                  <div id="refresh-meta-{html.escape(job_id)}" class="refresh-meta">상태: 대기 / 실행 ID: - / 시작: - / 종료: -</div>
                </div>
              </form>
              <div class="split-grid refresh-split">
                <div class="pane">
                  <div class="refresh-pane-title"><h3>실행 로그</h3><span>Live</span></div>
                  <div id="refresh-log-{html.escape(job_id)}" class="line-list refresh-log-list"><div class="line">아직 로그가 없습니다.</div></div>
                </div>
                <div class="pane">
                  <div class="refresh-pane-title"><h3>갱신 결과</h3><span>Output</span></div>
                  <div id="refresh-updates-{html.escape(job_id)}" class="line-list refresh-update-list"><div class="line">아직 갱신 결과가 없습니다.</div></div>
                </div>
              </div>
            </div>
            """
        )
    body = f"""
    {_message_block(ctx)}
    <div class="card refresh-hero">
      <div class="refresh-hero-grid">
        <div>
          <h2>공용 데이터 갱신</h2>
          <p class="hint" style="margin:0 0 8px 0;">
            포트폴리오, 스톡_뉴스, 스톡_분석 화면에서 함께 쓰는 데이터를 여기서 갱신합니다.
            분석 화면은 바로 열어볼 수 있지만, 최신 기준으로 다시 보려면 아래 작업을 먼저 실행하는 편이 안전합니다.
          </p>
          <p class="hint" style="margin:0;">
            전체 갱신을 한 번에 묶지 않고 {job_count}개 작업으로 나눴습니다.
            각 작업은 독립적으로 실행되며, 카드 안에서 최신 현황, 실행 로그, 갱신 결과를 바로 확인할 수 있습니다.
          </p>
        </div>
        <div class="refresh-summary">
          <div class="refresh-summary-title">사용 방법</div>
          <div class="refresh-summary-list">
            <div class="refresh-summary-item"><strong>1. 필요한 작업만 실행</strong>가격/시총, 분기 재무, 뉴스를 각각 따로 갱신합니다.</div>
            <div class="refresh-summary-item"><strong>2. 한 번에 하나씩 진행</strong>동시에 여러 작업을 돌리지 않고 순서대로 처리합니다.</div>
            <div class="refresh-summary-item"><strong>3. 완료 후 분석 화면 확인</strong>갱신이 끝나면 관련 페이지로 돌아가 최신 결과를 확인합니다.</div>
          </div>
        </div>
      </div>
    </div>
    <div class="stack">
      {"".join(cards)}
    </div>
    <script>
      const refreshLogCounts = {{}};
      let refreshStatusBootstrapped = false;

      function escapeHtml(value) {{
        return String(value ?? "")
          .replace(/&/g, "&amp;")
          .replace(/</g, "&lt;")
          .replace(/>/g, "&gt;");
      }}

      function renderItems(el, items, emptyText) {{
        if (!el) return;
        if (!items || items.length === 0) {{
          el.innerHTML = "<div class='line'>" + escapeHtml(emptyText) + "</div>";
          return;
        }}
        el.innerHTML = items.map((item) => {{
          const label = escapeHtml(item.label || item.path || "-");
          const latest = escapeHtml(item.latest_value || "-");
          const rows = item.rows == null ? "-" : escapeHtml(item.rows);
          const modified = escapeHtml(item.modified_at || "-");
          const detail = item.detail ? " / " + escapeHtml(item.detail) : "";
          return "<div class='line'><strong>" + label + "</strong> | latest=" + latest + " | rows=" + rows + " | modified=" + modified + detail + "</div>";
        }}).join("");
        el.scrollTop = el.scrollHeight;
      }}

      function renderLatest(summaryEl, itemsEl, summary, items) {{
        if (summaryEl) {{
          summaryEl.textContent = summary || "현재 DB 상태를 확인할 수 없습니다.";
        }}
        if (!itemsEl) {{ return; }}
        if (!items || items.length === 0) {{
          itemsEl.innerHTML = "";
          return;
        }}
        itemsEl.innerHTML = items.map((item) => {{
          const label = escapeHtml(item.label || item.path || "-");
          const latest = escapeHtml(item.latest_value || "-");
          const rows = item.rows == null ? "-" : escapeHtml(item.rows);
          return "<div>" + label + " · latest=" + latest + " · rows=" + rows + "</div>";
        }}).join("");
      }}

      function renderLogs(el, lines, emptyText) {{
        if (!el) {{ return; }}
        if (!lines || lines.length === 0) {{
          el.innerHTML = "<div class='line'>" + escapeHtml(emptyText) + "</div>";
          return;
        }}
        el.innerHTML = lines.map((line) => "<div class='line'>" + escapeHtml(line) + "</div>").join("");
        el.scrollTop = el.scrollHeight;
      }}

      async function pollRefreshStatus() {{
        try {{
          const res = await fetch("/refresh_status", {{ cache: "no-store", headers: {{ 'X-Requested-With': 'fetch' }} }});
          if (!res.ok) return;
          const data = await res.json();
          const jobs = data.jobs || [];
          jobs.forEach((job) => {{
            const jobId = job.job_id;
            const metaEl = document.getElementById("refresh-meta-" + jobId);
            const logEl = document.getElementById("refresh-log-" + jobId);
            const updatesEl = document.getElementById("refresh-updates-" + jobId);
            const btnEl = document.getElementById("refresh-btn-" + jobId);
            const latestSummaryEl = document.getElementById("refresh-latest-summary-" + jobId);
            const latestItemsEl = document.getElementById("refresh-latest-items-" + jobId);
            renderLatest(latestSummaryEl, latestItemsEl, job.latest_summary || "", job.latest_items || []);
            if (metaEl) {{
              metaEl.textContent =
                "상태: " + (job.status || "대기") +
                " / 실행 ID: " + (job.run_id || "-") +
                " / 시작: " + (job.started_at || "-") +
                " / 종료: " + (job.finished_at || "-");
            }}
            if (btnEl) {{ // 버튼 텍스트 및 활성화/비활성화 상태 업데이트
              const isBusy = Boolean(data.running);
              btnEl.disabled = isBusy;
              btnEl.style.opacity = isBusy ? "0.6" : "1";
              btnEl.textContent = isBusy && data.current_job_id === jobId ? "실행 중..." : (job.button_label || "갱신 실행");
            }}
            if (job.log_count !== refreshLogCounts[jobId]) {{
              refreshLogCounts[jobId] = job.log_count;
              renderLogs(logEl, job.logs || [], "아직 로그가 없습니다.");
            }}
            renderItems(updatesEl, job.updated_items || [], "아직 갱신 결과가 없습니다.");
          }});
          refreshStatusBootstrapped = true;
        }} catch (err) {{
          console.error(err);
        }}
      }}

      document.querySelectorAll(".refresh-run-form").forEach((formEl) => {{
        formEl.addEventListener("submit", async (event) => {{
          event.preventDefault();
          try {{
            const body = new URLSearchParams(new FormData(formEl));
            const res = await fetch(formEl.action, {{
              method: "POST",
              headers: {{
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                "Accept": "application/json",
                "X-Requested-With": "fetch",
              }},
              body,
            }});
            const data = await res.json().catch(() => ({{ ok: false, error: "응답을 해석할 수 없습니다." }}));
            if (!res.ok || !data.ok) {{
              const message = data.error || "작업 시작에 실패했습니다.";
              window.alert(message);
              return;
            }}
            await pollRefreshStatus();
          }} catch (err) {{
            console.error(err);
            window.alert("작업 요청 중 오류가 발생했습니다.");
          }}
        }});
      }});

      setInterval(pollRefreshStatus, 2000);
      pollRefreshStatus();
    </script>
    """
    return _layout(
        "Portfolio Lab | 데이터 갱신",
        "공용 데이터를 작업별로 나눠 갱신하고 바로 상태를 확인합니다.",
        "refresh",
        ctx,
        body,
        show_nav=False,
    )


_html_refresh_page = _refresh_page

def _render_strategy_impact(impact_df: pd.DataFrame | None, strategy_col: str) -> str:
    if impact_df is None or impact_df.empty:
        return ""
    
    # "현재" 포트폴리오와 특정 전략("복제"/"공격"/"방어")만 추출하여 비교
    try:
        subset = impact_df[["구분", "현재", strategy_col]].copy()
        subset.columns = ["지표", "현재(Before)", "조정후(After)"]
        
        # 변화량 계산 (Delta)
        subset["변화"] = subset["조정후(After)"] - subset["현재(Before)"]
        
        # 수치 포맷팅
        for c in ["현재(Before)", "조정후(After)"]:
            subset[c] = subset[c].map(lambda x: f"{x:.2f}")
        subset["변화"] = subset["변화"].map(lambda x: f"{x:+.2f}")
        
        return f"""
        <div style="margin-top: 15px; padding: 10px; background: #fff; border-radius: 6px; border: 1px solid var(--line);">
          <h4 style="margin: 0 0 5px 0; font-size: 13px;">전후 기대 성과 비교 (Impact)</h4>
          <p class="small muted" style="margin-bottom:8px;">※ 기대 수익률과 예상 변동성은 연율 기준 추정치이며, 실제 성과를 보장하지 않습니다.</p>
          {_safe_table(subset, max_rows=10)}
        </div>
        """
    except Exception:
        return ""

def _format_opt_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    rename_map = {
        "ticker": "티커",
        "sector": "섹터",
        "target_weight_pct": "제안(%)",
        "current_weight_pct": "현재(%)",
        "diff_weight_pct": "차이(%)",
        "suggested_trade": "추천 거래(주)",
        "expected_return_pct": "기대수익(연 %)",
        "volatility_pct": "변동성(연 %)",
        "integrated_score": "점수"
    }
    cols = ["ticker", "sector", "target_weight_pct", "current_weight_pct", "diff_weight_pct", "suggested_trade", "expected_return_pct", "volatility_pct", "integrated_score"]
    out = df[[c for c in cols if c in df.columns]].copy() # Ensure columns exist before selecting
    for c in out.columns:
        if c == "diff_weight_pct":
            out[c] = out[c].map(lambda x: f"{x:+.1f}" if pd.notna(x) else "-")
        elif c in ["target_weight_pct", "current_weight_pct", "expected_return_pct", "volatility_pct"]:
            out[c] = out[c].map(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
        elif c == "integrated_score":
            out[c] = out[c].map(lambda x: f"{x:.0f}" if pd.notna(x) else "-")
    return out.rename(columns=rename_map)

def _parse_lookback(raw: str | None) -> int:
    try:
        return max(int(raw or DEFAULT_LOOKBACK_DAYS), 21) # Use DEFAULT_LOOKBACK_DAYS from portfolio.analysis
    except Exception:
        return DEFAULT_LOOKBACK_DAYS


def _parse_date(raw: str | None, fallback: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return fallback
    try:
        return pd.Timestamp(text).normalize().strftime("%Y-%m-%d")
    except Exception:
        return fallback


def _resolve_range(raw_start: str | None, raw_end: str | None, lookback_days: int) -> tuple[str, str, int]:
    fallback_end = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    end_date = _parse_date(raw_end, fallback_end)
    fallback_start = "2025-12-31"
    start_date = _parse_date(raw_start, fallback_start)
    if pd.Timestamp(start_date) > pd.Timestamp(end_date):
        start_date = fallback_start
    resolved_lookback = max(int((pd.Timestamp(end_date) - pd.Timestamp(start_date)).days), 21)
    return start_date, end_date, resolved_lookback


def _parse_int(raw: str | None, default: int, minimum: int) -> int:
    try:
        return max(int(raw or default), minimum)
    except Exception:
        return default


def _parse_float(raw: str | None, default: float, minimum: float, maximum: float | None = None) -> float:
    try:
        value = float(raw or default)
    except Exception:
        value = default
    value = max(value, minimum)
    if maximum is not None:
        value = min(value, maximum)
    return value


def _should_run_analysis(params: dict[str, list[str]]) -> bool:
    intent = str(params.get("intent", [""])[0]).strip().lower()
    return intent in {"run", "refresh", "analyze"}


def _load_dashboard_or_error(lookback_days: int, start_date: str, end_date: str) -> tuple[PortfolioDashboard | None, str | None]:
    try:
        return build_portfolio_dashboard(lookback_days=lookback_days, start_date=start_date, end_date=end_date), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def launch_web_gui(host: str = "localhost", port: int = 8515, open_browser: bool = False) -> ThreadingHTTPServer:
    dashboard_cache_lock = threading.Lock()
    dashboard_cache_ttl_seconds = 45.0
    dashboard_cache: dict[tuple[int, str, str], tuple[float, PortfolioDashboard | None, str | None]] = {}

    session_cache_lock = threading.Lock()
    session_dashboard_cache: dict[str, dict[tuple[int, str, str], tuple[PortfolioDashboard | None, str | None]]] = {}

    optimization_cache_lock = threading.Lock()
    optimization_cache_ttl_seconds = 45.0
    optimization_cache: dict[
        tuple[int, int, float, float, float],
        tuple[float, OptimizationResult | None, str | None],
    ] = {}
    session_optimization_cache: dict[
        str,
        dict[tuple[int, int, float, float, float], tuple[OptimizationResult | None, str | None]],
    ] = {}

    def _remember_session_dashboard(session_id: str, lookback_days: int, start_date: str, end_date: str, dashboard: PortfolioDashboard | None, error: str | None) -> None:
        key = (int(lookback_days), str(start_date), str(end_date))
        with session_cache_lock:
            bucket = session_dashboard_cache.setdefault(str(session_id), {})
            bucket[key] = (dashboard, error)
            while len(bucket) > 12:
                bucket.pop(next(iter(bucket)))

    def _load_session_dashboard(session_id: str, lookback_days: int, start_date: str, end_date: str) -> tuple[PortfolioDashboard | None, str | None]:
        key = (int(lookback_days), str(start_date), str(end_date))
        with session_cache_lock:
            bucket = session_dashboard_cache.get(str(session_id), {})
            cached = bucket.get(key)
        if cached is None:
            return None, None
        return cached[0], cached[1]

    def _load_latest_session_dashboard(session_id: str) -> tuple[PortfolioDashboard | None, str | None]:
        with session_cache_lock:
            bucket = session_dashboard_cache.get(str(session_id), {})
            if not bucket:
                return None, None
            last_key = next(reversed(bucket))
            cached = bucket[last_key]
        return cached[0], cached[1]

    def _remember_session_optimization(
        session_id: str,
        *,
        lookback_days: int,
        universe_size: int,
        sector_cap_pct: float,
        max_position_pct: float,
        cash_buffer_pct: float,
        optimization: OptimizationResult | None,
        error: str | None,
    ) -> None:
        key = (
            int(lookback_days),
            int(universe_size),
            float(sector_cap_pct),
            float(max_position_pct),
            float(cash_buffer_pct),
        )
        with session_cache_lock:
            bucket = session_optimization_cache.setdefault(str(session_id), {})
            bucket[key] = (optimization, error)
            while len(bucket) > 12:
                bucket.pop(next(iter(bucket)))

    def _load_session_optimization(
        session_id: str,
        *,
        lookback_days: int,
        universe_size: int,
        sector_cap_pct: float,
        max_position_pct: float,
        cash_buffer_pct: float,
    ) -> tuple[OptimizationResult | None, str | None]:
        key = (
            int(lookback_days),
            int(universe_size),
            float(sector_cap_pct),
            float(max_position_pct),
            float(cash_buffer_pct),
        )
        with session_cache_lock:
            bucket = session_optimization_cache.get(str(session_id), {})
            cached = bucket.get(key)
        if cached is None:
            return None, None
        return cached[0], cached[1]

    def _load_latest_session_optimization(session_id: str) -> tuple[OptimizationResult | None, str | None]:
        with session_cache_lock:
            bucket = session_optimization_cache.get(str(session_id), {})
            if not bucket:
                return None, None
            last_key = next(reversed(bucket))
            cached = bucket[last_key]
        return cached[0], cached[1]

    def _clear_runtime_caches() -> None:
        with dashboard_cache_lock:
            dashboard_cache.clear()
        with optimization_cache_lock:
            optimization_cache.clear()
        with session_cache_lock:
            session_dashboard_cache.clear()
            session_optimization_cache.clear()

    def _load_dashboard_or_error_cached(lookback_days: int, start_date: str, end_date: str) -> tuple[PortfolioDashboard | None, str | None]:
        key = (int(lookback_days), str(start_date), str(end_date))
        now = time.time()
        with dashboard_cache_lock:
            cached = dashboard_cache.get(key)
            if cached is not None and (now - float(cached[0])) <= dashboard_cache_ttl_seconds:
                return cached[1], cached[2]
        dashboard, error = _load_dashboard_or_error(lookback_days, start_date, end_date)
        with dashboard_cache_lock:
            dashboard_cache[key] = (now, dashboard, error)
        return dashboard, error

    def _build_portfolio_optimization_cached(
        *,
        lookback_days: int,
        universe_size: int,
        sector_cap_pct: float,
        max_position_pct: float,
        cash_buffer_pct: float,
    ) -> tuple[OptimizationResult | None, str | None]:
        key = (
            int(lookback_days),
            int(universe_size),
            float(sector_cap_pct),
            float(max_position_pct),
            float(cash_buffer_pct),
        )
        now = time.time()
        with optimization_cache_lock:
            cached = optimization_cache.get(key)
            if cached is not None and (now - float(cached[0])) <= optimization_cache_ttl_seconds:
                return cached[1], cached[2]
        try:
            optimization = build_portfolio_optimization(
                lookback_days=lookback_days,
                universe_size=universe_size,
                sector_cap_pct=sector_cap_pct,
                max_position_pct=max_position_pct,
                cash_buffer_pct=cash_buffer_pct,
            )
            error = None
        except Exception as exc:
            optimization = None
            error = f"{type(exc).__name__}: {exc}"
        with optimization_cache_lock:
            optimization_cache[key] = (now, optimization, error)
        return optimization, error

    def _blank_job_state(job_id: str) -> dict[str, object]:
        return {
            "job_id": job_id,
            "status": "idle",
            "run_id": 0,
            "started_at": None,
            "finished_at": None,
            "logs": [],
            "updated_items": [],
            "latest_summary": "",
            "latest_items": [],
        }

    class RefreshState:
        def __init__(self) -> None:
            self.lock = threading.Lock()
            self.running = False
            self.current_job_id: str | None = None
            self.jobs = {item["job_id"]: _blank_job_state(item["job_id"]) for item in _refresh_job_defs()}

    refresh_state = RefreshState()
    root_dir = _project_root_dir()

    def _ensure_refresh_snapshots_initialized() -> None:
        with refresh_state.lock:
            job_ids = [job_id for job_id, state in refresh_state.jobs.items() if not state.get("latest_summary")]
        for job_id in job_ids:
            snapshot = _collect_refresh_snapshot(root_dir, job_id)
            with refresh_state.lock:
                state = refresh_state.jobs[job_id]
                if not state.get("latest_summary"):
                    state["latest_summary"] = snapshot["summary"]
                    state["latest_items"] = _copy_refresh_items(snapshot["items"])

    def _append_job_log(job_id: str, line: str) -> None:
        text = str(line).rstrip()
        if not text:
            return
        with refresh_state.lock:
            state = refresh_state.jobs[job_id]
            logs = list(state["logs"])
            logs.append(text)
            state["logs"] = logs[-400:]

    def _start_refresh_job(job_id: str) -> None:
        command = _refresh_subprocess_command(job_id)
        if not command:
            with refresh_state.lock:
                state = refresh_state.jobs[job_id]
                state["status"] = "command-missing"
                state["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return

        job_title = _refresh_job_title(job_id)
        try:
            process = subprocess.Popen(
                command,
                cwd=str(root_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except Exception as exc:
            _append_job_log(job_id, f"[refresh:{job_id}] failed to start: {type(exc).__name__}: {exc}")
            with refresh_state.lock:
                state = refresh_state.jobs[job_id]
                state["status"] = "start-failed"
                state["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                refresh_state.running = False
                refresh_state.current_job_id = None
                snapshot = _collect_refresh_snapshot(root_dir, job_id)
                state["latest_summary"] = snapshot["summary"]
                state["latest_items"] = _copy_refresh_items(snapshot["items"])
                state["updated_items"] = _copy_refresh_items(snapshot["items"])
            return

        _append_job_log(job_id, f"[refresh:{job_id}] started {' '.join(command)}")
        if process.stdout is not None:
            for line in process.stdout:
                _append_job_log(job_id, line)
        exit_code = process.wait()
        snapshot = _collect_refresh_snapshot(root_dir, job_id)
        finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with refresh_state.lock:
            state = refresh_state.jobs[job_id]
            state["status"] = "completed" if exit_code == 0 else f"failed({exit_code})"
            state["finished_at"] = finished_at
            state["updated_items"] = _copy_refresh_items(snapshot["items"])
            state["latest_summary"] = snapshot["summary"]
            state["latest_items"] = _copy_refresh_items(snapshot["items"])
            refresh_state.running = False
            refresh_state.current_job_id = None
        _clear_runtime_caches()
        _append_job_log(job_id, f"[refresh:{job_id}] {job_title} finished with exit_code={exit_code}")

    class Handler(BaseHTTPRequestHandler):
        def _get_or_create_session_id(self) -> str:
            existing = getattr(self, "_portfolio_session_id", None)
            if existing:
                return str(existing)
            cookie = SimpleCookie()
            try:
                cookie.load(str(self.headers.get("Cookie", "")))
            except Exception:
                cookie = SimpleCookie()
            session_id = str(cookie.get("portfolio_lab_sid").value).strip() if cookie.get("portfolio_lab_sid") else ""
            if not session_id:
                session_id = uuid4().hex
                self._portfolio_session_is_new = True
            else:
                self._portfolio_session_is_new = False
            self._portfolio_session_id = session_id
            return session_id

        def _wants_json(self) -> bool:
            accept = str(self.headers.get("Accept", ""))
            requested_with = str(self.headers.get("X-Requested-With", ""))
            return "application/json" in accept or requested_with.lower() == "fetch"

        def _send_html(self, content: str, status: int = 200) -> None:
            self._get_or_create_session_id()
            encoded = content.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            if getattr(self, "_portfolio_session_is_new", False):
                self.send_header("Set-Cookie", f"portfolio_lab_sid={self._portfolio_session_id}; Path=/; HttpOnly; SameSite=Lax")
                self._portfolio_session_is_new = False
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_json(self, payload: dict[str, object], status: int = 200) -> None:
            self._get_or_create_session_id()
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            if getattr(self, "_portfolio_session_is_new", False):
                self.send_header("Set-Cookie", f"portfolio_lab_sid={self._portfolio_session_id}; Path=/; HttpOnly; SameSite=Lax")
                self._portfolio_session_is_new = False
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _render_error(self, exc: Exception, lookback_days: int = DEFAULT_LOOKBACK_DAYS, start_date: str | None = None, end_date: str | None = None) -> None:
            resolved_start, resolved_end, resolved_lookback = _resolve_range(start_date, end_date, lookback_days)
            ctx = _PageContext(
                dashboard=None,
                lookback_days=resolved_lookback,
                start_date=resolved_start,
                end_date=resolved_end,
                error=f"{type(exc).__name__}: {exc}",
            )
            body = _layout(
                "Portfolio Lab | 오류",
                "분석 중 오류가 발생했습니다.",
                "overview",
                ctx,
                _message_block(ctx) + f"<div class='card'><pre>{html.escape(traceback.format_exc())}</pre></div>",
            )
            self._send_html(body, status=500)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            path = parsed.path
            session_id = self._get_or_create_session_id()
            run_analysis = _should_run_analysis(params)
            lookback_days = _parse_lookback(params.get("lookback_days", [None])[0])
            start_date, end_date, lookback_days = _resolve_range(
                params.get("start_date", [None])[0],
                params.get("end_date", [None])[0],
                lookback_days,
            )
            try:
                if path == "/redirect_stock_lab":
                    ticker = params.get("ticker", [""])[0]
                    intent = params.get("intent", [""])[0]
                    _ensure_stock_lab_running(8512)
                    target = f"http://localhost:8512/page6?ticker={ticker}&intent={intent}"
                    self.send_response(303)
                    self.send_header("Location", target)
                    self.end_headers()
                    return
                if path == "/sync_stock_lab":
                    ticker = str(params.get("ticker", [""])[0]).strip().upper()
                    ok, message = _dispatch_stock_lab_external_select(ticker, target="/page6", port=8512)
                    if ok:
                        self._send_json({"ok": True, "ticker": ticker, "message": message})
                    else:
                        self._send_json({"ok": False, "ticker": ticker, "error": message}, status=502)
                    return

                if path in ("/", "/index.html", "/data-entry"):
                    if run_analysis:
                        dashboard, error = _load_dashboard_or_error_cached(lookback_days, start_date, end_date)
                        _remember_session_dashboard(session_id, lookback_days, start_date, end_date, dashboard, error)
                    else:
                        dashboard, error = _load_session_dashboard(session_id, lookback_days, start_date, end_date)
                        if dashboard is None and error is None:
                            dashboard, error = _load_latest_session_dashboard(session_id)
                    self._send_html(_data_entry_page(_PageContext(dashboard=dashboard, lookback_days=lookback_days, start_date=start_date, end_date=end_date, error=error)))
                    return
                if path == "/overview":
                    if run_analysis:
                        dashboard, error = _load_dashboard_or_error_cached(lookback_days, start_date, end_date)
                        _remember_session_dashboard(session_id, lookback_days, start_date, end_date, dashboard, error)
                    else:
                        dashboard, error = _load_session_dashboard(session_id, lookback_days, start_date, end_date)
                        if dashboard is None and error is None:
                            dashboard, error = _load_latest_session_dashboard(session_id)
                    self._send_html(_overview_page(_PageContext(dashboard=dashboard, lookback_days=lookback_days, start_date=start_date, end_date=end_date, error=error)))
                    return
                if path == "/attribution":
                    if run_analysis:
                        dashboard, error = _load_dashboard_or_error_cached(lookback_days, start_date, end_date)
                        _remember_session_dashboard(session_id, lookback_days, start_date, end_date, dashboard, error)
                    else:
                        dashboard, error = _load_session_dashboard(session_id, lookback_days, start_date, end_date)
                        if dashboard is None and error is None:
                            dashboard, error = _load_latest_session_dashboard(session_id)
                    self._send_html(_attribution_page(_PageContext(dashboard=dashboard, lookback_days=lookback_days, start_date=start_date, end_date=end_date, error=error)))
                    return
                if path == "/risk":
                    if run_analysis:
                        dashboard, error = _load_dashboard_or_error_cached(lookback_days, start_date, end_date)
                        _remember_session_dashboard(session_id, lookback_days, start_date, end_date, dashboard, error)
                    else:
                        dashboard, error = _load_session_dashboard(session_id, lookback_days, start_date, end_date)
                        if dashboard is None and error is None:
                            dashboard, error = _load_latest_session_dashboard(session_id)
                    self._send_html(_risk_page(_PageContext(dashboard=dashboard, lookback_days=lookback_days, start_date=start_date, end_date=end_date, error=error)))
                    return
                if path == "/scoring":
                    if run_analysis:
                        dashboard, error = _load_dashboard_or_error_cached(lookback_days, start_date, end_date)
                        _remember_session_dashboard(session_id, lookback_days, start_date, end_date, dashboard, error)
                    else:
                        dashboard, error = _load_session_dashboard(session_id, lookback_days, start_date, end_date)
                        if dashboard is None and error is None:
                            dashboard, error = _load_latest_session_dashboard(session_id)
                    self._send_html(_scoring_page(_PageContext(dashboard=dashboard, lookback_days=lookback_days, start_date=start_date, end_date=end_date, error=error)))
                    return
                if path == "/virtual-trade":
                    if run_analysis:
                        dashboard, error = _load_dashboard_or_error_cached(lookback_days, start_date, end_date)
                        _remember_session_dashboard(session_id, lookback_days, start_date, end_date, dashboard, error)
                    else:
                        dashboard, error = _load_session_dashboard(session_id, lookback_days, start_date, end_date)
                        if dashboard is None and error is None:
                            dashboard, error = _load_latest_session_dashboard(session_id)
                    self._send_html(_virtual_trade_page(_PageContext(dashboard=dashboard, lookback_days=lookback_days, start_date=start_date, end_date=end_date, error=error)))
                    return
                if path == "/optimization":
                    universe_size = _parse_int(params.get("universe_size", [None])[0], DEFAULT_OPTIMIZATION_UNIVERSE_SIZE, 20)
                    sector_cap_pct = _parse_float(params.get("sector_cap_pct", [None])[0], DEFAULT_SECTOR_CAP_PCT, 1.0, 100.0)
                    max_position_pct = _parse_float(params.get("max_position_pct", [None])[0], DEFAULT_MAX_POSITION_PCT, 0.5, 100.0)
                    cash_buffer_pct = _parse_float(params.get("cash_buffer_pct", [None])[0], DEFAULT_CASH_BUFFER_PCT, 0.0, 95.0)
                    if run_analysis:
                        dashboard, error = _load_dashboard_or_error_cached(lookback_days, start_date, end_date)
                        _remember_session_dashboard(session_id, lookback_days, start_date, end_date, dashboard, error)
                        optimization, optimization_error = _build_portfolio_optimization_cached(
                            lookback_days=lookback_days,
                            universe_size=universe_size,
                            sector_cap_pct=sector_cap_pct,
                            max_position_pct=max_position_pct,
                            cash_buffer_pct=cash_buffer_pct,
                        )
                        _remember_session_optimization(
                            session_id,
                            lookback_days=lookback_days,
                            universe_size=universe_size,
                            sector_cap_pct=sector_cap_pct,
                            max_position_pct=max_position_pct,
                            cash_buffer_pct=cash_buffer_pct,
                            optimization=optimization,
                            error=optimization_error,
                        )
                    else:
                        dashboard, error = _load_session_dashboard(session_id, lookback_days, start_date, end_date)
                        if dashboard is None and error is None:
                            dashboard, error = _load_latest_session_dashboard(session_id)
                        optimization, optimization_error = _load_session_optimization(
                            session_id,
                            lookback_days=lookback_days,
                            universe_size=universe_size,
                            sector_cap_pct=sector_cap_pct,
                            max_position_pct=max_position_pct,
                            cash_buffer_pct=cash_buffer_pct,
                        )
                        if optimization is None and optimization_error is None:
                            optimization, optimization_error = _load_latest_session_optimization(session_id)
                    self._send_html(
                        _optimization_page(
                            _PageContext(
                                dashboard=dashboard,
                                lookback_days=lookback_days,
                                start_date=start_date,
                                end_date=end_date,
                                error=optimization_error or error,
                                optimization=optimization,
                                optimization_params={
                                    "universe_size": universe_size,
                                    "sector_cap_pct": sector_cap_pct,
                                    "max_position_pct": max_position_pct,
                                    "cash_buffer_pct": cash_buffer_pct,
                                },
                            )
                        )
                    )
                    return
                if path == "/refresh":
                    self._send_html(_html_refresh_page(_PageContext(dashboard=None, lookback_days=lookback_days, start_date=start_date, end_date=end_date)))
                    return
                if path == "/refresh_status":
                    _ensure_refresh_snapshots_initialized()
                    with refresh_state.lock:
                        jobs_payload: list[dict[str, object]] = []
                        for job in _refresh_job_defs():
                            state = refresh_state.jobs[job["job_id"]]
                            jobs_payload.append(
                                {
                                    "job_id": job["job_id"],
                                    "label": job["label"],
                                    "button_label": job["button_label"],
                                    "status": state["status"],
                                    "run_id": state["run_id"],
                                    "started_at": state["started_at"],
                                    "finished_at": state["finished_at"],
                                    "logs": state["logs"][-100:],
                                    "log_count": len(state["logs"]),
                                    "updated_items": state["updated_items"],
                                    "latest_summary": state["latest_summary"],
                                    "latest_items": state["latest_items"],
                                }
                            )
                        self._send_json(
                            {
                                "running": refresh_state.running,
                                "current_job_id": refresh_state.current_job_id,
                                "jobs": jobs_payload,
                            }
                        )
                    return
                if path == "/healthz":
                    self._send_json({"ok": True, "lookback_days": lookback_days, "start_date": start_date, "end_date": end_date})
                    return
                self._send_html("<h1>Not Found</h1>", status=404)
            except Exception as exc:
                self._render_error(exc, lookback_days=lookback_days, start_date=start_date, end_date=end_date)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            content_length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(content_length).decode("utf-8", errors="replace")
            form = {key: values[-1] for key, values in parse_qs(raw).items()}
            lookback_days = _parse_lookback(form.get("lookback_days"))
            start_date, end_date, lookback_days = _resolve_range(form.get("start_date"), form.get("end_date"), lookback_days)
            try:
                if path == "/run_add_trade":
                    add_trade(
                        trade_date=form.get("trade_date", ""),
                        ticker=form.get("ticker", ""),
                        side=form.get("side", ""),
                        quantity=float(form.get("quantity", "0") or 0),
                        price=float(form.get("price", "0") or 0),
                        fees=float(form.get("fees", "0") or 0),
                        notes=form.get("notes", ""),
                    )
                    _clear_runtime_caches()
                    dashboard, error = _load_dashboard_or_error_cached(lookback_days, start_date, end_date)
                    _remember_session_dashboard(self._get_or_create_session_id(), lookback_days, start_date, end_date, dashboard, error)
                    self._send_html(
                        _data_entry_page(
                            _PageContext(
                                dashboard=dashboard,
                                lookback_days=lookback_days,
                                start_date=start_date,
                                end_date=end_date,
                                message="거래가 저장되었습니다.",
                                error=error,
                            )
                        )
                    )
                    return
                if path == "/run_delete_trade":
                    trade_id_raw = str(form.get("trade_id", "")).strip()
                    if not trade_id_raw:
                        raise ValueError("삭제할 거래 id가 없습니다.")
                    deleted = delete_trade(int(trade_id_raw))
                    _clear_runtime_caches()
                    dashboard, error = _load_dashboard_or_error_cached(lookback_days, start_date, end_date)
                    _remember_session_dashboard(self._get_or_create_session_id(), lookback_days, start_date, end_date, dashboard, error)
                    message = f"거래 #{int(trade_id_raw)}를 삭제했습니다." if deleted else f"거래 #{int(trade_id_raw)}를 찾지 못했습니다."
                    response_error = error if deleted else (error or f"거래 #{int(trade_id_raw)}를 찾지 못했습니다.")
                    response_message = message if deleted else None
                    self._send_html(
                        _data_entry_page(
                            _PageContext(
                                dashboard=dashboard,
                                lookback_days=lookback_days,
                                start_date=start_date,
                                end_date=end_date,
                                message=response_message,
                                error=response_error if not deleted else error,
                            )
                        )
                    )
                    return
                if path == "/run_virtual_trade":
                    dashboard, error = _load_dashboard_or_error_cached(lookback_days, start_date, end_date)
                    _remember_session_dashboard(self._get_or_create_session_id(), lookback_days, start_date, end_date, dashboard, error)
                    virtual_result = analyze_virtual_trade(
                        ticker=form.get("ticker", ""),
                        side=form.get("side", ""),
                        quantity=float(form.get("quantity", "0") or 0),
                        price=float(form["price"]) if str(form.get("price", "")).strip() else None,
                        fees=float(form.get("fees", "0") or 0),
                        lookback_days=lookback_days,
                        forecast_horizon_days=max(int(form.get("forecast_horizon_days", DEFAULT_VIRTUAL_FORECAST_HORIZON_DAYS)), 1),
                    )
                    self._send_html(
                        _virtual_trade_page(
                            _PageContext(
                                dashboard=dashboard,
                                lookback_days=lookback_days,
                                start_date=start_date,
                                end_date=end_date,
                                message="가상 거래 계산이 완료되었습니다.",
                                error=error,
                                virtual_result=virtual_result,
                            )
                        )
                    )
                    return

                # Stock Analysis Lab POST handlers
                if path == "/run": # Forecast
                    self._handle_stock_forecast_run(form)
                    return
                if path == "/run_financial":
                    self._handle_stock_financials_run(form)
                    return
                if path == "/run_technical":
                    self._handle_stock_technical_run(form)
                    return
                if path == "/run_returns":
                    self._handle_stock_returns_run(form)
                    return
                if path == "/run_risk":
                    self._handle_stock_risk_run(form)
                    return
                if path == "/run_factor":
                    self._handle_stock_factor_run(form)
                    return
                if path == "/run_decision":
                    self._handle_stock_decision_run(form)
                    return
                if path == "/run_walk_forward":
                    self._handle_stock_wfv_run(form)
                    return

                # News Lab POST handlers
                news_page_key = news_web_gui._page_key_from_path(path)
                if path == "/run_overview":
                    self._handle_news_run(form, news_web_gui._overview_page, "overview")
                    return
                if path == "/run_event_study":
                    self._handle_news_run(form, news_web_gui._html_event_page, "event")
                    return
                if path == "/run_sector_spillover":
                    self._handle_news_run(form, news_web_gui._html_spillover_page, "spillover")
                    return
                if path == "/run_divergence":
                    self._handle_news_run(form, news_web_gui._html_divergence_page, "divergence")
                    return
                if path == "/run_expectation_reset":
                    self._handle_news_run(form, news_web_gui._html_expectation_page, "expectation")
                    return
                if path == "/run_volatility_regime":
                    self._handle_news_run(form, news_web_gui._html_volatility_page, "volatility")
                    return
                if path == "/run_topic_modeling":
                    self._handle_news_run(form, news_web_gui._html_topics_page, "topics")
                    return

                if path == "/run_refresh":
                    job_id = str(form.get("job_id", "")).strip().lower()
                    job = _refresh_job_def(job_id)
                    if job is None:
                        if self._wants_json():
                            self._send_json({"ok": False, "error": "알 수 없는 갱신 작업입니다."}, status=400)
                            return
                        self._send_html(
                            _refresh_page(
                                _PageContext(
                                    dashboard=None,
                                    lookback_days=lookback_days,
                                    start_date=start_date,
                                    end_date=end_date,
                                    error="알 수 없는 갱신 작업입니다.",
                                )
                            ),
                            status=400,
                        )
                        return
                    with refresh_state.lock:
                        if refresh_state.running:
                            busy_job_id = refresh_state.current_job_id or ""
                            busy_job_title = _refresh_job_title(busy_job_id) if busy_job_id else "다른 작업"
                            if self._wants_json():
                                self._send_json(
                                    {
                                        "ok": False,
                                        "error": f"현재 {busy_job_title} 작업이 실행 중입니다. 완료 후 다시 시도해 주세요.",
                                    },
                                    status=409,
                                )
                                return
                            self._send_html(
                                _refresh_page(
                                    _PageContext(
                                        dashboard=None,
                                        lookback_days=lookback_days,
                                        start_date=start_date,
                                        end_date=end_date,
                                        error=f"현재 {busy_job_title} 작업이 실행 중입니다. 완료 후 다시 시도해 주세요.",
                                    )
                                )
                            )
                            return
                        state = refresh_state.jobs[job_id]
                        state["run_id"] = int(state["run_id"]) + 1
                        state["status"] = "running"
                        state["started_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        state["finished_at"] = None
                        state["logs"] = [f"[refresh:{job_id}] queued {job['label']}"]
                        refresh_state.running = True
                        refresh_state.current_job_id = job_id
                    worker = threading.Thread(target=_start_refresh_job, args=(job_id,), daemon=True)
                    worker.start()
                    if self._wants_json():
                        self._send_json(
                            {
                                "ok": True,
                                "job_id": job_id,
                                "message": f"{job['label']} 갱신을 시작했습니다.",
                            }
                        )
                        return
                    self._send_html(
                        _refresh_page(
                            _PageContext(
                                dashboard=None,
                                lookback_days=lookback_days,
                                start_date=start_date,
                                end_date=end_date,
                                message=f"{job['label']} 갱신을 시작했습니다.",
                            )
                        )
                    )
                    return
                self._send_html("<h1>Not Found</h1>", status=404)
            except Exception as exc: # Catch all exceptions and render an error page
                self._render_error(exc, lookback_days=lookback_days, start_date=start_date, end_date=end_date)

        # Stock Analysis Lab Handlers
        def _handle_stock_forecast_run(self, form: dict[str, str]) -> None:
            for checkbox in ["use_sample", "auto_save", "insecure_ssl"]:
                if checkbox not in form:
                    form[checkbox] = ""

            intent = form.get("intent", "run").strip().lower() or "run"
            insecure_ssl = form.get("insecure_ssl", "") == "on"
            ca_bundle_path = form.get("ca_bundle_path", "").strip() or None

            effective_ticker = self._resolve_ticker_and_update_common_note(form.get("ticker", ""), ca_bundle_path, insecure_ssl)

            self.__class__.stock_forecast_form = {
                "ticker": effective_ticker,
                "forecast_horizon": form.get("forecast_horizon", "10"),
                "history_years": form.get("history_years", "8"),
                "start_date": form.get("start_date", ""),
                "end_date": form.get("end_date", datetime.utcnow().strftime("%Y-%m-%d")),
                "output_dir": form.get("output_dir", "outputs/stock_forecast"),
                "prices_csv_path": form.get("prices_csv_path", ""),
                "use_sample": form.get("use_sample", ""),
                "auto_save": form.get("auto_save", ""),
                "insecure_ssl": form.get("insecure_ssl", ""),
                "ca_bundle_path": form.get("ca_bundle_path", ""),
            }
            self.__class__._sync_cross_page_tickers(effective_ticker)

            if intent == "resolve_ticker":
                self.__class__.stock_forecast_error = None
                self.send_response(303)
                self.send_header("Location", "/stock-forecast")
                self.end_headers()
                return

            if self.common_ticker_note_error and not effective_ticker:
                self.__class__.stock_forecast_ctx = None
                self.__class__.stock_forecast_error = self.common_ticker_note or "Provide ticker, or set Local Prices CSV Path Override."
                self.send_response(303)
                self.send_header("Location", "/stock-forecast")
                self.end_headers()
                return

            try:
                self.__class__.stock_forecast_ctx = stock_web_gui._run_once(self.stock_forecast_form)
                self.__class__.stock_forecast_error = None
            except Exception as exc:
                self.__class__.stock_forecast_ctx = None
                out_dir = Path(self.stock_forecast_form.get("output_dir", "outputs/stock_forecast"))
                hint = stock_web_gui.security_hint(exc, output_dir=out_dir)
                if isinstance(exc, ValueError):
                    self.__class__.stock_forecast_error = str(exc)
                else:
                    self.__class__.stock_forecast_error = f"{hint}\n\nRaw error: {exc}" if hint else traceback.format_exc()
            self.send_response(303)
            self.send_header("Location", "/stock-forecast")
            self.end_headers()

        def _handle_stock_financials_run(self, form: dict[str, str]) -> None:
            for checkbox in ["auto_save", "insecure_ssl"]:
                if checkbox not in form:
                    form[checkbox] = ""

            intent = form.get("intent", "run").strip().lower() or "run"
            insecure_ssl = form.get("insecure_ssl", "") == "on"
            ca_bundle_path = form.get("ca_bundle_path", "").strip() or None

            effective_ticker = self._resolve_ticker_and_update_common_note(form.get("ticker", ""), ca_bundle_path, insecure_ssl)

            self.__class__.stock_financials_form = {
                "ticker": effective_ticker,
                "statement_periods": form.get("statement_periods", "4"),
                "output_dir": form.get("output_dir", "outputs/stock_forecast_finance"),
                "auto_save": form.get("auto_save", ""),
                "insecure_ssl": form.get("insecure_ssl", ""),
                "ca_bundle_path": form.get("ca_bundle_path", ""),
                "fmp_api_key": form.get("fmp_api_key", ""),
            }
            self.__class__._sync_cross_page_tickers(effective_ticker)

            if intent == "resolve_ticker":
                self.__class__.stock_financials_error = None
                self.send_response(303)
                self.send_header("Location", "/stock-financials")
                self.end_headers()
                return

            if self.common_ticker_note_error and not effective_ticker:
                self.__class__.stock_financials_ctx = None
                self.__class__.stock_financials_error = self.common_ticker_note or "Provide ticker for financial statements page."
                self.send_response(303)
                self.send_header("Location", "/stock-financials")
                self.end_headers()
                return

            cache_key = self._financial_cache_key(self.stock_financials_form)
            cached = self.stock_financials_cache.get(cache_key)
            if cached is not None:
                self.__class__.stock_financials_ctx = cached
                self.__class__.stock_financials_error = None
            else:
                try:
                    fin_ctx = stock_web_gui._run_financial_once(self.stock_financials_form)
                    self.__class__.stock_financials_ctx = fin_ctx
                    self.__class__.stock_financials_cache[cache_key] = fin_ctx
                    self.__class__.stock_financials_error = None
                except Exception as exc:
                    self.__class__.stock_financials_ctx = None
                    out_dir = Path(self.stock_financials_form.get("output_dir", "outputs/stock_forecast_finance"))
                    hint = stock_web_gui.security_hint(exc, output_dir=out_dir)
                    if isinstance(exc, ValueError):
                        self.__class__.stock_financials_error = str(exc)
                    else:
                        self.__class__.stock_financials_error = (
                            f"{hint}\n\nRaw error: {exc}" if hint else traceback.format_exc()
                        )
            self.send_response(303)
            self.send_header("Location", "/stock-financials")
            self.end_headers()

        def _handle_stock_technical_run(self, form: dict[str, str]) -> None:
            for checkbox in ["use_sample", "auto_save"]:
                if checkbox not in form:
                    form[checkbox] = ""

            intent = form.get("intent", "run").strip().lower() or "run"
            effective_ticker = self._resolve_ticker_and_update_common_note(form.get("ticker", ""), None, False)
            action = stock_web_gui._normalize_technical_action(form.get("action", "all"))

            self.__class__.stock_technical_form = {
                "ticker": effective_ticker,
                "output_dir": form.get("output_dir", "outputs/technical_analysis"),
                "use_sample": form.get("use_sample", ""),
                "auto_save": form.get("auto_save", ""),
                "action": action,
            }
            self.__class__._sync_cross_page_tickers(effective_ticker)

            if intent == "resolve_ticker":
                self.__class__.stock_technical_error = None
                self.send_response(303)
                self.send_header("Location", "/stock-technical")
                self.end_headers()
                return

            if self.common_ticker_note_error and not effective_ticker:
                self.__class__.stock_technical_ctx = None
                self.__class__.stock_technical_error = self.common_ticker_note or "Provide ticker for technical analysis."
                self.send_response(303)
                self.send_header("Location", "/stock-technical")
                self.end_headers()
                return

            try:
                ta_ctx, ta_cache = stock_web_gui.ta_web_gui._run_analysis(
                    form=self.stock_technical_form,
                    action=action,
                    cache=self.stock_technical_cache,
                )
                self.__class__.stock_technical_ctx = ta_ctx
                self.__class__.stock_technical_cache = ta_cache
                self.__class__.stock_technical_error = None
            except Exception as exc:
                self.__class__.stock_technical_ctx = None
                out_dir = Path(self.stock_technical_form.get("output_dir", "outputs/technical_analysis"))
                hint = stock_web_gui.security_hint(exc, output_dir=out_dir)
                if isinstance(exc, ValueError):
                    self.__class__.stock_technical_error = str(exc)
                else:
                    self.__class__.stock_technical_error = f"{hint}\n\nRaw error: {exc}" if hint else traceback.format_exc()
            self.send_response(303)
            self.send_header("Location", "/stock-technical")
            self.end_headers()

        def _handle_stock_returns_run(self, form: dict[str, str]) -> None:
            intent = form.get("intent", "run").strip().lower() or "run"
            effective_ticker = self._resolve_ticker_and_update_common_note(form.get("ticker", ""), None, False)
            self.__class__.stock_returns_form = {"ticker": effective_ticker}
            self.__class__._sync_cross_page_tickers(effective_ticker)

            if intent == "resolve_ticker":
                self.__class__.stock_returns_error = None
                self.send_response(303)
                self.send_header("Location", "/stock-returns")
                self.end_headers()
                return

            if self.common_ticker_note_error and not effective_ticker:
                self.__class__.stock_returns_ctx = None
                self.__class__.stock_returns_error = self.common_ticker_note or "Provide an S&P 500 ticker for return analysis."
                self.send_response(303)
                self.send_header("Location", "/stock-returns")
                self.end_headers()
                return

            try:
                self.__class__.stock_returns_ctx = stock_web_gui._run_returns_once(self.stock_returns_form)
                self.__class__.stock_returns_error = None
            except Exception as exc:
                self.__class__.stock_returns_ctx = None
                if isinstance(exc, ValueError):
                    self.__class__.stock_returns_error = str(exc)
                else:
                    self.__class__.stock_returns_error = traceback.format_exc()
            self.send_response(303)
            self.send_header("Location", "/stock-returns")
            self.end_headers()

        def _handle_stock_risk_run(self, form: dict[str, str]) -> None:
            intent = form.get("intent", "run").strip().lower() or "run"
            effective_ticker = self._resolve_ticker_and_update_common_note(form.get("ticker", ""), None, False)
            self.__class__.stock_risk_form = {"ticker": effective_ticker}
            self.__class__._sync_cross_page_tickers(effective_ticker)

            if intent == "resolve_ticker":
                self.__class__.stock_risk_error = None
                self.send_response(303)
                self.send_header("Location", "/stock-risk")
                self.end_headers()
                return

            if self.common_ticker_note_error and not effective_ticker:
                self.__class__.stock_risk_ctx = None
                self.__class__.stock_risk_error = self.common_ticker_note or "Provide an S&P 500 ticker for risk analysis."
                self.send_response(303)
                self.send_header("Location", "/stock-risk")
                self.end_headers()
                return

            try:
                self.__class__.stock_risk_ctx = stock_web_gui._run_risk_once(self.stock_risk_form)
                self.__class__.stock_risk_error = None
            except Exception as exc:
                self.__class__.stock_risk_ctx = None
                if isinstance(exc, ValueError):
                    self.__class__.stock_risk_error = str(exc)
                else:
                    self.__class__.stock_risk_error = traceback.format_exc()
            self.send_response(303)
            self.send_header("Location", "/stock-risk")
            self.end_headers()

        def _handle_stock_factor_run(self, form: dict[str, str]) -> None:
            intent = form.get("intent", "run").strip().lower() or "run"
            effective_ticker = self._resolve_ticker_and_update_common_note(form.get("ticker", ""), None, False)
            self.__class__.stock_factor_form = {"ticker": effective_ticker}
            self.__class__._sync_cross_page_tickers(effective_ticker)

            if intent == "resolve_ticker":
                self.__class__.stock_factor_error = None
                self.send_response(303)
                self.send_header("Location", "/stock-factor")
                self.end_headers()
                return

            if self.common_ticker_note_error and not effective_ticker:
                self.__class__.stock_factor_ctx = None
                self.__class__.stock_factor_error = self.common_ticker_note or "Provide an S&P 500 ticker for factor and regime analysis."
                self.send_response(303)
                self.send_header("Location", "/stock-factor")
                self.end_headers()
                return

            try:
                self.__class__.stock_factor_ctx = stock_web_gui._run_factor_once(self.stock_factor_form)
                self.__class__.stock_factor_error = None
            except Exception as exc:
                self.__class__.stock_factor_ctx = None
                if isinstance(exc, ValueError):
                    self.__class__.stock_factor_error = str(exc)
                else:
                    self.__class__.stock_factor_error = traceback.format_exc()
            self.send_response(303)
            self.send_header("Location", "/stock-factor")
            self.end_headers()

        def _handle_stock_decision_run(self, form: dict[str, str]) -> None:
            intent = form.get("intent", "run").strip().lower() or "run"
            effective_ticker = self._resolve_ticker_and_update_common_note(form.get("ticker", ""), None, False)
            self.__class__.stock_decision_form = {"ticker": effective_ticker}
            self.__class__._sync_cross_page_tickers(effective_ticker)

            if intent == "resolve_ticker":
                self.__class__.stock_decision_error = None
                self.send_response(303)
                self.send_header("Location", "/stock-decision")
                self.end_headers()
                return

            if self.common_ticker_note_error and not effective_ticker:
                self.__class__.stock_decision_ctx = None
                self.__class__.stock_decision_error = self.common_ticker_note or "Provide an S&P 500 ticker for decision analysis."
                self.send_response(303)
                self.send_header("Location", "/stock-decision")
                self.end_headers()
                return

            try:
                # Ensure returns and risk contexts are available
                if self.stock_returns_ctx is None or self.stock_returns_ctx.ticker != effective_ticker:
                    self.__class__.stock_returns_form = {"ticker": effective_ticker}
                    self.__class__.stock_returns_ctx = stock_web_gui._run_returns_once(self.stock_returns_form)
                    self.__class__.stock_returns_error = None
                if self.stock_risk_ctx is None or self.stock_risk_ctx.ticker != effective_ticker:
                    self.__class__.stock_risk_form = {"ticker": effective_ticker}
                    self.__class__.stock_risk_ctx = stock_web_gui._run_risk_once(self.stock_risk_form)
                    self.__class__.stock_risk_error = None

                fin_ctx = self.stock_financials_ctx if self.stock_financials_ctx is not None and self.stock_financials_ctx.ticker.strip().upper() == effective_ticker else None
                self.__class__.stock_decision_ctx = stock_web_gui._run_decision_once(
                    self.stock_decision_form,
                    returns_ctx=self.stock_returns_ctx,
                    risk_ctx=self.stock_risk_ctx,
                    fin_ctx=fin_ctx,
                )
                self.__class__.stock_decision_error = None
            except Exception as exc:
                self.__class__.stock_decision_ctx = None
                if isinstance(exc, ValueError):
                    self.__class__.stock_decision_error = str(exc)
                else:
                    self.__class__.stock_decision_error = traceback.format_exc()
            self.send_response(303)
            self.send_header("Location", "/stock-decision")
            self.end_headers()

        def _handle_stock_wfv_run(self, form: dict[str, str]) -> None:
            for checkbox in ["use_sample", "auto_save", "insecure_ssl"]:
                if checkbox not in form:
                    form[checkbox] = ""

            intent = form.get("intent", "run").strip().lower() or "run"
            insecure_ssl = form.get("insecure_ssl", "") == "on"
            ca_bundle_path = form.get("ca_bundle_path", "").strip() or None

            effective_ticker = self._resolve_ticker_and_update_common_note(form.get("ticker", ""), ca_bundle_path, insecure_ssl)

            self.__class__.stock_wfv_form = {
                "ticker": effective_ticker,
                "forecast_horizon": form.get("forecast_horizon", "10"),
                "history_years": form.get("history_years", "8"),
                "start_date": form.get("start_date", ""),
                "end_date": form.get("end_date", datetime.utcnow().strftime("%Y-%m-%d")),
                "wf_min_train_rows": form.get("wf_min_train_rows", "252"),
                "wf_step_size": form.get("wf_step_size", "21"),
                "wf_max_splits": form.get("wf_max_splits", "4"),
                "output_dir": form.get("output_dir", "outputs/walk_forward_validation"),
                "prices_csv_path": form.get("prices_csv_path", ""),
                "use_sample": form.get("use_sample", ""),
                "auto_save": form.get("auto_save", ""),
                "insecure_ssl": form.get("insecure_ssl", ""),
                "ca_bundle_path": form.get("ca_bundle_path", ""),
            }
            self.__class__._sync_cross_page_tickers(effective_ticker)

            if intent == "resolve_ticker":
                self.__class__.stock_wfv_error = None
                self.send_response(303)
                self.send_header("Location", "/stock-wfv")
                self.end_headers()
                return

            if self.common_ticker_note_error and not effective_ticker and not form.get("prices_csv_path", "").strip() and form.get("use_sample", "") != "on":
                self.__class__.stock_wfv_ctx = None
                self.__class__.stock_wfv_error = self.common_ticker_note or "Provide ticker, or set Local Prices CSV Path Override."
                self.send_response(303)
                self.send_header("Location", "/stock-wfv")
                self.end_headers()
                return

            try:
                self.__class__.stock_wfv_ctx = stock_web_gui._run_walk_forward_validation_once(self.stock_wfv_form)
                self.__class__.stock_wfv_error = None
            except Exception as exc:
                self.__class__.stock_wfv_ctx = None
                out_dir = Path(self.stock_wfv_form.get("output_dir", "outputs/walk_forward_validation"))
                hint = stock_web_gui.security_hint(exc, output_dir=out_dir)
                if isinstance(exc, ValueError):
                    self.__class__.stock_wfv_error = str(exc)
                else:
                    self.__class__.stock_wfv_error = f"{hint}\n\nRaw error: {exc}" if hint else traceback.format_exc()
            self.send_response(303)
            self.send_header("Location", "/stock-wfv")
            self.end_headers()

        # News Lab Handlers
        def _handle_news_run(self, form: dict[str, str], render_page_func, page_key: str) -> None:
            intent = str(form.get("intent", "run")).strip().lower()
            effective_ticker = self._resolve_ticker_and_update_common_note(form.get("ticker", ""), None, False)

            page_form = dict(form)
            page_form["ticker"] = effective_ticker
            self.__class__._news_store_page_form(page_key, page_form)
            self.__class__._sync_cross_page_tickers(effective_ticker)

            if intent == "resolve_ticker":
                self.__class__._news_store_page_note(page_key, page_form, self.common_ticker_note, self.common_ticker_note_error)
                self.send_response(303)
                self.send_header("Location", f"/{page_key}")
                self.end_headers()
                return

            if self.common_ticker_note_error and not effective_ticker:
                self.__class__._news_store_page_error(page_key, page_form, self.common_ticker_note or "Provide ticker for news analysis.")
                self.send_response(303)
                self.send_header("Location", f"/{page_key}")
                self.end_headers()
                return

            try:
                dashboard = news_web_gui._build_dashboard_from_form(page_form, page_key)
                self.__class__._news_store_page_result(page_key, page_form, dashboard)
            except Exception as exc:
                self.__class__._news_store_page_error(
                    page_key,
                    page_form,
                    f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc(limit=2)}",
                )
            self.send_response(303)
            self.send_header("Location", f"/{page_key}")
            self.end_headers()

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    server = ThreadingHTTPServer((host, port), Handler)
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(f"http://{host}:{port}/")).start()
    print(f"Portfolio Lab web GUI running on http://{host}:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return server

if __name__ == "__main__":
    launch_web_gui(port=8515, open_browser=True)
