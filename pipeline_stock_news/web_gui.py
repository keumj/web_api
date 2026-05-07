from __future__ import annotations

import base64
import html
import io
import json
import os
import re
import sqlite3
import subprocess
import sys
import threading
import traceback
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import matplotlib
import numpy as np
import pandas as pd

from .analysis import (
    DASHBOARD_SECTION_DIVERGENCE,
    DASHBOARD_SECTION_EVENT_STUDY,
    DASHBOARD_SECTION_EXPECTATION_RESET,
    DASHBOARD_SECTION_OVERVIEW,
    DASHBOARD_SECTION_SECTOR_SPILLOVER,
    DASHBOARD_SECTION_TOPICS,
    DASHBOARD_SECTION_VOLATILITY_REGIME,
    DEFAULT_DIVERGENCE_TOP_N,
    DEFAULT_EVENT_HORIZON_DAYS,
    DEFAULT_EVENT_KEYWORDS,
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_TOPIC_COUNT,
    StockNewsDashboard,
    build_stock_news_dashboard,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass
class _PageContext:
    dashboard: StockNewsDashboard | None
    form: dict[str, str]
    error: str | None = None
    ticker_note: str | None = None
    ticker_note_error: bool = False


PAGE_TO_SECTIONS: dict[str, frozenset[str]] = {
    "overview": frozenset({DASHBOARD_SECTION_OVERVIEW}),
    "event": frozenset({DASHBOARD_SECTION_EVENT_STUDY}),
    "spillover": frozenset({DASHBOARD_SECTION_SECTOR_SPILLOVER}),
    "divergence": frozenset({DASHBOARD_SECTION_DIVERGENCE}),
    "expectation": frozenset({DASHBOARD_SECTION_EXPECTATION_RESET}),
    "volatility": frozenset({DASHBOARD_SECTION_VOLATILITY_REGIME}),
    "topics": frozenset({DASHBOARD_SECTION_TOPICS}),
}


def _default_form() -> dict[str, str]:
    return {
        "event_keywords": DEFAULT_EVENT_KEYWORDS,
        "ticker": "",
        "lookback_days": str(DEFAULT_LOOKBACK_DAYS),
        "horizon_days": str(DEFAULT_EVENT_HORIZON_DAYS),
        "divergence_top_n": str(DEFAULT_DIVERGENCE_TOP_N),
        "topic_count": str(DEFAULT_TOPIC_COUNT),
    }


def _nav(active: str, is_sub_page: bool = False) -> str:
    items = [
        ("overview", "/overview", "뉴스 개요"),
        ("event", "/event-study", "이벤트 스터디"),
        ("spillover", "/sector-spillover", "섹터 전이"),
        ("divergence", "/divergence", "뉴스-프라이스 다이버전스"),
        ("expectation", "/expectation-reset", "기대 리셋"),
        ("volatility", "/volatility-regime", "변동성 레짐"),
        ("topics", "/topic-modeling", "토픽 모델링"),
    ]
    links = []
    for page_id, href, label in items:
        css_parts = []
        if page_id == active:
            css_parts.append("active")
        css = " ".join(css_parts)
        links.append(f'<a class="{css}" href="{href}">{html.escape(label)}</a>')
    
    return '<div class="nav">' + "".join(links) + "</div>"


def _base_css(is_sub_page: bool = False) -> str:
    return """
    :root {
      --bg: #f3f5f7;
      --card: #ffffff;
      --line: #d4dde8;
      --text: #1f2937;
      --muted: #5f6b7a;
      --brand: #0f4c81;
      --ok-bg: #e8f7ee;
      --ok-line: #99d5af;
      --err-bg: #fff2f2;
      --err-line: #efadad;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); font-family: "Segoe UI", "Noto Sans KR", sans-serif; }
    .wrap { width: 100%; max-width: 1460px; margin: 0 auto; padding: 20px; }
    h1 { margin: 0 0 5px; font-size: 24px; }
    .page-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; margin-bottom: 2px; }
    .page-head h1 { margin: 0; color: var(--text); }
    .page-credit { color: var(--muted); font-size: 11px; white-space: nowrap; padding-top: 8px; }
    .sub { color: var(--muted); font-size: 14px; margin-bottom: 15px; line-height: 1.4; }
    .card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 14px; }
    .nav { display: flex; gap: 8px; margin-bottom: 20px; flex-wrap: wrap; }
    .nav a { text-decoration: none; color: var(--brand); border: 1px solid var(--line); background: #fff; border-radius: 999px; padding: 7px 12px; font-size: 13px; }
    .nav a.active { background: var(--brand); color: #fff; border-color: var(--brand); font-weight: bold; }
    .form-grid { display: grid; grid-template-columns: repeat(6, minmax(150px, 1fr)); gap: 10px 12px; }
    .form-grid label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 4px; }
    .form-grid input { width: 100%; padding: 8px; border: 1px solid var(--line); border-radius: 6px; }
    button { background: var(--brand); border: 0; color: #fff; padding: 10px 16px; border-radius: 8px; cursor: pointer; font-weight: 600; }
    .notice { margin-top: 10px; border-radius: 8px; padding: 10px; }
    .notice.ok { background: var(--ok-bg); border: 1px solid var(--ok-line); }
    .notice.err { background: var(--err-bg); border: 1px solid var(--err-line); }
    .metrics { margin-top: 12px; display: grid; grid-template-columns: repeat(4, minmax(170px, 1fr)); gap: 10px; }
    .metric { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 10px; }
    .metric span { display: block; font-size: 12px; color: var(--muted); }
    .metric strong { display: block; margin-top: 5px; font-size: 17px; }
    .charts { margin-top: 12px; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
    .table-wrap { width: 100%; overflow-x: auto; border-bottom: 1px solid var(--line); }
    .data-table { width: 100%; border-collapse: collapse; font-size: 13px; }
    .data-table th, .data-table td { border: 1px solid var(--line); padding: 8px; text-align: left; }
    @media (max-width: 980px) {
      .form-grid, .metrics { grid-template-columns: repeat(2, 1fr); }
      .charts { grid-template-columns: 1fr; }
    }
    """


def _page_head() -> str:
    # 공통 헤드라인만 출력
    return (
        '<div class="page-head">'
        '<h1>Stock News Lab | S&P 500</h1>'
        '<div class="page-credit">Keumj 제작</div>'
        "</div>"
    )


def _layout_page(
    *,
    active: str,
    subtitle: str,
    ctx: _PageContext,
    action: str,
    button_label: str,
    content_html: str,
    is_sub_page: bool = False,
) -> str:
    notices = ""
    if ctx.error:
        notices += f'<div class="notice err"><pre>{html.escape(ctx.error)}</pre></div>'
    
    head_content = ""
    if not is_sub_page:
        head_content = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Stock News Lab</title>
  <style>{_base_css()}</style>
</head>
<body>"""
    
    # 1. 공통 헤드라인 -> 2. 페이지 설명 -> 3. 메뉴바 순서
    # h2(페이지 제목)를 제거하여 중복을 없앰
    body_content = f"""
  <div class="wrap">
    {_page_head()} 
    
    <div class="sub">{html.escape(subtitle)}</div>
    
    {_nav(active)} 
    
    {_shared_form(ctx.form, action=action, button_label=button_label)}
    {notices}
    {_summary_metrics(ctx)}
    {content_html}
  </div>"""

    if is_sub_page:
        return body_content
    return head_content + body_content + "</body>\n</html>\n"


def _overview_page(ctx: _PageContext, is_sub_page: bool = False) -> str:
    # 타이틀 인자(title="뉴스 개요")를 _layout_page에 전달하지 않거나 무시함
    dashboard = ctx.dashboard
    if dashboard is None or DASHBOARD_SECTION_OVERVIEW not in dashboard.computed_sections:
        body = '<div class="card" style="margin-top:12px;"><p class="hint">분석 결과가 없습니다. 실행 버튼을 눌러주세요.</p></div>'
    else:
        # (생략: 기존 데이터 렌더링 로직...)
        body = '<div style="margin-top:12px;">분석 데이터가 여기에 표시됩니다.</div>'

    return _layout_page(
        active="overview",
        subtitle="분석 계산 없이 최근 뉴스 데이터 자체를 빠르게 훑는 개요 페이지",
        ctx=ctx,
        action="/run_overview",
        button_label="뉴스 개요 갱신",
        content_html=body,
        is_sub_page=is_sub_page,
    )

# ... (다른 페이지 함수들도 동일하게 _layout_page를 호출) ...
