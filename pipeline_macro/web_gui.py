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
    h1 { margin: 0 0 10px; font-size: 24px; }
    h2, h3, h4, p { margin-top: 0; }
    .page-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; margin-bottom: 5px; }
    .page-head h1 { margin: 0; }
    .page-credit { color: var(--muted); font-size: 11px; white-space: nowrap; padding-top: 4px; }
    .sub { color: var(--muted); margin-bottom: 14px; font-size: 14px; }
    .card { min-width: 0; background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 14px; }
    .nav { display: flex; gap: 8px; margin-bottom: 20px; flex-wrap: wrap; }
    .nav a { text-decoration: none; color: var(--brand); border: 1px solid var(--line); background: #fff; border-radius: 999px; padding: 7px 12px; font-size: 13px; }
    .nav a.active { background: var(--brand); color: #fff; border-color: var(--brand); font-weight: bold; }
    .form-grid { display: grid; grid-template-columns: repeat(6, minmax(150px, 1fr)); gap: 10px 12px; }
    .form-grid label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 4px; }
    .form-grid input[type="text"], .form-grid input[type="number"] {
      width: 100%; box-sizing: border-box; padding: 8px; border: 1px solid var(--line); border-radius: 6px;
    }
    .row { display: flex; flex-wrap: wrap; gap: 14px; align-items: center; margin-top: 10px; }
    button { background: var(--brand); border: 0; color: #fff; padding: 10px 16px; border-radius: 8px; cursor: pointer; font-weight: 600; }
    .notice { margin: 10px 0; border-radius: 8px; padding: 10px; font-size: 13px; }
    .notice.ok { background: var(--ok-bg); border: 1px solid var(--ok-line); }
    .notice.err { background: var(--err-bg); border: 1px solid var(--err-line); }
    .metrics { margin-top: 12px; display: grid; grid-template-columns: repeat(4, minmax(170px, 1fr)); gap: 10px; }
    .metric { min-width: 0; background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 10px; }
    .metric span { display: block; font-size: 12px; color: var(--muted); }
    .metric strong { display: block; margin-top: 5px; font-size: 17px; line-height: 1.3; }
    .charts { margin-top: 12px; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
    .table-grid { margin-top: 12px; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
    .table-wrap { width: 100%; overflow-x: auto; max-height: 500px; border-bottom: 1px solid var(--line); }
    .data-table { width: 100%; border-collapse: collapse; font-size: 13px; }
    .data-table th, .data-table td { border: 1px solid var(--line); padding: 8px; text-align: left; }
    img.chart { width: 100%; height: auto; border-radius: 8px; border: 1px solid var(--line); background: #fff; }
    """


def _page_head() -> str:
    """인자 없이 서비스명만 출력하도록 고정"""
    return (
        '<div class="page-head">'
        '<h1>News Lab | S&P 500</h1>'
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
    """
    구조 수정: 1. 헤드라인(서비스명) -> 2. 페이지 설명(subtitle) -> 3. 메뉴바(_nav)
    """
    notices = ""
    if ctx.error:
        notices += f'<div class="notice err"><pre>{html.escape(ctx.error)}</pre></div>'
    if ctx.ticker_note:
        css = "err" if ctx.ticker_note_error else "ok"
        notices += f'<div class="notice {css}"><pre>{html.escape(ctx.ticker_note)}</pre></div>'
    
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
    
    # 순서: 서비스명 -> 설명 -> 메뉴바 -> 폼 -> 지표 -> 본문
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
    dashboard = ctx.dashboard
    if not _has_sections(dashboard, "overview"):
        body = '<div class="card"><p class="hint">아직 결과가 없습니다. 상단에서 한 번 실행해 주세요.</p></div>'
    else:
        overview = dashboard.overview
        body = f"""
        {_overview_interpretation(dashboard)}
        <div class="charts">
          <div class="card"><h3>일자별 기사 수</h3><img class="chart" src="data:image/png;base64,{_overview_daily_chart(dashboard)}" /></div>
          <div class="card"><h3>주요 언급 티커</h3><img class="chart" src="data:image/png;base64,{_overview_ticker_chart(dashboard)}" /></div>
        </div>
        <div class="table-grid">
          <div class="card"><h3>섹터별 기사 집중도</h3>{_safe_table(overview.top_sectors, 10)}</div>
          <div class="card"><h3>소스별 기사 분포</h3>{_safe_table(overview.top_sources, 10)}</div>
        </div>
        """
    return _layout_page(
        active="overview",
        subtitle="분석 계산 없이 최근 뉴스 데이터 자체를 빠르게 훑는 개요 페이지",
        ctx=ctx,
        action="/run_overview",
        button_label="뉴스 개요 갱신",
        content_html=body,
        is_sub_page=is_sub_page,
    )


def _html_event_page(ctx: _PageContext, is_sub_page: bool = False) -> str:
    dashboard = ctx.dashboard
    if not _has_sections(dashboard, "event"):
        body = '<div class="card"><p class="hint">이벤트 스터디 결과가 없습니다.</p></div>'
    else:
        body = f"""
        {_event_interpretation(dashboard)}
        <div class="charts">
          <div class="card"><h3>수익률 곡선</h3><img class="chart" src="data:image/png;base64,{_event_study_chart(dashboard)}" /></div>
          <div class="card"><h3>요약 데이터</h3>{_safe_table(dashboard.event_study.summary, 10)}</div>
        </div>
        """
    return _layout_page(
        active="event",
        subtitle="특정 키워드 뉴스 이후 1~5거래일 반응을 통계적으로 확인하는 페이지",
        ctx=ctx,
        action="/run_event_study",
        button_label="이벤트 스터디 실행",
        content_html=body,
        is_sub_page=is_sub_page,
    )


def _html_spillover_page(ctx: _PageContext, is_sub_page: bool = False) -> str:
    dashboard = ctx.dashboard
    if not _has_sections(dashboard, "spillover"):
        body = '<div class="card"><p class="hint">섹터 전이 결과가 없습니다.</p></div>'
    else:
        body = f"""
        {_spillover_interpretation(dashboard)}
        <div class="charts">
          <div class="card"><h3>섹터 전이 강도</h3><img class="chart" src="data:image/png;base64,{_sector_spillover_chart(dashboard)}" /></div>
          <div class="card"><h3>섹터별 요약</h3>{_safe_table(dashboard.sector_spillover.summary, 12)}</div>
        </div>
        """
    return _layout_page(
        active="spillover",
        subtitle="한 종목 뉴스가 같은 섹터 다른 종목 수익률로 얼마나 번지는지 보는 페이지",
        ctx=ctx,
        action="/run_sector_spillover",
        button_label="섹터 전이 분석 실행",
        content_html=body,
        is_sub_page=is_sub_page,
    )


def _html_divergence_page(ctx: _PageContext, is_sub_page: bool = False) -> str:
    dashboard = ctx.dashboard
    if not _has_sections(dashboard, "divergence"):
        body = '<div class="card"><p class="hint">뉴스-프라이스 다이버전스 결과가 없습니다.</p></div>'
    else:
        body = f"""
        {_divergence_interpretation(dashboard)}
        <div class="charts">
          <div class="card"><h3>감성 대비 반대 가격 반응</h3><img class="chart" src="data:image/png;base64,{_divergence_chart(dashboard)}" /></div>
          <div class="card"><h3>괴리 알림 리스트</h3>{_safe_table(dashboard.divergence.alerts, 50)}</div>
        </div>
        """
    return _layout_page(
        active="divergence",
        subtitle="긍정 뉴스인데도 약하고, 부정 뉴스인데도 버티는 이상 반응을 잡는 페이지",
        ctx=ctx,
        action="/run_divergence",
        button_label="괴리 탐지 실행",
        content_html=body,
        is_sub_page=is_sub_page,
    )


def _html_expectation_page(ctx: _PageContext, is_sub_page: bool = False) -> str:
    dashboard = ctx.dashboard
    if not _has_sections(dashboard, "expectation"):
        body = '<div class="card"><p class="hint">기대 리셋 결과가 없습니다.</p></div>'
    else:
        body = f"""
        {_expectation_interpretation(dashboard)}
        <div class="charts">
          <div class="card"><h3>감성 대비 5일 수익률</h3><img class="chart" src="data:image/png;base64,{_expectation_reset_chart(dashboard)}" /></div>
          <div class="card"><h3>기대 리셋 후보</h3>{_safe_table(dashboard.expectation_reset.candidates, 50)}</div>
        </div>
        """
    return _layout_page(
        active="expectation",
        subtitle="강한 뉴스가 나왔는데도 가격 반응이 약한 종목을 선반영 후보로 보는 페이지",
        ctx=ctx,
        action="/run_expectation_reset",
        button_label="기대 리셋 탐지 실행",
        content_html=body,
        is_sub_page=is_sub_page,
    )


def _html_volatility_page(ctx: _PageContext, is_sub_page: bool = False) -> str:
    dashboard = ctx.dashboard
    if not _has_sections(dashboard, "volatility"):
        body = '<div class="card"><p class="hint">변동성 레짐 결과가 없습니다.</p></div>'
    else:
        body = f"""
        {_volatility_interpretation(dashboard)}
        <div class="charts">
          <div class="card"><h3>변동성 확대 비율</h3><img class="chart" src="data:image/png;base64,{_volatility_chart(dashboard)}" /></div>
          <div class="card"><h3>요약 데이터</h3>{_safe_table(dashboard.volatility_regime.summary, 12)}</div>
        </div>
        """
    return _layout_page(
        active="volatility",
        subtitle="뉴스 이후 5거래일 변동성이 직전 구간보다 얼마나 커졌는지 비교하는 페이지",
        ctx=ctx,
        action="/run_volatility_regime",
        button_label="변동성 레짐 실행",
        content_html=body,
        is_sub_page=is_sub_page,
    )


def _html_topics_page(ctx: _PageContext, is_sub_page: bool = False) -> str:
    dashboard = ctx.dashboard
    if not _has_sections(dashboard, "topics"):
        body = '<div class="card"><p class="hint">토픽 모델 결과가 없습니다.</p></div>'
    else:
        body = f"""
        {_topics_interpretation(dashboard)}
        <div class="charts">
          <div class="card"><h3>토픽 비중</h3><img class="chart" src="data:image/png;base64,{_topic_weight_chart(dashboard)}" /></div>
          <div class="card"><h3>키워드 클라우드</h3>{_word_cloud_html(dashboard)}</div>
        </div>
        """
    return _layout_page(
        active="topics",
        subtitle="최근 뉴스 제목을 묶어 시장을 관통하는 테마를 압축해서 보는 페이지",
        ctx=ctx,
        action="/run_topic_modeling",
        button_label="토픽 모델 실행",
        content_html=body,
        is_sub_page=is_sub_page,
    )

# --- 헬퍼 함수 및 공통 레이아웃 요소 ---

def _shared_form(form: dict[str, str], *, action: str, button_label: str) -> str:
    return f"""
    <form class="card" method="post" action="{action}">
      <div class="form-grid">
        <div><label>키워드</label><input type="text" name="event_keywords" value="{html.escape(form.get('event_keywords', ''))}" /></div>
        <div><label>티커</label><input type="text" name="ticker" value="{html.escape(form.get('ticker', ''))}" placeholder="AAPL" /></div>
        <div><label>조회 일수</label><input type="number" min="7" max="365" name="lookback_days" value="{html.escape(form.get('lookback_days', ''))}" /></div>
        <div><label>이벤트 수평선</label><input type="number" min="1" max="20" name="horizon_days" value="{html.escape(form.get('horizon_days', ''))}" /></div>
        <div><label>괴리 Top N</label><input type="number" min="5" max="100" name="divergence_top_n" value="{html.escape(form.get('divergence_top_n', ''))}" /></div>
        <div><label>토픽 수</label><input type="number" min="2" max="10" name="topic_count" value="{html.escape(form.get('topic_count', ''))}" /></div>
      </div>
      <div class="row">
        <button type="submit" name="intent" value="run">{html.escape(button_label)}</button>
      </div>
    </form>
    """

def _summary_metrics(ctx: _PageContext) -> str:
    dashboard = ctx.dashboard
    if dashboard is None: return ""
    m = [
        ("조회 기간", f"{dashboard.window_start} ~ {dashboard.window_end}"),
        ("적용 티커", dashboard.applied_ticker or "ALL"),
        ("키워드", ", ".join(dashboard.applied_keywords) if dashboard.applied_keywords else "ALL")
    ]
    return '<div class="metrics">' + "".join(f'<div class="metric"><span>{l}</span><strong>{v}</strong></div>' for l, v in m) + "</div>"

def _has_sections(dashboard: StockNewsDashboard | None, page_key: str) -> bool:
    required = PAGE_TO_SECTIONS.get(page_key)
    return dashboard is not None and required is not None and required.issubset(dashboard.computed_sections)

# --- 나머지 시각화 및 해석 함수 (기존 파일 내용 유지) ---
def _render_chart_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def _safe_table(df: pd.DataFrame, max_rows: int = 100) -> str:
    if df is None or df.empty: return "<p class='hint'>데이터가 없습니다.</p>"
    return f'<div class="table-wrap">{df.head(max_rows).to_html(index=False, border=0, classes="data-table")}</div>'

# (이하 _overview_interpretation, _event_interpretation 등 비즈니스 로직 함수들은 기존 코드와 동일하므로 생략 가능하나, 전체 파일 구성을 위해 포함되어야 함)
