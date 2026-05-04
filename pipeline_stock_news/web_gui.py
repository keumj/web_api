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
        if page_id == "refresh":
            css_parts.append("refresh")
        if page_id == active:
            css_parts.append("active")
        css = " ".join(css_parts)
        links.append(f'<a class="{css}" href="{href}">{html.escape(label)}</a>')
    links.append('<a class="" href="/macro/overview">거시분석</a>')
    return (
        '<div class="nav">' + "".join(links) + "</div>"
        + """
        <script>
          (function () {
            if (window.__keumjStockNewsSyncInstalled) {
              return;
            }
            window.__keumjStockNewsSyncInstalled = true;
            let lastCommandId = null;

            async function pollExternalCommand() {
              try {
                const res = await fetch("/external_command_state", { cache: "no-store" });
                if (res.ok) {
                  const data = await res.json();
                  const commandId = Number(data.command_id || 0);
                  if (lastCommandId === null) {
                    lastCommandId = commandId;
                  } else if (commandId > lastCommandId && data.navigate_url) {
                    lastCommandId = commandId;
                    const nextUrl = String(data.navigate_url);
                    const currentUrl = window.location.pathname + window.location.search;
                    if (currentUrl !== nextUrl) {
                      window.location.href = nextUrl;
                      return;
                    }
                  } else {
                    lastCommandId = commandId;
                  }
                }
              } catch (err) {
              }
              window.setTimeout(pollExternalCommand, 1200);
            }

            window.setTimeout(pollExternalCommand, 1200);
          })();
        </script>
        """
    )


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
    .page-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; margin-bottom: 10px; }
    .page-head h1 { margin: 0; }
    .page-credit { color: var(--muted); font-size: 11px; white-space: nowrap; padding-top: 4px; }
    .sub { color: var(--muted); margin-bottom: 14px; }
    .card { min-width: 0; background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 14px; }
    .nav { display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap; }
    .nav a { text-decoration: none; color: var(--brand); border: 1px solid var(--line); background: #fff; border-radius: 999px; padding: 7px 12px; font-size: 13px; }
    .nav a.active { background: var(--brand); color: #fff; border-color: var(--brand); }
    .nav a.refresh { color: #111; border-color: #111; }
    .nav a.refresh.active { background: #111; color: #fff; border-color: #111; }
    .form-grid { display: grid; grid-template-columns: repeat(6, minmax(150px, 1fr)); gap: 10px 12px; }
    .form-grid label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 4px; }
    .form-grid input[type="text"], .form-grid input[type="number"] {
      width: 100%; box-sizing: border-box; padding: 8px; border: 1px solid var(--line); border-radius: 6px;
    }
    .row { display: flex; flex-wrap: wrap; gap: 14px; align-items: center; margin-top: 10px; }
    button { background: var(--brand); border: 0; color: #fff; padding: 10px 16px; border-radius: 8px; cursor: pointer; font-weight: 600; }
    .notice { margin-top: 10px; border-radius: 8px; padding: 10px; }
    .notice.ok { background: var(--ok-bg); border: 1px solid var(--ok-line); }
    .notice.err { background: var(--err-bg); border: 1px solid var(--err-line); }
    .metrics { margin-top: 12px; display: grid; grid-template-columns: repeat(4, minmax(170px, 1fr)); gap: 10px; }
    .metric { min-width: 0; background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 10px; }
    .metric span { display: block; font-size: 12px; color: var(--muted); }
    .metric strong { display: block; margin-top: 5px; font-size: 17px; line-height: 1.3; }
    .charts { margin-top: 12px; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
    .tables { margin-top: 12px; display: grid; grid-template-columns: 1fr; gap: 10px; }
    .table-grid { margin-top: 12px; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
    .table-wrap { width: 100%; max-width: 100%; min-width: 0; overflow-x: auto; max-height: 500px; overflow-y: auto; border-bottom: 1px solid var(--line); -webkit-overflow-scrolling: touch; }
    .data-table { width: 100%; min-width: 100%; border-collapse: collapse; font-size: 13px; line-height: 1.45; }
    .data-table th, .data-table td { border: 1px solid var(--line); padding: 8px; text-align: left; vertical-align: top; white-space: normal; overflow-wrap: anywhere; word-break: normal; }
    .stacked-table-group { display: grid; gap: 12px; }
    .stacked-table-block h4 { margin: 4px 0 8px; font-size: 13px; color: var(--muted); }
    .muted { color: var(--muted); }
    .hint { color: var(--muted); font-size: 13px; line-height: 1.45; }
    .analysis-text { display: grid; gap: 12px; }
    .analysis-text p { margin: 0; line-height: 1.6; }
    .analysis-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
    .memo-grid { display: grid; gap: 10px; }
    .memo-block { border: 1px solid var(--line); border-radius: 10px; padding: 12px; background: #fbfcfd; }
    .memo-block strong { display: block; margin-bottom: 6px; font-size: 13px; }
    .signal-badge { display: inline-flex; align-items: center; gap: 6px; padding: 6px 10px; border-radius: 999px; font-size: 12px; font-weight: 700; border: 1px solid transparent; }
    .signal-badge.bullish { background: #e8f7ee; color: #146c2e; border-color: #99d5af; }
    .signal-badge.neutral { background: #eef4f9; color: #31556f; border-color: #c9d7e5; }
    .signal-badge.caution { background: #fff2f2; color: #a12626; border-color: #efadad; }
    .word-cloud { display: flex; flex-wrap: wrap; gap: 10px 14px; align-items: baseline; line-height: 1.25; min-height: 96px; }
    .word-cloud span { color: var(--brand); }
    .chip-list { display: grid; gap: 10px; }
    .chip { border: 1px solid var(--line); border-radius: 10px; background: #fff; padding: 12px; }
    .chip strong { display: block; margin-bottom: 4px; }
    .latest-inline {
      margin-top: 8px; padding: 8px 10px; border-radius: 8px;
      background: #eef4fb; border: 1px solid #c7d9ee; color: #24425f; font-size: 12px; line-height: 1.45;
    }
    img.chart { width: 100%; height: auto; border-radius: 8px; border: 1px solid var(--line); background: #fff; }
    pre { margin: 0; white-space: pre-wrap; word-break: break-word; }
    @media (max-width: 980px) {
      .form-grid { grid-template-columns: repeat(2, minmax(160px, 1fr)); }
      .metrics { grid-template-columns: repeat(2, minmax(160px, 1fr)); }
      .charts { grid-template-columns: 1fr; }
      .table-grid { grid-template-columns: 1fr; }
    }
    """


def _page_head(title: str, is_sub_page: bool = False) -> str:
    return (
        '<div class="page-head">'
        f"<h1>{html.escape(title)}</h1>"
        '<div class="page-credit">Keumj 제작</div>'
        "</div>"
    )


def _render_chart_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _safe_table(df: pd.DataFrame, max_rows: int = 100) -> str:
    if df is None or df.empty:
        return "<p class='hint'>데이터가 없습니다.</p>"
    show = df.head(max_rows).copy()
    return f'<div class="table-wrap">{show.to_html(index=False, border=0, classes="data-table")}</div>'


def _format_metric(value: object, digits: int = 2) -> str:
    if value is None:
        return "-"
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(numeric):
        return "-"
    return f"{numeric:,.{digits}f}"


def _format_pct(value: object, digits: int = 2, *, signed: bool = False) -> str:
    try:
        numeric = float(value)
    except Exception:
        return "-"
    if not np.isfinite(numeric):
        return "-"
    if signed:
        return f"{numeric:+,.{digits}f}%"
    return f"{numeric:,.{digits}f}%"


def _safe_str(value: object, fallback: str = "-") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _analysis_card(
    title: str,
    sections: list[tuple[str, str]],
    *,
    signal_label: str = "중립",
    signal_tone: str = "neutral",
) -> str:
    if not sections:
        sections = [("핵심 관찰", "해석할 데이터가 아직 충분하지 않습니다.")]
    badge_html = f'<span class="signal-badge {html.escape(signal_tone)}">{html.escape(signal_label)}</span>'
    body = "".join(
        f'<div class="memo-block"><strong>{html.escape(label)}</strong><p>{html.escape(text)}</p></div>'
        for label, text in sections
    )
    return (
        f'<div class="card"><div class="analysis-head"><h3>{html.escape(title)}</h3>{badge_html}</div>'
        f'<div class="analysis-text"><div class="memo-grid">{body}</div></div></div>'
    )


def _overview_interpretation(dashboard: StockNewsDashboard) -> str:
    overview = dashboard.overview
    if overview.article_count == 0:
        return _analysis_card("결과 해석", [("핵심 관찰", "현재 조건에서 집계된 뉴스가 없어 개요를 만들 수 없습니다.")], signal_label="중립", signal_tone="neutral")
    observation = f"현재 조건에서는 뉴스 {overview.article_count:,d}건이 잡혔고, {overview.unique_ticker_count:,d}개 티커와 {overview.unique_source_count:,d}개 소스를 커버합니다. 최신 기사는 {_safe_str(overview.latest_publish_at)} 시각 기준입니다."
    opportunity = "이 페이지는 계산형 신호를 보기 전, 어디에 뉴스가 집중되는지 빠르게 잡아내는 출발점으로 읽으면 좋습니다."
    caution = "단순 기사 수는 방향성을 뜻하지 않으므로, 언급량이 높다고 바로 강세 신호로 해석하기보다 이후 계산형 페이지에서 반응을 확인하는 편이 안전합니다."
    if not overview.top_tickers.empty:
        top = overview.top_tickers.iloc[0]
        opportunity = f"가장 많이 언급된 티커는 {_safe_str(top.get('ticker'))}이며 기사 {_safe_str(top.get('article_count'))}건입니다. 뉴스 편중이 큰 종목은 이후 이벤트·괴리 페이지에서 우선 점검할 후보로 볼 만합니다."
    if not overview.top_sectors.empty and not overview.top_sources.empty:
        top_sector = overview.top_sectors.iloc[0]
        top_source = overview.top_sources.iloc[0]
        observation = (
            f"{observation} 섹터 기준으로는 {_safe_str(top_sector.get('sector'))}가 가장 많이 언급됐고, "
            f"소스 기준으로는 {_safe_str(top_source.get('source'))} 비중이 가장 높습니다."
        )
    if not overview.daily_counts.empty:
        counts = pd.to_numeric(overview.daily_counts["article_count"], errors="coerce").fillna(0)
        recent = int(counts.head(7).sum())
        prior = int(counts.iloc[7:14].sum()) if len(counts) > 7 else 0
        if prior > 0:
            direction = "증가" if recent > prior else "감소" if recent < prior else "유지"
            caution = f"최근 7일 기사 수는 직전 7일 대비 {direction} 흐름입니다. 최근 7일 {recent:,d}건, 직전 7일 {prior:,d}건으로 뉴스 유입 강도는 읽을 수 있지만, 방향성은 별도 검증이 필요합니다."
        else:
            caution = f"집계 기간이 짧거나 최근 구간 중심으로 데이터가 몰려 있어 최근 7일 기사 {recent:,d}건 중심 해석이 적절합니다. 기간이 짧을수록 일시적 뉴스 급증에 흔들릴 수 있습니다."
    return _analysis_card("결과 해석", [("핵심 관찰", observation), ("기회 요인", opportunity), ("주의 요인", caution)], signal_label="중립", signal_tone="neutral")


def _event_interpretation(dashboard: StockNewsDashboard) -> str:
    result = dashboard.event_study
    if result.summary.empty:
        return _analysis_card("결과 해석", [("핵심 관찰", "매치된 이벤트 뉴스가 부족해 통계적으로 읽을 수 있는 패턴이 아직 없습니다."), ("기회 요인", "키워드 범위를 넓히거나 조회 기간을 늘리면 샘플을 확보할 수 있습니다."), ("주의 요인", "표본이 적을 때는 평균 수익률이 쉽게 왜곡될 수 있습니다.")], signal_label="중립", signal_tone="neutral")
    summary = result.summary.sort_values("day").reset_index(drop=True)
    last_row = summary.iloc[-1]
    strongest = summary.iloc[summary["mean_return_pct"].abs().idxmax()]
    observation = f"이번 실행에서는 이벤트 뉴스 {result.article_count:,d}건이 매치됐고, {result.matched_ticker_count:,d}개 티커에서 후속 반응을 추적했습니다. 최종 Day {int(last_row['day'])} 기준 평균 수익률은 {_format_pct(last_row['mean_return_pct'], signed=True)}, 중앙값은 {_format_pct(last_row['median_return_pct'], signed=True)}, 상승 비율은 {_format_pct(last_row['positive_ratio_pct'])}입니다."
    opportunity = f"가장 강한 평균 반응은 Day {int(strongest['day'])}에서 {_format_pct(strongest['mean_return_pct'], signed=True)}로 나타났습니다. 특정 시차에서 반응이 집중된다면 해당 기간을 중심으로 후속 모니터링 포인트를 잡을 수 있습니다."
    caution = f"강한 반응 구간의 t-stat은 {_format_metric(strongest.get('t_stat'))}입니다. 절대값이 낮으면 방향성은 보여도 신뢰도는 약할 수 있으니 평균 수치만으로 단정하지 않는 편이 좋습니다."
    signal_label = "강세" if float(last_row["mean_return_pct"]) > 0.5 and float(last_row["positive_ratio_pct"]) >= 55.0 else "경계" if float(last_row["mean_return_pct"]) < -0.5 and float(last_row["positive_ratio_pct"]) <= 45.0 else "중립"
    signal_tone = "bullish" if signal_label == "강세" else "caution" if signal_label == "경계" else "neutral"
    return _analysis_card("결과 해석", [("핵심 관찰", observation), ("기회 요인", opportunity), ("주의 요인", caution)], signal_label=signal_label, signal_tone=signal_tone)


def _spillover_interpretation(dashboard: StockNewsDashboard) -> str:
    result = dashboard.sector_spillover
    if result.summary.empty:
        return _analysis_card("결과 해석", [("핵심 관찰", "같은 섹터 peer로 번지는 수익률 패턴이 확인되지 않았습니다."), ("기회 요인", "섹터 구성이 풍부한 티커나 더 넓은 기간으로 다시 보면 전이 효과가 드러날 수 있습니다."), ("주의 요인", "peer 수가 적거나 이벤트 수가 부족하면 전이 결과는 쉽게 비어 있을 수 있습니다.")], signal_label="중립", signal_tone="neutral")
    summary = result.summary.reset_index(drop=True)
    top = summary.iloc[0]
    col = "peer_day_5_return_pct" if "peer_day_5_return_pct" in summary.columns else summary.columns[-1]
    positive_count = int((pd.to_numeric(summary[col], errors="coerce") > 0).sum())
    observation = f"가장 강한 섹터 전이는 {_safe_str(top.get('sector'))}에서 관찰됐고, 최종 peer 평균 수익률은 {_format_pct(top.get(col), signed=True)}입니다. 평균 peer 수는 {_format_metric(top.get('avg_peer_count'))}개, 이벤트 수는 {_safe_str(top.get('event_count'))}건입니다."
    opportunity = f"상위 섹터 {len(summary.index):,d}개 중 {positive_count:,d}개가 플러스 방향이었습니다. 개별 뉴스가 업종 공통 기대를 움직이는 구간이라면 섹터 단위 확산을 추적하는 단서가 됩니다."
    caution = "전이 결과는 개별 종목 뉴스가 업종 전체에 번졌는지만 보여주며, 실제 매매에는 업종 내 편차와 source ticker 주도의 일시 효과를 함께 걸러봐야 합니다."
    top_value = float(pd.to_numeric(pd.Series([top.get(col)]), errors="coerce").fillna(0).iloc[0])
    signal_label = "강세" if top_value > 0.5 and positive_count >= max(1, len(summary.index) // 2) else "경계" if top_value < -0.5 else "중립"
    signal_tone = "bullish" if signal_label == "강세" else "caution" if signal_label == "경계" else "neutral"
    return _analysis_card("결과 해석", [("핵심 관찰", observation), ("기회 요인", opportunity), ("주의 요인", caution)], signal_label=signal_label, signal_tone=signal_tone)


def _divergence_interpretation(dashboard: StockNewsDashboard) -> str:
    alerts = dashboard.divergence.alerts
    if alerts.empty:
        return _analysis_card("결과 해석", [("핵심 관찰", "강한 뉴스 감성과 실제 가격 반응이 크게 어긋난 사례가 이번 조건에서는 잡히지 않았습니다."), ("기회 요인", "시장 반응이 비교적 정직한 구간일 수 있어 이벤트 스터디 쪽 해석이 더 유효할 수 있습니다."), ("주의 요인", "괴리가 없다는 뜻이지 기회가 없다는 뜻은 아니며, 단기 반응만으로 시장 효율성을 단정하면 안 됩니다.")], signal_label="중립", signal_tone="neutral")
    top = alerts.iloc[0]
    positive_news_fade = int(((pd.to_numeric(alerts["effective_sentiment"], errors="coerce") > 0) & (pd.to_numeric(alerts["divergence_return_pct"], errors="coerce") < 0)).sum())
    negative_news_absorb = int(((pd.to_numeric(alerts["effective_sentiment"], errors="coerce") < 0) & (pd.to_numeric(alerts["divergence_return_pct"], errors="coerce") > 0)).sum())
    observation = f"가장 강한 괴리 사례는 {_safe_str(top.get('ticker'))}이며, 감성 점수 {_format_metric(top.get('effective_sentiment'))} 대비 실제 가격 반응은 {_format_pct(top.get('divergence_return_pct'), signed=True)}로 반대로 움직였습니다. 가장 큰 반전은 Day {_safe_str(top.get('divergence_horizon_days'))}에서 나타났습니다."
    opportunity = f"괴리 상위 {len(alerts.index):,d}건 중 긍정 뉴스 후 하락은 {positive_news_fade:,d}건, 부정 뉴스 후 방어는 {negative_news_absorb:,d}건입니다. 과열 기대의 되돌림인지, 악재 소화인지 구분하면 후행 재평가 후보를 좁히는 데 도움이 됩니다."
    caution = "괴리는 강한 제목과 실제 가격이 엇갈렸다는 뜻일 뿐, 즉시 반대로 베팅하라는 신호는 아닙니다. 펀더멘털 변화나 이미 선반영된 포지션 해소도 함께 봐야 합니다."
    top_score = float(pd.to_numeric(pd.Series([top.get("divergence_score")]), errors="coerce").fillna(0).iloc[0])
    signal_label = "경계" if top_score >= 4.0 else "중립"
    signal_tone = "caution" if signal_label == "경계" else "neutral"
    return _analysis_card("결과 해석", [("핵심 관찰", observation), ("기회 요인", opportunity), ("주의 요인", caution)], signal_label=signal_label, signal_tone=signal_tone)


def _expectation_interpretation(dashboard: StockNewsDashboard) -> str:
    candidates = dashboard.expectation_reset.candidates
    if candidates.empty:
        return _analysis_card("결과 해석", [("핵심 관찰", "강한 뉴스 대비 반응이 약한 리셋 후보가 잡히지 않았습니다."), ("기회 요인", "이번 구간은 뉴스 기대가 가격에 비교적 자연스럽게 반영됐을 가능성이 큽니다."), ("주의 요인", "리셋 후보 부재가 곧 가격 효율성을 뜻하지는 않으므로 기간과 임계값을 바꿔 다시 확인할 필요가 있습니다.")], signal_label="중립", signal_tone="neutral")
    top = candidates.iloc[0]
    reset_counts = candidates["reset_type"].astype(str).value_counts()
    dominant_type = reset_counts.index[0] if not reset_counts.empty else "-"
    observation = f"가장 강한 리셋 후보는 {_safe_str(top.get('ticker'))}이며 유형은 {_safe_str(top.get('reset_type'))}, 점수는 {_format_metric(top.get('reset_score'))}입니다. 감성은 {_format_metric(top.get('effective_sentiment'))}였지만 5일 수익률은 {_format_pct(top.get('day_5_return_pct'), signed=True)}에 그쳤습니다."
    opportunity = f"후보 {len(candidates.index):,d}건 중 가장 많이 나온 패턴은 {_safe_str(dominant_type)}입니다. 강한 뉴스에도 후속 반응이 둔하면, 기대가 이미 가격에 반영된 종목을 골라내는 데 유용합니다."
    caution = "이 결과는 저평가 탐지라기보다 기대 소진 여부를 보는 도구입니다. headline이 좋아 보여도 추가 상승 여력이 작을 수 있다는 점을 먼저 의심하는 편이 좋습니다."
    top_score = float(pd.to_numeric(pd.Series([top.get("reset_score")]), errors="coerce").fillna(0).iloc[0])
    signal_label = "경계" if top_score >= 2.0 else "중립"
    signal_tone = "caution" if signal_label == "경계" else "neutral"
    return _analysis_card("결과 해석", [("핵심 관찰", observation), ("기회 요인", opportunity), ("주의 요인", caution)], signal_label=signal_label, signal_tone=signal_tone)


def _volatility_interpretation(dashboard: StockNewsDashboard) -> str:
    result = dashboard.volatility_regime
    if result.summary.empty:
        return _analysis_card("결과 해석", [("핵심 관찰", "뉴스 이후 변동성이 뚜렷하게 재구성된 사례가 부족합니다."), ("기회 요인", "거래일이 더 쌓이면 변동성 레짐 변화가 더 분명해질 수 있습니다."), ("주의 요인", "기준 기간과 이후 기간이 짧으면 변동성 비율은 쉽게 흔들릴 수 있습니다.")], signal_label="중립", signal_tone="neutral")
    summary = result.summary.reset_index(drop=True)
    top = summary.iloc[0]
    expansion_count = int((pd.to_numeric(summary["avg_volatility_ratio"], errors="coerce") > 1.0).sum())
    observation = f"가장 강한 변동성 확대 종목은 {_safe_str(top.get('ticker'))}이며 평균 변동성 비율은 {_format_metric(top.get('avg_volatility_ratio'))}배입니다. 기준 변동성은 {_format_pct(top.get('avg_baseline_vol_pct'))}, 이후 변동성은 {_format_pct(top.get('avg_post_vol_pct'))}였습니다."
    opportunity = f"상위 요약 {len(summary.index):,d}개 중 {expansion_count:,d}개가 1배를 넘었습니다. 뉴스 이후 가격 분산이 커진 종목은 방향성보다 트레이딩 레인지 확대 관점에서 볼 만합니다."
    caution = "변동성 확대는 기회이면서 동시에 리스크입니다. 수익률 신호가 아니라 포지션 관리 난도 신호에 가깝기 때문에 손절·보유 기간 가정을 같이 조정해야 합니다."
    top_ratio = float(pd.to_numeric(pd.Series([top.get("avg_volatility_ratio")]), errors="coerce").fillna(0).iloc[0])
    signal_label = "경계" if top_ratio > 1.25 else "중립"
    signal_tone = "caution" if signal_label == "경계" else "neutral"
    return _analysis_card("결과 해석", [("핵심 관찰", observation), ("기회 요인", opportunity), ("주의 요인", caution)], signal_label=signal_label, signal_tone=signal_tone)


def _topics_interpretation(dashboard: StockNewsDashboard) -> str:
    topics = dashboard.topics.topics
    if topics.empty:
        return _analysis_card("결과 해석", [("핵심 관찰", "제목을 묶을 만큼 반복 출현한 단어가 아직 부족해 토픽 클러스터를 만들지 못했습니다."), ("기회 요인", "기간을 늘리거나 키워드 제한을 풀면 시장 서사가 더 잘 드러날 수 있습니다."), ("주의 요인", "짧은 기간의 토픽은 일시적 이슈에 크게 좌우될 수 있습니다.")], signal_label="중립", signal_tone="neutral")
    top = topics.iloc[0]
    cloud = dashboard.topics.word_cloud
    lead_terms = ", ".join(cloud.head(5)["term"].astype(str).tolist()) if not cloud.empty else _safe_str(top.get("top_terms"))
    observation = f"가장 큰 토픽은 T{int(top['topic_id'])}이며 비중은 {_format_pct(top.get('topic_weight_pct'))}, 기사 수는 {_safe_str(top.get('headline_count'))}건입니다. 대표 헤드라인은 '{_safe_str(top.get('sample_headline'))}'입니다."
    opportunity = f"핵심 단어는 {lead_terms}입니다. 이 단어들이 반복될수록 최근 뉴스 흐름의 중심 서사를 먼저 파악한 뒤 관련 계산형 신호를 연결해서 볼 수 있습니다."
    caution = "토픽 비중은 관심의 크기를 말해줄 뿐 수익 방향을 보장하지 않습니다. 높은 비중의 테마라도 실제 가격 반응은 이벤트·괴리 페이지에서 따로 검증하는 편이 좋습니다."
    top_weight = float(pd.to_numeric(pd.Series([top.get("topic_weight_pct")]), errors="coerce").fillna(0).iloc[0])
    signal_label = "강세" if top_weight >= 35.0 else "중립"
    signal_tone = "bullish" if signal_label == "강세" else "neutral"
    return _analysis_card("결과 해석", [("핵심 관찰", observation), ("기회 요인", opportunity), ("주의 요인", caution)], signal_label=signal_label, signal_tone=signal_tone)


def _event_study_chart(dashboard: StockNewsDashboard) -> str:
    summary = dashboard.event_study.summary
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    if summary.empty:
        ax.text(0.5, 0.5, "No event-study observations", ha="center", va="center")
        ax.set_axis_off()
        return _render_chart_base64(fig)
    ax.plot(summary["day"], summary["mean_return_pct"], marker="o", linewidth=2.0, color="#0f4c81", label="Mean")
    ax.plot(summary["day"], summary["median_return_pct"], marker="s", linewidth=1.6, color="#b26a00", label="Median")
    ax.axhline(0.0, color="#8a98a8", linewidth=1.0, alpha=0.8)
    ax.set_title("Forward Return After Matched News")
    ax.set_xlabel("Trading Days After Event")
    ax.set_ylabel("Return (%)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    return _render_chart_base64(fig)


def _divergence_chart(dashboard: StockNewsDashboard) -> str:
    alerts = dashboard.divergence.alerts
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    if alerts.empty:
        ax.text(0.5, 0.5, "No divergence alerts", ha="center", va="center")
        ax.set_axis_off()
        return _render_chart_base64(fig)
    colors = np.where(alerts["effective_sentiment"] >= 0, "#0f4c81", "#b42318")
    ax.scatter(alerts["effective_sentiment"], alerts["divergence_return_pct"], c=colors, alpha=0.8)
    ax.axhline(0.0, color="#8a98a8", linewidth=1.0, alpha=0.7)
    ax.axvline(0.0, color="#8a98a8", linewidth=1.0, alpha=0.7)
    ax.set_title("Sentiment vs Opposing Price Move")
    ax.set_xlabel("Effective Sentiment")
    ax.set_ylabel("Contrarian Return (%)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return _render_chart_base64(fig)


def _topic_weight_chart(dashboard: StockNewsDashboard) -> str:
    topics = dashboard.topics.topics
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    if topics.empty:
        ax.text(0.5, 0.5, "No topic clusters", ha="center", va="center")
        ax.set_axis_off()
        return _render_chart_base64(fig)
    labels = [f"T{int(row.topic_id)}" for row in topics.itertuples(index=False)]
    ax.barh(labels, topics["topic_weight_pct"], color="#2e7d32")
    ax.set_title("Topic Share")
    ax.set_xlabel("Weight (%)")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return _render_chart_base64(fig)


def _overview_daily_chart(dashboard: StockNewsDashboard) -> str:
    daily_counts = dashboard.overview.daily_counts
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    if daily_counts.empty:
        ax.text(0.5, 0.5, "No recent article counts", ha="center", va="center")
        ax.set_axis_off()
        return _render_chart_base64(fig)
    show = daily_counts.copy().sort_values("date", ascending=True).tail(14)
    ax.bar(show["date"].astype(str), show["article_count"], color="#0f4c81")
    ax.set_title("Recent Daily Article Count")
    ax.set_xlabel("Date")
    ax.set_ylabel("Articles")
    ax.grid(axis="y", alpha=0.2)
    ax.tick_params(axis="x", rotation=40)
    fig.tight_layout()
    return _render_chart_base64(fig)


def _overview_ticker_chart(dashboard: StockNewsDashboard) -> str:
    top_tickers = dashboard.overview.top_tickers
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    if top_tickers.empty:
        ax.text(0.5, 0.5, "No ticker mentions", ha="center", va="center")
        ax.set_axis_off()
        return _render_chart_base64(fig)
    show = top_tickers.head(8).iloc[::-1]
    ax.barh(show["ticker"].astype(str), show["article_count"], color="#2e7d32")
    ax.set_title("Most Mentioned Tickers")
    ax.set_xlabel("Articles")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return _render_chart_base64(fig)


def _sector_spillover_chart(dashboard: StockNewsDashboard) -> str:
    summary = dashboard.sector_spillover.summary
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    if summary.empty:
        ax.text(0.5, 0.5, "No sector spillover rows", ha="center", va="center")
        ax.set_axis_off()
        return _render_chart_base64(fig)
    col = "peer_day_5_return_pct" if "peer_day_5_return_pct" in summary.columns else summary.columns[-1]
    show = summary.head(8).iloc[::-1]
    ax.barh(show["sector"].astype(str), show[col], color="#6f42c1")
    ax.axvline(0.0, color="#8a98a8", linewidth=1.0, alpha=0.7)
    ax.set_title("Sector Peer Move After News")
    ax.set_xlabel("Peer Return (%)")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return _render_chart_base64(fig)


def _volatility_chart(dashboard: StockNewsDashboard) -> str:
    summary = dashboard.volatility_regime.summary
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    if summary.empty:
        ax.text(0.5, 0.5, "No volatility regime rows", ha="center", va="center")
        ax.set_axis_off()
        return _render_chart_base64(fig)
    show = summary.head(8).iloc[::-1]
    ax.barh(show["ticker"].astype(str), show["avg_volatility_ratio"], color="#b26a00")
    ax.axvline(1.0, color="#8a98a8", linewidth=1.0, alpha=0.7)
    ax.set_title("Post-News Volatility Ratio")
    ax.set_xlabel("Post / Baseline")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return _render_chart_base64(fig)


def _expectation_reset_chart(dashboard: StockNewsDashboard) -> str:
    candidates = dashboard.expectation_reset.candidates
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    if candidates.empty:
        ax.text(0.5, 0.5, "No expectation-reset candidates", ha="center", va="center")
        ax.set_axis_off()
        return _render_chart_base64(fig)
    colors = np.where(candidates["effective_sentiment"] >= 0, "#0f4c81", "#b42318")
    ax.scatter(candidates["effective_sentiment"], candidates["day_5_return_pct"], c=colors, alpha=0.8)
    ax.axhline(0.0, color="#8a98a8", linewidth=1.0, alpha=0.7)
    ax.axvline(0.0, color="#8a98a8", linewidth=1.0, alpha=0.7)
    ax.set_title("Sentiment vs 5D Return")
    ax.set_xlabel("Effective Sentiment")
    ax.set_ylabel("5D Return (%)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return _render_chart_base64(fig)


def _word_cloud_html(dashboard: StockNewsDashboard) -> str:
    cloud = dashboard.topics.word_cloud
    if cloud.empty:
        return "<p class='hint'>키워드 클라우드를 만들 데이터가 부족합니다.</p>"
    max_weight = float(cloud["weight"].max()) if not cloud.empty else 1.0
    parts = ["<div class='word-cloud'>"]
    for row in cloud.itertuples(index=False):
        size = 0.95 + (float(row.weight) / max_weight) * 1.5 if max_weight > 0 else 1.0
        parts.append(f"<span style='font-size:{size:.2f}rem'>{html.escape(str(row.term))}</span>")
    parts.append("</div>")
    return "".join(parts)


def _summary_metrics(ctx: _PageContext) -> str:
    dashboard = ctx.dashboard
    if dashboard is None:
        return ""
    metrics = [
        ("적용 키워드", ", ".join(dashboard.applied_keywords) if dashboard.applied_keywords else "ALL"),
        ("적용 티커", dashboard.applied_ticker or "ALL"),
        ("티커 섹터", dashboard.ticker_sector or "ALL"),
        ("분석 윈도우", f"{dashboard.window_start} ~ {dashboard.window_end}"),
    ]
    if DASHBOARD_SECTION_OVERVIEW in dashboard.computed_sections:
        metrics.extend(
            [
                ("뉴스 건수", f"{dashboard.overview.article_count:,d}"),
                ("티커 수", f"{dashboard.overview.unique_ticker_count:,d}"),
                ("소스 수", f"{dashboard.overview.unique_source_count:,d}"),
                ("최신 기사", dashboard.overview.latest_publish_at or "-"),
            ]
        )
    if DASHBOARD_SECTION_EVENT_STUDY in dashboard.computed_sections:
        metrics.append(("이벤트 매치", f"{dashboard.event_study.article_count:,d}"))
    if DASHBOARD_SECTION_SECTOR_SPILLOVER in dashboard.computed_sections:
        metrics.append(("섹터 전이", f"{len(dashboard.sector_spillover.events.index):,d}"))
    if DASHBOARD_SECTION_DIVERGENCE in dashboard.computed_sections:
        metrics.append(("괴리 알림", f"{len(dashboard.divergence.alerts.index):,d}"))
    if DASHBOARD_SECTION_TOPICS in dashboard.computed_sections:
        metrics.append(("토픽 소스 기사", f"{len(dashboard.topics.source_articles.index):,d}"))
    return '<div class="metrics">' + "".join(
        f'<div class="metric"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>'
        for label, value in metrics
    ) + "</div>"


def _resolve_ticker_query(query: str) -> tuple[str | None, str | None, bool]:
    raw = str(query or "").strip()
    if not raw:
        return None, None, False
    try:
        from pipeline_stock.web_gui import _resolve_ticker_input as stock_resolve_ticker_input

        return stock_resolve_ticker_input(raw, ca_bundle_path=None, insecure_ssl=False)
    except Exception:
        upper_raw = raw.upper()
        if upper_raw and " " not in raw:
            return upper_raw, None, False
        return None, f"회사명 '{raw}'에서 티커를 찾지 못했습니다.", True


def _has_sections(dashboard: StockNewsDashboard | None, page_key: str) -> bool:
    required = PAGE_TO_SECTIONS.get(page_key)
    return dashboard is not None and required is not None and required.issubset(dashboard.computed_sections)


def _page_key_from_path(path: str) -> str:
    return {
        "/": "overview",
        "/index.html": "overview",
        "/overview": "overview",
        "/event-study": "event",
        "/sector-spillover": "spillover",
        "/divergence": "divergence",
        "/expectation-reset": "expectation",
        "/volatility-regime": "volatility",
        "/topic-modeling": "topics",
    }.get(path, "overview")


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
        <button type="submit" name="intent" value="resolve_ticker">회사이름으로 티커 찾기</button>
      </div>
    </form>
    """


def _layout_page(
    *,
    active: str,
    title: str,
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
    if ctx.ticker_note:
        css = "err" if ctx.ticker_note_error else "ok"
        notices += f'<div class="notice {css}"><pre>{html.escape(ctx.ticker_note)}</pre></div>'
    elif _has_sections(ctx.dashboard, active):
        notices += '<div class="notice ok">현재 페이지 기준으로 최신 분석 결과를 불러왔습니다.</div>' # This notice should be handled by the main GUI
    
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
    
    body_content = f"""
  <div class="wrap">
    {_page_head(title)}
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
          <div class="card"><h3>일자별 기사 수</h3><img class="chart" src="data:image/png;base64,{_overview_daily_chart(dashboard)}" alt="daily article count chart" /></div>
          <div class="card"><h3>주요 언급 티커</h3><img class="chart" src="data:image/png;base64,{_overview_ticker_chart(dashboard)}" alt="top ticker mentions chart" /></div>
        </div>
        <div class="table-grid">
          <div class="card"><h3>섹터별 기사 집중도</h3>{_safe_table(overview.top_sectors, 10)}</div>
          <div class="card"><h3>소스별 기사 분포</h3>{_safe_table(overview.top_sources, 10)}</div>
        </div>
        <div class="table-grid">
          <div class="card"><h3>티커별 기사 수</h3>{_safe_table(overview.top_tickers, 10)}</div>
          <div class="card"><h3>최근 기사</h3>{_safe_table(overview.recent_articles, 20)}</div>
        </div>
        """
    return _layout_page(
        active="overview",
        title="뉴스 개요",
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
        event_table = _safe_table(
            dashboard.event_study.events[
                ["ticker", "publish_date", "reference_price_date", "day_1_return", "day_3_return", "day_5_return", "title"]
            ],
            100,
        ) if not dashboard.event_study.events.empty else "<p class='hint'>이벤트 스터디 대상 뉴스가 없습니다.</p>"
        body = f"""
        {_event_interpretation(dashboard)}
        <div class="charts">
          <div class="card"><h3>이벤트 이후 수익률 곡선</h3><img class="chart" src="data:image/png;base64,{_event_study_chart(dashboard)}" alt="event study chart" /></div>
          <div class="card"><h3>이벤트 스터디 요약</h3>{_safe_table(dashboard.event_study.summary, 10)}</div>
        </div>
        <div class="tables">
          <div class="card"><h3>매치된 이벤트 뉴스</h3>{event_table}</div>
        </div>
        """
    return _layout_page(
        active="event",
        title="이벤트 스터디",
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
        events = _safe_table(
            dashboard.sector_spillover.events[
                ["source_ticker", "sector", "publish_date", "peer_count", "peer_day_1_return_pct", "peer_day_3_return_pct", "peer_day_5_return_pct", "title"]
            ],
            100,
        ) if not dashboard.sector_spillover.events.empty else "<p class='hint'>섹터 전이 이벤트가 없습니다.</p>"
        body = f"""
        {_spillover_interpretation(dashboard)}
        <div class="charts">
          <div class="card"><h3>섹터 전이 강도</h3><img class="chart" src="data:image/png;base64,{_sector_spillover_chart(dashboard)}" alt="sector spillover chart" /></div>
          <div class="card"><h3>섹터별 요약</h3>{_safe_table(dashboard.sector_spillover.summary, 12)}</div>
        </div>
        <div class="tables">
          <div class="card"><h3>섹터 전이 이벤트</h3>{events}</div>
        </div>
        """
    return _layout_page(
        active="spillover",
        title="섹터 전이",
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
        alerts = _safe_table(
            dashboard.divergence.alerts[
                ["ticker", "publish_date", "effective_sentiment", "divergence_horizon_days", "divergence_return_pct", "divergence_score", "title"]
            ],
            100,
        ) if not dashboard.divergence.alerts.empty else "<p class='hint'>괴리 알림이 없습니다.</p>"
        body = f"""
        {_divergence_interpretation(dashboard)}
        <div class="charts">
          <div class="card"><h3>감성 대비 반대 가격 반응</h3><img class="chart" src="data:image/png;base64,{_divergence_chart(dashboard)}" alt="divergence chart" /></div>
          <div class="card"><h3>괴리 알림</h3>{alerts}</div>
        </div>
        """
    return _layout_page(
        active="divergence",
        title="뉴스-프라이스 다이버전스",
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
        table = _safe_table(
            dashboard.expectation_reset.candidates[
                ["ticker", "publish_date", "effective_sentiment", "day_1_return_pct", "day_3_return_pct", "day_5_return_pct", "reset_type", "reset_score", "title"]
            ],
            100,
        ) if not dashboard.expectation_reset.candidates.empty else "<p class='hint'>기대 리셋 후보가 없습니다.</p>"
        body = f"""
        {_expectation_interpretation(dashboard)}
        <div class="charts">
          <div class="card"><h3>감성 대비 5일 수익률</h3><img class="chart" src="data:image/png;base64,{_expectation_reset_chart(dashboard)}" alt="expectation reset chart" /></div>
          <div class="card"><h3>기대 리셋 후보</h3>{table}</div>
        </div>
        """
    return _layout_page(
        active="expectation",
        title="기대 리셋",
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
        events = _safe_table(
            dashboard.volatility_regime.events[
                ["ticker", "publish_date", "baseline_vol_pct", "post_vol_pct", "volatility_ratio", "title"]
            ],
            100,
        ) if not dashboard.volatility_regime.events.empty else "<p class='hint'>변동성 레짐 이벤트가 없습니다.</p>"
        body = f"""
        {_volatility_interpretation(dashboard)}
        <div class="charts">
          <div class="card"><h3>뉴스 이후 변동성 확대</h3><img class="chart" src="data:image/png;base64,{_volatility_chart(dashboard)}" alt="volatility regime chart" /></div>
          <div class="card"><h3>종목별 변동성 레짐 요약</h3>{_safe_table(dashboard.volatility_regime.summary, 12)}</div>
        </div>
        <div class="tables">
          <div class="card"><h3>변동성 레짐 이벤트</h3>{events}</div>
        </div>
        """
    return _layout_page(
        active="volatility",
        title="변동성 레짐",
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
        source_articles = _safe_table(
            dashboard.topics.source_articles[["ticker", "publish_date", "title"]], 100
        ) if not dashboard.topics.source_articles.empty else "<p class='hint'>토픽 소스 뉴스가 없습니다.</p>"
        body = f"""
        {_topics_interpretation(dashboard)}
        <div class="charts">
          <div class="card"><h3>토픽 비중</h3><img class="chart" src="data:image/png;base64,{_topic_weight_chart(dashboard)}" alt="topic weight chart" /></div>
          <div class="card"><h3>키워드 클라우드</h3>{_word_cloud_html(dashboard)}</div>
        </div>
        <div class="table-grid">
          <div class="card"><h3>토픽 버킷</h3>{_safe_table(dashboard.topics.topics, 10)}</div>
          <div class="card"><h3>토픽 소스 뉴스</h3>{source_articles}</div>
        </div>
        """
    return _layout_page(
        active="topics",
        title="토픽 모델링",
        subtitle="최근 뉴스 제목을 묶어 시장을 관통하는 테마를 압축해서 보는 페이지",
        ctx=ctx,
        action="/run_topic_modeling",
        button_label="토픽 모델 실행",
        content_html=body,
        is_sub_page=is_sub_page,
    )


def _parse_form(body: bytes) -> dict[str, str]:
    parsed = parse_qs(body.decode("utf-8", errors="ignore"), keep_blank_values=True)
    form = _default_form()
    for key in form:
        if key in parsed and parsed[key]:
            form[key] = str(parsed[key][0]).strip()
    if "intent" in parsed and parsed["intent"]:
        form["intent"] = str(parsed["intent"][0]).strip()
    return form


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


def _refresh_subprocess_command(root_dir: Path) -> tuple[list[str], str]:
    if getattr(sys, "frozen", False):
        exe_path = Path(sys.executable).resolve()
        return [str(exe_path), "refresh-news"], f"{exe_path} refresh-news"
    batch_path = root_dir / "refresh_news_data.bat"
    if not batch_path.exists() or not batch_path.is_file():
        return [], f"refresh_news_data.bat not found ({batch_path})"
    return ["cmd.exe", "/c", str(batch_path)], str(batch_path)


def _news_sqlite_snapshot(sqlite_path: Path) -> tuple[str | None, int]:
    if not sqlite_path.exists() or not sqlite_path.is_file():
        return None, 0
    try:
        with sqlite3.connect(sqlite_path) as conn:
            row = conn.execute("SELECT MAX(publish_date), COUNT(*) FROM news_articles").fetchone()
    except Exception:
        return None, 0
    if not row:
        return None, 0
    latest = str(row[0]).strip() if row[0] is not None else None
    count = int(row[1] or 0)
    return latest, count


def _collect_post_refresh_items(root_dir: Path) -> list[dict[str, object]]:
    sqlite_path = root_dir / "data" / "sp500_shared_db" / "sp500_shared_prices.sqlite"
    latest, count = _news_sqlite_snapshot(sqlite_path)
    return [
        {
            "dataset": "news_articles",
            "latest_date": latest,
            "rows": count,
            "source": "refresh_news_data.bat",
            "path": str(sqlite_path.resolve()),
        }
    ]


def _html_refresh_page(is_sub_page: bool = False) -> str:
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Stock News Lab - 데이터 갱신</title>
  <style>
    {_base_css()}
    .small {{ font-size: 12px; color: var(--muted); }}
    .split-grid {{ margin-top: 12px; display: grid; grid-template-columns: repeat(2, minmax(360px, 1fr)); gap: 10px; }}
    .pane {{ background: #fff; border: 1px solid var(--line); border-radius: 10px; padding: 12px; }}
    .pane h3 {{ margin: 0 0 8px 0; }}
    .line-list {{ height: 420px; overflow: auto; border: 1px solid var(--line); border-radius: 8px; background: #fff; padding: 8px; }}
    .line {{ font-family: Consolas, "Courier New", monospace; font-size: 12px; line-height: 1.45; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; border-bottom: 1px solid #eef2f7; padding: 3px 2px; }}
    .line:last-child {{ border-bottom: 0; }}
    @media (max-width: 1180px) {{
      .split-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1 style="margin:0 0 12px;">Stock News Lab | S&P 500</h1>
    {_nav("refresh")}
    {_page_head("데이터 갱신")}
    <div class="sub">뉴스 데이터만 증분 또는 일회성 백필로 갱신하고 진행 상황을 바로 확인하는 페이지</div>
    <form class="card" method="post" action="/run_refresh">
      <div class="row">
        <button type="submit">뉴스 데이터 갱신 시작</button>
        <a href="/refresh-history" style="color: var(--brand); text-decoration: none; font-size: 13px;">실행 이력 보기</a>
      </div>
      <div id="refresh-latest" class="latest-inline">데이터 상태 확인 중...</div>
      <p id="refresh-meta" class="small">상태: 대기 / 실행 ID: - / 시작: - / 종료: -</p>
    </form>
    <div class="split-grid">
      <div class="pane">
        <h3>진행 로그</h3>
        <div id="refresh-log" class="line-list"><div class="line">아직 로그가 없습니다.</div></div>
      </div>
      <div class="pane">
        <h3>업데이트 항목</h3>
        <div id="refresh-updates" class="line-list"><div class="line">아직 업데이트 항목이 없습니다.</div></div>
      </div>
    </div>
  </div>
  <script>
    const metaEl = document.getElementById("refresh-meta");
    const latestEl = document.getElementById("refresh-latest");
    const logEl = document.getElementById("refresh-log");
    const updatesEl = document.getElementById("refresh-updates");
    let lastLogCount = -1;
    let lastUpdateKey = "";

    function esc(value) {{
      return String(value ?? "-")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }}

    async function pollStatus() {{
      try {{
        const res = await fetch("/refresh_status", {{ cache: "no-store" }});
        if (!res.ok) {{
          metaEl.textContent = "상태: 오류 (상태 조회 응답이 없습니다)";
          return;
        }}
        const data = await res.json();
        metaEl.textContent =
          "상태: " + (data.status || "대기")
          + " / 실행 ID: " + (data.run_id || "-")
          + " / 시작: " + (data.started_at || "-")
          + " / 종료: " + (data.finished_at || "-");

        const logCount = Number(data.log_count || 0);
        if (logCount !== lastLogCount) {{
          lastLogCount = logCount;
          const logs = Array.isArray(data.logs) ? data.logs : [];
          logEl.innerHTML = logs.length === 0
            ? "<div class='line'>아직 로그가 없습니다.</div>"
            : logs.map((line) => "<div class='line'>" + esc(line) + "</div>").join("");
          logEl.scrollTop = logEl.scrollHeight;
        }}

        const items = Array.isArray(data.updated_items) ? data.updated_items : [];
        const updateKey = JSON.stringify(items);
        if (updateKey !== lastUpdateKey) {{
          lastUpdateKey = updateKey;
          updatesEl.innerHTML = items.length === 0
            ? "<div class='line'>아직 업데이트 항목이 없습니다.</div>"
            : items.map((item) => {{
                const line = (item.dataset || "-")
                  + " | latest=" + (item.latest_date || "-")
                  + " | rows=" + (item.rows ?? "-")
                  + " | source=" + (item.source || "-");
                return "<div class='line'>" + esc(line) + "</div>";
              }}).join("");
        }}
      }} catch (err) {{
        metaEl.textContent = "상태: 오류 (" + String(err) + ")";
      }}
    }}

    pollStatus();
    setInterval(pollStatus, 2200);
  </script>
</body>
</html>
"""


def _html_refresh_history_page(is_sub_page: bool = False) -> str:
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Stock News Lab - 데이터 갱신 이력</title>
  <style>
    {_base_css()}
    .table-wrap {{ overflow: auto; max-height: 360px; border: 1px solid var(--line); border-radius: 10px; background: #fff; }}
    .caption {{ margin: 0 0 8px 0; color: var(--muted); font-size: 12px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1 style="margin:0 0 12px;">Stock News Lab | S&P 500</h1>
    {_nav("refresh")}
    {_page_head("데이터 갱신 이력")}
    <div class="sub">뉴스 갱신 실행 요약과 최신 업데이트 결과를 누적해서 보는 페이지</div>
    <div class="card">
      <h3>실행 요약</h3>
      <p id="history-generated" class="caption">생성 시각: -</p>
      <div class="table-wrap">
        <table class="data-table">
          <thead>
            <tr>
              <th>Run ID</th>
              <th>Status</th>
              <th>Started</th>
              <th>Finished</th>
              <th>Old Latest</th>
              <th>New Latest</th>
              <th>Rows Added</th>
              <th>Error</th>
            </tr>
          </thead>
          <tbody id="run-history-body"></tbody>
        </table>
      </div>
    </div>
    <div class="card">
      <h3>업데이트 항목</h3>
      <div class="table-wrap">
        <table class="data-table">
          <thead>
            <tr>
              <th>Run ID</th>
              <th>Dataset</th>
              <th>Latest Date</th>
              <th>Rows</th>
              <th>Source</th>
              <th>Path</th>
            </tr>
          </thead>
          <tbody id="update-history-body"></tbody>
        </table>
      </div>
    </div>
  </div>
  <script>
    function esc(value) {{
      return String(value ?? "-")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }}

    async function pollHistory() {{
      try {{
        const res = await fetch("/refresh_history_data", {{ cache: "no-store" }});
        if (!res.ok) {{
          return;
        }}
        const data = await res.json();
        const runs = Array.isArray(data.runs) ? data.runs : [];
        document.getElementById("history-generated").textContent = "생성 시각: " + (data.generated_at || "-");

        const runRows = runs.map((run) => `
          <tr>
            <td>${{esc(run.run_id)}}</td>
            <td>${{esc(run.status)}}</td>
            <td>${{esc(run.started_at)}}</td>
            <td>${{esc(run.finished_at)}}</td>
            <td>${{esc(run.old_latest_date)}}</td>
            <td>${{esc(run.new_latest_date)}}</td>
            <td>${{esc(run.news_rows_added)}}</td>
            <td>${{esc(run.error_message)}}</td>
          </tr>
        `).join("");
        document.getElementById("run-history-body").innerHTML = runRows || "<tr><td colspan='8'>아직 갱신 이력이 없습니다.</td></tr>";

        const updateRows = [];
        for (const run of runs) {{
          const items = Array.isArray(run.updated_items) ? run.updated_items : [];
          for (const item of items) {{
            updateRows.push(`
              <tr>
                <td>${{esc(run.run_id)}}</td>
                <td>${{esc(item.dataset)}}</td>
                <td>${{esc(item.latest_date)}}</td>
                <td>${{esc(item.rows)}}</td>
                <td>${{esc(item.source)}}</td>
                <td>${{esc(item.path)}}</td>
              </tr>
            `);
          }}
        }}
        document.getElementById("update-history-body").innerHTML = updateRows.join("") || "<tr><td colspan='6'>아직 업데이트 결과가 없습니다.</td></tr>";
      }} catch (err) {{
        return;
      }}
    }}

    pollHistory();
    setInterval(pollHistory, 2200);
  </script>
</body>
</html>
"""
def _build_dashboard_from_form(form: dict[str, str], page_key: str) -> StockNewsDashboard:
    ticker = form.get("ticker", "").strip().upper() or None
    light_mode = str(os.getenv("KEUMJM_NEWS_LIGHT_MODE", "false")).strip().lower() in {"1", "true", "yes", "on"}
    lookback_days = max(int(form.get("lookback_days", str(DEFAULT_LOOKBACK_DAYS)) or str(DEFAULT_LOOKBACK_DAYS)), 1)
    horizon_days = max(int(form.get("horizon_days", DEFAULT_EVENT_HORIZON_DAYS) or DEFAULT_EVENT_HORIZON_DAYS), 1)
    divergence_top_n = max(int(form.get("divergence_top_n", DEFAULT_DIVERGENCE_TOP_N) or DEFAULT_DIVERGENCE_TOP_N), 1)
    topic_count = max(int(form.get("topic_count", DEFAULT_TOPIC_COUNT) or DEFAULT_TOPIC_COUNT), 2)
    if light_mode:
        lookback_days = min(lookback_days, int(os.getenv("KEUMJM_NEWS_MAX_LOOKBACK_DAYS", "14") or "14"))
        horizon_days = min(horizon_days, int(os.getenv("KEUMJM_NEWS_MAX_HORIZON_DAYS", "3") or "3"))
        divergence_top_n = min(divergence_top_n, int(os.getenv("KEUMJM_NEWS_MAX_TOP_N", "10") or "10"))
        topic_count = min(topic_count, int(os.getenv("KEUMJM_NEWS_MAX_TOPIC_COUNT", "3") or "3"))
    return build_stock_news_dashboard( # This function needs to be exposed
        event_keywords=form.get("event_keywords", DEFAULT_EVENT_KEYWORDS),
        ticker=ticker,
        lookback_days=lookback_days,
        horizon_days=horizon_days,
        divergence_top_n=divergence_top_n,
        topic_count=topic_count,
        sections=PAGE_TO_SECTIONS.get(page_key),
    )


# Expose functions and dataclasses for external use
__all__ = [
    "_PageContext",
    "PAGE_TO_SECTIONS",
    "_default_form",
    "_build_dashboard_from_form",
    "_resolve_ticker_query",
    "_overview_page",
    "_html_event_page",
    "_html_spillover_page",
    "_html_divergence_page",
    "_html_expectation_page",
    "_html_volatility_page",
    "_html_topics_page",
]
