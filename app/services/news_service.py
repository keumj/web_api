from __future__ import annotations

import traceback
from dataclasses import dataclass, field

from pipeline_stock_news import web_gui as news_web

from app.web import add_start_page_link, inject_busy_cursor_overlay, rewrite_links


NEWS_REWRITES = {
    'href="/overview"': 'href="/stock-news/overview"',
    'href="/event-study"': 'href="/stock-news/event-study"',
    'href="/sector-spillover"': 'href="/stock-news/sector-spillover"',
    'href="/divergence"': 'href="/stock-news/divergence"',
    'href="/expectation-reset"': 'href="/stock-news/expectation-reset"',
    'href="/volatility-regime"': 'href="/stock-news/volatility-regime"',
    'href="/topic-modeling"': 'href="/stock-news/topic-modeling"',
    'action="/run_overview"': 'action="/stock-news/run-overview"',
    'action="/run_event_study"': 'action="/stock-news/run-event-study"',
    'action="/run_sector_spillover"': 'action="/stock-news/run-sector-spillover"',
    'action="/run_divergence"': 'action="/stock-news/run-divergence"',
    'action="/run_expectation_reset"': 'action="/stock-news/run-expectation-reset"',
    'action="/run_volatility_regime"': 'action="/stock-news/run-volatility-regime"',
    'action="/run_topic_modeling"': 'action="/stock-news/run-topic-modeling"',
}


@dataclass
class NewsPageState:
    form: dict[str, str] = field(default_factory=news_web._default_form)
    dashboard: object | None = None
    error: str | None = None


states: dict[str, NewsPageState] = {key: NewsPageState() for key in news_web.PAGE_TO_SECTIONS}


PAGE_ALIASES = {
    "overview": "overview",
    "event-study": "event",
    "sector-spillover": "spillover",
    "divergence": "divergence",
    "expectation-reset": "expectation",
    "volatility-regime": "volatility",
    "topic-modeling": "topics",
}


def _render_func(page_key: str):
    return {
        "overview": news_web._overview_page,
        "event": news_web._html_event_page,
        "spillover": news_web._html_spillover_page,
        "divergence": news_web._html_divergence_page,
        "expectation": news_web._html_expectation_page,
        "volatility": news_web._html_volatility_page,
        "topics": news_web._html_topics_page,
    }[page_key]


def render(page: str) -> str:
    page_key = PAGE_ALIASES.get(page, "overview")
    state = states[page_key]
    ctx = news_web._PageContext(
        dashboard=state.dashboard,
        form=state.form,
        error=state.error,
    )
    html = rewrite_links(_render_func(page_key)(ctx), NEWS_REWRITES)
    return inject_busy_cursor_overlay(add_start_page_link(html))


def run(page: str, form: dict[str, str]) -> str:
    page_key = PAGE_ALIASES.get(page, "overview")
    state = states[page_key]
    page_form = news_web._default_form()
    page_form.update({key: str(value).strip() for key, value in form.items()})
    page_form["ticker"] = page_form.get("ticker", "").strip().upper()
    state.form = page_form
    try:
        state.dashboard = news_web._build_dashboard_from_form(page_form, page_key)
        state.error = None
    except Exception as exc:
        state.dashboard = None
        state.error = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc(limit=3)}"
    return page
