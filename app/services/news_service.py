from __future__ import annotations

import threading
import traceback
from dataclasses import dataclass, field

from pipeline_stock_news import web_gui as news_web

from app.web import apply_service_chrome, rewrite_links
from app.services import user_result_service
from app.services.auth_service import AuthUser


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


_states: dict[str, dict[str, NewsPageState]] = {}
_states_lock = threading.RLock()


def _session_states(session_key: str) -> dict[str, NewsPageState]:
    with _states_lock:
        return _states.setdefault(session_key, {key: NewsPageState() for key in news_web.PAGE_TO_SECTIONS})


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


def _news_user_id(user: AuthUser | None) -> str | None:
    return user.id if user is not None else None


def _render_state(page_key: str, state: NewsPageState) -> str:
    ctx = news_web._PageContext(
        dashboard=state.dashboard,
        form=state.form,
        error=state.error,
    )
    html = rewrite_links(_render_func(page_key)(ctx), NEWS_REWRITES)
    return apply_service_chrome(html, active="news")


def render(page: str, *, session_key: str = "global", user: AuthUser | None = None) -> str:
    page_key = PAGE_ALIASES.get(page, "overview")
    state = _session_states(session_key)[page_key]
    if state.dashboard is None and state.error is None:
        latest = user_result_service.load_latest_result(_news_user_id(user), module="stock-news", page=page_key)
        if latest is not None:
            return user_result_service.with_loaded_notice(latest.html, latest, label="뉴스 분석")
    return _render_state(page_key, state)


def run(page: str, form: dict[str, str], *, session_key: str = "global", user: AuthUser | None = None) -> str:
    page_key = PAGE_ALIASES.get(page, "overview")
    state = _session_states(session_key)[page_key]
    page_form = news_web._default_form()
    page_form.update({key: str(value).strip() for key, value in form.items()})
    page_form["ticker"] = page_form.get("ticker", "").strip().upper()
    state.form = page_form
    try:
        state.dashboard = news_web._build_dashboard_from_form(page_form, page_key)
        state.error = None
        user_result_service.save_latest_result(
            _news_user_id(user),
            module="stock-news",
            page=page_key,
            html=_render_state(page_key, state),
            metadata={"ticker": page_form.get("ticker", ""), "page": page},
        )
    except Exception as exc:
        state.dashboard = None
        state.error = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc(limit=3)}"
    return page
