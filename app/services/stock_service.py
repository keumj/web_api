from __future__ import annotations

import re
import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime

from pipeline_stock import web_gui as stock_web

from app.web import apply_service_chrome, rewrite_links
from app.services import user_result_service
from app.services.auth_service import AuthUser


STOCK_REWRITES = {
    'href="/forecast"': 'href="/stock/forecast"',
    'href="/page2"': 'href="/stock/financials"',
    'href="/page3"': 'href="/stock/technical"',
    'href="/page4"': 'href="/stock/returns"',
    'href="/page5"': 'href="/stock/risk"',
    'href="/factor-regime"': 'href="/stock/factor-regime"',
    'href="/page6"': 'href="/stock/decision"',
    'href="/page8"': 'href="/stock/walk-forward"',
    'action="/run"': 'action="/stock/run"',
    'action="/run_financial"': 'action="/stock/run-financial"',
    'action="/run_technical"': 'action="/stock/run-technical"',
    'action="/run_returns"': 'action="/stock/run-returns"',
    'action="/run_risk"': 'action="/stock/run-risk"',
    'action="/run_factor"': 'action="/stock/run-factor"',
    'action="/run_decision"': 'action="/stock/run-decision"',
    'action="/run_walk_forward"': 'action="/stock/run-walk-forward"',
}


@dataclass
class StockState:
    forecast_form: dict[str, str] = field(
        default_factory=lambda: {
            "ticker": "AAPL",
            "forecast_horizon": "10",
            "history_years": "8",
            "start_date": "2025-12-31",
            "end_date": datetime.utcnow().strftime("%Y-%m-%d"),
            "output_dir": "outputs/stock_forecast",
            "prices_csv_path": "",
            "use_sample": "",
            "auto_save": "on",
            "insecure_ssl": "",
            "ca_bundle_path": "",
        }
    )
    forecast_ctx: object | None = None
    forecast_error: str | None = None
    financials_form: dict[str, str] = field(
        default_factory=lambda: {
            "ticker": "AAPL",
            "statement_periods": "4",
            "output_dir": "outputs/stock_forecast_finance",
            "auto_save": "on",
            "insecure_ssl": "",
            "ca_bundle_path": "",
            "fmp_api_key": "",
        }
    )
    financials_ctx: object | None = None
    financials_error: str | None = None
    technical_form: dict[str, str] = field(
        default_factory=lambda: {
            "ticker": "AAPL",
            "output_dir": "outputs/technical_analysis",
            "use_sample": "",
            "auto_save": "on",
            "action": "all",
        }
    )
    technical_ctx: object | None = None
    technical_error: str | None = None
    technical_cache: object | None = None
    returns_form: dict[str, str] = field(default_factory=lambda: {"ticker": "AAPL"})
    returns_ctx: object | None = None
    returns_error: str | None = None
    returns_running: bool = False
    returns_started_at: str | None = None
    risk_form: dict[str, str] = field(default_factory=lambda: {"ticker": "AAPL"})
    risk_ctx: object | None = None
    risk_error: str | None = None
    factor_form: dict[str, str] = field(default_factory=lambda: {"ticker": "AAPL"})
    factor_ctx: object | None = None
    factor_error: str | None = None
    decision_form: dict[str, str] = field(default_factory=lambda: {"ticker": "AAPL"})
    decision_ctx: object | None = None
    decision_error: str | None = None
    wfv_form: dict[str, str] = field(
        default_factory=lambda: {
            "ticker": "AAPL",
            "forecast_horizon": "10",
            "history_years": "8",
            "start_date": "2025-12-31",
            "end_date": datetime.utcnow().strftime("%Y-%m-%d"),
            "wf_min_train_rows": "252",
            "wf_step_size": "21",
            "wf_max_splits": "4",
            "output_dir": "outputs/walk_forward_validation",
            "prices_csv_path": "",
            "use_sample": "",
            "auto_save": "on",
            "insecure_ssl": "",
            "ca_bundle_path": "",
        }
    )
    wfv_ctx: object | None = None
    wfv_error: str | None = None


_states: dict[str, StockState] = {}
_states_lock = threading.RLock()


def _state(session_key: str) -> StockState:
    with _states_lock:
        return _states.setdefault(session_key, StockState())


def _clean_stock_html(html: str) -> str:
    return apply_service_chrome(rewrite_links(html, STOCK_REWRITES), active="stock")


def _stock_user_id(user: AuthUser | None) -> str | None:
    return user.id if user is not None else None


def _page_has_result(state: StockState, page: str) -> bool:
    return {
        "forecast": state.forecast_ctx is not None,
        "financials": state.financials_ctx is not None,
        "technical": state.technical_ctx is not None,
        "returns": state.returns_ctx is not None,
        "risk": state.risk_ctx is not None,
        "factor-regime": state.factor_ctx is not None,
        "decision": state.decision_ctx is not None,
        "walk-forward": state.wfv_ctx is not None,
    }.get(page, state.forecast_ctx is not None)


def _page_ticker(state: StockState, page: str) -> str:
    form = {
        "forecast": state.forecast_form,
        "financials": state.financials_form,
        "technical": state.technical_form,
        "returns": state.returns_form,
        "risk": state.risk_form,
        "factor-regime": state.factor_form,
        "decision": state.decision_form,
        "walk-forward": state.wfv_form,
    }.get(page, state.forecast_form)
    return str(form.get("ticker", "") or "")


def _render_page_html(page: str, state: StockState) -> str:
    if page == "forecast":
        html = stock_web._html_page(
            state.forecast_form,
            ctx=state.forecast_ctx,
            error=state.forecast_error,
            enable_technical_page=True,
        )
    elif page == "financials":
        html = stock_web._html_financial_page(
            state.financials_form,
            ctx=state.financials_ctx,
            error=state.financials_error,
            enable_technical_page=True,
        )
    elif page == "technical":
        html = stock_web._html_technical_page(
            state.technical_form,
            ctx=state.technical_ctx,
            error=state.technical_error,
        )
    elif page == "returns":
        ticker_note = None
        if state.returns_running:
            ticker_note = "수익률비교 계산 중입니다. 잠시 후 자동으로 새로고침되며, 계산이 끝나면 결과가 표시됩니다."
        html = stock_web._html_returns_page(
            state.returns_form,
            ctx=state.returns_ctx,
            error=state.returns_error,
            ticker_note=ticker_note,
        )
        if state.returns_running:
            html = html.replace("</body>", "<script>setTimeout(() => location.reload(), 5000);</script></body>", 1)
    elif page == "risk":
        html = stock_web._html_risk_page(
            state.risk_form,
            ctx=state.risk_ctx,
            error=state.risk_error,
        )
    elif page == "factor-regime":
        html = stock_web._html_factor_page(
            state.factor_form,
            ctx=state.factor_ctx,
            error=state.factor_error,
        )
    elif page == "decision":
        html = stock_web._html_decision_page(
            state.decision_form,
            ctx=state.decision_ctx,
            error=state.decision_error,
        )
    elif page == "walk-forward":
        html = stock_web._html_walk_forward_page(
            state.wfv_form,
            ctx=state.wfv_ctx,
            error=state.wfv_error,
        )
    else:
        html = stock_web._html_page(
            state.forecast_form,
            ctx=state.forecast_ctx,
            error=state.forecast_error,
            enable_technical_page=True,
        )
    return html


def _render_clean_page(page: str, state: StockState) -> str:
    return _clean_stock_html(_render_page_html(page, state))


def _save_latest_page(user_id: str | None, page: str, state: StockState) -> None:
    if not user_id or not _page_has_result(state, page):
        return
    user_result_service.save_latest_result(
        user_id,
        module="stock",
        page=page,
        html=_render_clean_page(page, state),
        metadata={"ticker": _page_ticker(state, page)},
    )


def render(
    page: str,
    ticker: str | None = None,
    intent: str | None = None,
    *,
    session_key: str = "global",
    user: AuthUser | None = None,
) -> str:
    state = _state(session_key)
    selected_ticker = _clean_ticker(ticker or "")
    if selected_ticker:
        _sync_ticker(state, selected_ticker)
    if not selected_ticker and not _page_has_result(state, page):
        latest = user_result_service.load_latest_result(_stock_user_id(user), module="stock", page=page)
        if latest is not None:
            return user_result_service.with_loaded_notice(latest.html, latest, label="주식 분석")
    return _render_clean_page(page, state)


def _clean_ticker(value: str) -> str:
    raw = str(value or "").strip().upper()
    raw = re.split(r"[?&#\s]", raw, maxsplit=1)[0]
    return re.sub(r"[^A-Z0-9.\-]", "", raw)


def _sync_ticker(state: StockState, ticker: str) -> None:
    if not ticker:
        return
    for form in [
        state.forecast_form,
        state.financials_form,
        state.technical_form,
        state.returns_form,
        state.risk_form,
        state.factor_form,
        state.decision_form,
        state.wfv_form,
    ]:
        form["ticker"] = ticker


def _matching_financials_ctx(state: StockState, ticker: str) -> object | None:
    fin_ctx = state.financials_ctx
    if fin_ctx is None:
        return None
    fin_ticker = str(getattr(fin_ctx, "ticker", "")).strip().upper()
    return fin_ctx if fin_ticker == ticker else None


def start_returns(form: dict[str, str], *, session_key: str = "global", user: AuthUser | None = None) -> str:
    state = _state(session_key)
    user_id = _stock_user_id(user)
    ticker = _clean_ticker(form.get("ticker", ""))
    if not ticker:
        with _states_lock:
            state.returns_error = "ValueError: Provide an S&P 500 ticker for return analysis."
            state.returns_running = False
        return "returns"

    with _states_lock:
        _sync_ticker(state, ticker)
        if state.returns_running:
            return "returns"
        state.returns_form = {"ticker": ticker}
        state.returns_error = None
        state.returns_running = True
        state.returns_started_at = datetime.utcnow().isoformat(timespec="seconds")

    def worker(local_form: dict[str, str]) -> None:
        try:
            ctx = stock_web._run_returns_once(local_form)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc(limit=3)}"
            with _states_lock:
                state.returns_error = error
                state.returns_running = False
            return
        with _states_lock:
            state.returns_ctx = ctx
            state.returns_error = None
            state.returns_running = False
        _save_latest_page(user_id, "returns", state)

    thread = threading.Thread(target=worker, args=({"ticker": ticker},), name=f"stock-returns-{session_key}", daemon=True)
    thread.start()
    return "returns"


def run(action: str, form: dict[str, str], *, session_key: str = "global", user: AuthUser | None = None) -> str:
    state = _state(session_key)
    user_id = _stock_user_id(user)
    try:
        ticker = _clean_ticker(form.get("ticker", ""))
        _sync_ticker(state, ticker)
        if action == "forecast":
            for checkbox in ["use_sample", "auto_save", "insecure_ssl"]:
                form.setdefault(checkbox, "")
            state.forecast_form = {**state.forecast_form, **form, "ticker": ticker}
            try:
                state.forecast_ctx = stock_web._run_once(state.forecast_form)
            except ValueError as exc:
                if "Not enough history" not in str(exc):
                    raise
                retry_form = {**state.forecast_form, "start_date": "", "end_date": ""}
                state.forecast_ctx = stock_web._run_once(retry_form)
            state.forecast_error = None
            _save_latest_page(user_id, "forecast", state)
            return "forecast"
        if action == "financials":
            state.financials_form = {**state.financials_form, **form, "ticker": ticker}
            state.financials_ctx = stock_web._run_financial_once(state.financials_form)
            state.financials_error = None
            _save_latest_page(user_id, "financials", state)
            return "financials"
        if action == "technical":
            form["action"] = stock_web._normalize_technical_action(form.get("action", "all"))
            state.technical_form = {**state.technical_form, **form, "ticker": ticker}
            state.technical_ctx, state.technical_cache = stock_web.ta_web_gui._run_analysis(
                form=state.technical_form,
                action=state.technical_form.get("action", "all"),
                cache=state.technical_cache,
            )
            state.technical_error = None
            _save_latest_page(user_id, "technical", state)
            return "technical"
        if action == "returns":
            state.returns_form = {"ticker": ticker}
            state.returns_ctx = stock_web._run_returns_once(state.returns_form)
            state.returns_error = None
            _save_latest_page(user_id, "returns", state)
            return "returns"
        if action == "risk":
            state.risk_form = {"ticker": ticker}
            state.risk_ctx = stock_web._run_risk_once(state.risk_form)
            state.risk_error = None
            _save_latest_page(user_id, "risk", state)
            return "risk"
        if action == "factor":
            state.factor_form = {"ticker": ticker}
            state.factor_ctx = stock_web._run_factor_once(state.factor_form)
            state.factor_error = None
            _save_latest_page(user_id, "factor-regime", state)
            return "factor-regime"
        if action == "decision":
            state.decision_form = {"ticker": ticker}
            if state.returns_ctx is None or getattr(state.returns_ctx, "ticker", "") != ticker:
                state.returns_ctx = stock_web._run_returns_once({"ticker": ticker})
            if state.risk_ctx is None or getattr(state.risk_ctx, "ticker", "") != ticker:
                state.risk_ctx = stock_web._run_risk_once({"ticker": ticker})
            state.decision_ctx = stock_web._run_decision_once(
                state.decision_form,
                returns_ctx=state.returns_ctx,
                risk_ctx=state.risk_ctx,
                fin_ctx=_matching_financials_ctx(state, ticker),
            )
            state.decision_error = None
            _save_latest_page(user_id, "decision", state)
            return "decision"
        if action == "walk-forward":
            state.wfv_form = {**state.wfv_form, **form, "ticker": ticker}
            try:
                state.wfv_ctx = stock_web._run_walk_forward_validation_once(state.wfv_form)
            except ValueError as exc:
                if "Not enough usable rows" not in str(exc):
                    raise
                retry_form = {
                    **state.wfv_form,
                    "start_date": "",
                    "end_date": "",
                    "wf_min_train_rows": "80",
                }
                state.wfv_ctx = stock_web._run_walk_forward_validation_once(retry_form)
            state.wfv_error = None
            _save_latest_page(user_id, "walk-forward", state)
            return "walk-forward"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc(limit=3)}"
        target = {
            "forecast": ("forecast_error", "forecast"),
            "financials": ("financials_error", "financials"),
            "technical": ("technical_error", "technical"),
            "returns": ("returns_error", "returns"),
            "risk": ("risk_error", "risk"),
            "factor": ("factor_error", "factor-regime"),
            "decision": ("decision_error", "decision"),
            "walk-forward": ("wfv_error", "walk-forward"),
        }.get(action, ("forecast_error", "forecast"))
        setattr(state, target[0], error)
        return target[1]
    return "forecast"
