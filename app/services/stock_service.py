from __future__ import annotations

import re
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from math import isfinite

from bs4 import BeautifulSoup

from pipeline_stock import web_gui as stock_web

from app.web import add_start_page_link, rewrite_links


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
    forecast_form: dict[str, str] = field(default_factory=lambda: {
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
    })
    forecast_ctx: object | None = None
    forecast_error: str | None = None
    financials_form: dict[str, str] = field(default_factory=lambda: {"ticker": "AAPL", "statement_periods": "4", "output_dir": "outputs/stock_forecast_finance", "auto_save": "on", "insecure_ssl": "", "ca_bundle_path": "", "fmp_api_key": ""})
    financials_ctx: object | None = None
    financials_error: str | None = None
    technical_form: dict[str, str] = field(default_factory=lambda: {"ticker": "AAPL", "output_dir": "outputs/technical_analysis", "use_sample": "", "auto_save": "on", "action": "all"})
    technical_ctx: object | None = None
    technical_error: str | None = None
    technical_cache: object | None = None
    returns_form: dict[str, str] = field(default_factory=lambda: {"ticker": "AAPL"})
    returns_ctx: object | None = None
    returns_error: str | None = None
    risk_form: dict[str, str] = field(default_factory=lambda: {"ticker": "AAPL"})
    risk_ctx: object | None = None
    risk_error: str | None = None
    factor_form: dict[str, str] = field(default_factory=lambda: {"ticker": "AAPL"})
    factor_ctx: object | None = None
    factor_error: str | None = None
    decision_form: dict[str, str] = field(default_factory=lambda: {"ticker": "AAPL"})
    decision_ctx: object | None = None
    decision_error: str | None = None
    wfv_form: dict[str, str] = field(default_factory=lambda: {"ticker": "AAPL", "forecast_horizon": "10", "history_years": "8", "start_date": "2025-12-31", "end_date": datetime.utcnow().strftime("%Y-%m-%d"), "wf_min_train_rows": "252", "wf_step_size": "21", "wf_max_splits": "4", "output_dir": "outputs/walk_forward_validation", "prices_csv_path": "", "use_sample": "", "auto_save": "on", "insecure_ssl": "", "ca_bundle_path": ""})
    wfv_ctx: object | None = None
    wfv_error: str | None = None


state = StockState()


PAGE_SUBTITLES = {
    "forecast": "선택한 S&P 500 종목의 가격 예측을 실행합니다.",
    "financials": "재무제표, 밸류에이션 지표, 데이터 제공 상태를 확인합니다.",
    "technical": "이동평균, 캔들, RSI, MACD 기반 기술적 분석을 실행합니다.",
    "returns": "선택 종목의 수익률을 섹터 및 S&P 500 전체와 비교합니다.",
    "risk": "변동성, 낙폭, 베타, VaR, 상대 위험 순위를 측정합니다.",
    "factor-regime": "종목 움직임을 시장, 섹터, 고유 요인으로 분해합니다.",
    "decision": "수익률, 위험, 팩터, 추세 신호를 종합해 의사결정 점수를 계산합니다.",
    "walk-forward": "반복 과거 검증으로 예측 품질을 점검합니다.",
}

PAGE_RUN_LABELS = {
    "forecast": "예측 실행",
    "financials": "재무 분석 실행",
    "returns": "수익률 비교 실행",
    "risk": "위험 분석 실행",
    "factor-regime": "팩터/국면 분석 실행",
    "decision": "의사결정 점수 실행",
    "walk-forward": "워크포워드 검증 실행",
}

FIELD_LABELS = {
    "ticker": "티커",
    "forecast_horizon": "예측 기간",
    "history_years": "과거 데이터 연수",
    "start_date": "시작일",
    "end_date": "종료일",
    "output_dir": "출력 폴더",
    "prices_csv_path": "로컬 가격 CSV",
    "ca_bundle_path": "CA 번들 경로",
    "statement_periods": "재무제표 기간 수",
    "fmp_api_key": "FMP API 키",
    "wf_min_train_rows": "최소 학습 행 수",
    "wf_step_size": "분할 간격",
    "wf_max_splits": "최대 분할 수",
}

CHECKBOX_LABELS = {
    "use_sample": "샘플 가격 사용(오프라인)",
    "auto_save": "결과 자동 저장",
    "insecure_ssl": "SSL 검증 임시 비활성화",
}

ACTION_BUTTON_LABELS = {
    "ma": "이동평균",
    "candle": "캔들차트",
    "rsi": "RSI",
    "macd": "MACD",
    "all": "전체 실행",
}

PAGE_NOTICE = {
    "forecast": "티커를 입력하고 가격 예측을 실행하세요. 오프라인 점검이 필요하면 샘플 데이터를 사용할 수 있습니다.",
    "financials": "티커를 입력하고 재무 분석을 실행하세요. yfinance가 불안정하면 SEC/FMP/공유 데이터 대체 경로를 사용합니다.",
    "technical": "최신 OHLCV 데이터로 기술적 차트를 계산합니다. 오프라인 미리보기에는 샘플 데이터를 사용할 수 있습니다.",
    "returns": "선택 종목을 같은 섹터 및 S&P 500 전체와 비교합니다.",
    "risk": "변동성, 낙폭, 베타, 꼬리위험 지표를 기반으로 위험을 분석합니다.",
    "factor-regime": "시장, 섹터, 종목 고유 움직임을 분리해 현재 국면을 해석합니다.",
    "decision": "티커를 선택한 뒤 의사결정 점수를 계산합니다.",
    "walk-forward": "워크포워드 검증을 실행합니다. 기간이 짧으면 학습 구간을 자동으로 넓혀 계산합니다.",
}

PAGE_H3_LABELS = {
    "forecast": ["가격 예측", "모델 가중치", "데이터 출처", "예측 요약", "모델 점수", "방향성 점수", "국면 스냅샷", "특성 중요도"],
    "financials": ["데이터 출처", "제공자 상태", "재무 지표", "최신 재무 요약", "손익계산서", "재무상태표", "현금흐름표"],
    "technical": ["데이터 출처", "실행 요약"],
    "returns": ["YTD 기준 100 지수", "최근 일간 수익률 비교", "기간 수익률 비교", "최근 종목 일간 수익률", "최근 섹터 일간 수익률", "데이터 출처", "섹터 YTD 상위 10", "섹터 YTD 하위 10", "S&P 500 YTD 상위 10", "S&P 500 YTD 하위 10"],
    "risk": ["위험 해석", "1년 낙폭 비교", "20일 이동 연율화 변동성", "변동성 및 낙폭 요약", "최근 충격 점검", "데이터 출처", "섹터 내 1년 변동성 상위", "섹터 내 1년 변동성 하위", "S&P 500 1년 변동성 상위", "S&P 500 1년 변동성 하위"],
    "factor-regime": ["팩터/국면 읽는 법", "60일 이동 베타", "누적 잔차 수익률", "팩터 요약", "해석 가이드", "최근 팩터 분해", "데이터 출처"],
    "decision": ["최종 판단", "의사결정 점수 분해", "추세 및 변동성 맥락", "긍정 요인", "부정 요인", "관찰 항목", "점수표", "신호 상세", "데이터 출처"],
    "walk-forward": ["워크포워드 검증 읽는 법", "예측 수익률 vs 실현 수익률", "오차 및 이동 적중률", "검증 요약", "해석 가이드", "거래 제외 기준 요약", "국면 요약", "데이터 출처", "모델 진단", "분할 결과"],
}

PAGE_METRIC_LABELS = {
    "forecast": ["티커", "기준일", "예측일", "예측 기간", "최근 종가", "예측 가격", "기대 수익률", "상승 확률", "방향 신뢰도", "신호", "거래 필터", "앙상블 로그수익률"],
    "financials": ["티커", "회사명", "통화", "PER(과거)", "PER(예상)", "PBR", "시가총액", "ROE"],
    "technical": ["티커", "데이터 출처", "행 수", "기간", "실행 항목", "조회 기준"],
    "walk-forward": ["티커", "가격 출처", "검증 분할 수", "예측 기간", "방향 적중률", "분류 적중률", "거래 커버리지", "거래 적중률", "MAE", "RMSE", "단순 기준 대비 개선", "편향", "수익률 상관", "최신 기준일", "최신 실현일"],
}

MOJIBAKE_MARKERS = ("?곗", "?섏", "?쒖", "?덉", "?뚯", "媛", "醫", "理", "由", "蹂", "寃", "遺", "湲", "嫄")

PAGE_TEXT_FALLBACK = {
    "forecast": "예측 결과는 선택 종목의 가격 이력과 모델 앙상블을 기반으로 계산됩니다.",
    "financials": "재무 결과는 사용 가능한 제공자 데이터, 대체 데이터, 파생 지표를 함께 반영합니다.",
    "technical": "기술적 분석 결과는 가격 추세, 모멘텀, 차트 진단을 요약합니다.",
    "returns": "수익률 결과는 종목을 같은 섹터 및 S&P 500 전체와 비교합니다.",
    "risk": "위험 결과는 공유 가격 데이터 기반 변동성, 낙폭, 베타, 꼬리위험을 요약합니다.",
    "factor-regime": "팩터/국면 결과는 시장, 섹터, 종목 고유 움직임을 분리해 현재 환경을 요약합니다.",
    "decision": "의사결정 결과는 긍정/부정 요인, 위험, 추세 신호를 하나의 대시보드로 종합합니다.",
    "walk-forward": "워크포워드 결과는 반복 과거 검증으로 예측 품질을 보여줍니다.",
}

PAGE_TITLES = {
    "forecast": "종목 분석 | 가격 예측",
    "financials": "종목 분석 | 재무",
    "technical": "종목 분석 | 기술적 분석",
    "returns": "종목 분석 | 수익률",
    "risk": "종목 분석 | 위험",
    "factor-regime": "종목 분석 | 팩터/국면",
    "decision": "종목 분석 | 의사결정",
    "walk-forward": "종목 분석 | 워크포워드",
}

STOCK_NAV_LABELS = {
    "/stock/forecast": "가격 예측",
    "/stock/financials": "재무",
    "/stock/technical": "기술적 분석",
    "/stock/returns": "수익률",
    "/stock/risk": "위험",
    "/stock/factor-regime": "팩터/국면",
    "/stock/decision": "의사결정",
    "/stock/walk-forward": "워크포워드",
}

REGIME_FALLBACKS = {
    "Trend Regime": "Neutral trend",
    "Volatility Regime": "Normal volatility",
    "Beta Regime": "Market-like beta",
    "Overall Regime": "Mixed regime",
}

FACTOR_EXPLANATION = {
    "intro": "이 페이지는 선택한 종목의 일간 수익률을 S&P 500 및 소속 섹터 수익률과 비교해 베타, 상관계수, 잔차수익률, 현재 레짐을 함께 보여줍니다.",
    "items": [
        "베타는 시장 또는 섹터가 1 움직일 때 종목이 평균적으로 얼마나 민감하게 반응하는지 보여줍니다.",
        "상관계수는 함께 움직이는 정도를, 잔차수익률은 공통 요인을 제외한 종목 고유 흐름을 나타냅니다.",
        "추세, 변동성, 베타 레짐은 현재 시장 환경에서 종목을 어떻게 해석할지 빠르게 요약합니다.",
    ],
}

FACTOR_GUIDE_ROWS = [
    ("베타 (beta)", "최근 60거래일 동안 시장 또는 섹터가 1 움직일 때 종목이 평균적으로 얼마나 민감하게 반응하는지 봅니다."),
    ("상관계수 (correlation)", "같은 방향으로 함께 움직이는 정도입니다. 높을수록 공통 요인의 영향이 큽니다."),
    ("잔차수익률 (residual return)", "시장 또는 섹터 설명분을 제외하고 남는 종목 고유 수익률입니다. 플러스면 상대 초과성과, 마이너스면 상대 열위로 해석할 수 있습니다."),
    ("추세 레짐 (trend regime)", "20, 60, 120일 이동평균과 현재 가격의 상대 위치로 방향성을 요약합니다."),
    ("변동성 레짐 (volatility regime)", "현재 변동성이 평소보다 높은지, 낮은지, 중립적인지 보여줍니다."),
    ("종합 레짐 (overall regime)", "추세, 변동성, 베타를 함께 묶어 현재 환경을 읽기 쉬운 문장으로 정리한 스냅샷입니다."),
]

FACTOR_REGIME_LABELS = {
    "uptrend": "상승 추세 (uptrend)",
    "bull trend": "강한 상승 추세 (bull trend)",
    "early upturn": "상승 전환 시도 (early upturn)",
    "downtrend": "하락 추세 (downtrend)",
    "bear trend": "강한 하락 추세 (bear trend)",
    "downtrend pressure": "하락 압력 지속 (downtrend pressure)",
    "mixed trend": "혼합 추세 (mixed trend)",
    "high volatility": "고변동성 (high volatility)",
    "calm volatility": "저변동성 (calm volatility)",
    "normal volatility": "중립 변동성 (normal volatility)",
    "high beta": "공격적 베타 (high beta)",
    "defensive beta": "방어적 베타 (defensive beta)",
    "market-like beta": "시장유사 베타 (market-like beta)",
    "high-beta rally": "공격적 상승 국면 (high-beta rally)",
    "high-beta trend": "고베타 추세 국면 (high-beta trend)",
    "stable uptrend": "안정 상승 국면 (stable uptrend)",
    "risk-off stress": "스트레스 국면 (risk-off stress)",
    "defensive calm": "방어 안정 국면 (defensive calm)",
    "mixed regime": "혼합 국면 (mixed regime)",
    "insufficient history": "판단 보류 (insufficient history)",
}

WALK_FORWARD_EXPLANATION = {
    "intro": "각 분할은 특정 기준일까지만 이용 가능한 과거 데이터로 모델을 다시 학습한 뒤, 이후 영업일의 실제 수익률과 예측값을 비교합니다.",
    "items": [
        "방향 적중률은 가격 예측 회귀 모델이 상승과 하락 방향을 얼마나 잘 맞혔는지 보여줍니다.",
        "분류 적중률은 별도 방향 분류 모델의 상승/하락 판단 성능을 평가합니다.",
        "거래 커버리지와 거래 적중률은 no-trade 필터를 통과한 강한 신호만 봤을 때 결과가 좋아지는지 확인합니다.",
        "MAE와 RMSE는 예측 선행수익률과 실제 선행수익률 사이의 오차 크기를 측정합니다.",
        "Skill vs Naive는 0% 수익률을 가정하는 단순 기준선보다 모델 오차가 얼마나 개선됐는지 비교합니다.",
        "Bias는 모델이 구조적으로 너무 낙관적이거나 보수적인지 보여줍니다.",
        "Regime Summary는 어떤 시장 환경에서 예측이 잘 맞거나 약했는지 요약합니다.",
    ],
}

DECISION_REASON_COPY = {
    "Bullish Reasons": [
        "가격이 주요 이동평균과 모멘텀 지표 기준에서 우호적인 흐름을 보이고 있습니다.",
        "단기 추세 신호가 아직 훼손되지 않아 상승 시나리오를 유지할 근거가 있습니다.",
        "기술적 지표와 상대 강도 신호를 함께 보면 매수 우위 요소가 일부 확인됩니다.",
    ],
    "Bearish Reasons": [
        "최근 상대수익률이 섹터 또는 S&P 500 대비 약한 구간이 있어 주의가 필요합니다.",
        "가격이 단기 과열권에 가까워질 경우 되돌림 가능성을 함께 봐야 합니다.",
        "리스크와 수익률 신호가 완전히 한 방향으로 정렬되지는 않았습니다.",
    ],
    "Watch Items": [
        "실적과 밸류에이션 데이터가 부족한 경우 가격, 기술적 지표, 리스크 신호 중심으로 판단합니다.",
        "다음 리밸런싱 전까지 섹터 대비 상대수익률과 변동성 변화를 확인하세요.",
        "주요 이동평균 이탈이나 MACD 약화가 나타나면 점수를 다시 점검하는 것이 좋습니다.",
    ],
}

BROKEN_CLOSE_TAG_RE = re.compile(
    r"(?<!<)/(?P<tag>title|h[1-6]|span|strong|b|em|p|li|a)>",
    flags=re.I,
)


def _repair_broken_stock_markup(html: str) -> str:
    """Repair mojibake-damaged closing tags before BeautifulSoup reparents nodes."""
    return BROKEN_CLOSE_TAG_RE.sub(r"</\g<tag>>", html)


def _has_mojibake(text: str) -> bool:
    return any(marker in text for marker in MOJIBAKE_MARKERS)


def _fmt_pct(value: object, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not isfinite(number):
        return "-"
    return f"{number * 100.0:.{digits}f}%"


def _fmt_number(value: object, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not isfinite(number):
        return "-"
    return f"{number:.{digits}f}"


def _replace_tag_text(tag, text: str) -> None:
    tag.clear()
    tag.append(text)


def _normalized_factor_regime(value: object) -> str:
    text = str(value or "")
    lower = text.lower()
    for key, label in FACTOR_REGIME_LABELS.items():
        if key in lower:
            return label
    if "low volatility" in lower:
        return FACTOR_REGIME_LABELS["calm volatility"]
    if "beta trend" in lower:
        return "고베타 추세 국면 (high-beta trend)"
    return "판단 보류" if _has_mojibake(text) else text


def _replace_table_with_rows(soup: BeautifulSoup, table, headers: list[str], rows: list[tuple[str, str]]) -> None:
    table.clear()
    thead = soup.new_tag("thead")
    head_row = soup.new_tag("tr")
    for header in headers:
        th = soup.new_tag("th")
        th.string = header
        head_row.append(th)
    thead.append(head_row)
    table.append(thead)

    tbody = soup.new_tag("tbody")
    for first, second in rows:
        tr = soup.new_tag("tr")
        td_first = soup.new_tag("td")
        td_first.string = first
        td_second = soup.new_tag("td")
        td_second.string = second
        tr.append(td_first)
        tr.append(td_second)
        tbody.append(tr)
    table.append(tbody)


def _clean_factor_tables(soup: BeautifulSoup) -> None:
    factor_summary_titles = {"Factor Summary", "팩터 요약"}
    guide_titles = {"Interpretation Guide", "해석 가이드"}

    for metric in soup.select(".metric"):
        span = metric.find("span")
        strong = metric.find("strong")
        label = span.get_text(" ", strip=True) if span else ""
        if strong is not None and (label in REGIME_FALLBACKS or "Regime" in label or "레짐" in label):
            value = strong.get_text(" ", strip=True)
            strong.string = _normalized_factor_regime(value)

    for table in soup.find_all("table"):
        heading = table.find_previous("h3")
        title = heading.get_text(" ", strip=True) if heading else ""
        if title in factor_summary_titles:
            for row in table.find_all("tr"):
                cells = row.find_all(["th", "td"])
                if len(cells) < 2:
                    continue
                metric = cells[0].get_text(" ", strip=True)
                if metric in REGIME_FALLBACKS or "Regime" in metric:
                    cells[1].string = _normalized_factor_regime(cells[1].get_text(" ", strip=True))
                for cell in cells[1:]:
                    text = cell.get_text(" ", strip=True)
                    normalized = _normalized_factor_regime(text)
                    if normalized != text:
                        cell.string = normalized
        elif title in guide_titles:
            _replace_table_with_rows(soup, table, ["개념", "이 페이지에서 보는 의미"], FACTOR_GUIDE_ROWS)


def _replace_label_text(label, text: str) -> None:
    input_tag = label.find("input")
    if input_tag is not None and input_tag.parent is label:
        input_tag.extract()
        label.clear()
        label.append(input_tag)
        label.append(" " + text)
        return
    label.clear()
    label.append(text)


def _risk_commentary_text() -> str:
    ctx = state.risk_ctx
    if ctx is None:
        return "공유 가격 데이터와 시가총액 데이터를 이용해 변동성, 낙폭, 베타, VaR를 함께 점검합니다."
    sector_rank = "-" if ctx.sector_vol_rank_1y is None else f"{ctx.sector_vol_rank_1y:,d}/{ctx.sector_count:,d}"
    market_rank = "-" if ctx.market_vol_rank_1y is None else f"{ctx.market_vol_rank_1y:,d}/{ctx.market_count:,d}"
    return (
        f"{ctx.ticker}의 1년 연율화 변동성은 {_fmt_pct(ctx.ticker_vol_252d)}, "
        f"1년 최대 낙폭은 {_fmt_pct(ctx.ticker_max_drawdown_1y)}입니다. "
        f"섹터 대비 베타는 {_fmt_number(ctx.beta_sector_1y)}, S&P 500 대비 베타는 {_fmt_number(ctx.beta_market_1y)}이며, "
        f"95% 기준 1일 VaR는 {_fmt_pct(ctx.var_95_1d)}입니다. "
        f"변동성 순위는 섹터 {sector_rank}, S&P 500 {market_rank} 수준으로 함께 확인하세요."
    )


def _factor_commentary_text() -> str:
    ctx = state.factor_ctx
    if ctx is None:
        return "시장과 섹터 요인을 분리해 선택 종목의 고유 흐름과 현재 레짐을 해석합니다."
    overall_regime = _normalized_factor_regime(ctx.regime_overall)
    return (
        f"{ctx.ticker}의 최근 60거래일 베타는 S&P 500 대비 {_fmt_number(ctx.beta_market_60d)}, "
        f"섹터 대비 {_fmt_number(ctx.beta_sector_60d)}입니다. "
        f"20거래일 잔차수익률은 시장 대비 {_fmt_pct(ctx.residual_market_20d)}, "
        f"섹터 대비 {_fmt_pct(ctx.residual_sector_20d)}이며, 현재 종합 레짐은 {overall_regime}입니다."
    )


def _decision_commentary_text() -> str:
    ctx = state.decision_ctx
    if ctx is None:
        return "추세, 상대수익률, 리스크, 기술적 신호를 합산해 종합 판단을 만듭니다."
    recommendation = _decision_recommendation_label(ctx.total_score)
    confidence = _decision_confidence_label(ctx.total_score)
    return (
        f"{ctx.ticker}의 종합 점수는 {ctx.total_score:+.2f}이며 현재 판단은 {recommendation}, "
        f"신뢰도는 {confidence}입니다. 추세, 상대수익률, 변동성, 기술적 신호를 함께 반영한 결과입니다."
    )


def _walk_forward_commentary_text() -> str:
    ctx = state.wfv_ctx
    if ctx is None:
        return "과거 여러 기준일에서 같은 예측 구조를 반복 검증해 모델의 방향성, 오차, 거래 필터 품질을 확인합니다."
    return (
        f"{ctx.ticker}의 워크포워드 검증은 {ctx.evaluation_splits}개 분할과 {ctx.horizon_days}영업일 예측기간으로 계산됐습니다. "
        f"방향 적중률은 {_fmt_pct(ctx.direction_hit_rate)}, 분류 적중률은 {_fmt_pct(ctx.classification_hit_rate)}, "
        f"거래 커버리지는 {_fmt_pct(ctx.trade_coverage_rate)}, 거래 적중률은 {_fmt_pct(ctx.trade_hit_rate)}입니다. "
        f"MAE는 {_fmt_pct(ctx.mae_return)}, RMSE는 {_fmt_pct(ctx.rmse_return)}, "
        f"수익률 상관계수는 {_fmt_number(ctx.return_correlation)}입니다."
    )


def _decision_recommendation_label(total_score: float) -> str:
    if total_score >= 4.5:
        return "매수 우위"
    if total_score >= 2.0:
        return "약한 매수 우위"
    if total_score <= -4.5:
        return "매도/차익실현 우위"
    if total_score <= -2.0:
        return "약한 매도 우위"
    return "Neutral / Watch"


def _decision_confidence_label(total_score: float) -> str:
    abs_score = abs(float(total_score))
    if abs_score >= 5.5:
        return "높음"
    if abs_score >= 3.0:
        return "보통"
    return "낮음"


def _clean_decision_labels(soup: BeautifulSoup) -> None:
    ctx = state.decision_ctx
    if ctx is None:
        return
    label_values = {
        "Recommendation": _decision_recommendation_label(ctx.total_score),
        "Confidence": _decision_confidence_label(ctx.total_score),
    }
    metadata_values = {
        "recommendation": label_values["Recommendation"],
        "confidence": label_values["Confidence"],
    }

    for metric in soup.select(".metric"):
        span = metric.find("span")
        strong = metric.find("strong")
        label = span.get_text(" ", strip=True) if span else ""
        if strong is not None and label in label_values:
            strong.string = label_values[label]

    for row in soup.find_all("tr"):
        cells = row.find_all(["th", "td"])
        if len(cells) < 2:
            continue
        key = cells[0].get_text(" ", strip=True)
        if key in metadata_values:
            cells[1].string = metadata_values[key]


def _replace_bad_commentary(page: str, soup: BeautifulSoup) -> None:
    if page == "risk":
        for tag in soup.find_all("p"):
            if _has_mojibake(tag.get_text(" ", strip=True)):
                _replace_tag_text(tag, _risk_commentary_text())
        return

    if page == "factor-regime":
        bad_paragraphs = [tag for tag in soup.find_all("p") if _has_mojibake(tag.get_text(" ", strip=True))]
        if bad_paragraphs:
            _replace_tag_text(bad_paragraphs[0], FACTOR_EXPLANATION["intro"])
            for tag in bad_paragraphs[1:]:
                _replace_tag_text(tag, _factor_commentary_text())
        bad_items = [tag for tag in soup.find_all("li") if _has_mojibake(tag.get_text(" ", strip=True))]
        for tag, text in zip(bad_items, FACTOR_EXPLANATION["items"], strict=False):
            _replace_tag_text(tag, text)
        _clean_factor_tables(soup)
        return

    if page == "decision":
        _clean_decision_labels(soup)
        reason_copy_by_title = {
            "Bullish Reasons": DECISION_REASON_COPY["Bullish Reasons"],
            "긍정 요인": DECISION_REASON_COPY["Bullish Reasons"],
            "Bearish Reasons": DECISION_REASON_COPY["Bearish Reasons"],
            "부정 요인": DECISION_REASON_COPY["Bearish Reasons"],
            "Watch Items": DECISION_REASON_COPY["Watch Items"],
            "관찰 항목": DECISION_REASON_COPY["Watch Items"],
        }
        for card in soup.select(".card"):
            heading = card.find("h3")
            title = heading.get_text(" ", strip=True) if heading else ""
            if title in {"Final Decision", "최종 판단"}:
                paragraph = card.find("p")
                if paragraph is not None:
                    _replace_tag_text(paragraph, _decision_commentary_text())
            if title in reason_copy_by_title:
                ul = card.find("ul")
                if ul is not None:
                    ul.clear()
                    for text in reason_copy_by_title[title]:
                        li = soup.new_tag("li")
                        li.string = text
                        ul.append(li)
        return

    if page == "walk-forward":
        bad_paragraphs = [tag for tag in soup.find_all("p") if _has_mojibake(tag.get_text(" ", strip=True))]
        if bad_paragraphs:
            _replace_tag_text(bad_paragraphs[0], WALK_FORWARD_EXPLANATION["intro"])
            for tag in bad_paragraphs[1:]:
                _replace_tag_text(tag, _walk_forward_commentary_text())
        bad_items = [tag for tag in soup.find_all("li") if _has_mojibake(tag.get_text(" ", strip=True))]
        for tag, text in zip(bad_items, WALK_FORWARD_EXPLANATION["items"], strict=False):
            _replace_tag_text(tag, text)


def _make_stock_text_readable(page: str, html: str) -> str:
    html = _repair_broken_stock_markup(html)
    soup = BeautifulSoup(html, "html.parser")
    if soup.title is not None:
        soup.title.string = PAGE_TITLES.get(page, "종목 분석")

    for link in soup.find_all("a"):
        href = link.get("href", "")
        if href in STOCK_NAV_LABELS:
            link.string = STOCK_NAV_LABELS[href]

    sub = soup.select_one(".sub")
    if sub is not None:
        sub.string = PAGE_SUBTITLES.get(page, page)

    for label in soup.find_all("label"):
        nested_input = label.find("input")
        if nested_input is not None:
            name = nested_input.get("name", "")
            if name in CHECKBOX_LABELS:
                _replace_label_text(label, CHECKBOX_LABELS[name])
            continue
        parent = label.parent
        input_tag = parent.find(["input", "select"]) if parent else None
        name = input_tag.get("name", "") if input_tag else ""
        if name in FIELD_LABELS:
            _replace_label_text(label, FIELD_LABELS[name])

    for button in soup.find_all("button"):
        value = button.get("value", "")
        name = button.get("name", "")
        if name == "action" and value in ACTION_BUTTON_LABELS:
            button.string = ACTION_BUTTON_LABELS[value]
        elif value == "resolve_ticker":
            button.string = "회사명으로 티커 찾기"
        elif value == "run" or button.get("type") == "submit":
            button.string = PAGE_RUN_LABELS.get(page, "분석 실행")

    for notice in soup.select(".notice.ok"):
        code = notice.find("code")
        if code is not None and _has_mojibake(notice.get_text(" ", strip=True)):
            code.extract()
            notice.clear()
            notice.append("결과가 ")
            notice.append(code)
            notice.append("에 저장되었습니다.")
        elif notice.find(["pre", "code", "table"]) is None:
            notice.clear()
            notice.append(PAGE_NOTICE.get(page, "준비되었습니다."))

    for span, text in zip(soup.select(".metric span"), PAGE_METRIC_LABELS.get(page, []), strict=False):
        span.string = text

    for metric in soup.select(".metric"):
        span = metric.find("span")
        strong = metric.find("strong")
        if span is not None and strong is not None:
            label = span.get_text(" ", strip=True)
            value = strong.get_text(" ", strip=True)
            if label in REGIME_FALLBACKS and _has_mojibake(value):
                strong.string = REGIME_FALLBACKS[label]

    for h3, text in zip(soup.find_all("h3"), PAGE_H3_LABELS.get(page, []), strict=False):
        h3.string = text

    for h4 in soup.find_all("h4"):
        text = h4.get_text(" ", strip=True)
        if _has_mojibake(text):
            h4.string = "결과 표"

    _replace_bad_commentary(page, soup)

    return str(soup)


def _clean_stock_html(page: str, html: str) -> str:
    html = _repair_broken_stock_markup(html)
    html = rewrite_links(html, STOCK_REWRITES)
    html = re.sub(r"<title>.*?</title>", f"<title>{PAGE_TITLES.get(page, '종목 분석')}</title>", html, count=1, flags=re.S)
    html = re.sub(
        r'<div class="page-head">.*?</div>\s*</div>',
        '<div class="page-head"><h1>종목 분석 | S&P 500</h1><div class="page-credit">Keumj 서비스</div></div>',
        html,
        count=1,
        flags=re.S,
    )
    html = re.sub(
        r'<div class="sub">.*?</div>',
        f'<div class="sub">{page}</div>',
        html,
        count=1,
        flags=re.S,
    )
    html = add_start_page_link(html)
    return _make_stock_text_readable(page, html)


def render(page: str, ticker: str | None = None, intent: str | None = None) -> str:
    selected_ticker = _clean_ticker(ticker or "")
    if selected_ticker:
        _sync_ticker(selected_ticker)

    if page == "forecast":
        html = stock_web._html_page(
            state.forecast_form,
            ctx=state.forecast_ctx,
            error=state.forecast_error,
            enable_technical_page=True,
        )
    elif page == "financials":
        html = stock_web._html_financial_page(state.financials_form, ctx=state.financials_ctx, error=state.financials_error, enable_technical_page=True)
    elif page == "technical":
        html = stock_web._html_technical_page(state.technical_form, ctx=state.technical_ctx, error=state.technical_error)
    elif page == "returns":
        html = stock_web._html_returns_page(state.returns_form, ctx=state.returns_ctx, error=state.returns_error)
    elif page == "risk":
        html = stock_web._html_risk_page(state.risk_form, ctx=state.risk_ctx, error=state.risk_error)
    elif page == "factor-regime":
        html = stock_web._html_factor_page(state.factor_form, ctx=state.factor_ctx, error=state.factor_error)
    elif page == "decision":
        html = stock_web._html_decision_page(state.decision_form, ctx=state.decision_ctx, error=state.decision_error)
    elif page == "walk-forward":
        html = stock_web._html_walk_forward_page(state.wfv_form, ctx=state.wfv_ctx, error=state.wfv_error)
    else:
        html = stock_web._html_page(state.forecast_form, ctx=state.forecast_ctx, error=state.forecast_error, enable_technical_page=True)
    return _clean_stock_html(page, html)


def _clean_ticker(value: str) -> str:
    raw = str(value or "").strip().upper()
    raw = re.split(r"[?&#\s]", raw, maxsplit=1)[0]
    return re.sub(r"[^A-Z0-9.\-]", "", raw)


def _sync_ticker(ticker: str) -> None:
    if not ticker:
        return
    for form in [state.forecast_form, state.financials_form, state.technical_form, state.returns_form, state.risk_form, state.factor_form, state.decision_form, state.wfv_form]:
        form["ticker"] = ticker


def _matching_financials_ctx(ticker: str) -> object | None:
    fin_ctx = state.financials_ctx
    if fin_ctx is None:
        return None
    fin_ticker = str(getattr(fin_ctx, "ticker", "")).strip().upper()
    return fin_ctx if fin_ticker == ticker else None


def run(action: str, form: dict[str, str]) -> str:
    try:
        ticker = _clean_ticker(form.get("ticker", ""))
        _sync_ticker(ticker)
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
            return "forecast"
        if action == "financials":
            state.financials_form = {**state.financials_form, **form, "ticker": ticker}
            state.financials_ctx = stock_web._run_financial_once(state.financials_form)
            state.financials_error = None
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
            return "technical"
        if action == "returns":
            state.returns_form = {"ticker": ticker}
            state.returns_ctx = stock_web._run_returns_once(state.returns_form)
            state.returns_error = None
            return "returns"
        if action == "risk":
            state.risk_form = {"ticker": ticker}
            state.risk_ctx = stock_web._run_risk_once(state.risk_form)
            state.risk_error = None
            return "risk"
        if action == "factor":
            state.factor_form = {"ticker": ticker}
            state.factor_ctx = stock_web._run_factor_once(state.factor_form)
            state.factor_error = None
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
                fin_ctx=_matching_financials_ctx(ticker),
            )
            state.decision_error = None
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
