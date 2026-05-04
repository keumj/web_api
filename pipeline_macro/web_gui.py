from __future__ import annotations

import base64
import html
import io

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pipeline_common.notebook_models import select_quarter_snapshots

from .analysis import MacroDashboard, build_macro_dashboard


YIELD_CURVE_SERIES_IDS = ["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS20", "DGS30"]
YIELD_CURVE_MATURITIES = np.array([0.08, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30], dtype=float)
YIELD_CURVE_LABELS = ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]


PAGES: dict[str, tuple[str, str]] = {
    "overview": ("개요", "거시 국면, 핵심 점수, 주요 지표를 한 화면에서 봅니다."),
    "regime": ("레짐", "성장, 원자재, 정책, 위험선호 축으로 현재 시장 환경을 분류합니다."),
    "rates": ("금리/커브", "2년, 10년, 30년 금리와 장단기 스프레드를 점검합니다."),
    "risk": ("위험자산", "S&P 500 수익률, 변동성, 낙폭을 기준으로 위험선호를 읽습니다."),
    "dollar": ("달러/원자재", "달러와 원유, 금, 은, 구리 흐름을 함께 봅니다."),
    "playbook": ("섹터 플레이북", "현재 거시 환경에서 업종별 민감도와 선호도를 정리합니다."),
}


def normalize_page(page: str | None) -> str:
    key = str(page or "overview").strip().lower()
    return key if key in PAGES else "overview"


def _fmt(value: object, ndigits: int = 2) -> str:
    try:
        if pd.isna(value):
            return "-"
    except Exception:
        pass
    if isinstance(value, (int, float, np.floating)):
        return f"{float(value):,.{ndigits}f}"
    return html.escape(str(value))


def _table(frame: pd.DataFrame, *, max_rows: int = 80) -> str:
    if frame is None or frame.empty:
        return "<p class='service-muted'>표시할 데이터가 없습니다.</p>"
    show = frame.head(max_rows).copy()
    for col in show.columns:
        if pd.api.types.is_numeric_dtype(show[col]):
            show[col] = show[col].map(lambda x: "-" if pd.isna(x) else f"{float(x):,.2f}")
    return f"<div class='service-table-wrap'>{show.to_html(index=False, border=0, classes='service-table')}</div>"


def _macro_nav(active: str) -> str:
    links = []
    for key, (label, _) in PAGES.items():
        cls = "active" if key == active else ""
        links.append(f'<a class="{cls}" href="/macro/{key}">{html.escape(label)}</a>')
    return '<div class="macro-nav">' + "".join(links) + "</div>"


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if clean.empty:
        return clean
    out = clean.copy()
    for col in out.columns:
        series = out[col].dropna()
        if series.empty or float(series.iloc[0]) == 0:
            continue
        out[col] = out[col] / float(series.iloc[0]) * 100.0
    return out


def _chart_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _line_chart(frame: pd.DataFrame, title: str, *, ylabel: str = "", normalize: bool = False, tail: int = 252) -> str:
    data = _normalize(frame) if normalize else frame.copy()
    data = data.tail(tail).apply(pd.to_numeric, errors="coerce").dropna(how="all")
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    if data.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        for col in data.columns:
            series = data[col].dropna()
            if not series.empty:
                ax.plot(series.index, series.values, linewidth=1.8, label=str(col))
        ax.legend(loc="best", fontsize=8, frameon=False)
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="x", labelrotation=20, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    return _chart_to_base64(fig)


def _bar_chart(labels: list[str], values: list[float], title: str, *, ylabel: str = "") -> str:
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    vals = [0.0 if not np.isfinite(v) else float(v) for v in values]
    colors = ["#0f766e" if v >= 0 else "#a12626" for v in vals]
    ax.bar(labels, vals, color=colors, alpha=0.88)
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelrotation=15, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    return _chart_to_base64(fig)


def _yield_curve_chart(frame: pd.DataFrame, title: str) -> str:
    data = frame.reindex(columns=YIELD_CURVE_SERIES_IDS).apply(pd.to_numeric, errors="coerce").dropna(how="all")
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    clean = data.ffill().dropna()
    if clean.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        selected = select_quarter_snapshots(clean, n_quarters=5)
        for d in selected.index:
            values = selected.loc[d].values.astype(float)
            ax.plot(YIELD_CURVE_MATURITIES, values, marker="o", linewidth=1.7, markersize=3.5, label=d.strftime("%Y-%m-%d"))
        ax.legend(loc="best", fontsize=7, frameon=False, ncol=2)
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlabel("Maturity")
    ax.set_ylabel("Yield (%)")
    ax.set_xticks(YIELD_CURVE_MATURITIES)
    ax.set_xticklabels(YIELD_CURVE_LABELS, rotation=0, fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="y", labelsize=8)
    return _chart_to_base64(fig)


def _score_bars(scores: pd.DataFrame) -> str:
    rows: list[str] = []
    for _, row in scores.iterrows():
        label = html.escape(str(row.get("점수", "")))
        value = row.get("값")
        pct = 0.0 if pd.isna(value) else max(0.0, min(float(value), 100.0))
        rows.append(
            f"""
            <div class="macro-score-row">
              <div class="macro-score-label">{label}</div>
              <div class="macro-score-track"><span style="width:{pct:.1f}%"></span></div>
              <div class="macro-score-value">{pct:.1f}</div>
            </div>
            """
        )
    return "<div class='macro-score-grid'>" + "".join(rows) + "</div>"


def _hero(dashboard: MacroDashboard) -> str:
    return f"""
    <div class="macro-hero">
      <div>
        <h1>거시분석 | S&P 500</h1>
        <p>기준일 {html.escape(dashboard.as_of_date)}. 금리, 달러, 원자재, S&P 500 위험선호를 묶어 종목/뉴스/포트폴리오 분석의 배경 환경을 정리합니다.</p>
      </div>
      <div class="macro-metrics">
        <div><span>리스크</span><strong>{html.escape(dashboard.risk_level)}</strong></div>
        <div><span>주식 의견</span><strong>{html.escape(dashboard.equity_bias)}</strong></div>
      </div>
    </div>
    """


def _chart_card(title: str, image: str) -> str:
    return f'<section class="service-card macro-chart-card"><h2>{html.escape(title)}</h2><img src="data:image/png;base64,{image}" alt="{html.escape(title)} chart" /></section>'


def _page_charts(page: str, dashboard: MacroDashboard) -> str:
    scores = dashboard.scores.set_index("점수")["값"]
    score_label_map = {
        "성장 모멘텀": "Growth",
        "인플레/원자재 압력": "Inflation",
        "정책 긴축도": "Policy",
        "위험선호": "Risk Appetite",
        "침체 리스크": "Recession",
        "유동성": "Liquidity",
    }
    plot_scores = scores.rename(index=score_label_map)
    if page == "overview":
        charts = [
            ("위험자산, 달러, 금 흐름", _line_chart(dashboard.market_series, "S&P 500 / DXY / Gold Base 100", normalize=True)),
            ("거시 점수", _bar_chart(plot_scores.index.astype(str).tolist(), plot_scores.astype(float).tolist(), "Macro Scores", ylabel="score")),
        ]
    elif page == "regime":
        charts = [
            ("레짐 점수 분해", _bar_chart(plot_scores.index.astype(str).tolist(), plot_scores.astype(float).tolist(), "Regime Score Breakdown", ylabel="score")),
            ("S&P 500과 10Y-2Y 커브", _line_chart(pd.concat([_normalize(dashboard.market_series[["S&P 500"]]), dashboard.rate_series[["10Y-2Y"]]], axis=1), "Equity Trend and Yield Curve")),
        ]
    elif page == "rates":
        charts = [
            ("미국 국채 일드커브", _yield_curve_chart(dashboard.yield_curve_series, "Treasury Yield Curves (Quarter-end + Latest)")),
            ("만기간 스프레드", _line_chart(dashboard.rate_series[["10Y-3M", "10Y-2Y", "5Y-2Y", "30Y-10Y"]], "Maturity Spreads", ylabel="%p")),
        ]
    elif page == "risk":
        risk = dashboard.risk_series
        charts = [
            ("S&P 500과 낙폭", _line_chart(pd.concat([_normalize(risk[["S&P 500"]]), risk[["Drawdown"]]], axis=1), "S&P 500 Base 100 and Drawdown")),
            ("20D 연율 변동성", _line_chart(risk[["20D Ann Vol"]], "Rolling 20D Annualized Volatility", ylabel="annual %")),
        ]
    elif page == "dollar":
        comm = dashboard.commodity_series
        returns_60d = [float((comm[c].dropna().iloc[-1] / comm[c].dropna().iloc[-61] - 1.0) * 100.0) if len(comm[c].dropna()) > 60 else np.nan for c in comm.columns]
        charts = [
            ("달러와 원자재", _line_chart(pd.concat([dashboard.market_series[["DXY"]], comm], axis=1), "DXY and Commodities Base 100", normalize=True)),
            ("원자재 60D 수익률", _bar_chart(comm.columns.astype(str).tolist(), returns_60d, "Commodity 60D Returns", ylabel="%")),
        ]
    else:
        playbook = dashboard.sector_playbook
        charts = [
            ("섹터 선호 점수", _bar_chart(playbook["섹터"].astype(str).tolist(), playbook["선호 점수"].astype(float).tolist(), "Sector Preference Scores", ylabel="score")),
            ("거시 점수", _bar_chart(plot_scores.index.astype(str).tolist(), plot_scores.astype(float).tolist(), "Macro Scores", ylabel="score")),
        ]
    return '<div class="macro-grid two macro-chart-grid">' + "".join(_chart_card(title, image) for title, image in charts) + "</div>"


def _overview_page(dashboard: MacroDashboard) -> str:
    return f"""
    {_hero(dashboard)}
    {_macro_nav("overview")}
    {_page_charts("overview", dashboard)}
    <div class="macro-grid two">
      <section class="service-card"><h2>핵심 요약</h2>{_table(dashboard.summary)}</section>
      <section class="service-card"><h2>거시 점수</h2>{_score_bars(dashboard.scores)}</section>
    </div>
    <section class="service-card"><h2>주요 지표</h2>{_table(dashboard.indicators)}</section>
    """


def _regime_page(dashboard: MacroDashboard) -> str:
    scores = dashboard.scores.set_index("점수")["값"]
    notes = pd.DataFrame(
        [
            {"축": "성장", "읽는 법": "S&P 500, 구리, 장단기 커브를 함께 봅니다.", "현재 점수": _fmt(scores.get("성장 모멘텀"))},
            {"축": "물가/원자재", "읽는 법": "금리, 달러, 원자재 바스켓이 높을수록 압력이 큽니다.", "현재 점수": _fmt(scores.get("인플레/원자재 압력"))},
            {"축": "정책", "읽는 법": "2년 금리 중심으로 긴축 부담을 봅니다.", "현재 점수": _fmt(scores.get("정책 긴축도"))},
            {"축": "위험선호", "읽는 법": "수익률, 낙폭, 변동성의 조합입니다.", "현재 점수": _fmt(scores.get("위험선호"))},
        ]
    )
    return f"""
    {_hero(dashboard)}
    {_macro_nav("regime")}
    {_page_charts("regime", dashboard)}
    <section class="service-card"><h2>레짐 분해</h2>{_table(notes)}</section>
    <section class="service-card"><h2>점수 상세</h2>{_score_bars(dashboard.scores)}</section>
    """


def _rates_page(dashboard: MacroDashboard) -> str:
    return f"""
    {_hero(dashboard)}
    {_macro_nav("rates")}
    {_page_charts("rates", dashboard)}
    <section class="service-card"><h2>금리와 커브</h2>{_table(dashboard.rates)}</section>
    <section class="service-card"><h2>해석</h2><p class="service-muted">2년 금리는 정책 기대, 10년 금리는 성장/물가 기대, 10Y-2Y 스프레드는 경기 사이클 신호로 봅니다.</p></section>
    """


def _risk_page(dashboard: MacroDashboard) -> str:
    return f"""
    {_hero(dashboard)}
    {_macro_nav("risk")}
    {_page_charts("risk", dashboard)}
    <section class="service-card"><h2>위험자산 온도</h2>{_table(dashboard.risk_assets)}</section>
    <section class="service-card"><h2>주요 지표</h2>{_table(dashboard.indicators)}</section>
    """


def _dollar_page(dashboard: MacroDashboard) -> str:
    return f"""
    {_hero(dashboard)}
    {_macro_nav("dollar")}
    {_page_charts("dollar", dashboard)}
    <section class="service-card"><h2>달러/원자재 압력</h2>{_table(dashboard.dollar_commodities)}</section>
    <section class="service-card"><h2>데이터 출처</h2>{_table(dashboard.sources)}</section>
    """


def _playbook_page(dashboard: MacroDashboard) -> str:
    return f"""
    {_hero(dashboard)}
    {_macro_nav("playbook")}
    {_page_charts("playbook", dashboard)}
    <section class="service-card"><h2>섹터 플레이북</h2>{_table(dashboard.sector_playbook)}</section>
    <section class="service-card"><h2>사용 방법</h2><p class="service-muted">이 표는 매수/매도 신호가 아니라 현재 거시 환경에서 어떤 업종 해석을 먼저 확인할지 정하는 우선순위입니다.</p></section>
    """


def render_body(page: str, *, start_date: str | None = None, lookback_days: int = 504) -> str:
    active = normalize_page(page)
    dashboard = build_macro_dashboard(start_date=start_date, lookback_days=lookback_days)
    page_html = {
        "overview": _overview_page,
        "regime": _regime_page,
        "rates": _rates_page,
        "risk": _risk_page,
        "dollar": _dollar_page,
        "playbook": _playbook_page,
    }[active](dashboard)
    return f"""
    <style>      
      .macro-nav {{ display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px; }}
      .macro-nav a {{ text-decoration:none; color:var(--brand); border:1px solid var(--line); background:#fff; border-radius:999px; padding:7px 12px; font-size:13px; }}
      .macro-nav a.active {{ background:var(--brand); color:#fff; border-color:var(--brand); }}
      .service-main > .macro-nav:first-of-type {{ display:none; }}
      .macro-hero {{ display:flex; justify-content:space-between; gap:16px; align-items:flex-start; background:none; border:none; border-radius:8px; padding:18px; margin-bottom:12px; }}
      .macro-hero h1 {{ margin:4px 0 8px; font-size:26px; letter-spacing:0; }}
      .macro-hero p {{ margin:0; color:var(--muted); line-height:1.5; }}
      .macro-metrics {{ display:grid; grid-template-columns:repeat(2, minmax(120px, 1fr)); gap:8px; min-width:280px; }}
      .macro-metrics div {{ border:1px solid var(--line); border-radius:8px; padding:10px; background:#f8fafc; }}
      .macro-metrics span {{ display:block; color:var(--muted); font-size:12px; margin-bottom:4px; }}
      .macro-metrics strong {{ font-size:18px; }}
      .macro-grid {{ display:grid; gap:12px; margin-bottom:12px; }}
      .macro-grid.two {{ grid-template-columns:minmax(0, 1fr) minmax(0, 1fr); }}
      .service-card {{ margin-bottom:12px; }}
      .service-card h2 {{ margin:0 0 10px; font-size:18px; }}
      .macro-chart-card {{ overflow:hidden; }}
      .macro-chart-card img {{ display:block; width:100%; max-width:100%; height:auto; }}
      .macro-score-grid {{ display:grid; gap:9px; }}
      .macro-score-row {{ display:grid; grid-template-columns:150px 1fr 52px; gap:10px; align-items:center; }}
      .macro-score-label {{ font-size:13px; color:var(--text); }}
      .macro-score-track {{ height:10px; background:#e5e7eb; border-radius:999px; overflow:hidden; }}
      .macro-score-track span {{ display:block; height:100%; background:var(--accent); }}
      .macro-score-value {{ font-variant-numeric:tabular-nums; text-align:right; font-size:12px; color:var(--muted); }}
      @media (max-width:900px) {{
        .macro-hero {{ flex-direction:column; }}
        .macro-metrics {{ width:100%; min-width:0; }}
        .macro-grid.two {{ grid-template-columns:1fr; }}
        .macro-score-row {{ grid-template-columns:120px 1fr 44px; }}
      }}
    </style>
    {page_html}
    """
