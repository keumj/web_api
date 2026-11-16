"""PostgreSQL-only research signal dashboard, protected by the existing auth middleware."""
from __future__ import annotations

import html
import logging
from datetime import timedelta, timezone

import psycopg
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from psycopg.rows import dict_row
from fastapi.encoders import jsonable_encoder

from app.web import shell
from global_signals.db import connect
from global_signals.universe import PROXIES

router = APIRouter()
logger = logging.getLogger(__name__)
COLORS = {'green': ('#047857', '초록 · 상대 우위'), 'yellow': ('#a16207', '노랑 · 혼조'),
          'red': ('#b91c1c', '빨강 · 상대 열위'), 'gray': ('#64748b', '회색 · 판단 보류')}
REASONS = {
    'not_connected': '데이터 미연결', 'required_series_missing': '필수 데이터 수집 실패',
    'stale_input': '입력 데이터 지연', 'forecast_expired': '갱신 유효기간 만료',
    'too_few_matured_labels': '학습 표본 부족', 'short_history': '가격 이력 부족',
    'input_gap': '가격 이력 공백', 'research_only': '연구용',
    'no_point_in_time_validation': '당시 공개 정보 기준 검증 전',
    'snapshot_after_cutoff': '데이터 시점 불일치',
}


def read_dashboard():
    with connect() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute('SELECT * FROM research_latest_signals ORDER BY asset_class, exposure_id')
            rows = cur.fetchall()
            cur.execute('SELECT run_id,started_at,completed_at,status FROM research_refresh_runs ORDER BY run_id DESC LIMIT 1')
            run = cur.fetchone()
    return {'signals': rows, 'latest_run': run, 'mode': 'research_only', 'reporting_currency': 'KRW'}


@router.get('/api/global-signals')
def signals_api():
    try:
        return JSONResponse(jsonable_encoder(read_dashboard()))
    except (ValueError, psycopg.Error) as exc:
        logger.warning('Global signals unavailable: %s', type(exc).__name__)
        return JSONResponse({'error': 'global_signals_unavailable', 'signals': []}, status_code=503)


def percentage(value):
    return '—' if value is None else f'{float(value) * 100:.1f}%'


def render_dashboard(payload):
    rows = payload['signals']
    cards = []
    for row in rows:
        if row['exposure_id'] not in PROXIES:
            continue
        metrics, validation = row.get('metrics') or {}, row.get('validation') or {}
        color, label = COLORS.get(row['research_color'], COLORS['gray'])
        reasons = ' · '.join(REASONS.get(s, s) for s in row['reason_codes'])
        samples = metrics.get('sample_count', 0)
        score = validation.get('brier_score')
        baseline = validation.get('baseline_brier_score')
        score_text = '평가 표본 부족' if score is None else f'{score:.3f} / 비교 기준 {baseline:.3f} (낮을수록 양호)'
        cards.append(f'''<article class="signal-card">
          <h2>{html.escape(row['label'])}</h2>
          <p>{html.escape(PROXIES[row['exposure_id']][0])} · 환노출 · 원화 기준</p>
          <strong class="signal-label" style="color:{color}"><span aria-hidden="true">●</span> {label}</strong>
          <dl><dt>현금성 ETF 초과 확률 · 과거 표본 추정</dt><dd>{percentage(metrics.get('probability_outperform_cash'))}</dd>
          <dt>1개월 초과수익 중앙값</dt><dd>{percentage(metrics.get('excess_return_p50'))}</dd>
          <dt>과거 표본 분포 10%–90%</dt><dd>{percentage(metrics.get('excess_return_p10'))} ~ {percentage(metrics.get('excess_return_p90'))}</dd>
          <dt>추정 표본 / 순차 평가 표본</dt><dd>{samples} / {validation.get('evaluation_count', 0)}</dd>
          <dt>Brier 점수</dt><dd>{score_text}</dd>
          <dt>가정한 왕복 비용 · 세금 제외</dt><dd>{percentage(metrics.get('assumed_round_trip_cost_fraction'))}</dd>
          <dt>공통 데이터 기준일</dt><dd>{html.escape(str(row.get('data_date') or '—'))}</dd></dl>
          <p class="signal-note">{html.escape(reasons)}</p></article>''')
    pending = [r for r in rows if r['exposure_id'] not in PROXIES]
    pending_rows = ''.join(f'<tr><td>{html.escape(r["label"])}</td><td>{"관찰 전용" if r["monitor_only"] else "연결 준비 중"}</td></tr>' for r in pending)
    run = payload.get('latest_run') or {}
    completed = run.get('completed_at')
    completed_label = completed.astimezone(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M KST') if completed else '아직 없음'
    return f'''<style>
      .signals-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px}}
      .signal-card{{background:white;border:1px solid #d7e0ea;border-radius:14px;padding:22px}}
      .signal-card h2{{font-size:19px;margin:0}} .signal-card p,.signal-note{{color:#475569;font-size:13px}}
      .signal-label{{display:block;font-size:21px;margin:20px 0}} .signal-card dl{{margin:0}}
      .signal-card dt{{color:#475569;font-size:13px;margin-top:13px}} .signal-card dd{{margin:3px 0;font-weight:600}}
      .signals-intro{{background:#eff6ff;border:1px solid #bfdbfe;padding:18px;border-radius:12px;margin:20px 0}}
      .signals-pending{{margin-top:24px}} .signals-pending td{{padding:7px 15px;border-bottom:1px solid #e2e8f0}}
    </style>
    <h1>글로벌 자산 신호등</h1>
    <div class="signals-intro"><strong>연구용 신호 · 1개월 · 원화 기준</strong>
      <p>현재 초록·노랑·빨강은 과거 유사 추세 표본의 비교 결과입니다. 실전 검증된 예측이나 매매 지시가 아닙니다.
      현금 비교 기준은 TIGER KOFR금리액티브(합성) ETF(449170)이며 예금과 다릅니다.</p>
      <p>초록: 초과 확률 60% 이상·중앙값 양수·하위 10% 수익률 −10% 이상.
      빨강: 초과 확률 40% 이하·중앙값 음수. 그 외 노랑. 데이터 부족·지연·실패 시 회색.</p>
      <p>과거 조정주가의 수정 이력과 시장별 종가 시차가 있어 검증된 예측 신호는 아직 보류합니다.
      환헤지 신호는 실제 헤지 가격·비용 확인 전까지 제공하지 않습니다.</p></div>
    <p>마지막 갱신: {html.escape(completed_label)} · 상태: {html.escape(str(run.get('status') or '미실행'))}</p>
    <section class="signals-grid">{''.join(cards) or '<p>아직 등록된 데이터가 없습니다.</p>'}</section>
    <details class="signals-pending"><summary>나머지 자산 {len(pending)}개 준비 상태</summary><table>{pending_rows}</table></details>'''


@router.get('/global-signals', response_class=HTMLResponse)
def signals_page(request: Request):
    admin = bool(getattr(getattr(request.state, 'user', None), 'is_admin', False))
    try:
        body = render_dashboard(read_dashboard())
        status = 200
    except (ValueError, psycopg.Error) as exc:
        logger.warning('Global signals unavailable: %s', type(exc).__name__)
        body = '<h1>글로벌 자산 신호등</h1><p>신호 데이터에 연결할 수 없습니다. 관리자에게 DB 연결과 초기화 상태를 확인해 주세요.</p>'
        status = 503
    return HTMLResponse(shell('글로벌 자산 신호등', body, active='global-signals', admin=admin), status_code=status)
