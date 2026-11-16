"""All-asset PostgreSQL research dashboard and per-exposure analysis pages."""
from __future__ import annotations

import html
import logging
from datetime import timedelta, timezone
from functools import lru_cache

import psycopg
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.encoders import jsonable_encoder
from psycopg.rows import dict_row

from app.web import shell
from global_signals.db import connect
from global_signals.dashboard import read_asset
from global_signals.universe import BINDINGS, CASH_TICKER

router = APIRouter()
logger = logging.getLogger(__name__)
COLORS = {'green': ('#047857', '초록 · 상대 우위'), 'yellow': ('#a16207', '노랑 · 혼조'),
          'red': ('#b91c1c', '빨강 · 상대 열위'), 'gray': ('#64748b', '회색 · 판단 보류')}
CLASSES = {'equity': '주식', 'government_bond': '국채', 'corporate_bond': '회사채',
           'listed_reit': '상장 리츠', 'physical_property': '실물 부동산', 'commodity': '원자재',
           'foreign_exchange': '외환', 'cryptoasset': '암호자산', 'cash': '원화 현금', 'foreign_cash': '외화 현금성 상품'}
MODES = {'forecast': '상품 대용 가격 분석', 'fx': '현물환 분석', 'context': '참고지표 분석'}
STATES = {'ready': '자료 준비', 'stale': '자료 지연 / 만료', 'insufficient': '자료 / 표본 부족', 'failed': '수집 / 계산 실패'}
REASONS = {'not_connected': '아직 수집 전', 'required_series_missing': '필수 자료 누락',
           'stale_input': '입력 자료 지연', 'forecast_expired': '갱신 유효기간 만료',
           'too_few_matured_labels': '학습 표본 부족', 'short_history': '가격 이력 부족',
           'input_gap': '가격 이력 공백', 'research_only': '연구용', 'context_only': '참고지표 · 상품 수익률 예측 보류',
           'no_point_in_time_validation': '당시 공개 정보 기준 검증 전', 'snapshot_after_cutoff': '자료 시점 불일치'}
CSS = '''<style>
.signals-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}
.signal-card{background:#fff;border:1px solid #d7e0ea;border-radius:14px;padding:20px;min-width:0}
.signal-card h2{font-size:17px;margin:0 0 8px}.signal-card p,.signal-note{color:#475569;font-size:13px}
.signal-label{display:block;font-size:19px;margin:16px 0}.signal-card dt{color:#475569;font-size:13px;margin-top:12px}
.signal-card dd{margin:3px 0;font-weight:600}.signals-intro{background:#eff6ff;border:1px solid #bfdbfe;padding:18px;border-radius:12px;margin:20px 0}
.signal-tools{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}.signal-tools input,.signal-tools select{padding:10px;border:1px solid #cbd5e1;border-radius:8px}
.signal-table{border-collapse:collapse;width:100%;font-size:14px}.signal-table td,.signal-table th{padding:10px;text-align:left;border-bottom:1px solid #e2e8f0}
.signal-scroll{overflow-x:auto}.signal-section{margin:28px 0}.signal-link{color:#1d4ed8;text-decoration:none;font-weight:600}
.signal-chart{min-height:300px}.signal-pill{font-size:12px;border:1px solid #cbd5e1;border-radius:6px;padding:3px 7px;display:inline-block}
[hidden]{display:none!important}
</style>'''


def escaped(value):
    return html.escape(str(value if value is not None else '—'))


def percentage(value):
    return '—' if value is None else f'{float(value) * 100:.1f}%'


def timestamp(value):
    return value.astimezone(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M KST') if value else '아직 없음'


def read_dashboard():
    with connect() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute('SELECT * FROM research_latest_signals ORDER BY asset_class, exposure_id')
        rows = cur.fetchall()
        cur.execute('SELECT run_id,started_at,completed_at,status FROM research_refresh_runs ORDER BY run_id DESC LIMIT 1')
        run = cur.fetchone()
    return {'signals': rows, 'latest_run': run, 'mode': 'research_only', 'reporting_currency': 'KRW'}


def unavailable(exc):
    logger.warning('Global signals unavailable: %s', type(exc).__name__)
    return JSONResponse({'error': 'global_signals_unavailable'}, status_code=503)


@router.get('/api/global-signals')
def signals_api():
    try:
        return JSONResponse(jsonable_encoder(read_dashboard()))
    except (ValueError, psycopg.Error) as exc:
        return unavailable(exc)


@router.get('/api/global-signals/{exposure_id}')
def asset_api(exposure_id: str):
    try:
        return JSONResponse(jsonable_encoder(read_asset(exposure_id)))
    except LookupError:
        return JSONResponse({'error': 'unknown_exposure'}, status_code=404)
    except (ValueError, psycopg.Error) as exc:
        return unavailable(exc)


@lru_cache(maxsize=1)
def chart_library():
    from plotly.offline import get_plotlyjs
    return get_plotlyjs()


@router.get('/global-signals/chart-library.js')
def plotly_library():
    return Response(chart_library(), media_type='application/javascript', headers={'Cache-Control': 'private, max-age=86400'})


def signal_label(row):
    metrics = row.get('metrics') or {}
    if metrics.get('analysis_mode') == 'context':
        return '#64748b', '회색 · 참고지표 분석'
    return COLORS.get(row['research_color'], COLORS['gray'])


def render_dashboard(payload):
    rows = payload['signals']
    cards = []
    for row in rows:
        exposure = row['exposure_id']
        binding = BINDINGS.get(exposure, {})
        metrics, validation = row.get('metrics') or {}, row.get('validation') or {}
        mode = binding.get('mode', 'context')
        color, label = signal_label(row)
        if mode == 'context':
            context = metrics.get('context_series') or []
            if context:
                first = context[0]
                summary = f'{escaped(first["label"])}<br><strong>{first["latest"]:,.3f} {escaped(first["unit"])}</strong>'
            else:
                summary = '참고자료 수집 대기 / 누락'
        else:
            summary = f'1개월 초과수익 중앙값 <strong>{percentage(metrics.get("excess_return_p50"))}</strong><br>현금성 ETF 초과 확률 {percentage(metrics.get("probability_outperform_cash"))}'
        asset_class = row.get('asset_class', '')
        cards.append(f'''<article class="signal-card" data-asset="{escaped(asset_class)}" data-mode="{escaped(mode)}" data-state="{escaped(row.get('status','insufficient'))}">
          <h2><a class="signal-link" href="/global-signals/{escaped(exposure)}">{escaped(row['label'])}</a></h2>
          <span class="signal-pill">{escaped(CLASSES.get(asset_class,asset_class))} · {MODES[mode]}</span>
          <strong class="signal-label" style="color:{color}">● {label}</strong>
          <p>{summary}</p><p>자료 기준일: {escaped(row.get('data_date'))}<br>{STATES.get(row.get('status'), '수집 전')}</p>
          <a class="signal-link" href="/global-signals/{escaped(exposure)}">차트 · 분석 · 출처 보기 →</a></article>''')
    classes = ''.join(f'<option value="{key}">{label}</option>' for key, label in CLASSES.items())
    run = payload.get('latest_run') or {}
    forecast = sum(BINDINGS.get(r['exposure_id'], {}).get('mode') in ('forecast', 'fx') for r in rows)
    return CSS + f'''<h1>글로벌 자산 신호등</h1>
      <div class="signals-intro"><strong>전체 {len(rows)}개 · 가격/현물환 분석 {forecast}개 · 참고지표 분석 {len(rows)-forecast}개</strong>
      <p>색상은 연구용 과거 표본 추정입니다. 기준 통화는 원화, 예측 기간은 1개월, 비교 기준은 TIGER KOFR금리액티브(합성) ETF입니다.
      금리·주택가격 지수는 상품의 총수익률과 다릅니다. 참고지표만 있는 자산은 회색으로 유지하며 자료의 변화와 한계를 분석합니다.</p>
      <p>초록: 초과 확률 60% 이상·중앙값 양수·하위 10% 수익률 −10% 이상. 빨강: 확률 40% 이하·중앙값 음수. 그 외 노랑.
      자료 누락·지연·표본 부족 시 회색. 실전 검증과 환헤지 신호는 보류합니다.</p></div>
      <p>마지막 갱신: {escaped(timestamp(run.get('completed_at')))} · {escaped(run.get('status') or '미실행')}</p>
      <div class="signal-tools"><input id="asset-search" aria-label="자산 검색" placeholder="국가·자산 이름 검색">
      <select id="asset-class" aria-label="자산군"><option value="">전체 자산군</option>{classes}</select>
      <select id="asset-mode" aria-label="분석 종류"><option value="">전체 분석</option><option value="forecast">상품 대용 가격</option><option value="fx">현물환</option><option value="context">참고지표</option></select>
      <select id="asset-state" aria-label="자료 상태"><option value="">전체 상태</option><option value="ready">자료 준비</option><option value="stale">자료 지연 / 만료</option><option value="insufficient">자료 / 표본 부족</option><option value="failed">수집 / 계산 실패</option></select>
      <span id="asset-count" role="status">{len(rows)}개 표시</span></div>
      <section class="signals-grid">{''.join(cards)}</section>
      <script>
      (()=>{{const controls=['asset-search','asset-class','asset-mode','asset-state'].map(id=>document.getElementById(id));
      const filter=()=>{{let count=0;document.querySelectorAll('.signal-card').forEach(card=>{{
        const visible=card.textContent.toLowerCase().includes(controls[0].value.toLowerCase()) &&
          (!controls[1].value||card.dataset.asset===controls[1].value)&&(!controls[2].value||card.dataset.mode===controls[2].value)&&
          (!controls[3].value||card.dataset.state===controls[3].value);card.hidden=!visible;if(visible)count++;}});
        document.getElementById('asset-count').textContent=count+'개 표시';}};
      controls.forEach(c=>c.addEventListener('input',filter));}})();
      </script>'''


def chart(traces, title, unit):
    import plotly.graph_objects as go
    fig = go.Figure()
    for label, values in traces:
        fig.add_trace(go.Scatter(x=[v[0] for v in values], y=[v[1] for v in values], name=label, mode='lines', connectgaps=False))
    fig.update_layout(title=title, yaxis_title=unit, template='plotly_white', height=360,
                      margin=dict(l=55,r=20,t=55,b=45), hovermode='x unified',
                      legend=dict(orientation='h'), xaxis=dict(rangeslider=dict(visible=False),
                      rangeselector=dict(buttons=[dict(count=1,label='1년',step='year',stepmode='backward'),dict(count=5,label='5년',step='year',stepmode='backward'),dict(step='all',label='전체')])) )
    return fig.to_html(full_html=False, include_plotlyjs=False, config={'displaylogo': False, 'responsive': True})


def render_asset(payload):
    from global_signals.data import Snapshot
    from global_signals.model import krw_asset
    row, binding = payload['signal'], payload['binding']
    metrics, validation = row.get('metrics') or {}, row.get('validation') or {}
    color, label = signal_label(row)
    reasons = ' · '.join(REASONS.get(r, r) for r in row.get('reason_codes', []))
    sources_html, charts = [], []
    available = {}
    for source in payload['sources']:
        values = source['values']
        if values:
            available[source['key']] = Snapshot(source['key'], source['currency'], source['first_seen'], '', values)
        status = {'available': '수집됨', 'stale': '자료 지연', 'missing': '최신 수집에 없음'}[source['collection_status']]
        sources_html.append(f'<tr><td><a href="{escaped(source["url"])}" target="_blank" rel="noopener noreferrer">{escaped(source["label"])}</a><br>{escaped(source["key"])}</td><td>{escaped(source["frequency"])}</td><td>{escaped(source["currency"])}</td><td>{len(values):,}</td><td>{escaped(values[-1][0] if values else None)}</td><td>{status}</td></tr>')
        if values:
            charts.append('<section class="signal-section">' + chart([(source['label'], values)], source['label'], source['currency']) + '</section>')
    combined = ''
    if binding['mode'] in ('forecast', 'fx') and all(s in available for s in binding['series']):
        import pandas as pd
        from global_signals.data import price_series
        price = krw_asset(available, binding)
        frame = pd.concat({'자산 (원화)': price, '현금성 ETF': price_series(available[CASH_TICKER])}, axis=1, join='inner').dropna()
        if not frame.empty:
            frame = frame / frame.iloc[0] * 100
            traces = [(name, [[d.date().isoformat(), float(v)] for d,v in frame[name].items()]) for name in frame]
            combined = chart(traces, '원화 환산 누적 가격 · 공통 시작일=100', '지수 (100 기준)')
    if binding['mode'] == 'context':
        cells = []
        for item in metrics.get('context_series', []):
            change = f'{item["change_pp"]:+.3f}%p' if 'change_pp' in item else percentage(item.get('change_fraction'))
            year = f'{item["year_change_pp"]:+.3f}%p' if 'year_change_pp' in item else percentage(item.get('year_change_fraction'))
            cells.append(f'<tr><td>{escaped(item["label"])}</td><td>{item["latest"]:,.4f} {escaped(item["unit"])}</td><td>{change}</td><td>{year}</td></tr>')
        summary = '<h2>참고지표 변화</h2><p>금리는 퍼센트포인트(%p), 가격·지수는 변화율(%)입니다. 기간은 각 자료의 발표 주기를 따릅니다.</p><div class="signal-scroll"><table class="signal-table"><tr><th>지표</th><th>최근 값</th><th>직전 관측 대비</th><th>1년 전 관측 대비</th></tr>' + ''.join(cells) + '</table></div>'
    else:
        statistics = [('1개월 초과수익 중앙값', percentage(metrics.get('excess_return_p50'))),
                      ('현금성 ETF 초과 확률', percentage(metrics.get('probability_outperform_cash'))),
                      ('표본 분포 10%–90%', percentage(metrics.get('excess_return_p10'))+' ~ '+percentage(metrics.get('excess_return_p90'))),
                      ('최근 1개월 / 3개월 원화 수익률', percentage(metrics.get('return_1m'))+' / '+percentage(metrics.get('return_3m'))),
                      ('최근 1년 원화 수익률', percentage(metrics.get('return_1y'))),
                      ('공통 이력 최대 낙폭', percentage(metrics.get('max_drawdown'))),
                      ('최근 연율 변동성', percentage(metrics.get('annualized_volatility'))),
                      ('추정 표본 / 순차 평가 수', str(metrics.get('sample_count',0))+' / '+str(validation.get('evaluation_count',0))),
                      ('가정 왕복 비용 · 세금 제외', percentage(metrics.get('assumed_round_trip_cost_fraction')))]
        summary = '<h2>원화 기준 분석</h2><div class="signals-grid">' + ''.join(f'<div class="signal-card"><p>{escaped(k)}</p><strong>{escaped(v)}</strong></div>' for k,v in statistics) + '</div>'
        if validation.get('brier_score') is not None:
            summary += f'<p>순차 검증 Brier 점수: {validation["brier_score"]:.3f} · 단순 비교 기준: {validation["baseline_brier_score"]:.3f} (낮을수록 양호). 진단 기준: {"통과" if validation.get("diagnostic_gate_passed") else "미통과"}.</p>'
        summary += '<p>수익률은 수집된 공통 이력에 대한 계산이며 미래 보장이 아닙니다. 과거 조정주가의 수정 이력, 시장별 종가 시차, 가정한 거래비용 때문에 실전 검증 승인은 보류되어 있습니다.</p>'
    history = ''.join(f'<tr><td>{escaped(timestamp(r["as_of"]))}</td><td>{escaped(r["data_date"])}</td><td>{COLORS.get(r["research_color"],COLORS["gray"])[1]}</td><td>{STATES.get(r["status"],r["status"])}</td></tr>' for r in payload['signal_history'])
    return CSS + f'''<script src="/global-signals/chart-library.js"></script>
      <a class="signal-link" href="/global-signals">← 전체 자산 목록</a><h1>{escaped(row['label'])}</h1>
      <span class="signal-pill">{escaped(CLASSES.get(row['asset_class'],row['asset_class']))} · {MODES[binding['mode']]}</span>
      <strong class="signal-label" style="color:{color}">● {label}</strong>
      <div class="signals-intro"><strong>분석 범위</strong><p>{escaped(binding['note'])}</p><p>{escaped(reasons)}</p>
      <p>계산 시각: {escaped(timestamp(row.get('as_of')))} · 자료 기준일: {escaped(row.get('data_date'))} · {STATES.get(row.get('status'),'수집 전')}</p></div>
      {summary}<section class="signal-section">{combined}</section>
      <h2>수집 자료와 출처</h2><p>각 차트는 해당 계산에 사용한 자료 버전입니다. 관측일은 공개 시각과 다르며, 수집 시점 이전에 알려져 있었다고 가정하지 않습니다.</p>
      <div class="signal-scroll"><table class="signal-table"><tr><th>출처 / 시계열</th><th>주기</th><th>단위</th><th>관측 수</th><th>최근 관측</th><th>자료 상태</th></tr>{''.join(sources_html)}</table></div>
      {''.join(charts) or '<p>표시할 수집 자료가 없습니다. 로컬 갱신 6번과 업로드를 실행해 주세요.</p>'}
      <h2>최근 분석 이력</h2><div class="signal-scroll"><table class="signal-table"><tr><th>계산 시각</th><th>자료 기준일</th><th>연구 색상</th><th>자료 상태</th></tr>{history}</table></div>'''


def page_response(request, title, render):
    admin = bool(getattr(getattr(request.state, 'user', None), 'is_admin', False))
    try:
        body, status = render(), 200
    except LookupError:
        body, status = '<h1>등록되지 않은 자산입니다.</h1><a href="/global-signals">전체 목록</a>', 404
    except (ValueError, psycopg.Error) as exc:
        logger.warning('Global signals unavailable: %s', type(exc).__name__)
        body, status = '<h1>글로벌 자산 신호등</h1><p>자료에 연결할 수 없습니다. DB 연결과 초기화 상태를 확인해 주세요.</p>', 503
    return HTMLResponse(shell(title, body, active='global-signals', admin=admin), status_code=status)


@router.get('/global-signals', response_class=HTMLResponse)
def signals_page(request: Request):
    return page_response(request, '글로벌 자산 신호등', lambda: render_dashboard(read_dashboard()))


@router.get('/global-signals/{exposure_id}', response_class=HTMLResponse)
def asset_page(exposure_id: str, request: Request):
    return page_response(request, '자산별 분석', lambda: render_asset(read_asset(exposure_id)))
