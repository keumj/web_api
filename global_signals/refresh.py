"""Local recovery collection, explicit upload, or direct source-to-PostgreSQL refresh."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import hashlib
import sqlite3
import time
from pathlib import Path

import psycopg
from psycopg.types.json import Jsonb

from .data import fetch_snapshot, load_snapshots, save_snapshot, save_manifest, period_end
from .db import connect
from .model import VERSION, evaluate, empty_result
from .analysis import analyze_context
from .returns import next_calendar_month
from .universe import BINDINGS, SERIES, SYMBOLS, SOURCE_ID, CASH_TICKER, series_id, source_id


def register_series(conn):
    conn.execute('INSERT INTO data_sources(source_id,label,category,documentation_url,access_notes) VALUES (%s,%s,%s,%s,%s) ON CONFLICT(source_id) DO NOTHING',
                 ('bis_policy_rates', 'BIS central bank policy rates', 'policy_rate_context',
                  'https://data.bis.org/topics/CBPOL', 'Official policy rates; not executable money-market or deposit quotes.'))
    conn.execute('INSERT INTO data_sources(source_id,label,category,documentation_url,access_notes) VALUES (%s,%s,%s,%s,%s) ON CONFLICT(source_id) DO NOTHING',
                 (SOURCE_ID, 'Yahoo Finance via yfinance', 'research_market_data',
                  'https://ranaroussi.github.io/yfinance/',
                  'Adjusted historical prices; first-seen availability only. Research proxy; source terms and redistribution access require separate review.'))
    mapping = {b['primary']: (key, SERIES[b['primary']]['label']) for key, b in BINDINGS.items() if b['mode'] == 'forecast'}
    mapping[CASH_TICKER] = ('cash_krw', 'TIGER KOFR active synthetic cash proxy')
    for ticker, currency in SYMBOLS.items():
        spec = SERIES[ticker]
        exposure, label = mapping.get(ticker, (None, spec['label']))
        instrument_id = 'yf:' + ticker if ticker in mapping else None
        if instrument_id:
            conn.execute('INSERT INTO instruments(instrument_id,exposure_id,label,ticker,quote_currency,hedge_type,investability_status,attributes) VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(instrument_id) DO NOTHING',
                         (instrument_id, exposure, label, ticker, currency, 'unhedged', 'monitor_only',
                          Jsonb({'purpose': 'research_proxy', 'personal_access': 'unverified'})))
        conn.execute('INSERT INTO series_definitions(series_id,source_id,source_series_code,label,series_kind,unit,frequency,reference_currency,exposure_id,instrument_id,return_basis,attributes) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(series_id) DO UPDATE SET label=excluded.label,series_kind=excluded.series_kind,unit=excluded.unit,frequency=excluded.frequency,attributes=excluded.attributes',
                     (series_id(ticker), source_id(ticker), spec['code'], label, spec['kind'],
                      currency, spec['frequency'], currency if spec['provider'] == 'yahoo' else None,
                      exposure, instrument_id,
                      {'adjusted_price': 'total_return', 'spot_price': 'price_only', 'yield': 'yield'}.get(spec['kind'], 'level'),
                      Jsonb({'availability': 'first_seen', 'source_url': spec['url'], 'research_only': True})))
    for exposure, binding in BINDINGS.items():
        for key in binding['series']:
            conn.execute('INSERT INTO exposure_series_bindings(exposure_id,series_id,analysis_mode,coverage_note) VALUES (%s,%s,%s,%s) ON CONFLICT(exposure_id,series_id) DO UPDATE SET analysis_mode=excluded.analysis_mode,coverage_note=excluded.coverage_note',
                         (exposure, series_id(key), binding['mode'], binding['note']))


def ingest_snapshot(conn, snapshot):
    provider_id = source_id(snapshot.ticker)
    previous = conn.execute('SELECT ingestion_run_id FROM raw_ingestion_runs WHERE source_id=%s AND artifact_sha256=%s AND status=%s',
                            (provider_id, snapshot.digest, 'complete')).fetchone()
    if previous:
        return previous[0]
    first_seen = datetime.fromisoformat(snapshot.first_seen)
    run_id = conn.execute('INSERT INTO raw_ingestion_runs(source_id,request_fingerprint,artifact_sha256,records_received,attributes) VALUES (%s,%s,%s,%s,%s) RETURNING ingestion_run_id',
                          (provider_id, snapshot.ticker, snapshot.digest, len(snapshot.values),
                           Jsonb({'ticker': snapshot.ticker, 'first_seen': snapshot.first_seen}))).fetchone()[0]
    with conn.cursor() as cursor:
        cursor.executemany('INSERT INTO observations(series_id,source_id,ingestion_run_id,observed_period_start,observed_period_end,value,first_seen_at,revision_key,availability_basis,retrieved_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,clock_timestamp()) ON CONFLICT(series_id,observed_period_start,observed_period_end,revision_key) DO NOTHING', [
            (series_id(snapshot.ticker), provider_id, run_id, day, period_end(snapshot.ticker, day), value, first_seen, snapshot.digest, 'first_seen')
            for day, value in snapshot.values
        ])
    count = conn.execute('SELECT count(*) FROM observations WHERE ingestion_run_id=%s', (run_id,)).fetchone()[0]
    conn.execute("UPDATE raw_ingestion_runs SET status='complete',completed_at=CURRENT_TIMESTAMP,records_inserted=%s WHERE ingestion_run_id=%s", (count, run_id))
    return run_id


def save_forecast(conn, exposure, ticker, result, snapshots, ingestions, now):
    """Persist research forecasts through the original schema's gray safety gates."""
    metrics = result['metrics']
    if 'probability_outperform_cash' not in metrics:
        return
    used = BINDINGS[exposure]['series']
    cutoff = max(datetime.fromisoformat(snapshots[s].first_seen) for s in used)
    code_hash = hashlib.sha256(Path(__file__).with_name('model.py').read_bytes()).hexdigest()
    model_id = conn.execute('INSERT INTO model_runs(model_key,model_version,training_data_cutoff,training_period_start,training_period_end,validation_status,validation_metrics,evaluation_design,configuration,code_version) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING model_run_id',
                            ('conditional_historical:' + exposure, VERSION, cutoff, metrics['data_start'], result['data_date'],
                             'unvalidated', Jsonb(result['validation']), Jsonb(result['validation']), Jsonb(metrics), code_hash)).fetchone()[0]
    quality = {'ready': 'ok', 'stale': 'stale', 'insufficient': 'insufficient', 'failed': 'unknown'}[result['status']]
    feature_id = conn.execute('INSERT INTO feature_snapshots(exposure_id,as_of,feature_version,input_max_available_at,feature_values,quality_status,quality_details,availability_policy) VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING feature_snapshot_id',
                              (exposure, now, VERSION, cutoff, Jsonb(metrics), quality,
                               Jsonb({'research_only': True, 'snapshot_hashes': metrics['snapshot_hashes']}), 'first_seen_only')).fetchone()[0]
    for symbol in used:
        conn.execute('INSERT INTO feature_observation_lineage(feature_snapshot_id,feature_name,observation_id) SELECT %s,%s,observation_id FROM observations WHERE ingestion_run_id=%s',
                     (feature_id, series_id(symbol), ingestions[symbol]))
    start = now.date() + timedelta(days=1)
    conn.execute('INSERT INTO signal_predictions(exposure_id,instrument_id,feature_snapshot_id,model_run_id,prediction_as_of,valid_until,target_period_start,target_period_end,currency_treatment,cash_benchmark_series_id,probability_outperform_cash,excess_return_p10,excess_return_p50,excess_return_p90,transaction_cost_fraction,cost_assumptions,risk_policy,forecast_status,signal_color,reason_codes) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)',
                 (exposure, 'yf:' + ticker, feature_id, model_id, now, now + timedelta(hours=36), start, next_calendar_month(start),
                  'unhedged', series_id(CASH_TICKER), metrics['probability_outperform_cash'], metrics['excess_return_p10'],
                  metrics['excess_return_p50'], metrics['excess_return_p90'], metrics['assumed_round_trip_cost_fraction'],
                  Jsonb({'basis': 'hypothetical_round_trip', 'excludes_tax': True, 'embedded_fund_expenses_not_deducted_again': True}),
                  Jsonb({'research_only': True, 'executable_calendar_pending': True}), 'unvalidated', 'gray', result['reason_codes']))


def collect(*, local=False, retry_missing=False):
    snapshots = load_snapshots() if retry_missing else {}
    failures = {}
    if local:
        save_manifest({})
    for ticker in SYMBOLS:
        if ticker in snapshots:
            continue
        try:
            for attempt in range(2):
                try:
                    snapshot = fetch_snapshot(ticker)
                    break
                except Exception:
                    if attempt:
                        raise
                    time.sleep(1)
            if local:
                save_snapshot(snapshot)
            snapshots[ticker] = snapshot
            print(f'{ticker}: {len(snapshot.values)} rows; latest={snapshot.values[-1][0]}', flush=True)
        except Exception as exc:
            # Provider exceptions can contain URLs; do not persist credential-bearing text.
            failures[ticker] = type(exc).__name__
            print(f'{ticker}: collection failed ({type(exc).__name__})', flush=True)
    if local:
        save_manifest(snapshots)
    return snapshots, failures


def refresh(mode='direct', cost_fraction=0.002, retry_missing=False):
    if retry_missing and mode != 'local':
        raise ValueError('--retry-missing is only supported for explicit local recovery collection')
    if mode == 'local':
        snapshots, failures = collect(local=True, retry_missing=retry_missing)
        return {'mode': mode, 'snapshots': len(snapshots), 'failures': failures}, int(bool(failures))
    # Session advisory lock is supported on this dedicated direct PostgreSQL connection.
    # Transaction lock below also prevents duplicate ingestion on repeated uploads.
    with connect() as conn:
        marker = conn.execute("SELECT value FROM global_signal_meta WHERE key='project_key'").fetchone()
        if marker != ('global_asset_signals_v1',):
            raise ValueError('Initialize the owned database first')
        if not conn.execute('SELECT pg_try_advisory_xact_lock(79244002)').fetchone()[0]:
            raise ValueError('Another global signals refresh is running')
        if mode == 'upload-local':
            snapshots = load_snapshots()
            failures = {s: 'missing_from_latest_local_refresh' for s in SYMBOLS if s not in snapshots}
        else:
            snapshots, failures = collect()
        register_series(conn)
        run_id = conn.execute('INSERT INTO research_refresh_runs(mode) VALUES (%s) RETURNING run_id', (mode,)).fetchone()[0]
        ingestions = {s: ingest_snapshot(conn, snap) for s, snap in snapshots.items()}
        # Timestamp after collection: first_seen must never be later than prediction time.
        now = datetime.now(timezone.utc)
        outcomes = {}
        for exposure, binding in BINDINGS.items():
            ticker = binding['primary']
            try:
                result = (analyze_context(snapshots, exposure, now=now) if binding['mode'] == 'context' else
                          evaluate(snapshots, ticker, cost_fraction=cost_fraction, now=now, exposure=exposure))
            except (ValueError, KeyError, TypeError) as exc:
                result = empty_result('failed', 'calculation_' + type(exc).__name__)
                failures[exposure] = type(exc).__name__
            result['metrics']['analysis_mode'] = binding['mode']
            result['metrics']['coverage_note'] = binding['note']
            result['metrics']['missing_series'] = [s for s in binding['series'] if s not in snapshots]
            result['metrics']['ingestion_run_ids'] = {s: ingestions[s] for s in binding['series'] if s in ingestions}
            conn.execute('INSERT INTO research_signals(run_id,exposure_id,as_of,data_date,valid_until,status,research_color,model_version,metrics,validation,reason_codes) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)',
                         (run_id, exposure, now, result['data_date'], now + timedelta(hours=36), result['status'],
                          result['research_color'], VERSION, Jsonb(result['metrics']), Jsonb(result['validation']), result['reason_codes']))
            outcomes[exposure] = {'status': result['status'], 'research_color': result['research_color'], 'data_date': str(result['data_date'])}
            if binding['mode'] == 'forecast':
                save_forecast(conn, exposure, ticker, result, snapshots, ingestions, now)
        status = 'partial' if failures else 'complete'
        if not snapshots:
            status = 'failed'
        conn.execute('UPDATE research_refresh_runs SET status=%s, completed_at=clock_timestamp(), details=%s WHERE run_id=%s',
                     (status, Jsonb({'failures': failures, 'outcomes': outcomes}), run_id))
    return {'run_id': run_id, 'status': status, 'outcomes': outcomes, 'failures': failures}, int(bool(failures))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['local', 'upload-local', 'direct'], default='direct')
    parser.add_argument('--retry-missing', action='store_true', help='Retry only series absent from the latest local collection; preserve original first-seen times')
    parser.add_argument('--cost-fraction', type=float, default=0.002, help='Hypothetical round-trip cost fraction; excludes tax, not an observed broker quote')
    args = parser.parse_args(argv)
    try:
        result, code = refresh(args.mode, args.cost_fraction, args.retry_missing)
        print(json.dumps(result, ensure_ascii=False))
        return code
    except (ValueError, psycopg.Error, sqlite3.Error) as exc:
        print('Refresh failed:', type(exc).__name__, 'SQLSTATE:', getattr(exc, 'sqlstate', None))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
