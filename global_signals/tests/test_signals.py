from __future__ import annotations

from contextlib import closing
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from global_signals.data import make_snapshot, save_snapshot, save_manifest, load_snapshots
from global_signals.db import ROOT
from global_signals.model import evaluate, distribution, color_for
from global_signals.returns import krw_total_return, static_principal_hedge_return, next_calendar_month
from global_signals.universe import SYMBOLS, FX_TICKER, CASH_TICKER


class ReturnTests(unittest.TestCase):
    def test_fx_conversion_once(self):
        self.assertAlmostEqual(krw_total_return(.1, 1000, 1100), .21)

    def test_static_hedge_leaves_gain_fx_risk(self):
        self.assertAlmostEqual(static_principal_hedge_return(.1, 1000, 1100, 1000, explicit_cost_fraction=0), .11)

    def test_calendar_month_clamps_leap_year(self):
        self.assertEqual(next_calendar_month(date(2024, 1, 31)), date(2024, 2, 29))
        self.assertEqual(next_calendar_month(date(2025, 12, 31)), date(2026, 1, 31))

    def test_invalid_fx(self):
        for value in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                krw_total_return(.1, value, 1000)


class SnapshotTests(unittest.TestCase):
    def test_completely_failed_first_collection(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'data') as temp:
            path = Path(temp) / 'recovery.sqlite'
            save_manifest({}, path)
            self.assertEqual(load_snapshots(path), {})

    def test_reject_invalid_provider_values(self):
        for value in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                make_snapshot('SPY', 'USD', [('2020-01-01', 100), ('2020-01-02', value)])

    def test_currency_and_duplicate_rejection(self):
        with self.assertRaises(ValueError):
            make_snapshot('SPY', 'KRW', [('2020-01-01', 100), ('2020-01-02', 101)])
        with self.assertRaises(ValueError):
            make_snapshot('SPY', 'USD', [('2020-01-01', 100), ('2020-01-01', 101)])

    def test_preserve_first_seen_and_failed_manifest(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'data') as temp:
            path = Path(temp) / 'recovery.sqlite'
            first = make_snapshot('SPY', 'USD', [('2020-01-01', 100), ('2020-01-02', 101)], '2020-01-03T00:00:00+00:00')
            second = make_snapshot('SPY', 'USD', first.values, '2020-01-04T00:00:00+00:00')
            save_snapshot(first, path)
            save_snapshot(second, path)
            save_manifest({'SPY': second}, path)
            self.assertEqual(load_snapshots(path)['SPY'].first_seen, first.first_seen)
            save_manifest({}, path)
            self.assertEqual(load_snapshots(path), {})

    def test_checksum_detects_local_corruption(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'data') as temp:
            path = Path(temp) / 'recovery.sqlite'
            snap = make_snapshot('SPY', 'USD', [('2020-01-01', 100), ('2020-01-02', 101)])
            save_snapshot(snap, path)
            save_manifest({'SPY': snap}, path)
            with closing(sqlite3.connect(path)) as conn, conn:
                conn.execute('UPDATE snapshots SET values_json=?', (json.dumps([['2020-01-01', 100], ['2020-01-02', 900]]),))
            with self.assertRaises(ValueError):
                load_snapshots(path)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.now = datetime(2026, 10, 29, tzinfo=timezone.utc)
        days = pd.bdate_range('2018-01-01', '2026-10-28')
        # Deterministic, changing regimes; synthetic fixtures never go to a DB.
        steps = np.arange(len(days))
        asset = 100 * np.exp(steps * .00015 + .12 * np.sin(steps / 80))
        cash = 100 * np.exp(steps * .00006)
        cls.snapshots = {
            ticker: make_snapshot(ticker, SYMBOLS[ticker], list(zip(days, values)), cls.now.isoformat())
            for ticker, values in [('SPY', asset), (FX_TICKER, np.full(len(days), 1300)), (CASH_TICKER, cash)]
        }

    def test_evaluation_is_never_production_approval(self):
        result = evaluate(self.snapshots, 'SPY', now=self.now)
        self.assertEqual(result['status'], 'ready')
        self.assertGreater(result['validation']['evaluation_count'], 12)
        self.assertFalse(result['validation']['production_approved'])
        self.assertEqual(result['validation']['status'], 'retrospective_only')

    def test_missing_fx_fails_closed(self):
        result = evaluate({'SPY': self.snapshots['SPY']}, 'SPY', now=self.now)
        self.assertEqual(result['research_color'], 'gray')

    def test_stale_data_fails_closed(self):
        result = evaluate(self.snapshots, 'SPY', now=self.now + timedelta(days=10))
        self.assertEqual(result['status'], 'stale')
        self.assertEqual(result['research_color'], 'gray')

    def test_snapshot_after_cutoff_fails_closed(self):
        result = evaluate(self.snapshots, 'SPY', now=self.now - timedelta(days=1))
        self.assertIn('snapshot_after_cutoff', result['reason_codes'])

    def test_cost_changes_net_forecast(self):
        a = evaluate(self.snapshots, 'SPY', cost_fraction=0, now=self.now)
        b = evaluate(self.snapshots, 'SPY', cost_fraction=.01, now=self.now)
        self.assertAlmostEqual(a['metrics']['excess_return_p50'] - b['metrics']['excess_return_p50'], .01)

    def test_training_labels_strictly_matured(self):
        import global_signals.model as model
        original = distribution
        calls = []
        def capture(history, regime):
            calls.append(history.copy())
            return original(history, regime)
        with patch.object(model, 'distribution', side_effect=capture):
            evaluate(self.snapshots, 'SPY', now=self.now)
        # The latest label in each expanding evaluation is never reused before expiry.
        ends = [max(r['end'] for r in h) for h in calls if h]
        self.assertEqual(ends, sorted(ends))
        self.assertTrue(all(end < pd.Timestamp('2026-10-28') for end in ends))

    def test_future_prices_do_not_change_forecast(self):
        extended = {}
        for ticker, snap in self.snapshots.items():
            extended[ticker] = make_snapshot(ticker, snap.currency, snap.values + [['2027-01-01', 999999]], snap.first_seen)
        a = evaluate(self.snapshots, 'SPY', now=self.now)
        b = evaluate(extended, 'SPY', now=self.now)
        self.assertEqual(a['metrics']['probability_outperform_cash'], b['metrics']['probability_outperform_cash'])
        self.assertEqual(a['metrics']['excess_return_p50'], b['metrics']['excess_return_p50'])

    def test_color_boundary_and_downside_guard(self):
        self.assertEqual(color_for({'probability_outperform_cash': .7, 'excess_return_p50': .02, 'excess_return_p10': -.2}), 'yellow')
        self.assertEqual(color_for({'probability_outperform_cash': .3, 'excess_return_p50': -.02}), 'red')
        self.assertIsNone(distribution([{'excess': .1, 'regime': 1}] * 23, 1))


class ApiTests(unittest.TestCase):
    def test_page_precedes_legacy_catch_all(self):
        from app.main import app
        from starlette.routing import Match
        scope = {'type': 'http', 'path': '/global-signals', 'method': 'GET', 'root_path': ''}
        route = next(r for r in app.routes if r.matches(scope)[0] == Match.FULL)
        self.assertEqual(route.path, '/global-signals')

    def test_unavailable_returns_503_without_secrets(self):
        from app.routers import global_signals as route
        with patch.object(route, 'read_dashboard', side_effect=ValueError('secret-connection-string')):
            response = route.signals_api()
        self.assertEqual(response.status_code, 503)
        self.assertNotIn(b'secret', response.body)

    def test_html_escapes_database_labels(self):
        from app.routers.global_signals import render_dashboard
        body = render_dashboard({'signals': [{'exposure_id': 'equity_us', 'label': '<script>alert(1)</script>', 'research_color': 'gray', 'reason_codes': [], 'monitor_only': False}]})
        self.assertNotIn('<script>alert(1)</script>', body)
        self.assertIn('&lt;script&gt;', body)


@unittest.skipUnless(os.environ.get('GLOBAL_SIGNALS_TEST_DB') == '1', 'Set GLOBAL_SIGNALS_TEST_DB=1 for owned-DB integration checks')
class PostgresTests(unittest.TestCase):
    def test_owned_database_and_forecast_guards(self):
        from global_signals.db import connect
        with connect() as conn:
            self.assertEqual(conn.execute("SELECT value FROM global_signal_meta WHERE key='project_key'").fetchone()[0], 'global_asset_signals_v1')
            self.assertEqual(conn.execute("SELECT count(*) FROM signal_predictions WHERE signal_color <> 'gray'").fetchone()[0], 0)
            self.assertGreater(conn.execute('SELECT count(*) FROM observations').fetchone()[0], 0)

    def test_new_failed_result_does_not_revive_old_green(self):
        from global_signals.db import connect
        with connect() as conn:
            try:
                run_id = conn.execute("INSERT INTO research_refresh_runs(mode) VALUES ('direct') RETURNING run_id").fetchone()[0]
                conn.execute("INSERT INTO research_signals(run_id,exposure_id,as_of,valid_until,status,research_color,model_version,metrics,validation,reason_codes) VALUES (%s,'equity_us',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP + interval '1 hour','failed','gray','test','{}','{}',ARRAY['required_series_missing'])", (run_id,))
                row = conn.execute("SELECT status,research_color FROM research_latest_signals WHERE exposure_id='equity_us'").fetchone()
                self.assertEqual(row, ('failed', 'gray'))
            finally:
                conn.rollback()

    def test_expired_signal_is_gray(self):
        from global_signals.db import connect
        with connect() as conn:
            try:
                conn.execute("UPDATE research_signals SET as_of=CURRENT_TIMESTAMP - interval '3 days', valid_until=CURRENT_TIMESTAMP - interval '1 day' WHERE exposure_id='equity_us'")
                self.assertEqual(conn.execute("SELECT research_color FROM research_latest_signals WHERE exposure_id='equity_us'").fetchone()[0], 'gray')
            finally:
                conn.rollback()

    def test_observations_are_immutable(self):
        from global_signals.db import connect
        import psycopg
        with connect() as conn:
            with self.assertRaises(psycopg.Error):
                conn.execute('UPDATE observations SET value=value WHERE observation_id=(SELECT min(observation_id) FROM observations)')
            conn.rollback()


if __name__ == '__main__':
    unittest.main()
