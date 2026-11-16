from datetime import datetime, timezone
import unittest
from unittest.mock import patch

import pandas as pd

from global_signals.catalog import EXPOSURES
from global_signals.universe import BINDINGS, SERIES, FX_TICKER, CASH_TICKER, series_id
from global_signals.data import make_snapshot, period_end
from global_signals.model import krw_fx, krw_asset
from global_signals.analysis import analyze_context


NOW = datetime(2026, 11, 16, tzinfo=timezone.utc)


def fixture(key, a, b, dates=('2026-11-12', '2026-11-13')):
    return make_snapshot(key, SERIES[key]['currency'], list(zip(dates, (a, b))), NOW.isoformat())


class CoverageTests(unittest.TestCase):
    def test_every_catalogue_item_has_data_and_explanation(self):
        self.assertEqual(set(BINDINGS), {r['exposure_id'] for r in EXPOSURES})
        self.assertEqual(len(BINDINGS), 114)
        for exposure, binding in BINDINGS.items():
            self.assertTrue(binding['series'], exposure)
            self.assertTrue(binding['note'], exposure)
            self.assertEqual(len(binding['series']), len(set(binding['series'])))
            self.assertTrue(set(binding['series']).issubset(SERIES), exposure)

    def test_context_never_treated_as_forecast(self):
        for row in EXPOSURES:
            if row['asset_class'] in ('physical_property', 'foreign_cash', 'cash'):
                self.assertEqual(BINDINGS[row['exposure_id']]['mode'], 'context')

    def test_source_ids_are_unique(self):
        ids = [series_id(key) for key in SERIES]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_asset_is_linked_in_dashboard(self):
        from app.routers.global_signals import render_dashboard
        rows = [{**r, 'research_color': 'gray', 'reason_codes': [], 'status': 'insufficient'} for r in EXPOSURES]
        page = render_dashboard({'signals': rows})
        for exposure in BINDINGS:
            self.assertIn('href="/global-signals/' + exposure + '"', page)


class CurrencyTests(unittest.TestCase):
    def test_jpy_divides_usdjpy_no_100_yen_scaling(self):
        snapshots = {FX_TICKER: fixture(FX_TICKER, 1300, 1400), 'JPY=X': fixture('JPY=X', 100, 140)}
        self.assertEqual(list(krw_fx(snapshots, 'JPY')), [13, 10])

    def test_eur_multiplies_usd_per_eur(self):
        snapshots = {FX_TICKER: fixture(FX_TICKER, 1300, 1400), 'EURUSD=X': fixture('EURUSD=X', 1.1, 1.2)}
        self.assertAlmostEqual(krw_fx(snapshots, 'EUR').iloc[-1], 1680)

    def test_usd_listed_country_etf_is_converted_once(self):
        snapshots = {FX_TICKER: fixture(FX_TICKER, 1300, 1400), 'EWJ': fixture('EWJ', 10, 11)}
        self.assertEqual(list(krw_asset(snapshots, BINDINGS['equity_jp'])), [13000, 15400])

    def test_krw_instrument_does_not_require_usd_fx(self):
        key = '329200.KS'
        self.assertEqual(list(krw_asset({key: fixture(key, 5000, 5100)}, BINDINGS['listed_reit_kr'])), [5000, 5100])

    def test_fx_is_spot_price_not_interest_bearing_cash(self):
        snapshots = {FX_TICKER: fixture(FX_TICKER, 1300, 1400)}
        self.assertEqual(list(krw_asset(snapshots, BINDINGS['fx_usd_krw'])), [1300, 1400])

    def test_missing_cross_leg_is_not_forward_filled(self):
        snapshots = {FX_TICKER: fixture(FX_TICKER, 1300, 1400),
                     'JPY=X': fixture('JPY=X', 100, 140, ('2026-11-11','2026-11-12'))}
        result = krw_fx(snapshots, 'JPY')
        self.assertEqual(len(result), 1)
        self.assertEqual(result.index[0], pd.Timestamp('2026-11-12'))


class ContextTests(unittest.TestCase):
    def test_negative_and_zero_interest_rates_are_valid(self):
        key = 'fred:ECBDFR'
        snapshot = fixture(key, -.5, 0)
        self.assertEqual(snapshot.values[-1][1], 0)

    def test_rate_changes_are_percentage_points(self):
        binding = BINDINGS['foreign_cash_eur_fixed_term_deposit']
        snapshots = {k: fixture(k, 2, 3) for k in binding['series']}
        result = analyze_context(snapshots, 'foreign_cash_eur_fixed_term_deposit', NOW)
        rate = next(r for r in result['metrics']['context_series'] if r['kind'] == 'yield')
        self.assertEqual(rate['change_pp'], 1)
        self.assertNotIn('change_fraction', rate)
        self.assertNotIn('probability_outperform_cash', result['metrics'])
        self.assertEqual(result['research_color'], 'gray')

    def test_quarterly_period_end_and_freshness(self):
        key = 'fred:QUSN628BIS'
        self.assertEqual(period_end(key, '2026-01-01').isoformat(), '2026-03-31')
        result = analyze_context({key: fixture(key, 100, 102, ('2025-10-01', '2026-01-01'))}, 'physical_property_us', NOW)
        self.assertEqual(result['data_date'].isoformat(), '2026-03-31')
        self.assertEqual(result['status'], 'ready')
        self.assertEqual(result['research_color'], 'gray')

    def test_missing_input_stays_visible_with_partial_analysis(self):
        key = 'fred:SOFR'
        result = analyze_context({key: fixture(key, 4, 4.1)}, 'foreign_cash_usd_treasury_bill', NOW)
        self.assertEqual(result['status'], 'insufficient')
        self.assertIn(FX_TICKER, result['metrics']['missing_series'])

    def test_future_snapshot_is_not_used(self):
        key = 'fred:QUSN628BIS'
        snapshot = fixture(key, 100, 102, ('2025-10-01', '2026-01-01'))
        snapshot.first_seen = '2026-12-01T00:00:00+00:00'
        result = analyze_context({key: snapshot}, 'physical_property_us', NOW)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['metrics']['context_series'], [])


if __name__ == '__main__':
    unittest.main()
