"""Descriptive analyses for non-tradable references and product candidates."""
from datetime import datetime, timezone

import pandas as pd

from .data import price_series, period_end
from .model import empty_result
from .universe import BINDINGS, SERIES


def analyze_context(snapshots, exposure, now=None):
    now = now or datetime.now(timezone.utc)
    binding = BINDINGS[exposure]
    missing, stale, rows = [], [], []
    for key in binding['series']:
        spec = SERIES[key]
        if key not in snapshots or pd.Timestamp(snapshots[key].first_seen) > pd.Timestamp(now):
            missing.append(key)
            continue
        values = price_series(snapshots[key])
        values = values[values.index < pd.Timestamp(now.date())]
        if len(values) < 2:
            missing.append(key)
            continue
        latest, previous = float(values.iloc[-1]), float(values.iloc[-2])
        end = period_end(key, values.index[-1])
        row = dict(series=key, label=spec['label'], kind=spec['kind'], unit=spec['currency'],
                   frequency=spec['frequency'], latest=latest, previous=previous,
                   latest_date=values.index[-1].date().isoformat(), observations=len(values),
                   latest_period_end=end.isoformat(),
                   source_url=spec['url'], first_seen=snapshots[key].first_seen)
        if spec['kind'] == 'yield':
            row['change_pp'] = latest - previous
        else:
            row['change_fraction'] = latest / previous - 1
        prior_year = values[values.index <= values.index[-1] - pd.DateOffset(years=1)]
        if not prior_year.empty:
            if spec['kind'] == 'yield':
                row['year_change_pp'] = latest - float(prior_year.iloc[-1])
            else:
                row['year_change_fraction'] = latest / float(prior_year.iloc[-1]) - 1
        row['stale'] = (now.date() - end).days > spec['max_age_days']
        if row['stale']:
            stale.append(key)
        rows.append(row)
    result = empty_result('ready', 'context_only')
    if not rows:
        result['status'] = 'failed'
    elif missing:
        result['status'] = 'insufficient'
    elif stale:
        result['status'] = 'stale'
    if missing:
        result['reason_codes'].append('required_series_missing')
    if stale:
        result['reason_codes'].append('stale_input')
    result['data_date'] = min((pd.Timestamp(r['latest_period_end']).date() for r in rows), default=None)
    result['metrics'] = {'analysis_mode': 'context', 'context_series': rows, 'missing_series': missing,
                         'stale_series': stale, 'coverage_note': binding['note'],
                         'snapshot_hashes': {s: snapshots[s].digest for s in binding['series'] if s in snapshots}}
    result['validation'] = {'status': 'context_only', 'production_approved': False,
                            'limitations': ['not_a_product_return', 'not_a_forecast']}
    return result
