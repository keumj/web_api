"""Auditable monthly conditional historical baseline, always research-only.

Full-history adjusted prices have revision/survivorship and first-seen limitations.
Walk-forward diagnostics below are retrospective; they NEVER mark a model validated.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math

import numpy as np
import pandas as pd

from .data import price_series
from .returns import next_calendar_month
from .universe import CASH_TICKER, FX_TICKER

VERSION = 'monthly-conditional-v1'
MIN_TRAIN = 24
MIN_REGIME = 8
MIN_EVALUATIONS = 12
MAX_AGE_DAYS = 7


def distribution(history, regime):
    """Callers provide only labels whose exits strictly precede decision time."""
    if len(history) < MIN_TRAIN:
        return None
    conditional = [r['excess'] for r in history if r['regime'] == regime]
    samples = conditional if len(conditional) >= MIN_REGIME else [r['excess'] for r in history]
    a = np.asarray(samples)
    q = np.quantile(a, [0.1, 0.5, 0.9])
    return {'probability_outperform_cash': float((np.sum(a > 0) + 1) / (len(a) + 2)),
            'excess_return_p10': float(q[0]), 'excess_return_p50': float(q[1]),
            'excess_return_p90': float(q[2]), 'sample_count': len(a),
            'conditional': len(conditional) >= MIN_REGIME}


def color_for(prediction):
    if prediction is None:
        return 'gray'
    p = prediction['probability_outperform_cash']
    median = prediction['excess_return_p50']
    if p >= 0.6 and median > 0 and prediction['excess_return_p10'] >= -0.1:
        return 'green'
    if p <= 0.4 and median < 0:
        return 'red'
    return 'yellow'


def evaluate(snapshots, ticker, *, cost_fraction=0.002, now=None):
    now = now or datetime.now(timezone.utc)
    if not math.isfinite(cost_fraction) or not 0 <= cost_fraction < 1:
        raise ValueError('Round-trip assumed cost must be a finite fraction in [0,1)')
    required = (ticker, FX_TICKER, CASH_TICKER)
    if any(s not in snapshots for s in required):
        return empty_result('failed', 'required_series_missing')
    if any(pd.Timestamp(snapshots[s].first_seen) > pd.Timestamp(now) for s in required):
        return empty_result('failed', 'snapshot_after_cutoff')
    prices = pd.concat({s: price_series(snapshots[s]) for s in required}, axis=1, join='inner').sort_index().dropna()
    prices = prices[prices.index < pd.Timestamp(now.date())]
    if len(prices) < 160:
        return empty_result('insufficient', 'short_history')
    krw = prices[ticker] * prices[FX_TICKER]
    cash = prices[CASH_TICKER]
    momentum = krw / krw.shift(63) - 1
    regime = ((momentum > 0) & (krw > krw.rolling(126).mean())).astype(int)
    # Use actual common trading dates, first tradable observation AFTER decision.
    # Cross-market fixings remain asynchronous, so this is not an executable backtest.
    decisions = prices.groupby(prices.index.to_period('M')).tail(1).index
    labels = []
    for decision in decisions:
        if prices.index.get_loc(decision) < 125:
            continue
        entry_pos = prices.index.searchsorted(decision, side='right')
        if entry_pos >= len(prices):
            continue
        entry = prices.index[entry_pos]
        target = pd.Timestamp(next_calendar_month(entry.date()))
        exit_pos = prices.index.searchsorted(target)
        if exit_pos >= len(prices):
            continue
        end = prices.index[exit_pos]
        if (entry - decision).days > 7 or (end - target).days > 7:
            continue
        excess = float(krw[end] / krw[entry] - cash[end] / cash[entry] - cost_fraction)
        labels.append(dict(decision=decision, entry=entry, end=end, regime=int(regime[decision]), excess=excess))
    diagnostics = []
    for row in labels:
        train = [past for past in labels if past['end'] < row['decision']]
        prediction = distribution(train, row['regime'])
        if prediction:
            baseline_p = (sum(x['excess'] > 0 for x in train) + 1) / (len(train) + 2)
            diagnostics.append((prediction['probability_outperform_cash'], baseline_p, float(row['excess'] > 0)))
    last = prices.index[-1]
    train = [row for row in labels if row['end'] < last]
    prediction = distribution(train, int(regime.iloc[-1]))
    validation = {'status': 'retrospective_only', 'production_approved': False,
                  'evaluation_count': len(diagnostics), 'minimum_evaluations': MIN_EVALUATIONS,
                  'availability_policy': 'downloaded_history_not_historical_vintages',
                  'entry_policy': 'next_common_daily_close_after_month_end',
                  'exit_policy': 'first_common_close_on_or_after_one_calendar_month',
                  'training_policy': 'expanding_window_label_exit_strictly_before_decision',
                  'limitations': ['adjusted_history_revisions', 'asynchronous_fx_and_market_closes',
                                  'cash_etf_proxy_not_deposit', 'assumed_costs_exclude_tax',
                                  'monthly_labels_may_overlap', 'no_point_in_time_archive']}
    if diagnostics:
        a = np.asarray(diagnostics)
        brier = float(np.mean((a[:, 0] - a[:, 2]) ** 2))
        baseline = float(np.mean((a[:, 1] - a[:, 2]) ** 2))
        validation.update(brier_score=brier, baseline_brier_score=baseline,
                          diagnostic_gate_passed=len(a) >= MIN_EVALUATIONS and brier < baseline)
    metrics = {**(prediction or {}), 'ticker': ticker, 'cash_benchmark': CASH_TICKER,
               'currency_treatment': 'unhedged', 'reporting_currency': 'KRW',
               'assumed_round_trip_cost_fraction': cost_fraction,
               'momentum_3m': float(momentum.iloc[-1]),
               'history_rows': len(prices), 'matured_labels': len(train),
               'data_start': prices.index[0].date().isoformat(),
               'snapshot_hashes': {s: snapshots[s].digest for s in required},
               'first_seen': {s: snapshots[s].first_seen for s in required}}
    age = (now.date() - last.date()).days
    gap = int(prices.index.to_series().diff().dt.days.tail(126).max())
    status = 'ready' if prediction else 'insufficient'
    reasons = ['research_only', 'no_point_in_time_validation']
    if age > MAX_AGE_DAYS:
        status = 'stale'
        reasons.append('stale_input')
    if gap > 10:
        status = 'insufficient'
        reasons.append('input_gap')
    if not prediction:
        reasons.append('too_few_matured_labels')
    return {'status': status, 'research_color': color_for(prediction) if status == 'ready' else 'gray',
            'data_date': last.date(), 'metrics': metrics, 'validation': validation, 'reason_codes': reasons}


def empty_result(status, reason):
    return {'status': status, 'research_color': 'gray', 'data_date': None, 'metrics': {},
            'validation': {'status': 'not_evaluated', 'production_approved': False}, 'reason_codes': [reason]}
