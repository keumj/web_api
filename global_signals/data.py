"""Full provider snapshots with immutable SQLite recovery copies and first-seen times."""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .db import ROOT
from .universe import FX_TICKER, SYMBOLS

LOCAL_DB = ROOT / 'data' / 'global_signals.sqlite'


@dataclass
class Snapshot:
    ticker: str
    currency: str
    first_seen: str
    digest: str
    values: list


def make_snapshot(ticker, currency, values, first_seen=None):
    if ticker not in SYMBOLS or currency != SYMBOLS[ticker]:
        raise ValueError('Unexpected symbol or quote currency')
    clean = []
    for day, value in values:
        day = pd.Timestamp(day).date().isoformat()
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError('Non-finite or nonpositive provider price')
        clean.append([day, value])
    clean.sort()
    if len(clean) < 2 or len({r[0] for r in clean}) != len(clean):
        raise ValueError('Insufficient or duplicate provider observations')
    digest = hashlib.sha256(json.dumps([ticker, currency, clean], separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return Snapshot(ticker, currency, first_seen or datetime.now(timezone.utc).isoformat(), digest, clean)


def fetch_snapshot(ticker, *, now=None):
    import yfinance as yf
    now = now or datetime.now(timezone.utc)
    cache = ROOT / 'data' / '.yfinance_cache'
    cache.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache))
    instrument = yf.Ticker(ticker)
    # Exclude today's potentially unfinished daily bar in every market.
    frame = instrument.history(start='2010-01-01', end=now.date().isoformat(),
                               auto_adjust=False, actions=False, raise_errors=True)
    field = 'Close' if ticker == FX_TICKER else 'Adj Close'
    if frame.empty or field not in frame:
        raise ValueError('Provider returned no required adjusted-price series')
    metadata = instrument.get_history_metadata()
    currency = metadata.get('currency')
    if currency != SYMBOLS[ticker]:
        raise ValueError('Provider quote currency does not match configured currency')
    values = [(d.date().isoformat(), v) for d, v in frame[field].items()]
    return make_snapshot(ticker, currency, values, datetime.now(timezone.utc).isoformat())


def save_snapshot(snapshot, path=LOCAL_DB):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute('CREATE TABLE IF NOT EXISTS snapshots (digest TEXT PRIMARY KEY, ticker TEXT NOT NULL, currency TEXT NOT NULL, first_seen TEXT NOT NULL, values_json TEXT NOT NULL)')
        conn.execute('INSERT OR IGNORE INTO snapshots VALUES (?,?,?,?,?)',
                     (snapshot.digest, snapshot.ticker, snapshot.currency, snapshot.first_seen, json.dumps(snapshot.values, allow_nan=False)))


def load_snapshots(path=LOCAL_DB):
    # Only explicit upload-local workflows call this. HTTP reads always use PostgreSQL.
    path = Path(path)
    if not path.exists():
        raise ValueError('Local snapshots missing; run local refresh first')
    snapshots = {}
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as conn:
        # A failed latest download must not silently revive yesterday's snapshot.
        current = conn.execute('SELECT payload FROM refresh_manifest WHERE id=1').fetchone()
        if current is None:
            raise ValueError('No completed local collection manifest')
        digests = set(json.loads(current[0]).values())
        if not digests:
            return {}
        for digest, ticker, currency, first_seen, payload in conn.execute('SELECT digest,ticker,currency,first_seen,values_json FROM snapshots ORDER BY first_seen,digest'):
            if digest not in digests:
                continue
            snap = make_snapshot(ticker, currency, json.loads(payload), first_seen)
            if snap.digest != digest:
                raise ValueError('Local snapshot checksum mismatch')
            snapshots[ticker] = snap
    return snapshots


def save_manifest(snapshots, path=LOCAL_DB):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute('CREATE TABLE IF NOT EXISTS refresh_manifest(id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL)')
        conn.execute('INSERT OR REPLACE INTO refresh_manifest VALUES (1,?)',
                     (json.dumps({s: v.digest for s, v in snapshots.items()}),))


def price_series(snapshot):
    return pd.Series([r[1] for r in snapshot.values], index=pd.to_datetime([r[0] for r in snapshot.values]), dtype=float)
