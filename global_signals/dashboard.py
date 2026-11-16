"""Read immutable PostgreSQL snapshot histories for a selected exposure."""
from datetime import datetime, timezone

from psycopg.rows import dict_row

from .db import connect
from .data import period_end
from .universe import BINDINGS, SERIES


def read_asset(exposure):
    if exposure not in BINDINGS:
        raise LookupError(exposure)
    with connect() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute('SELECT * FROM research_latest_signals WHERE exposure_id=%s', (exposure,))
        signal = cur.fetchone()
        if signal is None:
            raise LookupError(exposure)
        metrics = signal.get('metrics') or {}
        ingestions = metrics.get('ingestion_run_ids', {})
        sources = []
        for key in BINDINGS[exposure]['series']:
            source = {'key': key, **SERIES[key], 'values': [], 'first_seen': None}
            run_id = ingestions.get(key)
            if run_id is not None:
                # Only this signal's exact immutable snapshot, not a mixture of revisions.
                cur.execute('SELECT observed_period_start,value,first_seen_at FROM observations WHERE ingestion_run_id=%s ORDER BY observed_period_start', (int(run_id),))
                records = cur.fetchall()
                source['values'] = [[r['observed_period_start'].isoformat(), float(r['value'])] for r in records]
                if records:
                    source['first_seen'] = records[0]['first_seen_at'].isoformat()
            if source['values']:
                age = (datetime.now(timezone.utc).date() - period_end(key, source['values'][-1][0])).days
                source['collection_status'] = 'stale' if age > source['max_age_days'] else 'available'
            else:
                source['collection_status'] = 'missing'
            sources.append(source)
        cur.execute('SELECT as_of,data_date,research_color,status FROM research_signals WHERE exposure_id=%s ORDER BY as_of DESC,research_signal_id DESC LIMIT 30', (exposure,))
        history = cur.fetchall()
    return {'signal': signal, 'binding': BINDINGS[exposure], 'sources': sources, 'signal_history': history}
