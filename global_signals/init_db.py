"""Create or migrate an explicitly owned, dedicated PostgreSQL database."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict
from psycopg.types.json import Jsonb

from .catalog import SOURCES, EXPOSURES
from .db import connect, connection_info, database_name

PROJECT_KEY = 'global_asset_signals_v1'
SCHEMA_VERSION = '4'


def initialize(database=None):
    target = database or conninfo_to_dict(connection_info()).get('dbname') or database_name()
    if not target or len(target.encode('utf-8')) > 63 or target in {'postgres', 'template0', 'template1', 'krx'}:
        raise ValueError('Use a dedicated global signals database, not an existing shared database')
    created = False
    with connect(admin=True, autocommit=True) as admin:
        if not admin.execute('SELECT 1 FROM pg_database WHERE datname=%s', (target,)).fetchone():
            admin.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier(target)))
            created = True
    with connect(database=target) as conn:
        conn.execute('SELECT pg_advisory_xact_lock(79244001)')
        tables = {r[0] for r in conn.execute("SELECT tablename FROM pg_tables WHERE schemaname='public'")}
        if tables:
            if 'global_signal_meta' not in tables:
                raise ValueError('Existing database is not owned by global signals; no changes made')
            marker = conn.execute("SELECT value FROM global_signal_meta WHERE key='project_key'").fetchone()
            if marker != (PROJECT_KEY,):
                raise ValueError('Database project marker does not match; no changes made')
        conn.execute(Path(__file__).with_name('schema.sql').read_text(encoding='utf-8'), prepare=False)
        conn.execute(Path(__file__).with_name('research_schema.sql').read_text(encoding='utf-8'), prepare=False)
        with conn.cursor() as cursor:
            cursor.executemany('INSERT INTO global_signal_meta(key,value) VALUES (%s,%s) ON CONFLICT(key) DO NOTHING', [
                ('project_key', PROJECT_KEY), ('base_currency', 'KRW'),
                ('primary_horizon', '1_calendar_month'), ('system_status', 'research_baseline'),
            ])
            cursor.execute("INSERT INTO global_signal_meta(key,value) VALUES ('schema_version',%s) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (SCHEMA_VERSION,))
            cursor.executemany('INSERT INTO data_sources(source_id,label,category,documentation_url,access_notes) VALUES (%s,%s,%s,%s,%s) ON CONFLICT(source_id) DO NOTHING', [
                (r['source_id'], r['label'], r['category'], r['documentation_url'], r['access_notes']) for r in SOURCES
            ])
            cursor.executemany('INSERT INTO asset_exposures(exposure_id,label,asset_class,country_code,reference_currency,monitor_only,attributes) VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(exposure_id) DO NOTHING', [
                (r['exposure_id'], r['label'], r['asset_class'], r['country_code'], r['reference_currency'], r['monitor_only'], Jsonb(r['attributes'])) for r in EXPOSURES
            ])
        exposures = conn.execute('SELECT count(*) FROM asset_exposures').fetchone()[0]
    return dict(database=target, created=created, registered_exposures=exposures, schema_version=SCHEMA_VERSION)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database')
    args = parser.parse_args(argv)
    try:
        print(json.dumps(initialize(args.database), ensure_ascii=False))
    except (ValueError, psycopg.Error) as exc:
        print('Initialization failed:', type(exc).__name__, 'SQLSTATE:', getattr(exc, 'sqlstate', None))
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
