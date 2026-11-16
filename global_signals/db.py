"""Explicit PostgreSQL configuration; no runtime SQLite fallback."""
from __future__ import annotations

import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from psycopg.conninfo import conninfo_to_dict, make_conninfo

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = 'global_asset_signals'


def load_config():
    load_dotenv(ROOT / '.env')


def database_name():
    load_config()
    return os.getenv('KEUMJ_GLOBAL_SIGNALS_DB', DEFAULT_DATABASE)


def connection_info(*, admin=False, database=None):
    load_config()
    runtime = os.getenv('KEUMJ_GLOBAL_SIGNALS_DATABASE_URL', '')
    administrative = os.getenv('KEUMJ_GLOBAL_SIGNALS_ADMIN_URL', '')
    value = (administrative or runtime) if admin else (runtime or administrative)
    if not value:
        raise ValueError('Configure KEUMJ_GLOBAL_SIGNALS_DATABASE_URL or KEUMJ_GLOBAL_SIGNALS_ADMIN_URL')
    parts = conninfo_to_dict(value)
    if database is not None:
        parts['dbname'] = database
    elif admin:
        parts['dbname'] = 'postgres'
    elif not runtime:
        parts['dbname'] = database_name()
    return make_conninfo(**parts)


def connect(*, database=None, admin=False, autocommit=False):
    return psycopg.connect(connection_info(admin=admin, database=database),
                           connect_timeout=10, prepare_threshold=None,
                           autocommit=autocommit)
