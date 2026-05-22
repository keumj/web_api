from __future__ import annotations

import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SHARED_DB_ROOT = Path(os.getenv("KEUMJ_SP500_DB_DIR", "data/sp500_shared_db"))
DEFAULT_SQLITE_NAME = str(os.getenv("KEUMJ_SP500_DB_SQLITE_NAME", "sp500_shared_prices.sqlite")).strip() or "sp500_shared_prices.sqlite"
TREASURY_DATASET = "treasury_yields" # Define TREASURY_DATASET here
NEWS_ANALYSIS_STATUS_PENDING = "pending"
NEWS_ANALYSIS_STATUS_PROCESSING = "processing"
NEWS_ANALYSIS_STATUS_DONE = "done"
NEWS_ANALYSIS_STATUS_FAILED = "failed"
VALID_NEWS_ANALYSIS_STATUSES = {
    NEWS_ANALYSIS_STATUS_PENDING,
    NEWS_ANALYSIS_STATUS_PROCESSING,
    NEWS_ANALYSIS_STATUS_DONE,
    NEWS_ANALYSIS_STATUS_FAILED,
}
ALLOWED_NEWS_ANALYSIS_TRANSITIONS: dict[str, set[str]] = {
    NEWS_ANALYSIS_STATUS_PENDING: {NEWS_ANALYSIS_STATUS_PROCESSING, NEWS_ANALYSIS_STATUS_FAILED},
    NEWS_ANALYSIS_STATUS_PROCESSING: {
        NEWS_ANALYSIS_STATUS_PENDING,
        NEWS_ANALYSIS_STATUS_DONE,
        NEWS_ANALYSIS_STATUS_FAILED,
    },
    NEWS_ANALYSIS_STATUS_DONE: {NEWS_ANALYSIS_STATUS_PROCESSING},
    NEWS_ANALYSIS_STATUS_FAILED: {NEWS_ANALYSIS_STATUS_PENDING, NEWS_ANALYSIS_STATUS_PROCESSING},
}


@dataclass(frozen=True)
class SharedPricesBuildResult:
    db_path: Path
    source_dir: Path
    file_count: int
    row_count: int


@dataclass(frozen=True)
class NewsArticleRow:
    id: int
    ticker: str
    publish_date: str
    title: str
    link: str
    source: str
    sentiment_score: float | None
    analysis_status: str


@dataclass(frozen=True)
class FundamentalsSnapshotRow:
    symbol: str
    as_of_date: str
    roe: float | None
    per: float | None
    pbr: float | None
    source: str


@dataclass(frozen=True)
class QuarterlyFundamentalsRow:
    symbol: str
    fiscal_date: str
    filing_date: str | None
    period_type: str
    net_income: float | None
    diluted_eps: float | None
    stockholders_equity: float | None
    total_assets: float | None
    total_debt: float | None
    current_assets: float | None
    current_liabilities: float | None
    operating_cash_flow: float | None
    free_cash_flow: float | None
    source: str


def _normalize_symbol(symbol: str) -> str:
    return re.sub(r"[^A-Z0-9._-]+", "_", str(symbol or "").strip().upper())


def _resolve_shared_root(shared_db_root: Path | str | None = None) -> Path:
    if shared_db_root is None:
        return Path(os.getenv("KEUMJ_SP500_DB_DIR", "data/sp500_shared_db"))
    return Path(shared_db_root)


def shared_prices_csv_dir(shared_db_root: Path | str | None = None) -> Path:
    return _resolve_shared_root(shared_db_root) / "prices"


def shared_prices_sqlite_path(shared_db_root: Path | str | None = None) -> Path:
    explicit = str(os.getenv("KEUMJ_SP500_DB_SQLITE_PATH", "")).strip()
    if explicit:
        return Path(explicit)
    return _resolve_shared_root(shared_db_root) / DEFAULT_SQLITE_NAME


def _env_first(*names: str) -> str:
    for name in names:
        value = str(os.getenv(name, "")).strip()
        if value:
            return value
    return ""


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def shared_prices_database_url() -> str:
    try:
        from app.settings import settings

        if settings.sp500_database_url:
            return settings.sp500_database_url
    except Exception:
        pass
    return _env_first("KEUMJ_SP500_DATABASE_URL", "KEUMJ_SP500_TURSO_DATABASE_URL", "TURSO_SP500_DATABASE_URL")


def shared_prices_database_auth_token() -> str:
    try:
        from app.settings import settings

        if settings.sp500_database_auth_token:
            return settings.sp500_database_auth_token
    except Exception:
        pass
    return _env_first("KEUMJ_SP500_DATABASE_AUTH_TOKEN", "KEUMJ_SP500_TURSO_AUTH_TOKEN", "TURSO_SP500_AUTH_TOKEN")


def shared_prices_supabase_database_url() -> str:
    try:
        from app.settings import settings

        if settings.sp500_supabase_database_url:
            return settings.sp500_supabase_database_url
    except Exception:
        pass
    return _env_first("KEUMJ_SP500_SUPABASE_DATABASE_URL")


def shared_prices_supabase_config() -> dict[str, object]:
    try:
        from app.settings import settings

        host = settings.sp500_supabase_host
        user = settings.sp500_supabase_user
        password = settings.sp500_supabase_password
        database = settings.sp500_supabase_database
        port = settings.sp500_supabase_port
        sslmode = settings.sp500_supabase_sslmode
    except Exception:
        host = os.getenv("KEUMJ_SP500_SUPABASE_HOST", "").strip()
        user = os.getenv("KEUMJ_SP500_SUPABASE_USER", "").strip()
        password = os.getenv("KEUMJ_SP500_SUPABASE_PASSWORD", "")
        database = os.getenv("KEUMJ_SP500_SUPABASE_DATABASE", "postgres").strip() or "postgres"
        port = _env_int("KEUMJ_SP500_SUPABASE_PORT", 6543)
        sslmode = os.getenv("KEUMJ_SP500_SUPABASE_SSLMODE", "require").strip() or "require"
    if not host or not user:
        return {}
    config: dict[str, object] = {
        "host": host,
        "port": int(port),
        "dbname": database,
        "user": user,
        "sslmode": sslmode,
    }
    if password:
        config["password"] = password
    return config


def using_supabase_shared_prices_db() -> bool:
    return bool(shared_prices_supabase_database_url() or shared_prices_supabase_config())


def using_remote_shared_prices_db() -> bool:
    return bool(shared_prices_database_url() or using_supabase_shared_prices_db())


def shared_prices_embedded_replica_enabled() -> bool:
    return bool(shared_prices_database_url()) and not using_supabase_shared_prices_db() and _env_bool("KEUMJ_TURSO_EMBEDDED_REPLICA", True)


def shared_prices_replica_path() -> Path:
    explicit = str(os.getenv("KEUMJ_SP500_REPLICA_PATH", "")).strip()
    if explicit:
        return Path(explicit)
    replica_dir = Path(os.getenv("KEUMJ_TURSO_REPLICA_DIR", "data/turso_replicas"))
    return replica_dir / "sp500_shared_prices_replica.sqlite"


def shared_prices_storage_label(target: Path | str | None = None) -> str:
    supabase_url = shared_prices_supabase_database_url()
    supabase_config = shared_prices_supabase_config()
    if supabase_url:
        return "supabase-postgres"
    if supabase_config:
        return f"supabase-postgres:{supabase_config.get('host')}:{supabase_config.get('port')}/{supabase_config.get('dbname')}"
    url = shared_prices_database_url()
    if url:
        if shared_prices_embedded_replica_enabled():
            return f"turso-replica:{shared_prices_replica_path().as_posix()}<-{url}"
        return f"turso:{url}"
    path = Path(target) if target is not None else shared_prices_sqlite_path()
    return f"sqlite:{path.as_posix()}"


_SHARED_REPLICA_LAST_SYNC: dict[str, float] = {}


class _RemoteConnectionProxy:
    _keumj_remote_libsql = True

    def __init__(self, conn, *, replica_path: Path | None = None):
        self._conn = conn
        self.replica_path = replica_path
        self._keumj_embedded_replica = replica_path is not None

    def __getattr__(self, name: str):
        return getattr(self._conn, name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        close = getattr(self._conn, "close", None)
        if close is not None:
            close()


def _translate_qmark_params(query: str) -> str:
    query = re.sub(r"\bdate\(([^()]+)\)", r"\1::date", query)
    return query.replace("?", "%s")


class _PostgresCursorProxy:
    def __init__(self, cursor):
        self._cursor = cursor

    def __getattr__(self, name: str):
        return getattr(self._cursor, name)

    def fetchone(self):
        return self._cursor.fetchone()

    def fetchall(self):
        return self._cursor.fetchall()


class _PostgresConnectionProxy:
    _keumj_remote_postgres = True
    total_changes = 0

    def __init__(self, conn):
        self._conn = conn

    def __getattr__(self, name: str):
        return getattr(self._conn, name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def execute(self, query: str, params: list[object] | tuple[object, ...] | None = None):
        normalized = query.strip().upper()
        if normalized == "BEGIN IMMEDIATE":
            query = "BEGIN"
        cur = self._conn.execute(_translate_qmark_params(query), tuple(params or ()))
        return _PostgresCursorProxy(cur)

    def executemany(self, query: str, params_seq):
        cur = self._conn.cursor()
        cur.executemany(_translate_qmark_params(query), params_seq)
        return _PostgresCursorProxy(cur)


def _connect_shared_prices_postgres():
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError(
            "KEUMJ_SP500_SUPABASE_* is set, but the 'psycopg[binary]' package is not installed."
        ) from exc
    url = shared_prices_supabase_database_url()
    if url:
        return _PostgresConnectionProxy(psycopg.connect(url, prepare_threshold=None))
    config = shared_prices_supabase_config()
    if not config:
        raise RuntimeError(
            "Set KEUMJ_SP500_SUPABASE_DATABASE_URL or KEUMJ_SP500_SUPABASE_HOST/USER/PASSWORD."
        )
    return _PostgresConnectionProxy(psycopg.connect(**config, prepare_threshold=None))


def _maybe_sync_shared_replica(conn, replica_path: Path) -> None:
    sync = getattr(conn, "sync", None)
    if sync is None:
        return
    interval = max(_env_int("KEUMJ_TURSO_REPLICA_SYNC_SECONDS", 300), 1)
    key = str(replica_path.resolve())
    now = time.monotonic()
    if key in _SHARED_REPLICA_LAST_SYNC and interval > 0 and now - _SHARED_REPLICA_LAST_SYNC[key] < interval:
        return
    try:
        sync()
    except Exception:
        if replica_path.exists() and replica_path.stat().st_size > 0:
            return
        raise
    _SHARED_REPLICA_LAST_SYNC[key] = now


def _connect_shared_prices_read_db(target: Path | str | None = None):
    if using_supabase_shared_prices_db():
        return _connect_shared_prices_postgres()
    url = shared_prices_database_url()
    if url:
        try:
            import libsql
        except ImportError as exc:
            raise RuntimeError(
                "KEUMJ_SP500_DATABASE_URL is set, but the 'libsql' package is not installed."
            ) from exc
        token = shared_prices_database_auth_token()
        if shared_prices_embedded_replica_enabled():
            replica_path = shared_prices_replica_path()
            replica_path.parent.mkdir(parents=True, exist_ok=True)
            kwargs: dict[str, object] = {
                "database": str(replica_path),
                "sync_url": url,
                "sync_interval": max(_env_int("KEUMJ_TURSO_REPLICA_SYNC_SECONDS", 300), 1),
            }
            if token:
                kwargs["auth_token"] = token
            conn = libsql.connect(**kwargs)
            _maybe_sync_shared_replica(conn, replica_path)
            return _RemoteConnectionProxy(conn, replica_path=replica_path)
        kwargs: dict[str, object] = {"database": url}
        if token:
            kwargs["auth_token"] = token
        return _RemoteConnectionProxy(libsql.connect(**kwargs))
    path = Path(target) if target is not None else shared_prices_sqlite_path()
    return sqlite3.connect(path)


def read_sql_dataframe(conn, query: str, params: list[object] | tuple[object, ...] | None = None) -> pd.DataFrame:
    if isinstance(conn, sqlite3.Connection):
        return pd.read_sql_query(query, conn, params=params)
    if getattr(conn, "_keumj_remote_postgres", False):
        cur = conn.execute(query, tuple(params or ()))
        rows = cur.fetchall()
        description = getattr(cur, "description", None) or []
        columns = [str(col[0]) for col in description]
        return pd.DataFrame([tuple(row) for row in rows], columns=columns)
    cur = conn.execute(query, tuple(params or ()))
    rows = cur.fetchall()
    description = getattr(cur, "description", None) or []
    columns = [str(col[0]) for col in description]
    return pd.DataFrame([tuple(row) for row in rows], columns=columns)


def explain_shared_prices_auth_error(exc: Exception) -> RuntimeError:
    err = RuntimeError(
        "S&P500 Turso DB rejected the connection. Set KEUMJ_SP500_DATABASE_AUTH_TOKEN "
        "in Render if this database requires a token."
    )
    err.__cause__ = exc
    return err


def _ensure_postgres_news_context_views(conn) -> None:
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prices_symbol_date ON prices(symbol, date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_news_articles_ticker_publish_date ON news_articles(ticker, publish_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_news_articles_publish_date ON news_articles(publish_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_news_articles_analysis_status_publish_date ON news_articles(analysis_status, publish_date)")
    conn.execute(
        """
        CREATE OR REPLACE VIEW news_articles_price_context AS
        SELECT
            n.id,
            n.ticker,
            n.publish_date,
            n.publish_date::date AS publish_day,
            n.title,
            n.link,
            n.source,
            n.sentiment_score,
            n.analysis_status,
            p.date AS price_date,
            p.open,
            p.high,
            p.low,
            p.close,
            p.adj_close,
            p.volume,
            p.market_cap
        FROM news_articles AS n
        LEFT JOIN prices AS p
            ON p.symbol = n.ticker
           AND p.date::date = n.publish_date::date
        """
    )
    conn.execute(
        """
        CREATE OR REPLACE VIEW news_articles_market_context AS
        SELECT
            n.id,
            n.ticker,
            n.publish_date,
            n.publish_date::date AS publish_day,
            n.title,
            n.link,
            n.source,
            n.sentiment_score,
            n.analysis_status,
            p.date AS reference_price_date,
            CASE
                WHEN p.date::date = n.publish_date::date THEN 1
                ELSE 0
            END AS matched_on_publish_day,
            p.open,
            p.high,
            p.low,
            p.close,
            p.adj_close,
            p.volume,
            p.market_cap
        FROM news_articles AS n
        LEFT JOIN prices AS p
            ON p.symbol = n.ticker
           AND p.date::date = (
                SELECT MAX(p2.date::date)
                FROM prices AS p2
                WHERE p2.symbol = n.ticker
                  AND p2.date::date <= n.publish_date::date
           )
        """
    )
    conn.commit()


def _shared_prices_available(target: Path | str | None = None) -> bool:
    if using_remote_shared_prices_db():
        return True
    path = Path(target) if target is not None else shared_prices_sqlite_path()
    return path.exists() and path.is_file()


def _read_shared_price_csv(path: Path, symbol: str) -> pd.DataFrame | None:
    if not path.exists() or not path.is_file():
        return None

    try:
        raw = pd.read_csv(path)
    except Exception:
        return None

    if raw.empty:
        return None

    cols = {str(c).strip().lower(): c for c in raw.columns}
    date_col = cols.get("date") or cols.get("datetime") or cols.get("timestamp") or raw.columns[0]
    open_col = cols.get("open")
    high_col = cols.get("high")
    low_col = cols.get("low")
    close_col = cols.get("close") or cols.get("adj close") or cols.get("adjclose") or cols.get("adj_close")
    volume_col = cols.get("volume")
    adj_close_col = cols.get("adj close") or cols.get("adjclose") or cols.get("adj_close")
    dividends_col = cols.get("dividends") or cols.get("dividend")
    stock_splits_col = cols.get("stock splits") or cols.get("stocksplits") or cols.get("stock_splits") or cols.get("split")

    if open_col is None or high_col is None or low_col is None or close_col is None:
        return None

    selected = [date_col, open_col, high_col, low_col, close_col]
    if volume_col is not None:
        selected.append(volume_col)
    if adj_close_col is not None and adj_close_col not in selected:
        selected.append(adj_close_col)
    if dividends_col is not None:
        selected.append(dividends_col)
    if stock_splits_col is not None:
        selected.append(stock_splits_col)
    frame = raw[selected].copy()
    renamed = ["date", "open", "high", "low", "close"]
    if volume_col is not None:
        renamed.append("volume")
    if adj_close_col is not None and adj_close_col not in {date_col, open_col, high_col, low_col, close_col, volume_col}:
        renamed.append("adj_close")
    if dividends_col is not None:
        renamed.append("dividends")
    if stock_splits_col is not None:
        renamed.append("stock_splits")
    frame.columns = renamed

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume", "adj_close", "dividends", "stock_splits"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    if "volume" not in frame.columns:
        frame["volume"] = 0.0
    frame["volume"] = frame["volume"].fillna(0.0)
    if "adj_close" not in frame.columns:
        frame["adj_close"] = frame["close"]
    if "dividends" not in frame.columns:
        frame["dividends"] = 0.0
    if "stock_splits" not in frame.columns:
        frame["stock_splits"] = 0.0
    frame = frame.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date")
    if frame.empty:
        return None

    out = frame.copy()
    out["symbol"] = _normalize_symbol(symbol)
    out["date"] = out["date"].dt.normalize().dt.strftime("%Y-%m-%d")
    out = out[["symbol", "date", "open", "high", "low", "close", "volume", "adj_close", "dividends", "stock_splits"]]
    out = out.drop_duplicates(subset=["symbol", "date"], keep="last")
    return out if not out.empty else None


def _ensure_prices_table_schema(conn: sqlite3.Connection) -> None:
    if getattr(conn, "_keumj_remote_postgres", False):
        _ensure_postgres_news_context_views(conn)
        return
    if (
        getattr(conn, "_keumj_remote_libsql", False)
        or conn.__class__.__module__.split(".", 1)[0] == "libsql"
    ):
        return
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prices (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL DEFAULT 0,
            adj_close REAL,
            dividends REAL NOT NULL DEFAULT 0,
            stock_splits REAL NOT NULL DEFAULT 0,
            market_cap REAL,
            PRIMARY KEY (symbol, date)
        )
        """
    )
    existing_cols = {str(row[1]).strip().lower() for row in conn.execute("PRAGMA table_info(prices)").fetchall()}
    if "adj_close" not in existing_cols:
        conn.execute("ALTER TABLE prices ADD COLUMN adj_close REAL")
    if "dividends" not in existing_cols:
        conn.execute("ALTER TABLE prices ADD COLUMN dividends REAL NOT NULL DEFAULT 0")
    if "stock_splits" not in existing_cols:
        conn.execute("ALTER TABLE prices ADD COLUMN stock_splits REAL NOT NULL DEFAULT 0")
    if "market_cap" not in existing_cols:
        conn.execute("ALTER TABLE prices ADD COLUMN market_cap REAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS news_articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            publish_date DATETIME NOT NULL,
            title TEXT NOT NULL,
            link TEXT NOT NULL,
            source TEXT NOT NULL,
            sentiment_score REAL,
            analysis_status TEXT NOT NULL DEFAULT 'pending'
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fundamentals_snapshot (
            symbol TEXT PRIMARY KEY,
            as_of_date TEXT NOT NULL,
            roe REAL,
            per REAL,
            pbr REAL,
            source TEXT NOT NULL DEFAULT 'unknown',
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fundamentals_quarterly (
            symbol TEXT NOT NULL,
            fiscal_date TEXT NOT NULL,
            filing_date TEXT,
            period_type TEXT NOT NULL DEFAULT 'quarterly',
            net_income REAL,
            diluted_eps REAL,
            stockholders_equity REAL,
            total_assets REAL,
            total_debt REAL,
            current_assets REAL,
            current_liabilities REAL,
            operating_cash_flow REAL,
            free_cash_flow REAL,
            source TEXT NOT NULL DEFAULT 'unknown',
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, fiscal_date)
        )
        """
    )
    fundamentals_cols = {
        str(row[1]).strip().lower() for row in conn.execute("PRAGMA table_info(fundamentals_snapshot)").fetchall()
    }
    if "source" not in fundamentals_cols:
        conn.execute("ALTER TABLE fundamentals_snapshot ADD COLUMN source TEXT NOT NULL DEFAULT 'unknown'")
    if "updated_at" not in fundamentals_cols:
        conn.execute(
            "ALTER TABLE fundamentals_snapshot ADD COLUMN updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP"
        )
    quarterly_cols = {
        str(row[1]).strip().lower() for row in conn.execute("PRAGMA table_info(fundamentals_quarterly)").fetchall()
    }
    if "source" not in quarterly_cols:
        conn.execute("ALTER TABLE fundamentals_quarterly ADD COLUMN source TEXT NOT NULL DEFAULT 'unknown'")
    if "updated_at" not in quarterly_cols:
        conn.execute(
            "ALTER TABLE fundamentals_quarterly ADD COLUMN updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP"
        )
    news_cols = {str(row[1]).strip().lower() for row in conn.execute("PRAGMA table_info(news_articles)").fetchall()}
    if "analysis_status" not in news_cols:
        conn.execute("ALTER TABLE news_articles ADD COLUMN analysis_status TEXT NOT NULL DEFAULT 'pending'")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_news_articles_ticker_publish_date ON news_articles(ticker, publish_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_news_articles_publish_date ON news_articles(publish_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_news_articles_ticker_publish_day ON news_articles(ticker, date(publish_date))")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_news_articles_analysis_status_publish_date ON news_articles(analysis_status, publish_date)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_fundamentals_snapshot_as_of_date ON fundamentals_snapshot(as_of_date)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_fundamentals_quarterly_symbol_fiscal_date ON fundamentals_quarterly(symbol, fiscal_date)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_fundamentals_quarterly_fiscal_date ON fundamentals_quarterly(fiscal_date)"
    )
    conn.execute(
        """
        CREATE VIEW IF NOT EXISTS news_articles_price_context AS
        SELECT
            n.id,
            n.ticker,
            n.publish_date,
            date(n.publish_date) AS publish_day,
            n.title,
            n.link,
            n.source,
            n.sentiment_score,
            n.analysis_status,
            p.date AS price_date,
            p.open,
            p.high,
            p.low,
            p.close,
            p.adj_close,
            p.volume,
            p.market_cap
        FROM news_articles AS n
        LEFT JOIN prices AS p
            ON p.symbol = n.ticker
           AND p.date = date(n.publish_date)
        """
    )
    conn.execute(
        """
        CREATE VIEW IF NOT EXISTS news_articles_market_context AS
        SELECT
            n.id,
            n.ticker,
            n.publish_date,
            date(n.publish_date) AS publish_day,
            n.title,
            n.link,
            n.source,
            n.sentiment_score,
            n.analysis_status,
            p.date AS reference_price_date,
            CASE
                WHEN p.date = date(n.publish_date) THEN 1
                ELSE 0
            END AS matched_on_publish_day,
            p.open,
            p.high,
            p.low,
            p.close,
            p.adj_close,
            p.volume,
            p.market_cap
        FROM news_articles AS n
        LEFT JOIN prices AS p
            ON p.symbol = n.ticker
           AND p.date = (
                SELECT MAX(p2.date)
                FROM prices AS p2
                WHERE p2.symbol = n.ticker
                  AND p2.date <= date(n.publish_date)
           )
        """
    )


def _normalize_fundamentals_frame(
    frame: pd.DataFrame,
    *,
    default_as_of_date: str | None = None,
    source: str = "csv",
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["symbol", "as_of_date", "roe", "per", "pbr", "source"])
    raw = frame.copy()
    cols = {str(col).strip().lower(): col for col in raw.columns}
    symbol_col = cols.get("symbol") or cols.get("ticker")
    if symbol_col is None:
        raise ValueError("Fundamentals frame requires a symbol or ticker column")
    as_of_col = cols.get("as_of_date") or cols.get("date")
    roe_col = cols.get("roe")
    per_col = cols.get("per") or cols.get("pe")
    pbr_col = cols.get("pbr") or cols.get("pb")
    source_col = cols.get("source")
    out = pd.DataFrame()
    out["symbol"] = raw[symbol_col].astype(str).str.strip().str.upper()
    if as_of_col is not None:
        out["as_of_date"] = pd.to_datetime(raw[as_of_col], errors="coerce").dt.normalize()
    else:
        fallback = pd.Timestamp(default_as_of_date).normalize() if default_as_of_date else pd.Timestamp.today().normalize()
        out["as_of_date"] = fallback
    out["roe"] = pd.to_numeric(raw[roe_col], errors="coerce") if roe_col is not None else pd.Series(np.nan, index=raw.index)
    out["per"] = pd.to_numeric(raw[per_col], errors="coerce") if per_col is not None else pd.Series(np.nan, index=raw.index)
    out["pbr"] = pd.to_numeric(raw[pbr_col], errors="coerce") if pbr_col is not None else pd.Series(np.nan, index=raw.index)
    if source_col is not None:
        out["source"] = raw[source_col].astype(str).str.strip().replace({"": source})
    else:
        out["source"] = source
    out = out.dropna(subset=["symbol", "as_of_date"])
    out = out[(out["symbol"] != "")]
    out["as_of_date"] = pd.to_datetime(out["as_of_date"], errors="coerce").dt.normalize().dt.strftime("%Y-%m-%d")
    out = out.drop_duplicates(subset=["symbol"], keep="last")
    return out[["symbol", "as_of_date", "roe", "per", "pbr", "source"]].reset_index(drop=True)


def upsert_shared_fundamentals_snapshot(
    frame: pd.DataFrame,
    *,
    shared_db_root: Path | str | None = None,
    db_path: Path | str | None = None,
    default_as_of_date: str | None = None,
    source: str = "csv",
) -> int:
    normalized = _normalize_fundamentals_frame(frame, default_as_of_date=default_as_of_date, source=source)
    if normalized.empty:
        return 0
    root = _resolve_shared_root(shared_db_root)
    target = Path(db_path) if db_path is not None else shared_prices_sqlite_path(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = list(normalized[["symbol", "as_of_date", "roe", "per", "pbr", "source"]].itertuples(index=False, name=None))
    with sqlite3.connect(target) as conn:
        _ensure_prices_table_schema(conn)
        before_changes = conn.total_changes
        conn.executemany(
            """
            INSERT INTO fundamentals_snapshot(symbol, as_of_date, roe, per, pbr, source)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                as_of_date = excluded.as_of_date,
                roe = excluded.roe,
                per = excluded.per,
                pbr = excluded.pbr,
                source = excluded.source,
                updated_at = CURRENT_TIMESTAMP
            """,
            rows,
        )
        conn.commit()
        changed_rows = conn.total_changes - before_changes
    return int(changed_rows)


def sync_shared_fundamentals_snapshot_csv(
    csv_path: Path | str,
    *,
    shared_db_root: Path | str | None = None,
    db_path: Path | str | None = None,
    default_as_of_date: str | None = None,
    source: str | None = None,
) -> int:
    path = Path(csv_path)
    if not path.exists() or not path.is_file():
        return 0
    try:
        raw = pd.read_csv(path)
    except Exception:
        return 0
    if raw.empty:
        return 0
    source_value = source or f"csv:{path.as_posix()}"
    return upsert_shared_fundamentals_snapshot(
        raw,
        shared_db_root=shared_db_root,
        db_path=db_path,
        default_as_of_date=default_as_of_date,
        source=source_value,
    )


def load_shared_fundamentals_for_symbols(
    symbols: list[str],
    *,
    shared_db_root: Path | str | None = None,
    db_path: Path | str | None = None,
) -> tuple[pd.DataFrame | None, str | None]:
    normalized = []
    seen: set[str] = set()
    for symbol in symbols:
        clean = _normalize_symbol(symbol)
        if not clean or clean in seen:
            continue
        seen.add(clean)
        normalized.append(clean)
    if not normalized:
        return None, None
    root = _resolve_shared_root(shared_db_root)
    target = Path(db_path) if db_path is not None else shared_prices_sqlite_path(root)
    if not _shared_prices_available(target):
        return None, None
    with _connect_shared_prices_read_db(target) as conn:
        _ensure_prices_table_schema(conn)
        placeholders = ",".join("?" for _ in normalized)
        frame = read_sql_dataframe(
            conn,
            f"""
            SELECT symbol, as_of_date, roe, per, pbr, source, updated_at
            FROM fundamentals_snapshot
            WHERE symbol IN ({placeholders})
            ORDER BY symbol ASC
            """,
            params=normalized,
        )
    if frame.empty:
        return None, None
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in ["roe", "per", "pbr"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame, shared_prices_storage_label(target)


def _normalize_quarterly_fundamentals_frame(
    frame: pd.DataFrame,
    *,
    source: str = "unknown",
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "fiscal_date",
                "filing_date",
                "period_type",
                "net_income",
                "diluted_eps",
                "stockholders_equity",
                "total_assets",
                "total_debt",
                "current_assets",
                "current_liabilities",
                "operating_cash_flow",
                "free_cash_flow",
                "source",
            ]
        )
    raw = frame.copy()
    cols = {str(col).strip().lower(): col for col in raw.columns}
    symbol_col = cols.get("symbol") or cols.get("ticker")
    fiscal_col = cols.get("fiscal_date") or cols.get("period_end") or cols.get("as_of_date") or cols.get("date")
    if symbol_col is None or fiscal_col is None:
        raise ValueError("Quarterly fundamentals frame requires symbol/ticker and fiscal_date/date columns")
    filing_col = cols.get("filing_date")
    period_col = cols.get("period_type") or cols.get("period")
    out = pd.DataFrame()
    out["symbol"] = raw[symbol_col].astype(str).str.strip().str.upper()
    out["fiscal_date"] = pd.to_datetime(raw[fiscal_col], errors="coerce").dt.normalize()
    out["filing_date"] = (
        pd.to_datetime(raw[filing_col], errors="coerce").dt.normalize()
        if filing_col is not None
        else pd.NaT
    )
    out["period_type"] = (
        raw[period_col].astype(str).str.strip().replace({"": "quarterly"})
        if period_col is not None
        else "quarterly"
    )
    alias_map = {
        "net_income": ["net_income", "net income"],
        "diluted_eps": ["diluted_eps", "diluted eps", "eps", "basic_eps"],
        "stockholders_equity": ["stockholders_equity", "stockholders equity", "equity"],
        "total_assets": ["total_assets", "total assets", "assets"],
        "total_debt": ["total_debt", "total debt", "debt"],
        "current_assets": ["current_assets", "current assets"],
        "current_liabilities": ["current_liabilities", "current liabilities"],
        "operating_cash_flow": ["operating_cash_flow", "operating cash flow", "ocf"],
        "free_cash_flow": ["free_cash_flow", "free cash flow", "fcf"],
    }
    for target_col, aliases in alias_map.items():
        source_col = next((cols.get(alias) for alias in aliases if cols.get(alias) is not None), None)
        out[target_col] = pd.to_numeric(raw[source_col], errors="coerce") if source_col is not None else np.nan
    source_col = cols.get("source")
    out["source"] = (
        raw[source_col].astype(str).str.strip().replace({"": source})
        if source_col is not None
        else source
    )
    out = out.dropna(subset=["symbol", "fiscal_date"])
    out = out[(out["symbol"] != "")]
    out["fiscal_date"] = pd.to_datetime(out["fiscal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["filing_date"] = pd.to_datetime(out["filing_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out = out.sort_values(["symbol", "fiscal_date"]).drop_duplicates(subset=["symbol", "fiscal_date"], keep="last")
    return out.reset_index(drop=True)


def upsert_shared_quarterly_fundamentals(
    frame: pd.DataFrame,
    *,
    shared_db_root: Path | str | None = None,
    db_path: Path | str | None = None,
    source: str = "unknown",
) -> int:
    normalized = _normalize_quarterly_fundamentals_frame(frame, source=source)
    if normalized.empty:
        return 0
    root = _resolve_shared_root(shared_db_root)
    target = Path(db_path) if db_path is not None else shared_prices_sqlite_path(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = list(
        normalized[
            [
                "symbol",
                "fiscal_date",
                "filing_date",
                "period_type",
                "net_income",
                "diluted_eps",
                "stockholders_equity",
                "total_assets",
                "total_debt",
                "current_assets",
                "current_liabilities",
                "operating_cash_flow",
                "free_cash_flow",
                "source",
            ]
        ].itertuples(index=False, name=None)
    )
    with sqlite3.connect(target) as conn:
        _ensure_prices_table_schema(conn)
        before_changes = conn.total_changes
        conn.executemany(
            """
            INSERT INTO fundamentals_quarterly(
                symbol, fiscal_date, filing_date, period_type, net_income, diluted_eps,
                stockholders_equity, total_assets, total_debt, current_assets,
                current_liabilities, operating_cash_flow, free_cash_flow, source
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, fiscal_date) DO UPDATE SET
                filing_date = excluded.filing_date,
                period_type = excluded.period_type,
                net_income = excluded.net_income,
                diluted_eps = excluded.diluted_eps,
                stockholders_equity = excluded.stockholders_equity,
                total_assets = excluded.total_assets,
                total_debt = excluded.total_debt,
                current_assets = excluded.current_assets,
                current_liabilities = excluded.current_liabilities,
                operating_cash_flow = excluded.operating_cash_flow,
                free_cash_flow = excluded.free_cash_flow,
                source = excluded.source,
                updated_at = CURRENT_TIMESTAMP
            """,
            rows,
        )
        conn.commit()
        changed_rows = conn.total_changes - before_changes
    return int(changed_rows)


def load_shared_quarterly_fundamentals_for_symbols(
    symbols: list[str],
    *,
    limit_per_symbol: int | None = 4,
    shared_db_root: Path | str | None = None,
    db_path: Path | str | None = None,
) -> tuple[pd.DataFrame | None, str | None]:
    normalized = []
    seen: set[str] = set()
    for symbol in symbols:
        clean = _normalize_symbol(symbol)
        if not clean or clean in seen:
            continue
        seen.add(clean)
        normalized.append(clean)
    if not normalized:
        return None, None
    root = _resolve_shared_root(shared_db_root)
    target = Path(db_path) if db_path is not None else shared_prices_sqlite_path(root)
    if not _shared_prices_available(target):
        return None, None
    with _connect_shared_prices_read_db(target) as conn:
        _ensure_prices_table_schema(conn)
        placeholders = ",".join("?" for _ in normalized)
        query = f"""
            SELECT
                symbol,
                fiscal_date,
                filing_date,
                period_type,
                net_income,
                diluted_eps,
                stockholders_equity,
                total_assets,
                total_debt,
                current_assets,
                current_liabilities,
                operating_cash_flow,
                free_cash_flow,
                source,
                updated_at
            FROM fundamentals_quarterly
            WHERE symbol IN ({placeholders})
            ORDER BY symbol ASC, fiscal_date DESC
        """
        frame = read_sql_dataframe(conn, query, params=normalized)
    if frame.empty:
        return None, None
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["fiscal_date"] = pd.to_datetime(frame["fiscal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame["filing_date"] = pd.to_datetime(frame["filing_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in [
        "net_income",
        "diluted_eps",
        "stockholders_equity",
        "total_assets",
        "total_debt",
        "current_assets",
        "current_liabilities",
        "operating_cash_flow",
        "free_cash_flow",
    ]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if limit_per_symbol is not None and int(limit_per_symbol) > 0:
        frame = frame.groupby("symbol", as_index=False, group_keys=False).head(int(limit_per_symbol)).reset_index(drop=True)
    return frame, shared_prices_storage_label(target)


def _normalize_news_analysis_status(value: str) -> str:
    status = str(value or "").strip().lower()
    if status not in VALID_NEWS_ANALYSIS_STATUSES:
        raise ValueError(
            "Invalid news analysis status "
            f"'{value}'. Choose from: {', '.join(sorted(VALID_NEWS_ANALYSIS_STATUSES))}"
        )
    return status


def _news_article_row_from_tuple(row: tuple[object, ...]) -> NewsArticleRow:
    return NewsArticleRow(
        id=int(row[0]),
        ticker=str(row[1]),
        publish_date=str(row[2]),
        title=str(row[3]),
        link=str(row[4]),
        source=str(row[5]),
        sentiment_score=float(row[6]) if row[6] is not None else None,
        analysis_status=str(row[7]),
    )


def claim_pending_news_articles_for_analysis(
    limit: int,
    *,
    ticker: str | None = None,
    shared_db_root: Path | str | None = None,
    db_path: Path | str | None = None,
) -> list[NewsArticleRow]:
    if int(limit) <= 0:
        raise ValueError("limit must be positive")

    root = _resolve_shared_root(shared_db_root)
    target = Path(db_path) if db_path is not None else shared_prices_sqlite_path(root)
    if not _shared_prices_available(target):
        return []

    with _connect_shared_prices_read_db(target) as conn:
        _ensure_prices_table_schema(conn)
        conn.execute("BEGIN IMMEDIATE")
        query = (
            "SELECT id, ticker, publish_date, title, link, source, sentiment_score, analysis_status "
            "FROM news_articles "
            "WHERE analysis_status = ?"
        )
        params: list[object] = [NEWS_ANALYSIS_STATUS_PENDING]
        ticker_clean = str(ticker or "").strip().upper()
        if ticker_clean:
            query += " AND ticker = ?"
            params.append(ticker_clean)
        query += " ORDER BY publish_date ASC, id ASC LIMIT ?"
        params.append(int(limit))
        rows = conn.execute(query, params).fetchall()
        if not rows:
            conn.commit()
            return []
        ids = [int(row[0]) for row in rows]
        placeholders = ",".join("?" for _ in ids)
        conn.execute(
            f"UPDATE news_articles SET analysis_status = ? WHERE id IN ({placeholders})",
            [NEWS_ANALYSIS_STATUS_PROCESSING, *ids],
        )
        conn.commit()
    return [
        NewsArticleRow(
            id=int(row[0]),
            ticker=str(row[1]),
            publish_date=str(row[2]),
            title=str(row[3]),
            link=str(row[4]),
            source=str(row[5]),
            sentiment_score=float(row[6]) if row[6] is not None else None,
            analysis_status=NEWS_ANALYSIS_STATUS_PROCESSING,
        )
        for row in rows
    ]


def update_news_article_analysis_status(
    article_id: int,
    new_status: str,
    *,
    expected_current_status: str | None = None,
    shared_db_root: Path | str | None = None,
    db_path: Path | str | None = None,
) -> bool:
    status_to = _normalize_news_analysis_status(new_status)
    expected_from = _normalize_news_analysis_status(expected_current_status) if expected_current_status is not None else None

    root = _resolve_shared_root(shared_db_root)
    target = Path(db_path) if db_path is not None else shared_prices_sqlite_path(root)
    if not _shared_prices_available(target):
        return False

    with _connect_shared_prices_read_db(target) as conn:
        _ensure_prices_table_schema(conn)
        row = conn.execute(
            "SELECT analysis_status FROM news_articles WHERE id = ?",
            (int(article_id),),
        ).fetchone()
        if row is None:
            return False
        current_status = _normalize_news_analysis_status(str(row[0]))
        if expected_from is not None and current_status != expected_from:
            return False
        if status_to != current_status and status_to not in ALLOWED_NEWS_ANALYSIS_TRANSITIONS.get(current_status, set()):
            raise ValueError(f"Invalid news analysis status transition: {current_status} -> {status_to}")
        before = conn.total_changes
        conn.execute(
            "UPDATE news_articles SET analysis_status = ? WHERE id = ?",
            (status_to, int(article_id)),
        )
        conn.commit()
        return conn.total_changes > before


def mark_news_article_analysis_done(
    article_id: int,
    *,
    shared_db_root: Path | str | None = None,
    db_path: Path | str | None = None,
) -> bool:
    return update_news_article_analysis_status(
        article_id,
        NEWS_ANALYSIS_STATUS_DONE,
        expected_current_status=NEWS_ANALYSIS_STATUS_PROCESSING,
        shared_db_root=shared_db_root,
        db_path=db_path,
    )


def mark_news_article_analysis_failed(
    article_id: int,
    *,
    shared_db_root: Path | str | None = None,
    db_path: Path | str | None = None,
) -> bool:
    return update_news_article_analysis_status(
        article_id,
        NEWS_ANALYSIS_STATUS_FAILED,
        expected_current_status=NEWS_ANALYSIS_STATUS_PROCESSING,
        shared_db_root=shared_db_root,
        db_path=db_path,
    )


def build_sp500_shared_prices_sqlite(
    shared_db_root: Path | str | None = None,
    *,
    db_path: Path | str | None = None,
    overwrite: bool = True,
) -> SharedPricesBuildResult:
    root = _resolve_shared_root(shared_db_root)
    source_dir = shared_prices_csv_dir(root)
    target = Path(db_path) if db_path is not None else shared_prices_sqlite_path(root)

    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"shared prices directory not found: {source_dir}")

    csv_paths = sorted(source_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"no symbol CSV files found in: {source_dir}")

    target.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and target.exists():
        target.unlink()

    row_count = 0
    with sqlite3.connect(target) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        _ensure_prices_table_schema(conn)
        if overwrite:
            conn.execute("DELETE FROM prices")

        insert_sql = (
            "INSERT OR REPLACE INTO prices "
            "(symbol, date, open, high, low, close, volume, adj_close, dividends, stock_splits) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )

        for csv_path in csv_paths:
            frame = _read_shared_price_csv(csv_path, csv_path.stem)
            if frame is None or frame.empty:
                continue

            rows = list(frame.itertuples(index=False, name=None))
            conn.executemany(insert_sql, rows)
            row_count += len(rows)

        conn.execute("CREATE INDEX IF NOT EXISTS idx_prices_symbol_date ON prices(symbol, date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date)")
        conn.commit()

    return SharedPricesBuildResult(
        db_path=target,
        source_dir=source_dir,
        file_count=len(csv_paths),
        row_count=row_count,
    )


def _query_shared_prices_sqlite(
    query: str,
    params: list[object],
    *,
    db_path: Path,
) -> pd.DataFrame:
    with _connect_shared_prices_read_db(db_path) as conn:
        return read_sql_dataframe(conn, query, params=params)


def load_shared_ohlcv_for_symbol(
    symbol: str,
    *,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    shared_db_root: Path | str | None = None,
    db_path: Path | str | None = None,
) -> tuple[pd.DataFrame | None, str | None]:
    symbol_clean = _normalize_symbol(symbol)
    if not symbol_clean:
        return None, None

    root = _resolve_shared_root(shared_db_root)
    target = Path(db_path) if db_path is not None else shared_prices_sqlite_path(root)
    start_text = pd.Timestamp(start_date).normalize().strftime("%Y-%m-%d") if start_date is not None else None
    end_text = pd.Timestamp(end_date).normalize().strftime("%Y-%m-%d") if end_date is not None else None

    if _shared_prices_available(target):
        with _connect_shared_prices_read_db(target) as conn:
            _ensure_prices_table_schema(conn)
        query = "SELECT date, open, high, low, close, volume FROM prices WHERE symbol = ?"
        params: list[object] = [symbol_clean]
        if start_text is not None:
            query += " AND date >= ?"
            params.append(start_text)
        if end_text is not None:
            query += " AND date <= ?"
            params.append(end_text)
        query += " ORDER BY date"

        try:
            raw = _query_shared_prices_sqlite(query, params, db_path=target)
        except Exception:
            raw = pd.DataFrame()

        if not raw.empty:
            raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
            for col in ["open", "high", "low", "close", "volume"]:
                raw[col] = pd.to_numeric(raw[col], errors="coerce")
            raw = raw.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date")
            if not raw.empty:
                out = raw.set_index(raw["date"].dt.normalize())[["open", "high", "low", "close", "volume"]]
                out = out[~out.index.duplicated(keep="last")].sort_index()
                return out, shared_prices_storage_label(target)

    return None, None


def load_shared_close_prices_for_symbols(
    symbols: list[str],
    *,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp | None = None,
    shared_db_root: Path | str | None = None,
    db_path: Path | str | None = None,
) -> tuple[pd.DataFrame | None, str | None]:
    normalized = []
    seen: set[str] = set()
    for symbol in symbols:
        clean = _normalize_symbol(symbol)
        if not clean or clean in seen:
            continue
        seen.add(clean)
        normalized.append(clean)

    if not normalized:
        return None, None

    root = _resolve_shared_root(shared_db_root)
    target = Path(db_path) if db_path is not None else shared_prices_sqlite_path(root)
    start_text = pd.Timestamp(start_date).normalize().strftime("%Y-%m-%d")
    end_text = pd.Timestamp(end_date).normalize().strftime("%Y-%m-%d") if end_date is not None else None

    if _shared_prices_available(target):
        with _connect_shared_prices_read_db(target) as conn:
            _ensure_prices_table_schema(conn)
        placeholders = ",".join(["?"] * len(normalized))
        query = f"SELECT date, symbol, close FROM prices WHERE symbol IN ({placeholders})"
        params: list[object] = list(normalized)
        query += " AND date >= ?"
        params.append(start_text)
        if end_text is not None:
            query += " AND date <= ?"
            params.append(end_text)
        query += " ORDER BY date, symbol"

        pivot = _load_shared_metric_pivot(target, query=query, params=params, value_col="close", symbols=normalized)
        if pivot is not None:
            return pivot, shared_prices_storage_label(target)

    return None, None


def _load_shared_metric_pivot(
    target: Path,
    *,
    query: str,
    params: list[object],
    value_col: str,
    symbols: list[str],
) -> pd.DataFrame | None:
    try:
        raw = _query_shared_prices_sqlite(query, params, db_path=target)
    except Exception:
        raw = pd.DataFrame()

    if raw.empty:
        return None

    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw[value_col] = pd.to_numeric(raw[value_col], errors="coerce")
    raw = raw.dropna(subset=["date", "symbol", value_col])
    if raw.empty:
        return None
    pivot = raw.pivot_table(index="date", columns="symbol", values=value_col, aggfunc="last").sort_index()
    columns = [symbol for symbol in symbols if symbol in pivot.columns]
    if not columns:
        return None
    pivot = pivot[columns].dropna(how="all")
    return pivot if not pivot.empty else None


def load_shared_adjusted_close_prices_for_symbols(
    symbols: list[str],
    *,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp | None = None,
    shared_db_root: Path | str | None = None,
    db_path: Path | str | None = None,
) -> tuple[pd.DataFrame | None, str | None]:
    normalized = []
    seen: set[str] = set()
    for symbol in symbols:
        clean = _normalize_symbol(symbol)
        if not clean or clean in seen:
            continue
        seen.add(clean)
        normalized.append(clean)

    if not normalized:
        return None, None

    root = _resolve_shared_root(shared_db_root)
    target = Path(db_path) if db_path is not None else shared_prices_sqlite_path(root)
    start_text = pd.Timestamp(start_date).normalize().strftime("%Y-%m-%d")
    end_text = pd.Timestamp(end_date).normalize().strftime("%Y-%m-%d") if end_date is not None else None

    if _shared_prices_available(target):
        with _connect_shared_prices_read_db(target) as conn:
            _ensure_prices_table_schema(conn)
        placeholders = ",".join(["?"] * len(normalized))
        query = f"SELECT date, symbol, adj_close FROM prices WHERE symbol IN ({placeholders})"
        params: list[object] = list(normalized)
        query += " AND date >= ?"
        params.append(start_text)
        if end_text is not None:
            query += " AND date <= ?"
            params.append(end_text)
        query += " AND adj_close IS NOT NULL ORDER BY date, symbol"
        pivot = _load_shared_metric_pivot(target, query=query, params=params, value_col="adj_close", symbols=normalized)
        if pivot is not None:
            return pivot, shared_prices_storage_label(target)

    return None, None


def load_shared_dividends_for_symbols(
    symbols: list[str],
    *,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp | None = None,
    shared_db_root: Path | str | None = None,
    db_path: Path | str | None = None,
) -> tuple[pd.DataFrame | None, str | None]:
    normalized = []
    seen: set[str] = set()
    for symbol in symbols:
        clean = _normalize_symbol(symbol)
        if not clean or clean in seen:
            continue
        seen.add(clean)
        normalized.append(clean)

    if not normalized:
        return None, None

    root = _resolve_shared_root(shared_db_root)
    target = Path(db_path) if db_path is not None else shared_prices_sqlite_path(root)
    start_text = pd.Timestamp(start_date).normalize().strftime("%Y-%m-%d")
    end_text = pd.Timestamp(end_date).normalize().strftime("%Y-%m-%d") if end_date is not None else None

    if _shared_prices_available(target):
        with _connect_shared_prices_read_db(target) as conn:
            _ensure_prices_table_schema(conn)
        placeholders = ",".join(["?"] * len(normalized))
        query = f"SELECT date, symbol, dividends FROM prices WHERE symbol IN ({placeholders})"
        params: list[object] = list(normalized)
        query += " AND date >= ?"
        params.append(start_text)
        if end_text is not None:
            query += " AND date <= ?"
            params.append(end_text)
        query += " AND dividends IS NOT NULL ORDER BY date, symbol"
        pivot = _load_shared_metric_pivot(target, query=query, params=params, value_col="dividends", symbols=normalized)
        if pivot is not None:
            return pivot, shared_prices_storage_label(target)

    return None, None


def load_shared_stock_splits_for_symbols(
    symbols: list[str],
    *,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp | None = None,
    shared_db_root: Path | str | None = None,
    db_path: Path | str | None = None,
) -> tuple[pd.DataFrame | None, str | None]:
    normalized = []
    seen: set[str] = set()
    for symbol in symbols:
        clean = _normalize_symbol(symbol)
        if not clean or clean in seen:
            continue
        seen.add(clean)
        normalized.append(clean)

    if not normalized:
        return None, None

    root = _resolve_shared_root(shared_db_root)
    target = Path(db_path) if db_path is not None else shared_prices_sqlite_path(root)
    start_text = pd.Timestamp(start_date).normalize().strftime("%Y-%m-%d")
    end_text = pd.Timestamp(end_date).normalize().strftime("%Y-%m-%d") if end_date is not None else None

    if _shared_prices_available(target):
        with _connect_shared_prices_read_db(target) as conn:
            _ensure_prices_table_schema(conn)
        placeholders = ",".join(["?"] * len(normalized))
        query = f"SELECT date, symbol, stock_splits FROM prices WHERE symbol IN ({placeholders})"
        params: list[object] = list(normalized)
        query += " AND date >= ?"
        params.append(start_text)
        if end_text is not None:
            query += " AND date <= ?"
            params.append(end_text)
        query += " AND stock_splits IS NOT NULL ORDER BY date, symbol"
        pivot = _load_shared_metric_pivot(target, query=query, params=params, value_col="stock_splits", symbols=normalized)
        if pivot is not None:
            return pivot, shared_prices_storage_label(target)

    return None, None


def load_shared_market_caps_for_symbols(
    symbols: list[str],
    *,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp | None = None,
    shared_db_root: Path | str | None = None,
    db_path: Path | str | None = None,
) -> tuple[pd.DataFrame | None, str | None]:
    normalized = []
    seen: set[str] = set()
    for symbol in symbols:
        clean = _normalize_symbol(symbol)
        if not clean or clean in seen:
            continue
        seen.add(clean)
        normalized.append(clean)

    if not normalized:
        return None, None

    root = _resolve_shared_root(shared_db_root)
    target = Path(db_path) if db_path is not None else shared_prices_sqlite_path(root)
    start_text = pd.Timestamp(start_date).normalize().strftime("%Y-%m-%d")
    end_text = pd.Timestamp(end_date).normalize().strftime("%Y-%m-%d") if end_date is not None else None

    if _shared_prices_available(target):
        with _connect_shared_prices_read_db(target) as conn:
            _ensure_prices_table_schema(conn)
        placeholders = ",".join(["?"] * len(normalized))
        query = f"SELECT date, symbol, market_cap FROM prices WHERE symbol IN ({placeholders})"
        params: list[object] = list(normalized)
        query += " AND date >= ?"
        params.append(start_text)
        if end_text is not None:
            query += " AND date <= ?"
            params.append(end_text)
        query += " AND market_cap IS NOT NULL ORDER BY date, symbol"

        pivot = _load_shared_metric_pivot(target, query=query, params=params, value_col="market_cap", symbols=normalized)
        if pivot is not None:
            return pivot, shared_prices_storage_label(target)

    return None, None


def load_financial_market_series(
    dataset: str,
    *,
    series_id: str,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp | None = None,
    shared_db_root: Path | str | None = None,
    db_path: Path | str | None = None,
) -> tuple[pd.Series | None, str | None]:
    root = _resolve_shared_root(shared_db_root)
    target = Path(db_path) if db_path is not None else shared_prices_sqlite_path(root)
    start_text = pd.Timestamp(start_date).normalize().strftime("%Y-%m-%d")
    end_text = pd.Timestamp(end_date).normalize().strftime("%Y-%m-%d") if end_date is not None else None

    if not _shared_prices_available(target):
        return None, None

    with _connect_shared_prices_read_db(target) as conn:
        _ensure_prices_table_schema(conn) # Ensure table exists
        query = f"SELECT date, value FROM {dataset} WHERE series_id = ?"
        params: list[object] = [series_id]
        query += " AND date >= ?"
        params.append(start_text)
        if end_text is not None:
            query += " AND date <= ?"
            params.append(end_text)
        query += " ORDER BY date"

        try:
            raw = read_sql_dataframe(conn, query, params=params)
        except Exception:
            return None, None

    if raw.empty:
        return None, None

    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    raw = raw.dropna(subset=["date", "value"]).sort_values("date")
    if raw.empty:
        return None, None

    s = pd.Series(raw["value"].values, index=raw["date"].dt.normalize(), name=series_id)
    s = s[~s.index.duplicated(keep="last")]
    return s if not s.empty else None, shared_prices_storage_label(target)


def load_financial_market_frame(
    dataset: str,
    *,
    series_ids: list[str],
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp | None = None,
    shared_db_root: Path | str | None = None,
    db_path: Path | str | None = None,
) -> tuple[pd.DataFrame | None, str | None]:
    if not series_ids:
        return None, None

    root = _resolve_shared_root(shared_db_root)
    target = Path(db_path) if db_path is not None else shared_prices_sqlite_path(root)
    start_text = pd.Timestamp(start_date).normalize().strftime("%Y-%m-%d")
    end_text = pd.Timestamp(end_date).normalize().strftime("%Y-%m-%d") if end_date is not None else None

    if not _shared_prices_available(target):
        return None, None

    with _connect_shared_prices_read_db(target) as conn:
        _ensure_prices_table_schema(conn) # Ensure table exists
        placeholders = ",".join(["?"] * len(series_ids))
        query = f"SELECT date, series_id, value FROM {dataset} WHERE series_id IN ({placeholders})"
        params: list[object] = list(series_ids)
        query += " AND date >= ?"
        params.append(start_text)
        if end_text is not None:
            query += " AND date <= ?"
            params.append(end_text)
        query += " ORDER BY date, series_id"

        try:
            raw = read_sql_dataframe(conn, query, params=params)
        except Exception:
            return None, None

    if raw.empty:
        return None, None

    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    raw = raw.dropna(subset=["date", "series_id", "value"])
    if raw.empty:
        return None, None

    pivot = raw.pivot_table(index="date", columns="series_id", values="value", aggfunc="last").sort_index()
    columns = [sid for sid in series_ids if sid in pivot.columns]
    if not columns:
        return None, None
    pivot = pivot[columns].dropna(how="all")
    return pivot if not pivot.empty else None, shared_prices_storage_label(target)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build a SQLite database from per-ticker S&P 500 CSVs.")
    parser.add_argument("--shared-db-root", default=str(DEFAULT_SHARED_DB_ROOT), help="Shared DB root directory")
    parser.add_argument("--db-path", default="", help="Optional output SQLite path")
    parser.add_argument("--keep-existing", action="store_true", help="Keep existing DB rows instead of recreating the file")
    args = parser.parse_args()

    result = build_sp500_shared_prices_sqlite(
        args.shared_db_root,
        db_path=Path(args.db_path) if str(args.db_path).strip() else None,
        overwrite=not args.keep_existing,
    )
    print(
        f"Built SQLite prices DB: {result.db_path} "
        f"(source_dir={result.source_dir}, files={result.file_count}, rows={result.row_count})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
