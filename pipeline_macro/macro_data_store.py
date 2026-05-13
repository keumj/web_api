from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_MACRO_DB_PATH = Path(os.getenv("MACRO_PRICES_SQLITE_PATH", "data/macro_prices.sqlite"))


@dataclass(frozen=True)
class MacroSeriesSpec:
    series_id: str
    dataset: str
    frequency: str
    fred_id: str | None = None
    local_csv: str | None = None
    local_name: str | None = None
    yahoo_symbol: str | None = None


TREASURY_SERIES_IDS = ["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS3", "DGS5", "DGS7", "DGS10", "DGS20", "DGS30"]

FRED_SPECS: list[MacroSeriesSpec] = [
    *[MacroSeriesSpec(series_id=sid, dataset="treasury", frequency="daily", fred_id=sid, local_csv="data/treasury_yields.csv") for sid in TREASURY_SERIES_IDS],
    MacroSeriesSpec("SP500", "market", "daily", "SP500"),
    MacroSeriesSpec("VIX", "risk", "daily", "VIXCLS"),
    MacroSeriesSpec("IG_OAS", "credit", "daily", "BAMLC0A0CM"),
    MacroSeriesSpec("HY_OAS", "credit", "daily", "BAMLH0A0HYM2"),
    MacroSeriesSpec("BAA10Y", "credit", "daily", "BAA10Y"),
    MacroSeriesSpec("AAA10Y", "credit", "daily", "AAA10Y"),
    MacroSeriesSpec("UNRATE", "macro", "monthly", "UNRATE"),
    MacroSeriesSpec("PAYEMS", "macro", "monthly", "PAYEMS"),
    MacroSeriesSpec("CPIAUCSL", "macro", "monthly", "CPIAUCSL"),
    MacroSeriesSpec("PCEPILFE", "macro", "monthly", "PCEPILFE"),
    MacroSeriesSpec("INDPRO", "macro", "monthly", "INDPRO"),
    MacroSeriesSpec("RSAFS", "macro", "monthly", "RSAFS"),
    MacroSeriesSpec("HOUST", "macro", "monthly", "HOUST"),
    MacroSeriesSpec("UMCSENT", "macro", "monthly", "UMCSENT"),
    MacroSeriesSpec("M2SL", "macro", "weekly", "M2SL"),
    MacroSeriesSpec("FEDFUNDS", "macro", "monthly", "FEDFUNDS"),
    MacroSeriesSpec("GDPC1", "macro", "quarterly", "GDPC1"),
]

LOCAL_ONLY_SPECS: list[MacroSeriesSpec] = [
    MacroSeriesSpec("DXY", "market", "daily", local_csv="data/dxy.csv", local_name="DXY", yahoo_symbol="DX-Y.NYB"),
]

ALL_SPECS = [*FRED_SPECS, *LOCAL_ONLY_SPECS]


def macro_db_path(path: str | os.PathLike[str] | None = None) -> Path:
    return Path(path) if path else DEFAULT_MACRO_DB_PATH


def _env_first(*names: str) -> str:
    for name in names:
        value = str(os.getenv(name, "")).strip()
        if value:
            return value
    return ""


def macro_database_url() -> str:
    try:
        from app.settings import settings

        if settings.macro_database_url:
            return settings.macro_database_url
    except Exception:
        pass
    return _env_first("KEUMJ_MACRO_DATABASE_URL", "KEUMJ_MACRO_TURSO_DATABASE_URL", "TURSO_MACRO_DATABASE_URL")


def macro_database_auth_token() -> str:
    try:
        from app.settings import settings

        if settings.macro_database_auth_token:
            return settings.macro_database_auth_token
    except Exception:
        pass
    return _env_first("KEUMJ_MACRO_DATABASE_AUTH_TOKEN", "KEUMJ_MACRO_TURSO_AUTH_TOKEN", "TURSO_MACRO_AUTH_TOKEN")


def using_remote_macro_db() -> bool:
    return bool(macro_database_url())


def macro_storage_label(path: str | os.PathLike[str] | None = None) -> str:
    url = macro_database_url()
    if url:
        return f"turso:{url}"
    return f"macro_sqlite:{macro_db_path(path).as_posix()}"


class _RemoteConnectionProxy:
    _keumj_remote_libsql = True

    def __init__(self, conn):
        self._conn = conn

    def __getattr__(self, name: str):
        return getattr(self._conn, name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        close = getattr(self._conn, "close", None)
        if close is not None:
            close()


def _connect_macro_read_db(path: str | os.PathLike[str] | None = None):
    url = macro_database_url()
    if url:
        try:
            import libsql
        except ImportError as exc:
            raise RuntimeError(
                "KEUMJ_MACRO_DATABASE_URL is set, but the 'libsql' package is not installed."
            ) from exc
        kwargs: dict[str, str] = {"database": url}
        token = macro_database_auth_token()
        if token:
            kwargs["auth_token"] = token
        return _RemoteConnectionProxy(libsql.connect(**kwargs))
    return sqlite3.connect(macro_db_path(path))


def explain_macro_auth_error(exc: Exception) -> RuntimeError:
    err = RuntimeError(
        "Macro Turso DB rejected the connection. Set KEUMJ_MACRO_DATABASE_AUTH_TOKEN "
        "in Render if this database requires a token."
    )
    err.__cause__ = exc
    return err


def _macro_db_available(path: str | os.PathLike[str] | None = None) -> bool:
    return using_remote_macro_db() or macro_db_path(path).is_file()


def fred_api_key() -> str:
    key = str(os.getenv("FRED_API_KEY", "")).strip()
    if key:
        return key
    if os.name != "nt":
        return ""
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as reg_key:
            value, _ = winreg.QueryValueEx(reg_key, "FRED_API_KEY")
    except Exception:
        return ""
    return str(value).strip()


def ensure_macro_schema(conn: sqlite3.Connection) -> None:
    if getattr(conn, "_keumj_remote_libsql", False) or conn.__class__.__module__.split(".", 1)[0] == "libsql":
        return
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS macro_series (
            series_id TEXT NOT NULL,
            date TEXT NOT NULL,
            value REAL NOT NULL,
            dataset TEXT NOT NULL,
            frequency TEXT NOT NULL,
            source TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (series_id, date)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_macro_series_date ON macro_series(date)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS macro_metadata (
            series_id TEXT PRIMARY KEY,
            dataset TEXT NOT NULL,
            frequency TEXT NOT NULL,
            source TEXT NOT NULL,
            min_date TEXT,
            max_date TEXT,
            row_count INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def normalize_series(series: pd.Series, *, series_id: str, start_date: str | pd.Timestamp | None = None) -> pd.Series:
    clean = pd.Series(series).copy()
    clean.index = pd.to_datetime(clean.index, errors="coerce")
    clean = clean[~clean.index.isna()]
    clean = pd.to_numeric(clean, errors="coerce").dropna().sort_index()
    clean.index = clean.index.normalize()
    clean = clean[~clean.index.duplicated(keep="last")]
    if start_date is not None:
        clean = clean[clean.index >= pd.Timestamp(start_date).normalize()]
    clean.name = series_id
    return clean


def read_local_series(csv_path: str | os.PathLike[str], *, series_id: str, start_date: str | pd.Timestamp | None = None) -> pd.Series | None:
    path = Path(csv_path)
    if not path.is_file():
        return None
    try:
        raw = pd.read_csv(path)
    except Exception:
        return None
    if raw.empty:
        return None
    cols = {str(col).strip().lower(): col for col in raw.columns}
    date_col = cols.get("date") or cols.get("datetime") or raw.columns[0]
    value_col = None
    for candidate in [series_id, series_id.lower(), "close", "adj close", "adj_close", "value", "price"]:
        mapped = cols.get(str(candidate).lower())
        if mapped is not None:
            value_col = mapped
            break
    if value_col is None and series_id in raw.columns:
        value_col = series_id
    if value_col is None:
        value_col = raw.columns[-1]
    out = raw[[date_col, value_col]].copy()
    out.columns = ["date", "value"]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna().sort_values("date")
    if out.empty:
        return None
    series = pd.Series(out["value"].values, index=out["date"], name=series_id)
    series = normalize_series(series, series_id=series_id, start_date=start_date)
    return series if not series.empty else None


def read_macro_series(series_id: str, *, start_date: str | pd.Timestamp, db_path: str | os.PathLike[str] | None = None) -> tuple[pd.Series | None, str | None]:
    path = macro_db_path(db_path)
    if not _macro_db_available(db_path):
        return None, None
    try:
        with _connect_macro_read_db(db_path) as conn:
            ensure_macro_schema(conn)
            raw = pd.read_sql_query(
                "SELECT date, value FROM macro_series WHERE series_id = ? AND date >= ? ORDER BY date",
                conn,
                params=[series_id, pd.Timestamp(start_date).normalize().strftime("%Y-%m-%d")],
            )
    except Exception as exc:
        if "401" in str(exc) or "Unauthorized" in str(exc) or "empty JWT token" in str(exc):
            raise explain_macro_auth_error(exc)
        raise
    if raw.empty:
        return None, None
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    raw = raw.dropna().sort_values("date")
    if raw.empty:
        return None, None
    series = pd.Series(raw["value"].values, index=raw["date"].dt.normalize(), name=series_id)
    return series, macro_storage_label(path)


def read_macro_frame(series_ids: list[str], *, start_date: str | pd.Timestamp, db_path: str | os.PathLike[str] | None = None) -> tuple[pd.DataFrame | None, str | None]:
    path = macro_db_path(db_path)
    if not _macro_db_available(db_path) or not series_ids:
        return None, None
    placeholders = ",".join(["?"] * len(series_ids))
    params: list[object] = [*series_ids, pd.Timestamp(start_date).normalize().strftime("%Y-%m-%d")]
    try:
        with _connect_macro_read_db(db_path) as conn:
            ensure_macro_schema(conn)
            raw = pd.read_sql_query(
                f"SELECT date, series_id, value FROM macro_series WHERE series_id IN ({placeholders}) AND date >= ? ORDER BY date, series_id",
                conn,
                params=params,
            )
    except Exception as exc:
        if "401" in str(exc) or "Unauthorized" in str(exc) or "empty JWT token" in str(exc):
            raise explain_macro_auth_error(exc)
        raise
    if raw.empty:
        return None, None
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    raw = raw.dropna()
    if raw.empty:
        return None, None
    frame = raw.pivot_table(index="date", columns="series_id", values="value", aggfunc="last").sort_index()
    frame.index = pd.to_datetime(frame.index).normalize()
    frame = frame.reindex(columns=series_ids)
    return frame if not frame.empty else None, macro_storage_label(path)
