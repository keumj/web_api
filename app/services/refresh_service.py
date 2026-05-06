from __future__ import annotations

import html
import sqlite3
import subprocess
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from app.settings import settings
from app.services import refresh_state
from app.web import shell
from pipeline_portfolio import web_gui as portfolio_web


@dataclass(frozen=True)
class RefreshJob:
    job_id: str
    label: str
    button_label: str
    command: list[str]
    description: str


@dataclass
class JobState:
    status: str = "idle"
    run_id: int = 0
    started_at: str | None = None
    finished_at: str | None = None
    logs: list[str] = field(default_factory=list)
    error: str | None = None
    updated_items: list[dict[str, object]] = field(default_factory=list)
    latest_summary: str = ""
    latest_items: list[dict[str, object]] = field(default_factory=list)


JOB_TEXTS = {
    "stock": {
        "label": "S&P 500 가격/시총",
        "button_label": "가격/시총 갱신",
        "description": "원본 가격 갱신 모듈을 실행해 가격 CSV, 시가총액 CSV, shared SQLite 가격 테이블을 갱신합니다.",
    },
    "quarterly": {
        "label": "분기 재무",
        "button_label": "분기 재무 갱신",
        "description": "원본 분기 재무 갱신 모듈을 실행해 shared SQLite의 fundamentals_quarterly 테이블을 갱신합니다.",
    },
    "news": {
        "label": "뉴스",
        "button_label": "뉴스 갱신",
        "description": "원본 뉴스 갱신 모듈을 실행해 S&P 500 뉴스 기사와 분석 대기열을 적재합니다.",
    },
}


def _refresh_job_defs() -> list[dict[str, object]]:
    """Use the original refresh job definitions, but hide the duplicated Stock Lab price card."""
    return [
        dict(job)
        for job in portfolio_web._refresh_job_defs()
        if str(job.get("job_id", "")) in JOB_TEXTS and not job.get("is_stock_lab_refresh")
    ]


JOBS: dict[str, RefreshJob] = {}
for _job in _refresh_job_defs():
    job_id = str(_job["job_id"])
    text = JOB_TEXTS[job_id]
    JOBS[job_id] = RefreshJob(
        job_id=job_id,
        label=text["label"],
        button_label=text["button_label"],
        command=portfolio_web._refresh_subprocess_command(job_id),
        description=text["description"],
    )

states: dict[str, JobState] = {job_id: JobState() for job_id in JOBS}
lock = threading.Lock()


def _path_label(path: Path, root_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(root_dir.resolve())).replace("\\", "/")
    except Exception:
        return str(path)


def _format_timestamp(ts: float | None) -> str | None:
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def _csv_latest_date(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None
    if frame.empty:
        return None
    columns = {str(col).strip().lower(): col for col in frame.columns}
    date_col = columns.get("date") or columns.get("datetime") or frame.columns[0]
    dates = pd.to_datetime(frame[date_col], errors="coerce").dropna()
    return dates.max().strftime("%Y-%m-%d") if not dates.empty else None


def _csv_row_count(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return max(sum(1 for _ in handle) - 1, 0)
    except Exception:
        return None


def _sqlite_fetchone(path: Path, query: str) -> tuple[object, ...] | None:
    if not path.exists():
        return None
    try:
        with sqlite3.connect(path) as conn:
            return conn.execute(query).fetchone()
    except Exception:
        return None


def _file_item(
    path: Path,
    *,
    root_dir: Path,
    label: str,
    latest_value: str | None = None,
    rows: int | None = None,
    detail: str | None = None,
) -> dict[str, object]:
    exists = path.exists()
    return {
        "label": label,
        "path": _path_label(path, root_dir),
        "exists": exists,
        "latest_value": latest_value,
        "rows": rows,
        "modified_at": _format_timestamp(path.stat().st_mtime) if exists else None,
        "detail": detail,
    }


def _stock_snapshot(root_dir: Path) -> dict[str, object]:
    data_dir = root_dir / "data"
    metrics_path = data_dir / "sp500_all_metrics_prices.csv"
    market_cap_path = data_dir / "sp500_market_caps.csv"
    sqlite_path = data_dir / "sp500_shared_db" / "sp500_shared_prices.sqlite"
    sqlite_row = _sqlite_fetchone(sqlite_path, "SELECT MAX(date), COUNT(*) FROM prices")
    sqlite_max_date = str(sqlite_row[0]) if sqlite_row and sqlite_row[0] else None
    sqlite_rows = int(sqlite_row[1]) if sqlite_row and sqlite_row[1] is not None else None
    items = [
        _file_item(metrics_path, root_dir=root_dir, label="가격 CSV", latest_value=_csv_latest_date(metrics_path), rows=_csv_row_count(metrics_path)),
        _file_item(market_cap_path, root_dir=root_dir, label="시가총액 CSV", latest_value=_csv_latest_date(market_cap_path), rows=_csv_row_count(market_cap_path)),
        _file_item(sqlite_path, root_dir=root_dir, label="Shared SQLite 가격 테이블", latest_value=sqlite_max_date, rows=sqlite_rows, detail="table=prices"),
    ]
    summary = f"가격 {items[0]['latest_value'] or '-'} / 시총 {items[1]['latest_value'] or '-'} / DB 최신일 {sqlite_max_date or '-'}"
    return {"summary": summary, "items": items}


def _quarterly_snapshot(root_dir: Path) -> dict[str, object]:
    sqlite_path = root_dir / "data" / "sp500_shared_db" / "sp500_shared_prices.sqlite"
    row = _sqlite_fetchone(
        sqlite_path,
        "SELECT MAX(fiscal_date), COUNT(*), COUNT(DISTINCT symbol), MAX(updated_at) FROM fundamentals_quarterly",
    )
    max_fiscal = str(row[0]) if row and row[0] else None
    total_rows = int(row[1]) if row and row[1] is not None else None
    symbol_count = int(row[2]) if row and row[2] is not None else None
    updated_at = str(row[3]) if row and row[3] else None
    item = _file_item(
        sqlite_path,
        root_dir=root_dir,
        label="Shared SQLite 분기 재무 테이블",
        latest_value=max_fiscal,
        rows=total_rows,
        detail=f"symbols={symbol_count or 0}, updated_at={updated_at or '-'}",
    )
    return {"summary": f"최신 fiscal_date {max_fiscal or '-'} / 종목 {symbol_count or 0}개 / rows {total_rows or 0}", "items": [item]}


def _news_snapshot(root_dir: Path) -> dict[str, object]:
    sqlite_path = root_dir / "data" / "sp500_shared_db" / "sp500_shared_prices.sqlite"
    row = _sqlite_fetchone(sqlite_path, "SELECT MAX(publish_date), COUNT(*), COUNT(DISTINCT ticker) FROM news_articles")
    max_publish = str(row[0]) if row and row[0] else None
    total_rows = int(row[1]) if row and row[1] is not None else None
    ticker_count = int(row[2]) if row and row[2] is not None else None
    item = _file_item(
        sqlite_path,
        root_dir=root_dir,
        label="Shared SQLite 뉴스 테이블",
        latest_value=max_publish,
        rows=total_rows,
        detail=f"tickers={ticker_count or 0}, table=news_articles",
    )
    return {"summary": f"최신 publish_date {max_publish or '-'} / 기사 {total_rows or 0}건 / 티커 {ticker_count or 0}개", "items": [item]}


def _snapshot(job_id: str) -> dict[str, object]:
    root_dir = settings.project_root
    if job_id == "stock":
        return _stock_snapshot(root_dir)
    if job_id == "quarterly":
        return _quarterly_snapshot(root_dir)
    if job_id == "news":
        return _news_snapshot(root_dir)
    return {"summary": "알 수 없는 작업입니다.", "items": []}


def _ensure_latest_snapshots() -> None:
    with lock:
        missing = [job_id for job_id, state in states.items() if not state.latest_summary]
    for job_id in missing:
        snap = _snapshot(job_id)
        with lock:
            state = states[job_id]
            state.latest_summary = str(snap["summary"])
            state.latest_items = [dict(item) for item in snap["items"]]


def list_jobs() -> list[dict[str, object]]:
    _ensure_latest_snapshots()
    with lock:
        return [
            {
                "job_id": job.job_id,
                "label": job.label,
                "button_label": job.button_label,
                "description": job.description,
                "status": states[job.job_id].status,
                "run_id": states[job.job_id].run_id,
                "started_at": states[job.job_id].started_at,
                "finished_at": states[job.job_id].finished_at,
                "error": states[job.job_id].error,
                "logs": states[job.job_id].logs[-100:],
                "log_count": len(states[job.job_id].logs),
                "updated_items": states[job.job_id].updated_items,
                "latest_summary": states[job.job_id].latest_summary,
                "latest_items": states[job.job_id].latest_items,
            }
            for job in JOBS.values()
        ]


def _append(job_id: str, line: str) -> None:
    with lock:
        state = states[job_id]
        state.logs.append(str(line).rstrip())
        state.logs = state.logs[-300:]


def _record_refresh_state(event: str, *, job_id: str, exit_code: int | None = None) -> None:
    try:
        refresh_state.record_state(event, source=f"web:{job_id}", exit_code=exit_code)
    except Exception as exc:
        _append(job_id, f"[refresh:{job_id}] state record failed: {type(exc).__name__}: {exc}")


def _run_job(job: RefreshJob) -> None:
    root = settings.project_root
    exit_code: int | None = None
    try:
        if not job.command:
            raise FileNotFoundError(f"No command configured for {job.job_id}")
        _append(job.job_id, f"[refresh:{job.job_id}] started {' '.join(job.command)}")
        _record_refresh_state("started", job_id=job.job_id)
        proc = subprocess.Popen(
            job.command,
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if proc.stdout is not None:
            for line in proc.stdout:
                if line.strip():
                    _append(job.job_id, line)
        exit_code = int(proc.wait())
        snap = _snapshot(job.job_id)
        with lock:
            state = states[job.job_id]
            state.status = "completed" if exit_code == 0 else f"failed({exit_code})"
            state.error = None if exit_code == 0 else f"exit_code={exit_code}"
            state.finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            state.updated_items = [dict(item) for item in snap["items"]]
            state.latest_items = [dict(item) for item in snap["items"]]
            state.latest_summary = str(snap["summary"])
        _record_refresh_state("finished", job_id=job.job_id, exit_code=exit_code)
        _append(job.job_id, f"[refresh:{job.job_id}] finished with exit_code={exit_code}")
    except Exception as exc:
        snap = _snapshot(job.job_id)
        with lock:
            state = states[job.job_id]
            state.status = "failed"
            state.error = f"{type(exc).__name__}: {exc}"
            state.finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            state.updated_items = [dict(item) for item in snap["items"]]
            state.latest_items = [dict(item) for item in snap["items"]]
            state.latest_summary = str(snap["summary"])
        _record_refresh_state("finished", job_id=job.job_id, exit_code=exit_code if exit_code is not None else 1)
        _append(job.job_id, f"[refresh:{job.job_id}] failed: {type(exc).__name__}: {exc}")
    finally:
        if exit_code is None:
            _append(job.job_id, f"[refresh:{job.job_id}] stopped")


def start_job(job_id: str) -> dict[str, object]:
    job = JOBS.get(job_id)
    if job is None:
        return {"ok": False, "error": "unknown job"}
    with lock:
        if any(state.status == "running" for state in states.values()):
            return {"ok": False, "error": "another refresh job is already running"}
        state = states[job_id]
        state.status = "running"
        state.run_id += 1
        state.started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        state.finished_at = None
        state.error = None
        state.logs = [f"[refresh:{job_id}] queued {job.label}"]
        state.updated_items = []
    thread = threading.Thread(target=_run_job, args=(job,), daemon=True)
    thread.start()
    return {"ok": True, "job_id": job_id}


def start_original_job(job_id: str) -> dict[str, object]:
    return start_job(job_id)


def original_status_payload() -> dict[str, object]:
    with lock:
        running = any(state.status == "running" for state in states.values())
        current_job_id = next((job_id for job_id, state in states.items() if state.status == "running"), None)
    return {"running": running, "current_job_id": current_job_id, "jobs": list_jobs()}


def diagnostics_payload() -> dict[str, object]:
    root = settings.project_root
    sqlite_path = root / "data" / "sp500_shared_db" / "sp500_shared_prices.sqlite"
    info: dict[str, object] = {
        "project_root": str(root),
        "sqlite_path": str(sqlite_path),
        "sqlite_exists": sqlite_path.exists(),
        "sqlite_size_bytes": sqlite_path.stat().st_size if sqlite_path.exists() else None,
        "jobs": {
            job_id: {
                "command": job.command,
                "status": states[job_id].status,
                "error": states[job_id].error,
                "last_logs": states[job_id].logs[-20:],
            }
            for job_id, job in JOBS.items()
        },
    }
    if sqlite_path.exists():
        try:
            with sqlite3.connect(sqlite_path) as conn:
                tables = [
                    str(row[0])
                    for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
                ]
                info["tables"] = tables
                for table, date_col in {
                    "prices": "date",
                    "news_articles": "publish_date",
                    "fundamentals_quarterly": "fiscal_date",
                }.items():
                    if table not in tables:
                        info[f"{table}_exists"] = False
                        continue
                    row = conn.execute(f"SELECT COUNT(*), MAX({date_col}) FROM {table}").fetchone()
                    info[f"{table}_exists"] = True
                    info[f"{table}_rows"] = int(row[0] or 0) if row else 0
                    info[f"{table}_max_{date_col}"] = str(row[1]) if row and row[1] else None
        except Exception as exc:
            info["sqlite_error"] = f"{type(exc).__name__}: {exc}"
    return info


def _render_items_js() -> str:
    return """
      function escapeHtml(value) {
        return String(value ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
      }
      function renderLogs(el, lines, emptyText) {
        if (!el) return;
        if (!lines || lines.length === 0) {
          el.innerHTML = "<div class='line'>" + escapeHtml(emptyText) + "</div>";
          return;
        }
        el.innerHTML = lines.map((line) => "<div class='line'>" + escapeHtml(line) + "</div>").join("");
        el.scrollTop = el.scrollHeight;
      }
      function renderItems(el, items, emptyText) {
        if (!el) return;
        if (!items || items.length === 0) {
          el.innerHTML = "<div class='line'>" + escapeHtml(emptyText) + "</div>";
          return;
        }
        el.innerHTML = items.map((item) => {
          const label = escapeHtml(item.label || item.path || "-");
          const latest = escapeHtml(item.latest_value || "-");
          const rows = item.rows == null ? "-" : escapeHtml(item.rows);
          const modified = escapeHtml(item.modified_at || "-");
          const detail = item.detail ? " / " + escapeHtml(item.detail) : "";
          return "<div class='line'><strong>" + label + "</strong> | latest=" + latest + " | rows=" + rows + " | modified=" + modified + detail + "</div>";
        }).join("");
        el.scrollTop = el.scrollHeight;
      }
    """


def render_original_refresh_page(
    *,
    lookback_days: int,
    start_date: str | None,
    end_date: str | None,
    admin: bool = False,
) -> str:
    from app.services.portfolio_service import resolve_range

    date_range = resolve_range(start_date, end_date, lookback_days)
    cards = []
    for job in JOBS.values():
        job_id = html.escape(job.job_id)
        cards.append(
            f"""
            <section class="service-card refresh-card" data-job-id="{job_id}">
              <div class="refresh-card-head">
                <div>
                  <h2>{html.escape(job.label)}</h2>
                  <p class="service-muted">{html.escape(job.description)}</p>
                </div>
                <div class="refresh-latest">
                  <strong>실행 전 최신 현황</strong>
                  <div id="refresh-latest-summary-{job_id}">확인 중...</div>
                  <div id="refresh-latest-items-{job_id}" class="refresh-mini"></div>
                </div>
              </div>
              <form method="post" action="/run_refresh" class="refresh-run-form">
                <input type="hidden" name="lookback_days" value="{int(date_range.lookback_days)}">
                <input type="hidden" name="start_date" value="{html.escape(date_range.start_date)}">
                <input type="hidden" name="end_date" value="{html.escape(date_range.end_date)}">
                <input type="hidden" name="job_id" value="{job_id}">
                <div class="refresh-action-row">
                  <button class="service-button" type="submit" id="refresh-btn-{job_id}">{html.escape(job.button_label)}</button>
                  <div id="refresh-meta-{job_id}" class="refresh-meta">상태: 대기 / 실행 ID: - / 시작: - / 종료: -</div>
                </div>
              </form>
              <div class="refresh-split">
                <div>
                  <div class="refresh-pane-title"><h3>실행 로그</h3><span>Live</span></div>
                  <div id="refresh-log-{job_id}" class="refresh-list"><div class="line">아직 로그가 없습니다.</div></div>
                </div>
                <div>
                  <div class="refresh-pane-title"><h3>갱신 결과</h3><span>Output</span></div>
                  <div id="refresh-updates-{job_id}" class="refresh-list"><div class="line">아직 갱신 결과가 없습니다.</div></div>
                </div>
              </div>
            </section>
            """
        )

    body = f"""
    <style>
      .refresh-stack {{ display: grid; gap: 12px; }}
      .refresh-card {{ display: grid; gap: 12px; }}
      .refresh-card h2 {{ margin: 0 0 6px; font-size: 18px; }}
      .refresh-card-head {{ display: grid; grid-template-columns: minmax(260px, 1fr) minmax(260px, 420px); gap: 12px; align-items: start; }}
      .refresh-latest {{ border: 1px solid var(--line); background: #f8fafc; border-radius: 8px; padding: 10px; font-size: 12px; }}
      .refresh-latest strong {{ display: block; margin-bottom: 6px; }}
      .refresh-mini {{ display: grid; gap: 3px; margin-top: 6px; color: var(--muted); }}
      .refresh-action-row {{ display: grid; grid-template-columns: max-content 1fr; gap: 12px; align-items: center; }}
      .refresh-meta {{ color: var(--muted); font-size: 13px; }}
      .refresh-split {{ display: grid; grid-template-columns: repeat(2, minmax(280px, 1fr)); gap: 12px; }}
      .refresh-pane-title {{ display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 6px; }}
      .refresh-pane-title h3 {{ margin: 0; font-size: 15px; }}
      .refresh-pane-title span {{ color: var(--muted); font-size: 11px; }}
      .refresh-list {{ height: 260px; overflow: auto; border: 1px solid var(--line); border-radius: 8px; background: #fff; padding: 8px; font-size: 12px; }}
      .refresh-list .line {{ padding: 4px 0; border-bottom: 1px solid #eef2f6; white-space: pre-wrap; word-break: break-word; }}
      .refresh-list .line:last-child {{ border-bottom: 0; }}
      @media (max-width: 900px) {{
        .refresh-card-head, .refresh-action-row, .refresh-split {{ grid-template-columns: 1fr; }}
      }}
    </style>
    <div class="refresh-stack">
      <section class="service-card">
        <h1>데이터 갱신</h1>
        <p class="service-muted">원본 데이터 갱신 명령을 그대로 실행합니다. 중복 Stock Lab 가격 카드는 표시하지 않습니다.</p>
      </section>
      {''.join(cards)}
    </div>
    <script>
      const refreshLogCounts = {{}};
      {_render_items_js()}
      function renderLatest(summaryEl, itemsEl, summary, items) {{
        if (summaryEl) summaryEl.textContent = summary || "현재 데이터 상태를 확인할 수 없습니다.";
        if (!itemsEl) return;
        itemsEl.innerHTML = (items || []).map((item) => {{
          const label = escapeHtml(item.label || item.path || "-");
          const latest = escapeHtml(item.latest_value || "-");
          const rows = item.rows == null ? "-" : escapeHtml(item.rows);
          return "<div>" + label + " | latest=" + latest + " | rows=" + rows + "</div>";
        }}).join("");
      }}
      async function pollRefreshStatus() {{
        try {{
          const res = await fetch("/refresh_status", {{ cache: "no-store", headers: {{ "X-Requested-With": "fetch" }} }});
          if (!res.ok) return;
          const data = await res.json();
          (data.jobs || []).forEach((job) => {{
            const jobId = job.job_id;
            const metaEl = document.getElementById("refresh-meta-" + jobId);
            const logEl = document.getElementById("refresh-log-" + jobId);
            const updatesEl = document.getElementById("refresh-updates-" + jobId);
            const btnEl = document.getElementById("refresh-btn-" + jobId);
            renderLatest(document.getElementById("refresh-latest-summary-" + jobId), document.getElementById("refresh-latest-items-" + jobId), job.latest_summary || "", job.latest_items || []);
            if (metaEl) {{
              metaEl.textContent = "상태: " + (job.status || "대기") + " / 실행 ID: " + (job.run_id || "-") + " / 시작: " + (job.started_at || "-") + " / 종료: " + (job.finished_at || "-");
            }}
            if (btnEl) {{
              const isBusy = Boolean(data.running);
              btnEl.disabled = isBusy;
              btnEl.style.opacity = isBusy ? "0.6" : "1";
              btnEl.textContent = isBusy && data.current_job_id === jobId ? "실행 중..." : (job.button_label || "갱신 실행");
            }}
            if (job.log_count !== refreshLogCounts[jobId]) {{
              refreshLogCounts[jobId] = job.log_count;
              renderLogs(logEl, job.logs || [], "아직 로그가 없습니다.");
            }}
            renderItems(updatesEl, job.updated_items || [], "아직 갱신 결과가 없습니다.");
          }});
        }} catch (err) {{
          console.error(err);
        }}
      }}
      document.querySelectorAll(".refresh-run-form").forEach((formEl) => {{
        formEl.addEventListener("submit", async (event) => {{
          event.preventDefault();
          try {{
            const res = await fetch(formEl.action, {{
              method: "POST",
              headers: {{"Content-Type": "application/x-www-form-urlencoded; charset=UTF-8", "Accept": "application/json", "X-Requested-With": "fetch"}},
              body: new URLSearchParams(new FormData(formEl)),
            }});
            const data = await res.json().catch(() => ({{ ok: false, error: "응답을 해석할 수 없습니다." }}));
            if (!res.ok || !data.ok) {{
              window.alert(data.error || "작업 시작에 실패했습니다.");
              return;
            }}
            await pollRefreshStatus();
          }} catch (err) {{
            console.error(err);
            window.alert("작업 요청 중 오류가 발생했습니다.");
          }}
        }});
      }});
      setInterval(pollRefreshStatus, 2000);
      pollRefreshStatus();
    </script>
    """
    return shell("데이터 갱신", body, active="refresh", admin=admin)


def refresh_page() -> str:
    rows = []
    for job in list_jobs():
        job_id = html.escape(str(job["job_id"]))
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(job['label']))}</td>"
            f"<td>{html.escape(str(job['status']))}</td>"
            f"<td>{html.escape(str(job['started_at'] or '-'))}</td>"
            f"<td>{html.escape(str(job['finished_at'] or '-'))}</td>"
            "<td>"
            f"<form method='post' action='/refresh/{job_id}/run'>"
            "<button class='service-button' type='submit'>실행</button>"
            "</form>"
            "</td>"
            "</tr>"
        )
    return (
        "<div class='service-stack'><div class='service-card'><h1>데이터 갱신</h1></div>"
        "<div class='service-card'><div class='service-table-wrap'><table class='service-table'>"
        "<thead><tr><th>작업</th><th>상태</th><th>시작</th><th>종료</th><th>실행</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div></div></div>"
    )
