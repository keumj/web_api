from __future__ import annotations

import traceback

from pipeline_macro import web_gui as macro_web

from app.web import shell


def render(page: str, *, start_date: str | None = None, lookback_days: int = 504) -> str:
    page_key = macro_web.normalize_page(page)
    try:
        body = macro_web.render_body(page_key, start_date=start_date, lookback_days=lookback_days)
    except Exception as exc:
        body = f"""
        <div class="service-card">
          <h1>거시분석</h1>
          <p class="service-error">{type(exc).__name__}: {exc}</p>
          <pre class="service-error">{traceback.format_exc(limit=4)}</pre>
        </div>
        """
    return shell("거시분석 | Keumj Portfolio Lab", body, active="macro", start_page_only=True)
