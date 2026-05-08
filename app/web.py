from __future__ import annotations

import html

from app.settings import settings


def rewrite_links(page: str, replacements: dict[str, str]) -> str:
    out = page
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def add_start_page_link(page: str) -> str:
    link = '<div class="nav" style="margin-bottom:12px;"><a href="/">시작 페이지로 돌아가기</a></div>'
    return page.replace('<div class="wrap">', '<div class="wrap">' + link, 1)


def inject_busy_cursor_overlay(page: str) -> str:
    marker = "data-busy-cursor-overlay"
    if marker in page:
        return page

    overlay_style = """
  <style data-busy-cursor-overlay>
    .busy-cursor-overlay {
      position: fixed;
      left: 0;
      top: 0;
      z-index: 9999;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
      padding: 7px 11px;
      border-radius: 999px;
      background: rgba(17, 24, 39, 0.94);
      color: #fff;
      box-shadow: 0 14px 28px rgba(15, 23, 42, 0.22);
      border: 1px solid rgba(255, 255, 255, 0.14);
      pointer-events: none;
      opacity: 0;
      transform: translate3d(-9999px, -9999px, 0) scale(0.96);
      transition: opacity 140ms ease, transform 140ms ease;
      white-space: nowrap;
      font-family: "Segoe UI", "Noto Sans KR", sans-serif;
    }
    .busy-cursor-overlay.is-visible {
      opacity: 1;
    }
    .busy-cursor-overlay__spinner {
      width: 14px;
      height: 14px;
      border-radius: 50%;
      border: 1.5px solid rgba(255, 255, 255, 0.32);
      border-top-color: #fff;
      animation: busy-cursor-spin 0.8s linear infinite;
      flex: 0 0 auto;
    }
    .busy-cursor-overlay__text {
      font-size: 12px;
      font-weight: 650;
      letter-spacing: -0.01em;
    }
    body.busy-cursor-active,
    body.busy-cursor-active * {
      cursor: progress !important;
    }
    @keyframes busy-cursor-spin {
      to { transform: rotate(360deg); }
    }
  </style>
"""
    overlay_markup = """
  <div class="busy-cursor-overlay" id="busy-cursor-overlay" aria-live="polite" aria-hidden="true">
    <span class="busy-cursor-overlay__spinner" aria-hidden="true"></span>
    <span class="busy-cursor-overlay__text" id="busy-cursor-overlay-text">실행중...</span>
  </div>
  <script data-busy-cursor-overlay>
    (() => {
      const overlay = document.getElementById("busy-cursor-overlay");
      const textEl = document.getElementById("busy-cursor-overlay-text");
      if (!overlay || !textEl) return;

      const state = {
        active: false,
        x: Math.max(window.innerWidth * 0.5, 24),
        y: Math.max(window.innerHeight * 0.35, 24),
      };

      const trackedButtons = new WeakMap();

      const positionOverlay = () => {
        const rect = overlay.getBoundingClientRect();
        const offsetX = 18;
        const offsetY = 22;
        const maxX = Math.max(window.innerWidth - rect.width - 12, 12);
        const maxY = Math.max(window.innerHeight - rect.height - 12, 12);
        const nextX = Math.min(Math.max(state.x + offsetX, 12), maxX);
        const nextY = Math.min(Math.max(state.y + offsetY, 12), maxY);
        overlay.style.transform = `translate3d(${nextX}px, ${nextY}px, 0) scale(${state.active ? 1 : 0.96})`;
      };

      const setPointer = (x, y) => {
        state.x = x;
        state.y = y;
        if (state.active) {
          positionOverlay();
        }
      };

      const isAnalysisIntent = (form) => {
        if (!(form instanceof HTMLFormElement)) return false;
        if (form.dataset.noBusyCursor === "true") return false;

        const method = (form.getAttribute("method") || "get").toLowerCase();
        const action = (form.getAttribute("action") || window.location.pathname).toLowerCase();
        const intentField = form.querySelector('input[name="intent"]');
        const intent = (intentField ? intentField.value : "").trim().toLowerCase();

        if (method === "get" && ["run", "analyze", "refresh"].includes(intent)) {
          return true;
        }

        if (method !== "post") return false;

        if (action.includes("/stock/run")) return true;
        if (action.includes("/stock-news/run-")) return true;
        if (action.includes("/run_virtual_trade")) return true;
        if (action.includes("/run_refresh")) return true;
        return false;
      };

      const isPageNavigationLink = (event, link) => {
        if (!(link instanceof HTMLAnchorElement)) return false;
        if (event.defaultPrevented) return false;
        if (event.button !== 0) return false;
        if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return false;
        if (link.dataset.noBusyCursor === "true") return false;
        if (link.hasAttribute("download")) return false;

        const target = (link.getAttribute("target") || "").trim().toLowerCase();
        if (target && target !== "_self") return false;

        const rawHref = (link.getAttribute("href") || "").trim();
        if (!rawHref || rawHref.startsWith("#")) return false;
        if (/^(javascript|mailto|tel):/i.test(rawHref)) return false;

        let url;
        try {
          url = new URL(rawHref, window.location.href);
        } catch (err) {
          return false;
        }
        if (url.origin !== window.location.origin) return false;
        if (
          url.pathname === window.location.pathname &&
          url.search === window.location.search &&
          url.hash
        ) {
          return false;
        }
        return true;
      };

      const resolveLabel = (form, submitter) => {
        return "실행중...";
      };

      const activate = (label) => {
        state.active = true;
        textEl.textContent = label;
        overlay.classList.add("is-visible");
        overlay.setAttribute("aria-hidden", "false");
        document.body.classList.add("busy-cursor-active");
        positionOverlay();
      };

      const deactivate = () => {
        state.active = false;
        overlay.classList.remove("is-visible");
        overlay.setAttribute("aria-hidden", "true");
        document.body.classList.remove("busy-cursor-active");
        positionOverlay();
      };

      document.addEventListener("mousemove", (event) => {
        setPointer(event.clientX, event.clientY);
      }, { passive: true });

      document.addEventListener("pointerdown", (event) => {
        setPointer(event.clientX, event.clientY);
      }, { passive: true });

      document.addEventListener("click", (event) => {
        const target = event.target instanceof Element ? event.target.closest("button, input[type=submit]") : null;
        if (target) {
          trackedButtons.set(target.form || document.body, target);
        }
      }, true);

      document.addEventListener("click", (event) => {
        const link = event.target instanceof Element ? event.target.closest("a[href]") : null;
        if (isPageNavigationLink(event, link)) {
          activate(resolveLabel(null, null));
        }
      });

      document.addEventListener("submit", (event) => {
        const form = event.target;
        if (!isAnalysisIntent(form)) return;
        const submitter = event.submitter || trackedButtons.get(form) || null;
        activate(resolveLabel(form, submitter));
      }, true);

      window.addEventListener("pageshow", deactivate);
      window.addEventListener("pagehide", deactivate);
      window.addEventListener("focus", () => {
        if (!document.hidden) deactivate();
      });

      positionOverlay();
    })();
  </script>
"""

    page = page.replace("</head>", overlay_style + "\n</head>", 1) if "</head>" in page else overlay_style + page
    if "</body>" in page:
        return page.replace("</body>", overlay_markup + "\n</body>", 1)
    return page + overlay_markup


def shell(
    title: str,
    body: str,
    *,
    active: str = "portfolio",
    admin: bool = False,
    start_page_only: bool = False,
) -> str:
    active_class = {
        "portfolio": "active" if active == "portfolio" else "",
        "stock": "active" if active == "stock" else "",
        "news": "active" if active == "news" else "",
        "macro": "active" if active == "macro" else "",
        "admin": "active" if active == "admin" else "",
    }
    admin_link = f'<a class="{active_class["admin"]}" href="/admin/users">사용자 관리</a>' if admin else ""
    api_link = '<a href="/docs">API</a>' if admin else ""
    macro_link = f'<a class="{active_class["macro"]}" href="/macro/overview">거시 분석</a>' if settings.enable_macro else ""
    header_display = "display:none;" if start_page_only else "block"
    top_nav_style = "display:none;" if start_page_only else ""
    brand_style = "display:none;" if start_page_only else ""
    return_button = (
        '<div class="service-nav" style="margin-bottom:15px; justify-content: flex-start;"><a href="/">시작 페이지로 돌아가기</a></div>'
        if start_page_only
        else ""
    )
    default_nav = f"""
        <a class="{active_class["portfolio"]}" href="/portfolio/overview">포트폴리오</a>
        <a class="{active_class["stock"]}" href="/stock/financials">종목 분석</a>
        <a class="{active_class["news"]}" href="/stock-news/overview">뉴스 분석</a>
        {macro_link}
        {admin_link}
        {api_link}
    """
    return inject_busy_cursor_overlay(f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f5f7fa;
      --panel: #ffffff;
      --line: #d7e0ea;
      --text: #1f2937;
      --muted: #667085;
      --brand: #111827;
      --accent: #0f766e;
      --danger: #a12626;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; color: var(--text); background: var(--bg); font-family: "Segoe UI", "Noto Sans KR", sans-serif; }}
    .service-top {{ position: sticky; top: 0; z-index: 20; background: rgba(255,255,255,.96); border-bottom: 1px solid var(--line); }}
    .service-top-inner {{ width: 100%; max-width: 1460px; margin: 0 auto; padding: 10px 20px; display: flex; align-items: center; justify-content: space-between; gap: 12px; }}
    .service-brand {{ font-weight: 750; letter-spacing: 0; white-space: nowrap; }}
    .service-nav {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }}
    .service-nav a {{ color: var(--brand); border: 1px solid var(--line); background: #fff; text-decoration: none; border-radius: 8px; padding: 7px 11px; font-size: 13px; }}
    .service-nav a.active {{ background: var(--brand); color: #fff; border-color: var(--brand); }}
    .service-main {{ width: 100%; max-width: 1460px; margin: 0 auto; padding: 16px 20px 30px; }}
    .service-card {{ max-width: 100%; min-width: 0; overflow-x: auto; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }}
    .service-grid {{ display: grid; grid-template-columns: repeat(3, minmax(220px, 1fr)); gap: 12px; }}
    .service-grid a {{ display: block; color: var(--text); text-decoration: none; }}
    .service-grid h3 {{ margin: 0 0 6px; font-size: 16px; }}
    .service-grid p {{ margin: 0; color: var(--muted); font-size: 13px; line-height: 1.45; }}
    .service-stack {{ display: grid; gap: 12px; }}
    .service-login-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
    .service-login-grid h3 {{ margin: 0 0 10px; font-size: 15px; }}
    .service-login-grid label {{ display: block; margin: 10px 0 4px; font-size: 12px; color: var(--muted); }}
    .service-login-grid input {{ width: 100%; border: 1px solid var(--line); border-radius: 8px; padding: 10px; font-size: 14px; }}
    .service-login-grid button {{ margin-top: 14px; }}
    .service-table-wrap {{ width: 100%; max-width: 100%; min-width: 0; overflow-x: auto; -webkit-overflow-scrolling: touch; }}
    .service-table {{ width: max-content; min-width: 100%; border-collapse: collapse; font-size: 13px; line-height: 1.45; }}
    .service-table th, .service-table td {{ border: 1px solid var(--line); padding: 8px; text-align: left; vertical-align: top; white-space: normal; overflow-wrap: break-word; word-break: keep-all; }}
    .service-actions {{ display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }}
    .service-actions input {{ border: 1px solid var(--line); border-radius: 8px; padding: 9px 10px; font-size: 13px; }}
    .service-button {{ border: 0; background: var(--brand); color: #fff; border-radius: 8px; padding: 9px 13px; cursor: pointer; font-weight: 650; }}
    .service-button.secondary {{ background: var(--accent); }}
    .service-muted {{ color: var(--muted); }}
    .service-error {{ color: var(--danger); white-space: pre-wrap; }}
    @media (max-width: 900px) {{
      .service-top-inner {{ align-items: flex-start; flex-direction: column; }}
      .service-nav {{ justify-content: flex-start; }}
      .service-grid {{ grid-template-columns: 1fr; }}
      .service-login-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header class="service-top" style="{header_display}">
    <div class="service-top-inner">
      <div class="service-brand" style="{brand_style}">Keumj Portfolio Lab</div>
      <nav class="service-nav" style="{top_nav_style}">{default_nav}</nav>
    </div>
  </header>
  <main class="service-main">
    {return_button}
    {body}
  </main>
</body>
</html>
""")
