from __future__ import annotations

import html


def rewrite_links(page: str, replacements: dict[str, str]) -> str:
    out = page
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def add_start_page_link(page: str) -> str:
    link = '<div class="nav" style="margin-bottom:12px;"><a href="/">시작 페이지로 돌아가기</a></div>'
    return page.replace('<div class="wrap">', '<div class="wrap">' + link, 1)


def shell(title: str, body: str, *, active: str = "portfolio", admin: bool = False, start_page_only: bool = False) -> str:
    active_class = {
        "portfolio": "active" if active == "portfolio" else "",
        "stock": "active" if active == "stock" else "",
        "news": "active" if active == "news" else "",
        "macro": "active" if active == "macro" else "",
        "admin": "active" if active == "admin" else "",
    }
    admin_link = f'<a class="{active_class["admin"]}" href="/admin/users">사용자 관리</a>' if admin else ""
    api_link = '<a href="/docs">API</a>' if admin else ""
    default_nav = f"""
        <a class="{active_class["portfolio"]}" href="/portfolio/overview">포트폴리오</a>
        <a class="{active_class["stock"]}" href="/stock/forecast">종목 분석</a>
        <a class="{active_class["news"]}" href="/stock-news/overview">뉴스 분석</a>
        <a class="{active_class["macro"]}" href="/macro/overview">거시분석</a>
        {admin_link}
        {api_link}
    """
    service_nav = (
        '<a href="/">기본페이지로 돌아가기</a>'
        if start_page_only
        else f"""
        {default_nav}
        """
    )
    top_nav_style = "display:none;" if start_page_only else ""
    brand_style = "display:none;" if start_page_only else ""
    start_page_link = '<nav class="service-nav"><a href="/">기본페이지로 돌아가기</a></nav>' if start_page_only else ""
    start_page_link = '<nav class="service-nav"><a href="/">기본페이지로 돌아가기</a></nav>' if start_page_only else ""
    return f"""<!doctype html>
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
    .service-table {{ width: 100%; min-width: 100%; border-collapse: collapse; font-size: 13px; line-height: 1.45; }}
    .service-table th, .service-table td {{ border: 1px solid var(--line); padding: 8px; text-align: left; vertical-align: top; white-space: normal; overflow-wrap: anywhere; word-break: normal; }}
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
  <header class="service-top">
    <div class="service-top-inner">
      <div class="service-brand" style="{brand_style}">Keumj Portfolio Lab</div>
      <nav class="service-nav" style="{top_nav_style}">
        <a class="{active_class["portfolio"]}" href="/portfolio/overview">포트폴리오</a>
        <a class="{active_class["stock"]}" href="/stock/forecast">종목 분석</a>
        <a class="{active_class["news"]}" href="/stock-news/overview">뉴스 분석</a>
        <a class="{active_class["macro"]}" href="/macro/overview">거시분석</a>
        {admin_link}
        {api_link}
      </nav>
      {start_page_link}
    </div>
  </header>
  <main class="service-main">{body}</main>
</body>
</html>
"""
