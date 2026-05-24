# Codex Collaboration Notes

This repository is maintained with a preference for fixing the direct issue
and checking nearby failure modes that share the same root cause.

When a bug appears in one command, option, table, route, or script:

- Fix the reported failure first.
- Search for the same error pattern or implementation pattern in related
  paths before stopping.
- If the same root cause clearly affects sibling options or workflows, fix
  those related paths in the same pass.
- If the related change would alter product behavior, data policy, migration
  strategy, or a broad operational workflow, propose it first and explain the
  tradeoff.
- If the related change is a small compatibility or reliability guard in the
  same layer, implement it proactively.

For data refresh and upload workflows:

- Treat `refresh_local_data.cmd` as the local SQLite refresh path.
- Treat `refresh_supabase_data.cmd` as the local SQLite to Supabase upload
  path.
- Treat scheduled application refreshes as direct source-to-remote refreshes
  unless the code says otherwise.
- When changing one refresh target, check sibling targets such as prices,
  fundamentals, news, and macro for the same failure mode.
- For Supabase/PostgreSQL changes, check constraints, identity/default values,
  prepared statement behavior, and SQLite-to-PostgreSQL type/SQL differences
  across all affected targets.

Data store policy:

- Use local SQLite as the recoverable local source of truth for development,
  backup, and repair workflows.
- Use Supabase/PostgreSQL for shared reference data such as prices, market
  caps, fundamentals, news, and macro series.
- If Turso becomes available again, use it for per-user or per-session data,
  not for large shared reference datasets.
- Good Turso candidates include user-specific run history, saved parameters,
  recent selections, favorites, lightweight summaries, and small cached result
  tables.
- Do not store large shared market/news/macro datasets in Turso. Avoid storing
  full raw DataFrames or bulky analysis artifacts there.
- When implementing Turso-backed user history, include a stable user key and
  enough metadata to understand the run: module, run type, created timestamp,
  params JSON, summary JSON, and the reference data freshness used by the run.
- Turso must be an optional enhancement. If Turso is unavailable or rate
  limited, the app should fall back gracefully and continue without historical
  user runs.
- Keep storage responsibilities separate: local SQLite for local truth and
  recovery, Supabase for shared data serving, Turso for lightweight personal
  state.
- Do not add local SQLite fallback reads for normal Render/runtime paths that
  should use Supabase shared reference data or Turso user data. Local DB
  fallback is only for emergency recovery when Supabase shared price/news/
  fundamental/macro data or Turso user data cannot be used.

Do not use this guidance to justify unrelated refactors. Keep edits scoped to
the demonstrated failure pattern and its obvious siblings.
