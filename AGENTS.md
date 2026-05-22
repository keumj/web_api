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

Do not use this guidance to justify unrelated refactors. Keep edits scoped to
the demonstrated failure pattern and its obvious siblings.
