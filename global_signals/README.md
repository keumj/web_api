# Global asset signals

Initial connected universe: SPY (US large-cap equities), IEF (US 7–10 year
Treasuries), GLD (gold trust), VNQ (US listed real estate). These are explicit
proxies, not complete coverage of their asset classes. 449170.KS (TIGER KOFR
active synthetic ETF) supplies a KRW cash-like comparator; it is not a deposit.
KRW=X supplies KRW per USD. All six series are checked for their quote currency.

## PostgreSQL and recovery

Set `KEUMJ_GLOBAL_SIGNALS_DATABASE_URL` in `.env` to the dedicated PostgreSQL
database. `KEUMJ_GLOBAL_SIGNALS_ADMIN_URL` can optionally provide a separate
database-creation role. Never use the existing KRX or shared `postgres` database.
Credentials are not checked into Git. The initializer reuses databases bearing
the project's ownership marker and refuses unrelated existing tables.

```powershell
.\.venv\Scripts\python.exe -B -m global_signals.init_db
refresh_local_data.cmd global-signals
refresh_supabase_data.cmd global-signals
```

The first refresh writes immutable snapshots to `data/global_signals.sqlite`.
The second explicitly uploads that collection to PostgreSQL and computes signals.
The latest local manifest excludes failed downloads; older successes cannot mask
a current collection failure. Identical snapshots retain their original first-seen
time. A checksum validates recovery payloads. PostgreSQL observations are append-only.
Full snapshot revision hashes preserve internally consistent adjusted-price histories.

An independent command is also available:

```powershell
refresh_global_signals.cmd init
refresh_global_signals.cmd local
refresh_global_signals.cmd upload-local
refresh_global_signals.cmd direct
```

Existing `all` refresh commands retain their original prices/fundamentals/news/macro
scope; choose option 6 or `global-signals` for this new isolated workflow.
The FastAPI runtime reads PostgreSQL only, never the recovery SQLite file.

## Model and validation

`monthly-conditional-v1` is a **research baseline**, not an approved trading model.
It estimates a one-calendar-month distribution of KRW returns minus the cash ETF
return and an explicitly assumed round-trip cost (default 0.2%, tax excluded).
Adjusted fund prices already embed fund expenses; those expenses are not deducted again.
The conditional regime uses positive 63-common-session momentum and price above a
126-common-session moving average. It requires 24 matured labels overall and 8 in
the matching regime; otherwise it uses the unconditional sample or abstains.

Historical decisions occur at month-end. Entry is the next common trading date,
exit the first common date on/after one calendar month from entry. Each expanding
training window uses only labels with exits **strictly before** its decision.
Data-dependent hyperparameter fitting is not performed. The probability uses
Laplace smoothing; p10/p50/p90 describe the empirical outcome distribution, not
confidence bounds for an individual forecast.

Sequential diagnostics compare Brier scores to the expanding unconditional
baseline. A diagnostic gate needs 12 evaluations and an improvement in Brier score.
Passing it does **not** approve a production model: downloaded adjusted history is
not a point-in-time archive, cross-market closing times differ, some labels can
overlap, and observed execution costs/taxes are absent. The persisted model stays
`unvalidated`; original `exposure_latest_signals` predictions remain gray.
`research_latest_signals` contains separately labelled exploratory colors.

- Green: probability >= 60%, positive median excess return, p10 >= -10%.
- Red: probability <= 40% and negative median excess return.
- Yellow: other computed results.
- Gray: missing/failed inputs, insufficient history, >7-day input age,
  >10-day recent common-data gaps, or 36-hour forecast expiry.

These thresholds are versioned research choices. A research color is not an
investment recommendation or evidence that the model beats its benchmark.
Hedged forecasts remain unavailable pending executable hedge quotes, investor
access and costs. Remaining catalogue assets are visibly unconnected/context-only.

## UI, scheduling and operations

Open `/global-signals` in the existing FastAPI service; `/api/global-signals`
returns the same PostgreSQL data. Existing authentication applies to both routes.
Database failures return 503 without exposing a connection string. The page shows
sample sizes, Brier scores, dates and cost assumptions, with timestamps in KST.
Restart the existing service after deploying these source changes.

For a local Windows schedule use `scripts/refresh_global_signals.ps1`; it refreshes
SQLite then PostgreSQL and logs to `outputs/global_signals`. Even a partial collection
is uploaded so affected colors fail closed. The machine and PostgreSQL must be on.
This workstation has task `Keumjm Global Signals Daily` registered at 08:15 KST
daily, with interactive-user logon, missed-run catch-up, and a 30-minute time limit.
It runs hidden while the user is logged on; overlapping task instances are skipped.

For a remotely reachable dedicated PostgreSQL database, the supplied GitHub workflow
uses **direct source-to-remote** refresh. Set repository secret
`KEUMJ_GLOBAL_SIGNALS_DATABASE_URL` and variable `GLOBAL_SIGNALS_REFRESH_ENABLED=true`
after deploying and initializing the target DB. Its schedule is 08:15 KST Tue–Sat.
A GitHub runner cannot connect to this workstation's `127.0.0.1` database. For a
managed Supabase deployment provision a dedicated project/database connection;
do not aim this dedicated-database initializer at shared application tables.

```powershell
$env:GLOBAL_SIGNALS_TEST_DB='1'
.\.venv\Scripts\python.exe -B -m unittest global_signals.tests.test_signals -v
```

Integration tests use the configured owned DB and roll back their changes.
Without that environment flag only unit tests run.

Source/method references:
- https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html
- https://www.psycopg.org/psycopg3/docs/advanced/prepare.html
- https://www.postgresql.org/docs/current/sql-createdatabase.html
- https://www.tigeretf.com/upload/etf/20250611092637008958.pdf
- https://www.ishares.com/us/literature/fact-sheet/ief-ishares-7-10-year-treasury-bond-etf-fund-fact-sheet-en-us.pdf
- https://www.ssga.com/us/en/individual/etfs/spdr-gold-shares-gld
- https://advisors.vanguard.com/investments/products/vnq/vanguard-real-estate-etf
