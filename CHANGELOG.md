# Changelog

All notable changes to the FMA Python ecosystem (data-pipeline, camera-traps, plataforma-territorial, literatura-agent, schedule-agent, visualizaciones-artisticas) will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) loosely — dated sections, grouped by Added / Changed / Fixed / Deferred / Closed-rejected. Internal personal project, no public versioning.

---

## [2026-05-11] — 🏁 First full code review complete

The independent code review of the FMA ecosystem (started 2026-04-21) is finished. **Every finding across data-pipeline, camera-traps, and plataforma-territorial is now closed or explicitly deferred with a re-open trigger.** No silently-open work remains.

**Headline numbers:**
- 1 Critical (C1 — CR800 state-before-commit) — resolved
- ~25 Warnings (W8…W52) — all resolved
- ~50 Suggestions (S8…S78) — resolved, closed-rejected, or deferred with re-open conditions

### Fixed (today's session — Tier 1 through Tier 4)
- **Tier 1 reliability** — S50 (FastAPI health 503), S37 (mtime-keyed Streamlit cache), S10 (`managed_conn` context manager), S11 (per-event DB connections in watcher).
- **Tier 2 high-value** — S44 (`backend/paths.py`), S48 (stations.yaml coords), S55 (DuckDB CTE for `days_without_rain`), S64 (`demo_report.js`), S66 (RiskGauge pure component), S39 (`classify_all` / `apply_classifications` split).
- **Bundle A — schema authority pass** — S49 (startup drift-check; surfaced 27 real extras in the DB including `battery_voltage` accidentally absent from `ALLOWED_COLS`).
- **Bundle B — API/hook ergonomics** — S53 (docstrings + `common_name` symmetry), S59 (`useAPI` `refetch`).
- **Bundle C — camera-traps `setup/` cleanup** — S29 (`crop_to_bbox` reuse), S30 + S32 (argparse), S36 (env + CLI for `CAMPAIGNS_BASE`), S43 (extracted `setup/_fileops.py`). Net **−80 lines** despite one new file.
- **Tier 4 finalization** — S35 (`AnimalRow` dataclass), S38 (UI strings standardized to Spanish).

### Closed-rejected (with documented rationale)
- **S57** — DuckDB pushdown for `strftime` (resample is pandas-side; pushdown would add round-trips).
- **S47** — bootstrap_windows_db.py inline SCHEMA (intentional for cross-machine portability; revisit at Windows→Linux migration).
- **S14** — DuckDB secondary indices (columnar engine + composite PKs already cover hot reads at current data scale).
- **S72** — react-router-dom (overkill for 4-page internal tool; migration path documented).

### Deferred (with re-open triggers)
- **S58** — `stations.yaml` TC-11 / TC-18 both list `sd_card: M15`. Re-open when field records produce an authoritative answer.
- **S76** — Vitest tests in `plataforma-demo/`. Re-open when CI exists.

### Process
Six batches in a single day. Full narrative in `~/Documents/Obsidian FG/SecondBrain/Sessions/2026-05-11-fma-ecosystem-code-review-tier-1-and-tier-2.md`. Review state snapshots in `~/Documents/Obsidian FG/SecondBrain/Reviews/review-state-{data-pipeline,camera-traps,plataforma-territorial}.md`.

---

## [Prior to changelog adoption]

Pre-2026-05-11 history is preserved in:
- `~/Documents/Obsidian FG/SecondBrain/Sessions/` — per-session narrative logs
- `~/Documents/Obsidian FG/SecondBrain/Reviews/review-plan-fma-ecosystem.md` — master review plan with Track A–K log
- Per-project README "Last Updated" sections and `PROJECT_STATUS.md`

This `CHANGELOG.md` is the new top-level history starting 2026-05-11. Going forward, every significant change lands a one-line entry.
