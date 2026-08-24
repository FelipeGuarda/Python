# data-pipeline — test plan

**Written:** 2026-08-24 · **Status:** designed, not written. Deferred by Felipe on the day
the `ct_*` rebuild landed, so that documentation could catch up first.
**Audience:** whoever writes these — possibly a fresh session with no memory of the rebuild.

---

## The gap this closes

`src/` is 1,642 lines and has **no tests at all**. As of 2026-08-24 four modules carry
guarantees that rest on nothing more durable than having been run once each by hand:

| module | lines | what it guarantees |
|---|---:|---|
| `src/parsers/canonical_ct.py` | 188 | the canonical table becomes `ct_*` without re-deriving anything |
| `src/canonical_gate.py` | 189 | a stale database is a detectable condition, not a silent one |
| `src/recovery.py` | 211 | the irreplaceable weather tables survive a machine change |
| `src/ingest.py` (`_reconcile`) | 229 | the rebuild landed what it meant to land |

Compare `camera-traps`, which holds 226 tests over a much larger tree. The asymmetry is
the point: camera-traps is where the data is *produced* and it has been hardened session by
session; data-pipeline is where the data is *consumed*, and every defect this project has
spent months on lived at exactly that boundary.

## Framework: stdlib `unittest`, no new dependencies

**pytest is installed in neither environment** — not `data-pipeline`, not
`species-classifier`. That is the same gap V2-REVIEW 1.10 records as the reason a claimed
"152 tests" went unverified on 2026-08-18. Use `unittest`, matching camera-traps exactly:

```bash
conda run -n data-pipeline python -m unittest discover -s tests
```

Do **not** add pytest to `environment.yml` to make this nicer. The cost of a dependency
that must be present on two machines outweighs the ergonomics, and camera-traps has already
made this choice.

## Fixtures: synthetic, plus exactly one integration test

Unit tests build small canonical-shaped frames in code. They are fast, deterministic, and —
critically — they do not break when campaign data legitimately changes.

**One** integration test reads the real parquets and asserts they agree with
`data/CANONICAL_STATE.json`. It stays green through legitimate republishes, because the
parquet and the state file move together, and fails exactly when someone rewrites a parquet
without publishing. That is the failure the whole contract exists to catch.

### The one piece with a design gate

A `_frame(**overrides)` helper that builds a minimal valid canonical DataFrame.

- **Knowledge it owns:** what a canonical row minimally looks like. A schema change then
  breaks one place instead of thirty.
- **Interface:** `_frame(n=1, **column_overrides) -> pd.DataFrame`, defaults valid, any
  column overridable per test.
- **Why it earns its place:** the canonical contract is 16 columns. Without it every test
  restates all 16, and the next `schema_version` bump becomes a day of editing tests
  rather than one line.

Everything else in these files is assertions, not structure. Do not build a framework.

---

## `test_canonical_ct.py` — highest value per line, because the projection is pure

- **Identity is deterministic across runs.** Same input → same `mediaID`/`observationID`.
  If it were not, `INSERT OR REPLACE` would duplicate on every rebuild instead of
  replacing, and the table would grow without anyone noticing.
- **`med_` and `obs_` namespaces never collide** for the same still.
- **Ids are derived from the image, never inherited.** No output value is a UUID shape.
  Timelapse mints GUIDs per *project*: primavera's legacy and current projects share 2,387
  filenames and **0** mediaIDs, so a project rebuild silently forks any table keyed on them.
- **A null datetime survives into `ct_media`.** 4,013 of 35,807 rows have no clock and must
  stay: presence needs a station, not a clock.
- **`scientificName` is NULL unless `observationType='animal'`.**
- **`classificationMethod`:** `sweep_only` → `machine`, every other `review_resolution` →
  `human`.
- **`ct_media` and `ct_observations` are 1:1.**
- **A station whose clock failed entirely still gets a deployment row** — NaT window, not
  dropped. It was still deployed.
- **The regression that matters — the 815-row defect class.** Given a row typed `blank` but
  still carrying a stale `species_latin`, the output must **not** resurrect the species.
  This is the property that made the previous implementation wrong on 515 live rows, and it
  is currently asserted by nothing.

## `test_canonical_gate.py` — cheap, in-memory DuckDB

- Missing state file → refuse. Malformed JSON → refuse. Unknown `schema_version` → refuse.
  Fail-closed is the whole posture; a test that only checks the happy path checks nothing.
- **The fingerprint catches a change where `n_rows` and `n_stations` are identical.** The
  815-row review repair moved `observation_types` and `n_animal_rows` and left `n_rows`
  untouched — invisible to a row-count check, and exactly what a consumer most needs to see.
- A campaign present in the database but absent from the published state → flagged. This is
  the `pv_2025_2026` shape: a retired campaign still being served.
- State stamped while the tables are empty → flagged.

## `test_recovery.py` — temp dir plus temp DB

- **export → restore reproduces row counts, column set AND column order.** Row counts alone
  would hide a lost column; `weather_station` carries 33 dynamically-added TOA5 columns.
- The **refuse-to-shrink** guard fires when the database has more rows than the archive.
  That direction means this machine holds readings the archive has never seen.
- The **orphaned-year** guard fires when a year-file exists that the table has no rows for.
- Partitioning actually splits by year, and re-exporting an unchanged year is a no-op.

## `test_ingest.py`

- `_reconcile` raises on a per-campaign count mismatch and on `ct_media` ≠ `ct_observations`.
- `_read_canonical` refuses a parquet whose row count disagrees with the contract, and one
  missing a canonical column.

**Scale:** roughly 30–35 tests across four files.

---

## Deliberately out of scope, and why

`toa5.py`, `met_csv.py`, `cr800.py`, `tz_utils.py`, `species.py`, `stations.py`. All
pre-existing and all untested — worth doing, but a different job from locking down the
2026-08-24 rebuild, and folding them in would make this plan large enough to stall.

`cr800.py` additionally needs either hardware or a recorded PakBus session.

`tz_utils.py` deserves its own pass whenever the DST decision's piece 2 is taken up
(storing the instant plus a fixed per-deployment offset), which V2-REVIEW puts out of scope.

---

## See also

- `camera-traps/docs/V2-REVIEW.md` §2 — what the rebuild had to satisfy
- `camera-traps/tests/` — the conventions to match
- `camera-traps/docs/DATA-HEALTH-MANUAL.md` §9 — the tiered open-items register
