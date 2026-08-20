# HANDOFF — Camera-trap clock repair: segment-aware diagnosis and anchored repair

**Written:** 2026-07-30 · **Status update 2026-08-20: IMPLEMENTED — this is now the
specification of record, not a plan.** `camtrap/clocks.py` (757 lines) implements §5's
two preconditions, one rule and three flags, with 605 lines of fixtures in
`tests/test_clocks.py`. Read §5 as the spec for what the code does. **§8.1 and §8.2 are
still open field questions** — CT18's install date, and whether the older campaigns have
install photos — and are the reason this file is kept rather than archived.
**Audience:** a fresh AI session with no memory of the 2026-07-30 discussion.
**Read this whole file before touching code.** The plan at the end depends on facts
established in the middle, and the middle contains two traps that already produced
wrong answers once during analysis.

---

## 0. TL;DR

Our protocol for deciding whether a camera's broken clock can be repaired was
**structurally blind**: it diagnosed clock resets from a CSV containing only *animal*
photos. Reading the same campaign's **all-images** CSV revealed that camera **CT18 in
otoño 2026 reset its clock 4 times, not once**. The shipped repair applies a single
offset to all four segments, so most CT18 datetimes are fabricated. Those fabricated
dates reached the **pehuen research analysis** (65 focal-species records). The annual
report is unaffected.

The fix is not a patch to CT18. It is:

1. A new module `camtrap/clocks.py` that owns clock-failure diagnosis (segments,
   ordering evidence, coherence, repairability).
2. A specification — **two preconditions, one rule, three flags** — replacing the
   current binary `year < 2024` heuristic.
3. An **ingest protocol change**: clock diagnosis must read an all-images export with
   *all* detection categories labelled (empty / animal / person / vehicle), enforced by
   a validation gate, not documentation.

There is also **one unresolved question** (§7) about SD-card DCIM subfolders and our
flatten step which must be answered before the ordering logic can be trusted for
high-volume cameras.

---

## 1. Environment and orientation

| | |
|---|---|
| Repo | `C:\Users\USUARIO\Dev\Python\camera-traps` (branch `main`) |
| Python env | `C:/Users/USUARIO/miniforge3/envs/camera-traps/python.exe` — **the only env with pandas.** Never use base. Deps pinned in `Anual-reports/2025/requirements.txt` |
| pehuen repo | `C:\Users\USUARIO\Dev\Python\Research\pehuen-species-interactions` (separate repo, R) |
| Prior session logs | `C:\Users\USUARIO\Dev\obsidian-secondbrain\SecondBrain\Sessions\` |

**Immediately relevant prior session logs** — read these two if anything below seems to
lack background:
- `2026-07-30-camera-traps-canonical-observation-table-implemented.md` — the `camtrap/`
  package and canonical observation table landed earlier the same day. This handoff is
  a *second, separate* discovery from later that day.
- `2026-07-29-camera-traps-design-notes-and-canonical-observation-schema.md` — the
  design gate that produced `camtrap/`, plus findings F001–F010.

**Uncommitted work is on disk.** `git status` shows `camtrap/` (new package),
`observations.parquet` for all four campaigns, regenerated annual-report figures and
parquets, and `data/campaigns/otono_2026/ImageData_total.csv` (the new all-images
export that exposed this bug) — all untracked or modified, **none committed**. Nothing
in this handoff has been committed either. See §9.

### Key files

| File | Why it matters |
|---|---|
| `timestamps.py` | Owns clock repair. `classify_epochs` (line ~206) is the root-cause function. `repair_campaign` (~221) applies one offset per station. |
| `camtrap/stations.py` | Canonical station IDs `CT01`–`CT27`; `resolve()` maps legacy spellings via `data/campaigns/station_aliases.csv`. |
| `camtrap/observations.py` | `CANONICAL_COLUMNS`, `write_canonical`, `read_campaigns`. The canonical schema that needs a new flag. |
| `data/campaigns/*/deployment_anchors.csv` | Field anchors. Schema documented in `timestamps.py` lines 21–50. |
| `data/campaigns/otono_2026/ImageData_total.csv` | **The all-images export. 12,068 rows. Only otoño 2026 has one.** |
| `data/campaigns/*/new_labeled_data_reviewed.csv` | Animal-only export. **This is what ingest currently reads — the blind spot.** |
| `setup/flatten_for_camtrapdp.py` | Flattens DCIM subfolders. `resolve_dest` (line ~60) is implicated in §7. |
| `Anual-reports/2025/py/01_data_prep.py` | Annual report. `REPORT_CAMPAIGNS` (line 71) — note it **excludes** `otono_2026`. |
| `README.md:129` | Documents the export step that *specifies* the animal-only filter. |
| `README.md:363` | `## DESIGN_NOTES` — project coupling risk the `design-first` skill must apply. |
| pehuen `R/01_load_data.R` | Lines ~50–54 paths, ~308 `records_all`, ~415 `record_table`. The spill path. |

---

## 2. The bug, precisely

### 2.1 What was blind

`timestamps.py` reads `new_labeled_data_reviewed.csv` — animal photos only. For otoño
2026 that is **1,785 rows out of 12,068 actual images**. Clock resets that occur
between two animal photos are invisible. `README.md:129` *documents* this filter
("File → Export data as CSV → save as `ImageData_animals.csv`"), so this is the
specified protocol failing, not an oversight.

### 2.2 What CT18 actually did

Derived from `ImageData_total.csv`, JPGs only, ordered by filename counter
(reproduce with the recipe in §6):

| Segment | Images | Animal | Camera-clock range | Span |
|---|---:|---:|---|---|
| 0 (real) | 10 | 2 | 2025-11-19 06:41:40 → 2025-11-28 15:42:55 | 9.4 d |
| 1 | 32 | 14 | 2017-01-01 00:00:00 → 2017-01-19 07:53:24 | 18.3 d |
| 2 | 40 | 9 | 2017-01-01 00:00:00 → 2017-01-13 23:20:46 | 13.0 d |
| 3 | 3 | 1 | 2017-01-01 00:00:00 (2 seconds) | ~0 |
| 4 | 227 | 111 | 2017-01-01 00:00:00 → 2017-04-02 19:48:32 | 91.8 d |

**Four resets, five segments.** The current anchor asserts one bogus block.

### 2.3 What the shipped repair does with that

`data/campaigns/otono_2026/deployment_anchors.csv` has a `last_real_proxy` anchor
mapping camera-time `2017-04-02 19:48:30` → real `2026-05-15 12:10:00` (retrieval),
producing offset **+3329 d 16:21:30**, applied to *all* bogus rows. Consequences:

- Segments 1, 2 and 3 all restart at `2017-01-01`, so they are all stamped starting
  `2026-02-12 16:21:30` — **overlapping each other and segment 4**.
- 24 animal photos that actually occurred in Nov/Dec 2025 are dated Feb/Mar 2026.
- Segment 4's 111 photos are biased **late** by the unknown camera-death gap.
- Bogus camera-time = 123.1 d inside a 167.9 d window ⇒ **44.8 d unaccounted**.

The anchor's own note says *"2 real-time photos captured before reset."* That was
written from the animal-only view. The camera actually ran correctly for **9.4 days /
10 photos**. **The premise of the anchor is itself an artefact of the blind spot.**

### 2.4 CT18 has a second, independent failure

CT18 is the **only** camera in otoño 2026 where the filename's `MMDD` prefix disagrees
with its own `DateTime` — **166 of 312 JPGs**, with impossible month values (`0008`,
`0019`, `1921`, `1926`). Its month/day registers are corrupt, so datetimes may not tick
coherently *even inside a segment*.

This matters because it distinguishes CT18 from the otoño 2025 cases. The verdict notes
in `manual_review_verdicts_2026-06-02.csv` describe otoño 2025 CT15 as a clean **+8 yr**
offset where *"filename codifica 09-10"* — i.e. a **year-only** error where the filename
still encodes the true month/day. **That is repairable from the filename; CT18 is not.**
Do not apply one recipe to both.

### 2.5 CT18's install anchor is not corroborated by any photo

`deployment_anchors.csv` asserts `install, real=2025-11-14 14:00:00,
camera=2025-11-14 14:00:00` (offset zero). But CT18's counter-`0001` image — the first
image ever written to the card — is `11190001.JPG` at **2025-11-19 06:41:40**, and it is
an **animal**, not an install photo.

Every other camera's counter-`0001` image is a midday `unclassified` frame (CT_19 11-21
12:19, CT_04 11-22 12:00, CT_09 11-22 14:19, CT_12/13/15/10/16 all 11-25 11:25–13:30,
CT_01 11-26 13:39, …) — those are almost certainly **installation photos of the
technician, invisible because nobody labelled people**. CT18 has no such frame, and
there is a 5-day hole before its first image.

⇒ **CT18 currently has zero verified anchors.** We cannot even distinguish Scenario A
(clock correct) from Scenario B (clock misconfigured at install) for its 10
good-looking photos. Requires Felipe's field notebook. See §8.

---

## 3. The spill into pehuen

**The annual report is CLEAN.** `01_data_prep.py:71` sets
`REPORT_CAMPAIGNS = ("otono_2025", "primavera_2025", "pv_2025_2026")` — `otono_2026` is
not in it. **Nothing to redo in the annual report for CT18.**

**pehuen is contaminated.** `R/01_load_data.R` reads `PATH_OT26`. `records_all` (~line
308) filters only `!is.na(datetime)` — it **never checks `valid_date`**. So 65
focal-species CT18 records with fabricated dates are in `data/records_all.rds`:

| Species | Records | Trustworthy |
|---|---:|---:|
| Sus scrofa | 33 | 0 |
| Lycalopex culpaeus | 16 | 1 |
| Lepus europaeus | 12 | 0 |
| Canis lupus familiaris | 2 | 0 |
| Puma concolor | 2 | 0 |

The single trustworthy row: `11190001.JPG`, *Lycalopex culpaeus*, 2025-11-19 06:41:40.

Affected consumers:

| Script | Uses | Verdict |
|---|---|---|
| `02_detection_summary.R` | `records_all` + effort normalisation | **contaminated** in numerator *and* denominator |
| `05_spatial_distribution.R` Fig B1 | `records_all`, station counts, no dates | **spatially valid** — should KEEP these records |
| `06_seasonal_detection_maps.R` | `records_all`, `month(datetime)` → season | **worst case** — segment 4's 111 photos split 66 Otoño / 45 Verano purely as an offset artefact |
| `03_activity_patterns.R`, `04_temporal_overlap.R` | `record_table` | **SAFE** — `record_table` filters `valid_time_of_day == TRUE` (line ~415), which caught it |

The `valid_time_of_day` guard worked. The `valid_date` guard was never consumed.

**Decided:** CT18 is **excluded from rate figures entirely** — from the effort
**denominator** as well as the numerator, because it died at an unknown date so its
trap-nights are unknowable. It contributes no effort for *any* species at that station.

---

## 4. Scope of the blind spot beyond CT18

**Only otoño 2026 has an all-images export.**

| Campaign | All-images export? |
|---|---|
| `otono_2026` | ✅ `ImageData_total.csv` (12,068 rows) |
| `otono_2025` | ❌ only `ImageData_animals.csv` (animals only) |
| `primavera_2025` | ❌ none |
| `pv_2025_2026` | ❌ none |

Therefore **every existing `unrepairable_pending` diagnosis was written from the same
blind view** and is a lower bound:

- `otono_2025`: CT15 ("49 bogus rows"), CT16 ("9 bogus rows"), CT19 ("101 bogus rows,
  entire deployment may be affected")
- `primavera_2025`: CT16 ("68 bogus rows")
- `pv_2025_2026`: CT16 ("3 bogus rows")

Those row counts are **animal counts, not reset counts**. CT16 is the chronic offender
(clock failures in three campaigns — a fact the earlier design session noted the code
cannot see). None of these diagnoses can be trusted until re-done from full exports.

**Upside:** the annual report currently drops **143 records** from otoño 2025
CT15/CT16/CT19 (including 6 puma, 3 guiña, 2 pudú) as `unrepairable_pending`. If
install/retrieval photos exist in those campaigns, the new protocol may recover them.

**Risk:** otoño 2025 **is** in `REPORT_CAMPAIGNS`, so re-diagnosis may move the annual
report's numbers again. `Anual-reports/2025/figures_pre_canonical/` already preserves
the previous generation for diffing; do the same before any regeneration.

---

## 5. The specification (agreed with Felipe)

Felipe proposed a scenario taxonomy; it was verified correct and then compressed into a
rule that subsumes it. **Implement the rule; use the scenarios as test fixtures.**

### 5.1 Two preconditions — both must hold or the camera fails closed

- **P1 — Ordering established.** Capture order must be provable (filename counter
  unique and monotonic). If not, no repair reasoning applies at all.
- **P2 — Segment coherence.** Within a segment, filename `MMDD` must equal `DateTime`
  MMDD, and deltas must be non-negative and plausible. CT18 fails this.

### 5.2 The rule

> **A segment is repairable if and only if it is coherent AND contains at least one
> anchor.** The number of repairable segments equals the number of segments an anchor
> falls inside.

Anchors come from: the install photo, **every** mid-deployment maintenance visit, and
the retrieval photo — each needing a real wall-clock datetime recorded in the field
*and* an identifiable frame in the image sequence.

The protocol lever this exposes: **anchors are cheap and each one buys a segment.** A
camera with 4 resets is fully repairable if visits happened to land in each segment. The
`mid_visit` anchor type already exists (`timestamps.py:32`) and is currently unused.

### 5.3 Split detection

Splits are **discontinuity relative to capture order, or datetimes outside the
deployment window** — *not* `year < 2024`. The current threshold misses forward jumps
(a clock set to 2030, or a subtle 2025→2024 slip).

### 5.4 Three independent flags per row

`valid_date`, `valid_time_of_day` (both exist), **plus new `valid_effort`**.

**They must stay independent.** A pure year error (2017 instead of 2025, same
MM-DD HH:MM:SS) **preserves time-of-day exactly** — those photos are valid for
activity/overlap analysis before anyone fixes the year. Conversely `last_real_proxy`
rotates time-of-day while roughly preserving date order. A single usable/not-usable
switch would discard recoverable data.

### 5.5 Felipe's scenarios → test fixtures

| | Setup | Anchors | Repairable | Verified |
|---|---|---|---|---|
| **A** | clock correct, no split | install (offset 0) | all | ✓ camera-off-at-retrieval is irrelevant; no error to propagate |
| **B** | clock misconfigured at install, no split | install (offset ≠ 0) | all | ✓ one constant offset |
| **C** | 1 split, retrieval photo + real datetime | install + retrieval | both segments | ✓ |
| **D** | >1 split, retrieval photo + real datetime | install + retrieval | first + last only | ✓ middles → presence-only, flagged |
| **E** | 1 split, camera dead at retrieval | install only | first only | ✓ |
| **F** | **zero anchors** | none | none | ← not in Felipe's list; status of the legacy archive |
| **G** | **>1 split AND dead at retrieval** | install only | first only | ← not in Felipe's list; **CT18's actual case** |

Also add fixtures for: P1 failure (counter restarts), P2 failure (register corruption),
and a **forward** clock jump.

### 5.6 Explicitly rejected: the "slack" heuristic

An earlier proposal scored repairability by *slack* S = window − Σ(segment durations),
arguing dates are pinned when S ≈ 0. **Felipe rejected it** because it assumes prompt
reboots after power loss, which is unprovable. **Do not reintroduce it as a repair
criterion.** `unaccounted_days` may be *reported in the audit log* as a diagnostic
("44.8 d unaccounted at CT18") — never used to decide validity.

---

## 6. Two analysis traps that already produced wrong answers

**Reproduce any CT18 analysis with these guards, or you will get garbage.**

**Trap 1 — `.MP4` rows carry a bogus `DateTime`.** Video rows are stamped **+1 hour**
from their paired JPG, and some carry the **2026-06-10** file-copy date. Including them
in reset detection produced **61 phantom resets** for CT18 instead of the true 4.
⇒ **Filter to `.JPG` before any chronology analysis.**

**Trap 2 — the filename counter wraps at 999.** Sorting by counter produced **987
phantom resets** for CT_14 (whose dates are perfectly sane). Cameras with >999 images
have colliding counters:

| Camera | Images | max counter | colliding |
|---|---:|---:|---:|
| CT_14 | 2,608 | 999 | 1,609 |
| CT_20 | 1,377 | 999 | 418 |
| CT_08 | 873 | 999 | 88 |
| CT_23 | 776 | 999 | 46 |
| CT_15 | 999 | 999 | 166 |

CT18 has 312 JPGs with counters 1–455, all unique ⇒ counter order **is** valid for
CT18, which is why its 4-segment result stands. **This is precisely why P1 is a
precondition.** See §7 — the wrap has a cause we have not yet resolved.

**Working recipe** (filename grammar is `MMDD` + 4-digit counter):

```python
import pandas as pd, re
tot = pd.read_csv('data/campaigns/otono_2026/ImageData_total.csv', low_memory=False)
tot['dt'] = pd.to_datetime(tot['DateTime'], errors='coerce')
c = tot[tot['Deployments'] == 'CT_18'].drop_duplicates(subset=['File'])
c = c[c['File'].str.upper().str.endswith('.JPG')].copy()          # Trap 1
c['counter'] = c['File'].str.replace(r'\.JPG$', '', regex=True, flags=re.I).str[4:].astype(int)
assert c['counter'].is_unique                                      # Trap 2 / P1
c = c.sort_values('counter')
c['seg'] = (c['dt'].diff() < pd.Timedelta(0)).cumsum()
```

Scratch scripts from the 2026-07-30 analysis are in the session scratchpad
(`ct18d.py`, `allcams.py`, `mmdd.py`, `damage.py`, `spill.py`, `anchorcheck.py`,
`sdcard.py`, `person.py`) — they will be gone next session; the recipe above is the
part worth keeping.

---

## 7. ⚠️ UNRESOLVED — SD-card DCIM subfolders vs. our flatten step

**This must be answered before the ordering logic (P1) can be trusted. Felipe raised it
and it is not resolved.**

**Felipe's hypothesis:** SD cards store ~999 images per auto-created subfolder (DCIM
folders with names like `100EK113`, `101EK113`), then start a new folder. Our
preparation step flattens all subfolders into one `CT_xx` directory, pooling
`0001.JPG` from every folder — which is why counters repeat.

**Evidence gathered 2026-07-30 — hypothesis confirmed:**
- Exactly the five cameras with >999 images have `max_counter == 999` and colliding
  counters. Cameras with ≤999 images have none. The counter is a **per-DCIM-folder**
  field that wraps at 999.
- `RelativePath` in `ImageData_total.csv` contains **only** the deployment name
  (`CT_14`, `CT_18`, …). **The DCIM subfolder is already gone from the data.**
- `setup/flatten_for_camtrapdp.py:resolve_dest` collects the subfolder in `rel_parts`
  but **only uses it when the flat filename already exists**. Distinct DCIM folders
  whose files have different `MMDD` prefixes land flat with **no prefix** — the folder
  is permanently discarded.
- Corroborating: `primavera_2025` has a station literally named **`100EK113`** — a DCIM
  folder that was mistaken for a station (documented in the 2026-07-29 session log; it
  is camera 5).

**Questions for tomorrow:**

1. **Is the DCIM folder recoverable?** Does it still exist on Synology / in the
   originals, or has flattening already destroyed the ordering evidence for the five
   high-volume cameras? *(Highest priority — determines whether P1 can ever be
   satisfied for them.)*
2. **Does any camera have BOTH >999 images AND a broken clock?** If so, the counter
   wraps *and* the `MMDD` prefix is bogus, so **ordering may be genuinely
   unrecoverable** from filenames alone. In otoño 2026 the five wrapping cameras all
   have sane dates, so the problem is latent, not yet realised — **but it must be
   checked for the other three campaigns once their exports exist.**
3. **How should flatten preserve chronology?** Options to weigh: prefix the DCIM folder
   into the filename unconditionally; write a sidecar manifest of
   `original_path → flat_name`; or keep `RelativePath` populated with the subfolder in
   the Timelapse export. Note the folder name (`100EK113` → `101EK113`) is itself
   monotonic, so folder + counter gives a total order.
4. **Possible silent data loss.** `resolve_dest` treats *same name + same size* as a
   duplicate and **skips** it. A broken-clock camera resetting to `2017-01-01`
   repeatedly generates colliding filenames (`0101xxxx`) across DCIM folders — if sizes
   also match, a genuinely distinct image is discarded. **Unverified whether this has
   actually occurred**; the flatten log (written by `process_deployment`) would show
   `skipped_duplicate` rows. Check before assuming any campaign is intact.
5. Is 999 the exact threshold, and is it camera-model dependent?

---

## 8. ⚠️ UNRESOLVED — needs Felipe's manual verification

Neither can be answered from the data. **These gate steps 6–7 of the plan.**

1. **CT18 otoño 2026 install.** Was it really 2025-11-14? Is there an install photo?
   The first card image is an animal on 11-19 06:41, and the rest of the first batch
   went in 11-19 → 11-26, so 11-14 looks a week early. Also: **any CT18
   maintenance-visit date**, which would supply an interior anchor. Without this, CT18
   has zero verified anchors (§2.5).
2. **Do otoño 2025 / primavera 2025 / pv 2025-2026 have install photos at all?** An
   earlier draft asserted "zero anchors" for these; that was **too strong and is
   retracted** — Felipe has not checked. Status is **unverified**, not absent. The
   midday-`unclassified` pattern in otoño 2026 suggests install photos are routine, so
   they may well exist. If they do, the 143 dropped otoño 2025 records may be
   recoverable.

**Note on why these are hand checks:** MegaDetector *can* find person frames — the JSON
joins to the total CSV on filename (that is how CT18 was checked: exactly one person
detection, conf **0.151** at 22:30, a false positive). But MD is a detector, not a
verdict, so it can only **pre-filter** candidates. Reviewing every counter-`0001` frame
plus every person detection is a short list, not a card-by-card hunt.

---

## 9. THE PLAN

`design-first` **must be active**; apply `README.md:363` DESIGN_NOTES.

### 9.1 Design gate (already passed — reproduced so it need not be re-derived)

**Knowledge owned.** `camtrap/clocks.py` owns *how a camera's clock failure is
characterised and what may be concluded from it*: what a segment is, what evidence
establishes capture order, what makes a segment coherent, and the anchor-per-segment
rule. It does **not** own anchor storage or file I/O. Deliberately **not** "detect then
repair" — that temporal split is what leaked last time.

**Interface, committed before implementation:**

```python
BOGUS_YEAR_THRESHOLD          # moves here from timestamps.py — one home for the decision

@dataclass(frozen=True)
class Anchor:                 # what a field observation asserts about a clock
    real_datetime: datetime
    camera_datetime: datetime | None
    kind: str                 # install | mid_visit | retrieval | unrepairable_pending
    exact: bool               # False => date only, time-of-day not trustworthy

@dataclass(frozen=True)
class Segment:
    index: int
    n_images: int
    camera_start: datetime
    camera_end: datetime
    coherent: bool            # precondition P2
    def contains(self, camera_dt: datetime) -> bool

@dataclass(frozen=True)
class ClockDiagnosis:
    station: str
    ordered: bool             # precondition P1
    order_evidence: str       # 'counter' | 'none'
    segments: list[Segment]
    unaccounted_days: float | None   # AUDIT DIAGNOSTIC ONLY — never a criterion (§5.6)

@dataclass(frozen=True)
class SegmentRepair:
    segment_index: int
    offset: timedelta | None  # None => unrepairable
    valid_date: bool
    valid_time_of_day: bool
    reason: str               # names the rule that fired

def diagnose(images: pd.DataFrame, station: str) -> ClockDiagnosis
def repair_plan(d: ClockDiagnosis, anchors: list[Anchor]) -> list[SegmentRepair]
```

Hidden behind it: JPG/video separation (Trap 1), counter parsing with restart detection
that fails to `ordered=False` **rather than lying** (Trap 2), MMDD register-corruption
check, segment splitting, anchor→segment assignment, and the repairability rule.

**Designed twice.** Alternative: expose only `segments()` and let `timestamps.py` assign
anchors and decide. **Rejected** — that puts the repairability rule in the caller, which
is exactly how the current single-offset bug exists. `repair_plan` returns *decisions*,
not raw facts.

**Common case.** Healthy camera: one coherent segment; `repair_plan` returns one
`SegmentRepair(offset=None, valid_date=True, valid_time_of_day=True,
reason='clock_clean')`. Two calls, no options.

**Change localization.** A new failure mode (forward jumps, a per-model filename
grammar, a fourth validity axis) changes `clocks.py` only. `timestamps.py` retains
anchor-CSV I/O, offset application, audit rendering, CLI — real work, not a
pass-through.

**Duplicated decisions.** `BOGUS_YEAR_THRESHOLD`, `ANCHOR_TYPES_EXACT`,
`ANCHOR_TYPES_APPROXIMATE` consolidate into `clocks.py`; `timestamps.py` imports them.
Station resolution reuses `camtrap.stations`. The reviewed-vs-total export rule lives in
`camtrap/`, per DESIGN_NOTES.

### 9.2 Steps

**Do 1–5 now against otoño 2026 (the only campaign with a total export). Treat 6–7 as a
second pass once the other exports exist.**

1. **`camtrap/clocks.py`** per the interface above, with §5.5 fixtures A–G plus P1
   failure, P2 failure and forward-jump cases as unit tests over synthetic sequences.
2. **`timestamps.py`** — read the **total** export for clock diagnosis, keep
   `reviewed.csv` for species; consume `repair_plan()` and apply **per-segment**
   offsets; **delete `classify_epochs`**. **Hard-fail when no total export exists** — no
   silent fallback to animal-only. Audit log reports segments, anchor assignment per
   segment, and `unaccounted_days`.
3. **Schema** — add optional `segment_index` to the anchor CSV (resolved from
   `camera_datetime` when absent) so a `mid_visit` anchor can rescue an interior
   segment; add **`valid_effort`** to `CANONICAL_COLUMNS` in `camtrap/observations.py`.
4. **Ingest gate — Felipe's protocol addition.** Three enforcement points:
   - `camtrap/` validates the total export and **rejects it when `observationType`
     contains only `{animal, unclassified}`** — i.e. the categories were never
     assigned. *(Verified 2026-07-30: otoño 2026's export contains exactly `animal`
     1,785 and `unclassified` 10,283 — so today's file would be rejected and must be
     re-exported.)* **This gate is what makes the protocol enforceable rather than
     documentary.**
   - `setup/flatten_for_camtrapdp.py` gains the same check beside `--check-stations`.
   - `README.md:129` rewritten: the export must be a full-category CSV
     (`empty` / `animal` / `person` / `vehicle`) over **all** images, replacing
     `ImageData_animals.csv`. Add the field protocol: photo at install **and** at
     retrieval with wall-clock time written down, plus the camera's own displayed time
     recorded at install (this is what distinguishes Scenario A from B).
5. **Anchor-candidate report** — small helper joining the MegaDetector JSON to the total
   CSV, listing every counter-`0001` frame and every person detection with its camera
   datetime, so Felipe's review is a short list (§8).
6. **Re-diagnose all four campaigns**, regenerate `observations.parquet`. Preserve the
   current `figures/` first (mirroring `figures_pre_canonical/`) because otoño 2025 is
   in `REPORT_CAMPAIGNS` and its numbers may move (§4).
7. **pehuen** — relax the load-time `!is.na(datetime)` filter so spatially-valid records
   survive; filter on `valid_date` in `06_seasonal_detection_maps.R`; exclude stations
   with `valid_effort == FALSE` from `02_detection_summary.R` **denominators as well as
   numerators**; keep `05_spatial_distribution.R` inclusive; regenerate
   `records_all.rds` and affected figures.

### 9.3 Blocking dependencies

- Steps 6–7 need full-category total exports for otoño 2025, primavera 2025 and
  pv 2025-2026 — **manual Timelapse2 work on Synology, the long pole.**
- §7 Q1 (is the DCIM folder recoverable?) gates whether P1 can ever hold for the five
  high-volume cameras.
- §8 gates CT18's final verdict and any otoño 2025 record recovery.

### 9.4 Definition of done

- `timestamps.py` cannot run on an animal-only or single-category export.
- CT18 otoño 2026: segments 1–4 are `valid_date=False`, `valid_time_of_day=False`,
  `valid_effort=False`; segment 0's 10 photos keep whatever status its anchor supports
  once §8.1 is answered.
- pehuen: `06` excludes CT18; `02` excludes CT18 from both numerator and denominator;
  `05` retains CT18's presence records.
- Every §5.5 fixture passes as a unit test.

---

## 10. Git state — nothing committed

`git status` on branch `main` shows a **large uncommitted body of work** predating this
handoff (the `camtrap/` canonical-observation refactor from earlier on 2026-07-30):

```
 M PROJECT_STATUS.md · README.md · timestamps.py · setup/flatten_for_camtrapdp.py
 M Anual-reports/2025/py/{01_data_prep,apply_verdicts}.py
 M Anual-reports/2025/data/*.parquet · figures/*.png · corrections_report.md
 M data/campaigns/*/deployment_anchors.csv
 D  data/campaigns/primavera_2025/new_labeled_data_reviewed.dedup.csv   (staged)
 ?? camtrap/ · data/campaigns/station_aliases.csv
 ?? data/campaigns/*/observations.parquet
 ?? data/campaigns/otono_2026/{ImageData_total.csv,timelapse_recognition_file.json}
 ?? Anual-reports/2025/figures_pre_canonical/
 ?? docs/HANDOFF-clock-repair.md        (this file)
```

Also uncommitted from an older session: deletion of
`Python/Backups/Fotos UCT.2024-05-29.17-28-40.csv` (outside this repo).

**Commit before doing anything else tomorrow** — work has been lost between machines
before. No branch was created; everything is on `main`. Suggested split:

```
feat(camtrap): canonical observation table + station convention   # the earlier refactor
docs(clock-repair): handoff for segment-aware clock diagnosis     # this file
```

⚠️ `ImageData_total.csv` is 2.4 MB and `timelapse_recognition_file.json` is 1.9 MB —
decide deliberately whether they belong in git.
