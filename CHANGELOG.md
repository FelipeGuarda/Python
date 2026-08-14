# Changelog

All notable changes to the FMA Python ecosystem (data-pipeline, camera-traps, plataforma-territorial, literatura-agent, schedule-agent, visualizaciones-artisticas) will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) loosely — dated sections, grouped by Added / Changed / Fixed / Deferred / Closed-rejected. Internal personal project, no public versioning.

---

## [2026-08-14] — Camera-traps: attribution becomes a flatten precondition; two long-open chores closed

Worked entirely from the Linux laptop, which turned out to be more capable than the docs assumed.

### Added
- **`camtrap/stations.names_a_station(folder_name) -> bool`** — what a station folder looks like, in every spelling the project has used (`CT23`, `CT_23`, `TC23_M20.2`). Matched by **shape**, not by membership in `station_aliases.csv`, for one concrete reason: `100EK113` **is** an alias row (an unrenamed SD-card folder that became primavera_2025's camera 5), so a membership test would call every DCIM folder a station and refuse every deployment that contains one. Shape excludes it without needing to know what a DCF folder is — that stays owned by `clocks.dcim_folder_key`. Returns `bool` precisely so it cannot become a second route from a name to a camera number; `resolve()` and the alias table remain the only one.
- **`flatten_for_camtrapdp.find_nested_stations(files)`** — `{rel path: files beneath it}` for station folders nested inside a deployment. Reports only the **shallowest** station-shaped component (`TC23_M20.2/100EK113` is one offence, not two — the operator moves one folder, and naming its children would bury the instruction), and counts files from the already-collected walk, so a station folder holding no media misattributes nothing and is not reported.
- **8 fixtures** in `tests/test_flatten.py`. The one that matters reads `station_aliases.csv` and asserts every spelling is recognised **except `100EK113`** — the 2026-08-13 hand-check ("34 TC-style rows, 0 disagreements") turned into a test that re-runs whenever a row is added. **104 total, all passing** (was 96).

### Changed
- **Attribution is now the third flatten precondition**, alongside conservation and ordering. `flatten_for_camtrapdp.py` refuses a deployment containing a station-shaped subfolder, checked after discovery and **before a single file moves** — under `--dry-run` too, since a dry run exists to be trusted. **Deliberately fatal-always, with no override flag** (Felipe's call, asked explicitly): no arrangement puts one station folder legitimately inside another, so there is nothing to override and the fix is always to move the folder up and rename it. This closes the gap `TC23_M20.2`-inside-`TC22_M19.2` exposed on 2026-08-13 — 2,460 files that would have been attributed to camera 22 at camera 22's coordinates, with `moved=2460 renamed=0 lost=0` and every existing check passing.

### Verified
- End-to-end on scratch trees: the TC23 arrangement is refused (`CT22/TC23_M20.2 (2 file(s) would be attributed to CT22)`, exit 1); a clean tree with grid folders, DCIM folders and a real filename collision flattens exactly as before.
- **Informe Anual 2025 v2 DOCX rendered** — `bash Anual-reports/2025/render.sh`, 1.4 MB with figures embedded. Open since 2026-05-20; it only ever needed pandoc on Linux (3.1.3 present).
- **`Anual-reports/2025/figures/` mirrored to `figures_pre_reingest/`** — the ⚠️ precondition the README, CHANGELOG and PROJECT_STATUS all repeat before any re-ingest, now actually satisfied.

### Machine audit — what the Linux box can and cannot do
- **MegaDetector can run here.** RTX 4070 Laptop (8 GB), AddaxAI installed at `~/.AddaxAI_files/` with `models/det/MegaDetector 5a/md_v5a.0.0.pt`. Only the **Timelapse2 sweep** still requires Windows — it is a .NET app, and the `Timelapse/` folder on Synology holds `.dll`/`.exe` only.
- **The ingest chain never opens an image.** `anchor_candidates.py`, `propose_anchors.py` and `timestamps.py` read the export CSV, the DCIM manifest and the MegaDetector JSON. So **otoño 2025's ingest is blocked on Linux by exactly one missing file** — `ImageData_total.csv` (8,997 rows), which passed the gate on 2026-08-13 but exists only on the Windows box. Copying that one CSV unblocks the whole chain here. Caveat: confirming a `NEEDS_REVIEW` anchor still means looking at the photograph.
- **The campaign images are not on this machine.** `SynologyDrive/Datos/2. Camaras trampa/…/CAMPAÑAS DE RECOLECCION DE IMAGENES/{Otoño 2025, Primavera 2025}` exist but are empty (selective sync); 309 JPGs total under that tree, all legacy pre-Sept-2024 material.

### Added — later the same day, after Felipe reviewed the gates
Felipe's objection: every gate quotes the incident that produced it, which makes them read as point-fixes rather than rules. **Audited all six.** Five are derived from a stated premise and name the incident only as a regression witness (`clocks` P1/P2 + the repairability rule, `dcim_folder_key`, the export gate's unknown-value rejection, `establish_order`'s partial-manifest refusal, `stations.resolve()`). **One was not**: `names_a_station` enumerates the three spellings this project has used, so `Camara 23` or `Cam23` walks past it.

The sharper finding: **the pipeline already saw the alien frames.** Pooled into CT22, `establish_order` reports `2460 filename(s) do not match the MMDD+counter grammar` and returns `ordered=False` — verified by running it. But that routes the evidence to the **ordering** question, and per the documented P1 asymmetry a camera that cannot be ordered is not thereby condemned, so 2,460 frames kept camera 22's identity and coordinates. The missing gate was not a new observation; it was the right question asked of an observation we already had.

- **`camtrap/provenance.py`** — owns *how many capture stories does this folder tell?* The rule: **two or more filename shapes each forming a counter run** means more than one camera. A shape is the stem with digit runs collapsed (`IMAG0001` → `IMAG#`), **extension excluded** because these cameras fire three stills and a video and `01120001.JPG`/`01120004.AVI` are one story. A *run* rather than a group, because otoño 2026 CT_27's hand-renamed `01060117_fiscalizador.JPG` is one frame and one frame is not a sequence. It **enumerates nothing**, so a naming convention nobody has seen forms its own group automatically — the property `names_a_station` lacks.
- **Validated before being wired in, not after:** all four campaigns, **28,178 files/rows, 0 false positives.** The one false positive the first version produced was real and instructive — pv 2025-2026 CT14's 13 `101EK113_`-prefixed names, written by our own `resolve_dest`, formed their own run. **Folded into the rule rather than tuned away**: a shape that is a strict suffix of another at a separator boundary is that shape wearing a prefix, resolved transitively (a single pass leaves `#`, `X_#`, `Y_X_#` in two groups).
- **16 fixtures** in `tests/test_provenance.py`, including the cases that must *not* fire: stills-plus-video, the hand-renamed one-off, our rename prefixes, repeated names, and CT16's impossible months — a clock failure that belongs to `clocks.py`, and this module must stay out of it. **120 total, all passing** (was 104).
- It imports **nothing** from `clocks` and is stdlib-only. The design gate predicted it would consume `parse_filename`; it does not need to, because shapes are grammar-agnostic — a stronger position than the one designed, and the docstring was corrected to say so rather than left claiming the coupling.

### Changed — later the same day
- **The top-level station-convention check is fatal by default.** It warned unless `--check-stations` was passed, which left the *weaker* guard on the older failure — `100EK113` reaching Timelapse2 as a deployment cost 252 rows of camera 5 from the 2025 report for a year — while its sibling, a station folder nested inside another, refuses outright. Both answer "is this folder the camera we think it is?", and two severities for one question is how the cheaper one gets skipped. `--check-stations` is now accepted and ignored, so command lines and notes written before today keep working.
- **`names_a_station` deliberately stays narrow**, with a docstring recording why: it can name *which folder to move*, which the general check cannot; the general check catches names nobody has thought of, which it cannot. Do not widen the regex to chase a name the other already covers.

### Still open
- The **horario-de-invierno** shift — deferred a third time, though it is fully doable here (otoño 2026's data is all local). Needs Felipe's decision first.
- **`primavera_2025` × `pv_2025_2026`** as two readings of one campaign — both parquets are local, so this is analysable on Linux and is a prerequisite for the primavera re-ingest.
- **pehuen's hardcoded `C:/Users/USUARIO/...` paths** (`R/01_load_data.R:50–58`), plus three more in `Anual-reports/2025/py/`.

Session log: `SecondBrain/Sessions/2026-08-14-camera-traps-nested-station-precondition-and-linux-audit.md`.

---

## [2026-08-13] — Camera-traps: the last campaign flattened, and `pv_2025_2026` turns out not to be a campaign

The primavera-verano download reviewed and flattened (19,522 files, 26 stations, **0 lost**). Two findings outrank the flatten itself.

### Fixed (on disk, before flattening)
- **🛑 A whole station was nested inside another.** `TC23_M20.2` — 2,460 files — sat inside `TC22_M19.2`. Flattening would have attributed all of them to camera 22, at camera 22's coordinates. **Nothing in the pipeline would have caught it:** the two cameras use different filename schemes (`IMAG####` vs `MMDDnnnn`), so there were zero collisions — the run prints `moved=2460 renamed=0 lost=0` and the conservation check passes. The already-ingested `pv_2025_2026` parquet proves the nesting is new: `TC23_M20.2` appears there as a *top-level* `rel_path` prefix, and camera 22 has 0 rows (never reviewed). → **Conservation and ordering are checked; station attribution is not.** A precondition for this goes through the design gate next session.
- 26 folders renamed `TC<n>_M<grid>` → canonical `CT01`–`CT26`. The mapping rule was **checked against all 34 TC-style rows** in `station_aliases.csv` (0 disagreements) rather than assumed.

### Changed
- **`pv_2025_2026` is not a campaign.** The field record has exactly three transitions — `otono_2025` → `primavera_2025` → `otono_2026` — and campaigns are named for the season they are **retrieved** in. The campaign opened 2025-05-14/06-11 and closed 2025-11-12/2026-01-14 is `primavera_2025`; the download's span matches it exactly. `pv_2025_2026` is a **second Timelapse2 review pass** over the same cards (396 shared camera+filename keys; `label_conflicts_primavera_vs_pv_2026-05-27.csv`). This sharpens the 2026-07-30 note: they are not consecutive campaigns to dedup by precedence, they are two readings of one campaign, and `CAMPAIGN_ORDER` currently encodes them as sequential.

### Added
- **`data/campaigns/primavera_2025/dcim_manifest.csv`** — 13,814 rows; 19,522 files in → 19,522 out, 1,935 renamed (all CT14), **0 lost**. CT02, CT08, CT11 and CT14 all earn `ORDER_MANIFEST`.

### Verified
- **The deployment window holds on data it was not written from.** Every station with a working clock has its frame span *inside* its field-record window, often to the day: CT04 opened 05-14 / closed 11-21 with frames 0514..1121; CT06 05-14/11-13 with 0514..1113; CT20 06-09/12-03 with 0609..1203; CT15's first frame *is* its install date. First real test of yesterday's `camtrap/anchors.py` against unseen data.
- **Otoño 2025's export passes the gate** — `full_category_sweep`, 8,997 rows (animal 818, human 478, vehicle 99, blank 7,602), matching the flatten exactly. Ingest is unblocked.
- **Unicode: nothing to do** — 39,173 paths, 0 non-ASCII, 0 NFD, 0 control characters.
- No code changed; **96 tests still pass.**

### Notes for the sweeps
- **~9 primavera_2025 stations show clock resets detectable from the field window alone** (CT03, CT05, CT08, CT14, CT17, CT23, CT24, CT26 carry January frames in a deployment opened in May/June). Filename-MMDD evidence — a preliminary signal, not a pipeline verdict.
- **CT16's clock is impossible, not merely wrong** — filenames `00300001.JPG` (month 00) and `16300071.JPG` (month 16). Corrupt RTC; no anchor repairs a clock emitting invalid dates.
- **Eight otoño 2025 images cannot be decoded** — six 0-byte in CT04, two ~4.6 MB in CT13 whose bytes are all-zero. MegaDetector skipped all eight; Timelapse2 labels them `blank`. **Felipe's call: leave them `blank`** (I recommended `unknown`); 8 rows of 8,997, no figure moves — recorded as an accepted limitation. A scanner reproducing MegaDetector's error log found the same 8, and **0 in primavera_2025**.
- ⚠️ **Both `primavera_2025` and `pv_2025_2026` are in `REPORT_CAMPAIGNS`.** Re-ingesting primavera_2025 at full size — 26 stations vs the current parquet's 14, 19,522 files vs 1,960 observations — will move the 2025 report substantially more than otoño 2025 will. Mirror `Anual-reports/2025/figures/` before either.
- **The horario-de-invierno shift is deferred a second time** — it keeps losing to campaign work.

---

## [2026-08-12b] — Camera-traps: otoño 2025 re-downloaded and flattened; ordering evidence tightened

The campaign was re-pulled from the Synology originals **with its SD-card subfolders intact** — the capture-order evidence otoño 2026 lost permanently. Flattening it exposed a defect in what the DCIM manifest is allowed to claim.

### Fixed
- **Only a camera-created folder counts as ordering evidence.** The manifest's claim is *frames in folder A precede frames in folder B, because the camera fills folders in name order*. That has two preconditions; only one was enforced (`establish_order` refuses a partially-described deployment). The other — that every group actually **is** a camera folder — was missing, so `flatten` recorded the whole intermediate path and any hand-made directory became evidence. Otoño 2025 CT04 held 723 loose frames under `M5` beside `M5/100EK113` and `M5/101EK113`; `M5` sorted **first** and asserted its January frames preceded the October ones. Sorted on, that is a backwards step in capture order — read by the diagnosis as a **clock reset on 2,097 frames**.
- `clocks.dcim_folder_key()` keeps only a DCF-shaped last component and runs at **both** ends — `flatten` on write, `timestamps.load_manifest` on read — so a manifest already written is corrected **by being read**. Necessary here: flattening consumes the tree it describes, so otoño 2025's manifest cannot be regenerated. The rule holds whether or not a camera can leave files beside its own DCIM folders, so it rests on no assumption about firmware.
- **Rename prefixes use the DCIM folder alone**, not the whole path. Otoño 2025 would otherwise have produced `M 11_101EK113_01160002.JPG` — a space imported into 28 filenames to disambiguate nothing. Unsafe characters are stripped (`M17 (TC20)`, `M18 (vacía, TC mala)` were real folder names).
- `--check-export` help still said `person`/`vehicle`; it is `human` now.

### Changed
- Eighteen otoño 2025 stations with a single constant folder (`M7`) previously reported `ORDER_MANIFEST` while the manifest contributed nothing — sorting by a constant is a counter sort. They now correctly report `ORDER_COUNTER`. CT14 and CT20 keep full `ORDER_MANIFEST`; CT04 is refused, fail-closed. Otoño 2026 unchanged.

### Added
- **`data/campaigns/otono_2025/dcim_manifest.csv`** — 8,997 files flattened, 8,969 moved, 28 renamed, **0 lost**.
- `tests/test_flatten.py` (9) and `TestDcimFolderKey` in `test_clocks.py` (6). **96 total, all passing.**

### Notes for the otoño 2025 sweep
- The download holds **no video at all** (8,997 files, every one `.JPG`); counter gaps run 19–25% at most stations and 0% at three. That is the 3-stills-plus-1-video trigger pattern with videos absent from this download, **not** lost stills. Confirm the videos exist on the NAS.
- CT04's 723 loose frames carry their own counter run restarting at 1, an MMDD range disjoint from and later than both DCIM folders, and no filename in common with either — a third DCIM folder flattened by hand at some earlier point.
- The existing export is animal-only (830 rows), so the campaign stays rejected by the gate until swept in full.
- **otoño 2025 is in `REPORT_CAMPAIGNS`** — re-ingest will move the 2025 annual report's numbers. Mirror `Anual-reports/2025/figures/` first.

---

## [2026-08-12] — Camera-traps: the field record becomes a pipeline input; otoño 2026 ingested

`data/campaigns/field_notes.csv` stops being a reference document and starts driving the clock diagnosis. The change that matters is the **deployment window**: it used to be derived from the anchors, and anchors exist only where a clock already broke — so 26 of otoño 2026's 27 stations had **no window at all**, and a forward clock jump (a clock set *ahead*, which keeps every capture delta positive and is invisible to backwards-step detection) could not have been detected for any of them.

Also lands two days of uncommitted work: the Camtrap DP vocabulary correction to the export gate, and the one-time migration of the monitoring workbook.

### Added
- **`camtrap/anchors.py`** — owns what the field record asserts about a camera's clock, deliberately split into two assertions with different preconditions: the **deployment window** (every station, needs no photograph) and an **anchor** (only where the clock failed, needs a datable frame). Absorbs the anchor-CSV schema and `load_anchors` from `timestamps.py`, so the file format is stated in one place.
- **`propose_anchors.py`** — joins the visit record to `anchor_candidates.csv` and writes `anchor_proposals.csv` with a `READY` / `NEEDS_REVIEW` / `NOT_NEEDED` status per segment. Promotes nothing automatically.
- **`setup/build_field_notes.py` + `data/campaigns/field_notes.csv`** — 106 visits, 27 stations, one row per physical visit (a revision closes one campaign and opens the next). The workbook held three date conventions at once — Chilean `d/m/y` as text, `m/d/y` off camera screens, and cells Excel had already parsed with the machine locale; the last are the dangerous ones because a wrong reading looks clean. Swaps are detected against each sheet's plausible window and recorded in `data_flags`; a value plausible both ways is flagged, never picked. 57 of 106 rows carry a flag. `clock_state` defaults to `unknown`, never `ok`.
- **`visit_date_only`** anchor type (APPROXIMATE). All 27 otoño 2026 opening visits record a date and no time, so an exact anchor would assert an hour nobody wrote down — which is how CT18's install anchor came to claim `14:00:00` against a notebook that says only `2025-11-14`.
- **`tests/test_anchors.py`** — 19 fixtures. **81 total, all passing** (was 59).

### Changed
- **`anchor_candidates.py`** now ranks the swept export **above** MegaDetector: `human_labelled` / `vehicle_labelled` (someone looked and said so) outrank `person_detection` / `vehicle_detection` (an unconfirmed guess). On otoño 2026 that is 584 + 25 confirmed frames and **zero** unconfirmed — the sweep corroborated every MegaDetector person hit.
- **A station with no deployment window is now reported as `unverified` clean, not clean.** CT27 is the live case: no install record, so the in-window test never ran. Its passing verdict is an absence of evidence, and both `timestamps.py` and `clocks.diagnose` now say so instead of claiming "every frame is in-window".
- **`camtrap/exports.py`** — Camtrap DP's vocabulary named constant-by-constant (`TYPE_HUMAN`, `TYPE_BLANK`, …) so consumers stop restating it; likewise MegaDetector's in `camtrap/detections.py` (`CATEGORY_PERSON`, …). The two vocabularies are deliberately spelled differently because they are different vocabularies.

### Fixed
- **The export gate used our own invented vocabulary** (`person`/`empty`) instead of Camtrap DP's (`human`/`blank`), which the Timelapse2 template emits verbatim. Otoño 2026's first properly swept export passed **only because `vehicle` is spelled the same in both** — its 584 `human` rows counted as neither assigned nor proof, so the same campaign with no vehicle frames would have been rejected as unswept *after* the sweep was done. An unrecognised `observationType` is now a hard rejection (`unrecognised_category_values`) rather than a note: a value the gate cannot interpret vanishes from the tally, which is exactly the failure above.
- **A visit is not an anchor.** CT01's notebook says the deployment ran 2025-11-24 → 2026-05-13 while its frames run 2025-11-26 → 2026-05-14 across one coherent segment with no reset. Turning that visit into an anchor would apply a two-day offset to a clock that was never wrong. Anchors are now proposed only where the segment would otherwise be refused.
- **Witness vs navigational evidence.** The first version of the proposer paired CT18 segment 0 with `11190001.JPG` — a counter-`0001` frame — for a **−5 day offset applied to ten frames whose clock was correct**. A counter-`0001` frame is the first file on a card, not a photograph of the technician; the camera simply did not trigger for five days after install. Only a frame that *witnesses* a visit can date one.

### Verified
- The 3-day visit-window tolerance is **measured, not guessed**: across the 20 otoño 2026 stations provably coherent from capture order alone (so any gap to the notebook is the notebook's imprecision), the largest excursion past a recorded visit date is **+1.67 d**. The constraint is one-sided at each edge — a frame before the install or after the retrieval is impossible, while a quiet stretch inside the window is evidence of nothing (CT06 and CT11 went 35 and 41 days from install to first trigger; CT19 stopped firing 91 days before retrieval).
- Applying the window changes **zero** verdicts while giving 26 stations a check they never had.
- **otoño 2026 ingested end-to-end**: 1,785 rows, 26 clean stations, CT18 refused on all five segments (10/32/40/3/227, `valid_effort=FALSE`) — reproducing the 2026-08-03 hand analysis mechanically.

### Open
- **The Mayo 2026 horario-de-invierno shift.** Every camera was set back 1 h at that visit, and Chile left summer time 2026-04-04, so otoño 2026 frames between those dates read **1 h ahead of local time**. A ~40-day systematic time-of-day error that no reset detector can see — an hour never breaks segment coherence. Awaiting Felipe's decision on whether to correct it.
- The other three campaigns still need Timelapse2 sweeps; nothing else can move them.

Session log: `SecondBrain/Sessions/2026-08-12-camera-traps-field-record-as-pipeline-input.md`.

---

## [2026-07-09] — Pehuen research: Monterroso overlap categorisation on Dhat4 CI

Reframed `Research/pehuen-species-interactions/R/04_temporal_overlap.R` around the Monterroso et al. (2014) classification — the ecological standard for interpreting Dhat4. Explored Watson's U² and a Dhat4 randomisation test first; both were dropped after user pushback identified a conceptual mismatch: those tests answer "**are the two activity curves different?**", not the actual biological question **"is the observed 0.79 overlap meaningful?"**. Dhat4 has no natural null value (two nocturnal species can share Dhat4 ≈ 0.8 just from both being nocturnal), so a "significance of Dhat4" test isn't well-posed. The Monterroso threshold interpretation — applied to the *whole CI*, not the point estimate — sidesteps that trap by classifying the pair rather than testing it.

### Changed
- **`R/04_temporal_overlap.R`** — single-loop stats computation (Dhat4 + 1000-boot 95% CI + Monterroso category). Category assigned from the CI: clean `Low/Moderate/High` when the whole CI sits in one band; compound labels (`Low–Moderate`, `Moderate–High`, `Low–High`) when the CI straddles a threshold, to avoid overstating confidence. Per-pair PNGs now carry an annotation strip `Overlap: <category>   (Dhat4=x, 95% CI [l, h])`. Summary figure has two vertical cutoffs (0.50, 0.75), subtle background band shading, and category appended to each pair label. Override camtrapR's default title so it uses species names, not argument identifiers (`sp1`/`sp2`).
- **`data/overlap_stats.csv`** — new output. `sp1, sp2, guild_type, n1, n2, dhat4, ci_low, ci_high, category, small_sample`. Suitable for direct inclusion in the annual report.
- **`README.md`** (pehuen) — Monterroso classification documented in the "Running the analysis" section.

### Removed
- **10 stale `2026-06-25` PNGs** in `figures/overlap_pairs/` — regenerated with today's annotations, so the old copies were duplicates without categories.

### Results at a glance (10 pairs)
- **1 pair clean High**: Guiña × Zorro culpeo — CI [0.77, 0.85] entirely above 0.75. Statistically defensible statement that these two native carnivores share a diel niche.
- **5 pairs Moderate–High** (CI straddles 0.75): Guina × Liebre, Zorro × Liebre, Puma × Guiña, Puma × Liebre, Puma × Jabalí.
- **4 pairs clean Moderate**: Guina × Perro, Zorro × Perro, Zorro × Jabalí, Puma × Zorro culpeo.
- **No pair in the Low band** — nothing at Bosque Pehuén shows unambiguous temporal avoidance under the Monterroso cutoffs.

Session log: `SecondBrain/Sessions/2026-07-09-pehuen-overlap-monterroso-categorisation.md`.

---

## [2026-06-17] — Camera-traps: Otoño 2026 campaign integrated, +Quique

May 2026 SD pull (campaign name **Otoño 2026**, slug `otono_2026`) reviewed end-to-end and staged for ingestion. CSV registered in `data-pipeline/config.yaml`. New species — Quique (*Galictis cuja*) — added to the canonical catalog with a CLIP English prompt; first project record (5 obs in this campaign). Yesterday's Vaca addition validated: 579 rows tagged Vaca in this campaign, all of which would have been mislabeled Caballo. Ingestion itself is held until CT_18's clock-reset timestamps are corrected.

### Added
- **`data-pipeline/species.yaml`** — Quique (*Galictis cuja*). CLIP prompt: `"lesser grison small mustelid weasel"`. Native, no `is_invasive` / `is_priority` flag. 29 CLIP species + 4 reviewer-discovered non-CLIP entries = 33 total (was 28+4=32).
- **`camera-traps/data/campaigns/otono_2026/new_labeled_data_reviewed.csv`** — 1785 rows, 25 deployments (CT_02 and CT_12 produced no animal triggers; timelapse parser is observation-centric so they're correctly absent from `ct_deployments`). Date range covers 2025-? through 2026-05-15, except CT_18 which has 135 rows stuck at 2017-01-01 (see below).
- **`data-pipeline/config.yaml`** — 4th `camera_traps.campaigns` entry: `name: "Otoño 2026"`. Comment block immediately above the entry flags the CT_18 timestamp issue and instructs not to run `--ct` until corrected.

### Changed
- **`camera-traps/README.md`** — Status header rewritten for 2026-06-17; species table +Quique row + filled-in Invasive/Priority cells; CLIP species count 26 → 29 in Step 3; Campaign History table now includes Otoño 2026 and corrected paths for the prior three campaigns.
- **`PROJECT_STATUS.md`** — top "Last updated" line rewritten; section 1 species.yaml count 31 → 33; section 3 Last Updated/What Changed/Integration/Blockers refreshed; component table +Otoño 2026 row; Open Items: new CT_18 timestamp-fix entry.

### Notes
- **CT_18 clock reset**: 135 rows on CT_18 carry `DateTime` 2017-01-01 (camera clock reverted to factory default at some point during the deployment). Felipe has the real deployment-start date in his field notebook; until it's transcribed, `python run_fetch.py --ct` is held on the Linux box. Once the anchor is in hand, one re-stamp + re-ingest finishes the integration. Until then the `otono_2026` entry sits dormant in config.yaml behind a comment.
- **Zero cross-campaign overlap** verified via stdlib CSV-vs-CSV check against otono_2025 / primavera_2025 / pv_2025_2026. No dedup script needed (unlike the primavera_2025 case, this is a fresh pull, not a partial re-pull).
- **CLIP horse/cow confusion** that motivated yesterday's Vaca prompt is now quantifiable: 579 Vaca rows (#1 species in this campaign) vs 70 Caballo rows — strong evidence the prompt distinguishes correctly. Revisit `clip_confidence_threshold` (0.28) only if the false-positive rate on tightly-similar pairs (Vaca↔Caballo, side/rear shots) looks bad after ingestion.

Session log: `SecondBrain/Sessions/2026-06-17-camera-traps-otono-2026-ingest-prep-and-quique.md`.

---

## [2026-06-16] — Camera-traps: review UI burst context, full-frame display, resume loader, +Vaca

Reworked Phase-1 review UI to better support species disambiguation. Reviewers now see burst context (prev/current/next thumbnails sourced from the MD JSON, including empty triggers) and full frames instead of bbox crops. CLIP classifier untouched — it still receives the bbox crop, keeping its subject-isolation accuracy. Added a startup loader that rehydrates review progress from the previously-exported CSV, eliminating the in-memory-only footgun that bit during this session.

### Added
- **`phase1_labeling/app.py`** — `load_station_index()` builds `{station: [files sorted alphabetically]}` from the MD JSON (includes empty triggers, so reviewer sees what happened before/after the animal trigger). `neighbors(fp)` returns prev/next within station, `None` at deployment boundaries. **Resume loader**: on a fresh session with empty `confirmed`, reads `new_labeled_data_reviewed.csv` and rehydrates `st.session_state.confirmed` / `outcomes`, auto-jumping to the first species batch with unconfirmed images.
- **`data-pipeline/species.yaml`** — Vaca (*Bos taurus*). Aliases: `vaca, vacuno, ganado vacuno, bovino`. CLIP prompt: `"domestic cow cattle bovine"`. `is_invasive: true`. 28 CLIP species now (was 27). Cows were being misclassified as Caballo on BP Mayo 2026.

### Changed
- **`classify_campaign/crop_utils.py` → `cropping.py`** — the file is cohesive (only cropping code, no junk drawer), so the `_utils` suffix was lazy. Imports updated in `run_classification.py`; the now-unused `crop_to_bbox` import was dropped from `phase1_labeling/app.py`. General principle to apply elsewhere when `_utils` suffixes are spotted on cohesive modules.
- **Review UI grid** — was 5-col bbox-cropped thumbnails, now 3-col triptych grid `[anterior | actual | siguiente]` with full-frame thumbnails. Burst context (typically 2-3 frames per trigger) is now the primary species-disambiguation cue alongside the proposed species label.
- **`THUMB_SIZE`** 280 → 1280, **`JPEG_QUALITY`** 75 → 85 in `phase1_labeling/app.py`. Streamlit's expand-icon lightbox shows the cached JPEG as-is, so the cached resolution determines expand quality. In-grid display is unchanged (`use_container_width=True` clamps the visual width). Memory tradeoff acceptable for local desktop review (~10-18MB cache per page).
- **`camera-traps/README.md`** — project tree (`cropping.py`), Step 4 review-UI section rewritten to describe the burst triptych + full-frame display, species table now includes Vaca and the missing Invasive cells for Caballo / Gato doméstico.

### Fixed
- **Sibling import error when launching Streamlit** — `streamlit run phase1_labeling/app.py` was failing with `ModuleNotFoundError: No module named 'classify_campaign'` because Streamlit puts the script's directory on `sys.path`, not the project root. Added a 2-line `sys.path.insert(...)` at the top of `app.py`. This was a latent issue that newer Streamlit versions surfaced (older versions added CWD to sys.path more aggressively).

### Notes
- **Habit to keep**: hit "Exportar CSV revisado" periodically during long sessions — it's now the durable checkpoint and the resume loader will pick it up on next launch.
- **CLIP horse/cow confusion** may persist on side/rear shots even with the new Vaca prompt. If false-positive rate is high after re-classification, consider tightening `clip_confidence_threshold` (currently 0.28) — but only after seeing the data.

Session log: `SecondBrain/Sessions/2026-06-16-camera-traps-review-ui-context-strip-and-resume.md`.

---

## [2026-06-02] — Observatorio: piso vegetacional layer (plataforma-territorial)

New toggleable Leaflet layer on the Observatorio page: photointerpretation of Bosque Pehuén's vegetational floor (48 polygons). Off by default; click a polygon for `BIOTOPO / Distrito / Superficie (ha)`.

### Added
- **`plataforma-demo/src/components/PisoVegetacionalLayer.jsx`** — self-contained React layer: owns its own fetch, color map, and popup. `Observatorio.jsx` only adds 1 import, 1 `useState(false)`, 1 conditional render, and 1 checkbox. Future GIS layers should follow this shape.
- **`scripts/convert_piso_vegetacional.py`** — pure-Python shapefile → GeoJSON converter (`pyshp` + `pyproj`, no GDAL). Handles ESRI ring orientation (CW = outer), reprojection EPSG:32718 → EPSG:4326, and the actual UTF-8 dbf encoding. Writes the same payload to `data/`, `public/data/`, and `dist/data/`.
- **`data/piso_vegetacional.geojson`** (+ public/dist copies) — 48 features. Source shapefile preserved under `data/piso_vegetacional_source/veg_foto_BP.*` (outside the served frontend) so the script remains reproducible.

### Changed
- **Palette designed in two iterations.** First attempt (greens only, dark → light by density) collapsed visually to ~6 distinguishable swatches across 10 classes. Final palette groups by ecological type: greens for Bosque (4 classes), ochres for Renoval (3), blues/violets for Matorral / Pradera / Estepa.
- **No polygon borders** — `stroke: false`. White borders fought the Esri satellite imagery underneath.

### Fixed
- **Mojibake (`MesÃ³fito`) in popups.** First conversion used `encoding="latin-1"` based on a misleading test; the .dbf is genuinely UTF-8 (matching the .cpg). Codepoint inspection on Windows is authoritative — the cmd console can't render `ó` and shows `�`, which is not the same as a decode failure. Switched to `encoding="utf-8"`; also resolved a silent color-fallthrough where the mismatched key was hitting `FALLBACK_COLOR` for 4 polygons.

### Documentation
- README top block, "Four Pages" table, and "Real Data" list updated.
- `PROJECT_STATUS.md` Plataforma section: Observatorio row + Priority 3 list.

### Notes
- **`DISTRITO` field documented**: physiographic / terrain-relief classification (`Plano / Ondulado / Cerrano / Montano`) — geomorphological, independent of vegetation. Currently a popup property; could become a separate layer.
- **`ESPECIES_D` field omitted** from popups for now — Felipe doesn't have a key for the species codes (NF / ND / NA / NP / AA / SC). Re-enable when a legend surfaces.

Session log: `~/Documents/Obsidian FG/SecondBrain/Sessions/2026-06-02-plataforma-piso-vegetacional-layer.md`.

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
