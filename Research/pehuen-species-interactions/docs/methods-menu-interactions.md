---
title: "Camera-trap interaction analysis — critical methods menu"
project: Bosque Pehuén / FMA monitoreo de fauna
site: Bosque Pehuén, Andean Araucanía, Chile
tags: [camera-traps, camtrapR, occupancy, activity-patterns, spatiotemporal-interactions, methods, FMA]
status: reference
created: 2026-07-23
---

# Camera-trap interaction analysis: critical methods menu

Scope: peer-reviewed analytical methods for **spatial and temporal analysis of species
interactions** from camera-trap data, filtered against the specific constraints of the
Bosque Pehuén array. Focus is native-vs-invasive displacement (puma / güiña / zorro culpeo
vs jabalí / liebre europea / perro asilvestrado) and niche partitioning among the three
native carnivores.

This is a **menu with feasibility judgements**, not an implementation guide. Full
reference list at the end.

---

## 0. Framing: the binding constraint

The limiting factor is **sample size and single-site replication, not method availability**.
Nearly every method below is technically runnable on the data; the question is whether the
parameters are *identifiable* and the result *defensible* at ~26 stations with
tens-of-detections cells.

Two prerequisites gate most of what follows:

1. **A real effort matrix.** True per-station deployment start/end dates → `camtrapR::cameraOperation()`
   → `camtrapR::detectionHistory()`. Without this, everything in Bucket B is blocked, and
   detection rates remain approximations. This is the single highest-value action in the whole
   document — higher-value than any individual model.
2. **An independence filter.** No minimum-interval thinning has been applied yet. Activity
   and time-interval methods assume independent events. The 30-minute convention for
   medium-sized solitary mammals is the defensible operational standard (O'Brien et al. 2003;
   see also the Goldstein et al. 2024 critique of arbitrary thresholds — declare the choice
   and justify it rather than leaving it implicit).

Everything below assumes these two are done or explicitly flagged as not done.

---

## A. TEMPORAL — beyond pairwise Δ4

### A1. Activity-level estimation and formal comparison — HIGH VALUE

- **Canonical paper:** Rowcliffe, J.M., Kays, R., Kranstauber, B., Carbone, C. & Jansen, P.A.
  (2014) Quantifying levels of animal activity using camera trap data. *Methods in Ecology and
  Evolution* 5(11): 1170–1179. doi:10.1111/2041-210X.12278
- **Implementation:** R package `activity` (Rowcliffe). Key functions: `fitact()`,
  `compareAct()`, `compareCkern()`, `transtime()`.
- **Minimum data:** times of detection in radians. Bootstrapped SEs work at modest n but
  intervals widen fast below ~50 records.
- **Feasible here:** Yes.
- **What it adds that Δ4 does not:** This directly addresses the objection that led to
  dropping Watson U² and the Δ4 randomization test. Activity level — the *proportion of the
  day an animal spends active* — is an interpretable effect size with real units, not a
  shape-comparison statistic. `compareAct()` tests whether two species differ in that
  quantity. It gives the defensible null and effect size that Δ4 structurally lacks.
- **Practical note:** `transtime()` handles solar anchoring (sunrise/sunset or double
  anchoring), which matters for Southern-Hemisphere campaigns spanning very different day
  lengths (Otoño vs Primavera-verano). Without anchoring, seasonal comparisons partly measure
  photoperiod rather than behaviour.

### A2. Small-sample estimator choice — Δ1 vs Δ4 — ACT ON THIS

- **Canonical paper:** Ridout, M.S. & Linkie, M. (2009) Estimating overlap of daily activity
  patterns from camera trap data. *Journal of Agricultural, Biological, and Environmental
  Statistics* 14: 322–337. doi:10.1198/jabes.2009.08038
- **Implementation:** R package `overlap` (Meredith & Ridout). `overlapEst()`, `bootEst()`,
  `bootCI()`.
- **The guidance:** Δ4 is appropriate when the **smaller** sample has ≥50 observations
  (package documentation); the package vignette places the crossover nearer 75, recommending
  Δ1 below 50 with a grey zone between. Δ5 should never be used — it is unstable and can
  exceed 1.

**Implication for this dataset:** with tens-not-hundreds cells, **most pairs should be
reported with Δ1, not Δ4.** Note that `camtrapR::activityOverlap()` returns Δ1 by default —
so if `overlap` was called directly with `type="Dhat4"` on sparse cells, the estimator with
the worse small-sample behaviour was used. Worth re-running and comparing; if the two
estimators disagree materially on a pair, that pair is underpowered and should be presented
as such.

**Absolute sample-size floors:**

- Lashley, M.A., Cove, M.V., Chitwood, M.C., Penido, G., Gardner, B., DePerno, C.S. &
  Moorman, C.E. (2018) Estimating wildlife activity curves: comparison of methods and sample
  size. *Scientific Reports* 8: 4173. doi:10.1038/s41598-018-22638-6
- Findings: overlap error decreased rapidly with sample size up to an **asymptote near 100
  detections**, which they recommend as a minimum. However, sub-samples remained significantly
  correlated with the full dataset down to **as few as 10 detections**.
- **Operational rule adopted here:** ≥100 = confident; 20–100 = report with explicit
  precision caveat; 10–20 = plot but do not interpret; <10 = do not plot a curve.

### A3. Seasonal / intraspecific activity shifts across campaigns

- **Implementation:** same `activity` machinery, applied *within species across campaigns*
  (Otoño 2025 / Primavera-verano 2025-26 / Otoño 2026). `compareCkern()` for curve shape,
  `compareAct()` for activity level.
- **Framing references:**
  - Frey, S., Fisher, J.T., Burton, A.C. & Volpe, J.P. (2017) Investigating animal activity
    patterns and temporal niche partitioning using camera-trap data: challenges and
    opportunities. *Remote Sensing in Ecology and Conservation* 3(3): 123–132.
    doi:10.1002/rse2.60
  - Monterroso, P., Alves, P.C. & Ferreras, P. (2014) Plasticity in circadian activity
    patterns of mesocarnivores in Southwestern Europe: implications for species coexistence.
    *Behavioral Ecology and Sociobiology* 68: 1403–1417. doi:10.1007/s00265-014-1748-1
    *(already in use for the Low/Moderate/High overlap classification)*
- **What it adds:** asks whether a single species shifts its own diel schedule between
  campaigns — a plausible signature of seasonal response or of pressure from invasives. The
  current pairwise-species framing cannot ask this at all.
- **Feasible here:** Exploratory only. Campaign-level cells are the smallest in the dataset,
  and the missing Invierno deployment breaks the seasonal sequence.

### A4. Diel phenotype classification (optional extension)

Iannarilli et al. (2024) and related work formalise classification of diel activity
*phenotypes* (nocturnal / diurnal / crepuscular / cathemeral) with hypothesis testing, rather
than only comparing curves. Worth checking the current package ecosystem before committing —
this space has moved quickly and the tooling should be verified against CRAN at time of use.
Lower priority than A1–A3.

---

## B. SPATIAL — beyond bubble maps and naive occupancy

### B0. PREREQUISITE — build the detection history and effort matrix

- **Implementation:** `camtrapR::cameraOperation()` → `camtrapR::detectionHistory()`
- **Reference:** Niedballa, J., Sollmann, R., Courtiol, A. & Wilting, A. (2016) camtrapR: an
  R package for efficient camera trap data management. *Methods in Ecology and Evolution* 7:
  1457–1462. doi:10.1111/2041-210X.12600
- Requires: true deployment start/end per station, plus any malfunction gaps. Everything in
  Bucket B depends on this.

### B1. Single-season occupancy with altitude and guild covariates — HIGHEST-VALUE SPATIAL ADDITION

- **Canonical papers:**
  - MacKenzie, D.I., Nichols, J.D., Lachman, G.B., Droege, S., Royle, J.A. & Langtimm, C.A.
    (2002) Estimating site occupancy rates when detection probabilities are less than one.
    *Ecology* 83(8): 2248–2255. doi:10.1890/0012-9658(2002)083[2248:ESORWD]2.0.CO;2
  - MacKenzie, D.I., Nichols, J.D., Royle, J.A., Pollock, K.H., Bailey, L.L. & Hines, J.E.
    (2017) *Occupancy Estimation and Modeling*, 2nd ed. Academic Press. — the standard
    book-length treatment.
- **Implementation:** `unmarked::occu()` (Fiske & Chandler 2011) or `spOccupancy::PGOcc()`
  (Doser et al. 2022). Bayesian alternative: `ubms` (Kellner et al. 2022).
- **Minimum data:** detection-history matrix + effort. Rule of thumb: ≳20–30 sites with
  several occasions each, and enough detections that naive occupancy is not pinned near 0 or 1.
- **Feasible here:** **Yes for the commoner species** (culpeo, liebre, jabalí, perro).
  Marginal for puma and güiña if detections are sparse — check naive occupancy first.
- **What it adds:** separates true occupancy (ψ) from detectability (p), and lets **altitude
  enter as a continuous covariate on ψ** — converting the elevation stratifier from a binning
  variable into a tested ecological gradient. This is the single clearest upgrade over
  presence/absence and bubble maps.
- **Note on spatial random effects:** `spOccupancy` offers NNGP spatial models, but 26
  stations at one site almost certainly cannot estimate a spatial range parameter. Start
  non-spatial.

### B2. Multi-season / dynamic occupancy (colonisation–extinction)

- **Canonical paper:** MacKenzie, D.I., Nichols, J.D., Hines, J.E., Knutson, M.G. & Franklin,
  A.B. (2003) Estimating site occupancy, colonization, and local extinction when a species is
  detected imperfectly. *Ecology* 84(8): 2200–2207. doi:10.1890/02-3090
- **Implementation:** `unmarked::colext()`, `spOccupancy::tPGOcc()`
- **Minimum data:** ≥2–3 primary periods with secondary occasions nested inside each.
- **Feasible here:** **Borderline.** Three campaigns is the bare minimum to estimate
  colonisation (γ) and extinction (ε), per-season detections are thin, and the missing
  Invierno creates an irregular interval. Treat as exploratory and do not oversell.
- **What it adds:** asks whether species are *colonising or vacating* stations across
  campaigns — conceptually much closer to a displacement question than static occupancy.

### B3. Multi-species co-occurrence occupancy (Rota) — RECOMMEND AGAINST, BUT CITE

- **Canonical paper:** Rota, C.T., Ferreira, M.A.R., Kays, R.W., Forrester, T.D., Kalies,
  E.L., McShea, W.J., Parsons, A.W. & Millspaugh, J.J. (2016) A multispecies occupancy model
  for two or more interacting species. *Methods in Ecology and Evolution* 7: 1164–1173.
  doi:10.1111/2041-210X.12587
- **Implementation:** `unmarked::occuMulti()`
- **Penalised-likelihood improvement:** Clipp, H.L., Evans, A., Kessinger, B.E., Kellner, K.F.
  & Rota, C.T. (2021) A penalized likelihood for multi-species occupancy models improves
  predictions of species interactions. *Ecology* 102(10): e03520. — use via the `penalty=`
  argument in `occuMulti()`.

**Why not to use it here.** This is the obvious "spatial exclusion with imperfect detection"
tool and it is the wrong choice at this sample size. A simulation study of the Rota model
found **high bias and low coverage in the natural parameters used for inference at fewer than
400 sites**; strong co-occurrence was detected consistently only above 400 sites, and weak
co-occurrence was never consistently detected even at 3000 sites. At 26 stations, a
native × invasive interaction term is not defensibly estimable. Penalised likelihood mitigates
but does not rescue this.

- Sample-size critique: simulation study on co-occurrence model sample size requirements,
  CREEM / University of St Andrews, bioRxiv 2024. doi:10.1101/2024.09.20.614180
  *(verify final journal version at time of citing)*
- Broader critique: Twining, J.P. et al. (2026) Can hierarchical modelling of co-occurrence
  data provide accurate inference into species interactions? *Methods in Ecology and
  Evolution*. doi:10.1111/2041-210X.70210
- Related evaluation: Dorazio, R.M. (2025) An evaluation of multi-species occupancy models
  with correlated species occurrences. *Methods in Ecology and Evolution*.
  doi:10.1111/2041-210X.70168
- Fitting failures beyond ~10 species: Kéry, M. & Royle, J.A. (2021) *Applied Hierarchical
  Modeling in Ecology, Volume 2*, section 8.2. Academic Press.

**Recommended use:** cite Rota (2016) plus the sample-size critique to explain *why this model
was deliberately not fitted*, and route the exclusion question to Bucket C. Reviewers
familiar with this literature will read that as a strength.

### B4. Density without individual ID (REM / REST / CT-DS) — NOT FEASIBLE RETROACTIVELY

- **REM:** Rowcliffe, J.M., Field, J., Turvey, S.T. & Carbone, C. (2008) Estimating animal
  density using camera traps without the need for individual recognition. *Journal of Applied
  Ecology* 45(4): 1228–1236. doi:10.1111/j.1365-2664.2008.01473.x
- **CT-DS:** Howe, E.J., Buckland, S.T., Després-Einspenner, M.-L. & Kühl, H.S. (2017)
  Distance sampling with camera traps. *Methods in Ecology and Evolution* 8: 1558–1565.
  doi:10.1111/2041-210X.12790 — implementation: `Distance`
- **REST:** Nakashima, Y., Fukasawa, K. & Samejima, H. (2018) Estimating animal density
  without individual recognition using information derivable exclusively from camera traps.
  *Journal of Applied Ecology* 55: 735–744. doi:10.1111/1365-2664.13059
- **TTE / STE / IS:** Moeller, A.K., Lukacs, P.M. & Horne, J.S. (2018) Three novel methods to
  estimate abundance of unmarked animals using remote cameras. *Ecosphere* 9: e02331.
  doi:10.1002/ecs2.2331
- **Speed / day range (REM input):** Rowcliffe, J.M., Jansen, P.A., Kays, R., Kranstauber, B.
  & Carbone, C. (2016) Wildlife speed cameras: measuring animal travel speed and day range
  using camera traps. *Remote Sensing in Ecology and Conservation* 2: 84–94.
- **Effective detection distance:** Hofmeester, T.R., Rowcliffe, J.M. & Jansen, P.A. (2017)
  A simple method for estimating the effective detection distance of camera traps.
  *Remote Sensing in Ecology and Conservation* 3: 81–89.
- **Multi-species REM:** Wearn, O.R., Bell, T., Bolitho, A., Durrant, J., Haysom, J.,
  Nijhawan, S., Thorley, J. & Rowcliffe, J.M. (2022) Estimating animal density for a community
  of species using information obtained only from camera-traps. *Methods in Ecology and
  Evolution* 13: 2248–2261.
- **Reviews / comparisons:** Gilbert, N.A., Clare, J.D.J., Stenglein, J.L. & Zuckerberg, B.
  (2021) Abundance estimation of unmarked animals based on camera-trap data. *Conservation
  Biology* 35: 88–100. doi:10.1111/cobi.13517 · Twining, J.P. et al. (2022) A comparison of
  density estimation methods for monitoring marked and unmarked animal populations.
  *Ecosphere* 13: e4165.

**The common disqualifier.** None of these require individual recognition, but **all require
an estimate of the effective area surveyed by each camera** — detection zone distance and
angle, plus (for REM) animal speed and day range, or (for REST) staying time in a focal area,
ideally from video. The current deployments are not distance-calibrated and effort is
approximate.

**Verdict:** do not chase density-without-ID retroactively. It requires a purpose-built field
protocol on a future campaign (marked distance intervals in the field of view, video
triggers, recorded camera height/angle). Worth designing into the next deployment if absolute
density is a genuine institutional need.

### B5. Spatial capture–recapture for güiña — **CORRECTED: NOT FEASIBLE WITH CURRENT PLACEMENT**

> **Correction to earlier advice.** I previously recommended güiña SECR as the one readily
> available absolute-density opportunity, on the assumption that spotted güiña are reliably
> individually identifiable from standard lateral camera images. The literature does not
> support that assumption.

- **Key paper:** Gálvez, N., Kramer, T., Gallardo, B., Minte, E., Alarcón, V. &
  Palomo-Muñoz, G. (2026) Zenith placement of camera traps can individually identify the
  güiña *Leopardus guigna*: implications for population studies and conservation. *Oryx*
  (First View), pp. 1–8. doi:10.1017/S003060532510238X — **Open Access (CC-BY)**

**What it establishes:**

- Güiña **flanks do not carry enough individual variation**; the **dorsal** zone does
  (following Blair 2014, MSc thesis, University of Kent). Standard horizontal/lateral camera
  placement therefore does not reliably support individual ID.
- The workable approach is **zenith placement** — cameras mounted vertically, facing down.
  Recommended height from body length *L* and field-of-view angle *α*:
  **h = 3L / (2·tan(α/2))**. For güiña (L ≈ 0.6 m) with α = 42°, this gives ≈ 2.34 m;
  empirical focus calibration selected **2.5 m**.
- Effort required in their study: **12,784 trap-days across 40 stations** yielded 586
  independent events and **only 12 identified individuals**, 9 with at least one spatial
  recapture.
- **Melanistic morphs cannot be identified at all** — 57.1% of their güiña photographs were
  melanistic or indeterminate and were excluded outright. This alone can halve an effective
  sample.
- Sex determination is not possible from zenith images (testicles not visible).
- Identification succeeded in 83.3% of independent events of spotted güiña, comparable to
  leopard cat work (82.2%, Bashir et al. 2013).
- As of Gálvez et al. (2023), **no published density estimate exists for güiña anywhere**,
  despite ~20 years of camera-trap effort in Chile.

**Revised verdict:** SECR for güiña is **not achievable from the existing Bosque Pehuén
lateral-camera data**. It is, however, a well-defined and genuinely novel future project —
and note that it would be a *first for the species*, which is a strong publication argument.
If pursued, it needs: zenith stations at ~2.5 m, high shutter speed, 3-image bursts,
station spacing informed by güiña home range (Dunstone et al. 2002), and a paired
zenith/lateral station design to quantify the detection-rate trade-off.

**SECR references for when the data exist:**

- Efford, M.G. (2004) Density estimation in live-trapping studies. *Oikos* 106: 598–610.
- Royle, J.A., Karanth, K.U., Gopalaswamy, A.M. & Kumar, N.S. (2009) Bayesian inference in
  camera trapping studies for a class of spatial capture–recapture models. *Ecology* 90(11):
  3233–3244. doi:10.1890/08-1481.1
- Implementation: `secr` (Efford); Bayesian alternatives `oSCR`, `nimbleSCR`.
- Unmarked/partially-marked SCR (imprecise, Bayesian-only): Chandler, R.B. & Royle, J.A.
  (2013) Spatially explicit models for inference about density in unmarked or partially marked
  populations. *Annals of Applied Statistics* 7(2): 936–954. doi:10.1214/12-AOAS610 · see also
  Augustine, B.C. et al. (2019) *Ecosphere* 10: e02627.
- Güiña home range / spatial organisation (for buffer and spacing decisions): Dunstone, N.,
  Durbin, L., Wyllie, I., Freer, R., Jamett, G.A., Mazzolli, M. & Rose, S. (2002) Spatial
  organization, ranging behaviour and habitat use of the kodkod (*Oncifelis guigna*) in
  southern Chile. *Journal of Zoology* 257: 1–11.

### B6. Detection-rate GLMM with altitude — honest fallback

- **Implementation:** `glmmTMB` or `lme4`. Structure:
  `detections ~ altitude + season + guild + offset(log(effort)) + (1|station)`,
  negative-binomial family (camera-trap counts are reliably overdispersed).
- **Minimum data:** counts + a real effort offset (B0 again).
- **Feasible here:** Yes, and it is the sensible fallback for species where occupancy is
  underpowered.
- **Caveat to state explicitly:** detection rate conflates abundance with detectability.
  Frame results as a *relative index*, not as an equal alternative to occupancy. See
  Sollmann, R., Mohamed, A., Samejima, H. & Wilting, A. (2013) Risky business or simple
  solution — relative abundance indices from camera-trapping. *Biological Conservation* 159:
  405–412, and Burton, A.C. et al. (2015) Wildlife camera trapping: a review and
  recommendations for linking surveys to ecological processes. *Journal of Applied Ecology*
  52: 675–685. doi:10.1111/1365-2664.12432

---

## C. JOINT SPATIO-TEMPORAL — the main gap

### C1. Niedballa comparative framework — READ FIRST, USE AS PRIMARY TOOL

- **Canonical paper:** Niedballa, J., Wilting, A., Sollmann, R., Hofer, H. & Courtiol, A.
  (2019) Assessing analytical methods for detecting spatiotemporal interactions between
  species from camera trapping data. *Remote Sensing in Ecology and Conservation* 5(3):
  272–285. doi:10.1002/rse2.107
- **Implementation:** flexible R simulation function provided with the paper; the tests
  themselves are linear models, Mann–Whitney U, a permutation test, and a test based on
  randomly generated records.
- **What it is:** a systematic comparison of methods for detecting (a) spatiotemporal
  avoidance — avoidance of a *site* by one species after another's presence — and (b) temporal
  segregation, on simulated data with *known* interaction strength.

**Why this is the most important entry in the document:** it quantifies detectability as a
function of sample size, which converts "we found no avoidance" from a weak claim into a
defensible one.

- For narrow unimodal activity patterns (concentration κ=3), peak differences as small as
  **2 hours** were reliably detected given **≥40 records**.
- Peak differences of **5 h or more** were detectable with **fewer than 10 records** (κ≥2).
- But where avoidance was **subtle (odds ratio ≈ 2), even 100 records per species were
  insufficient**.
- With modest bimodal activity (κ=1), even considerable temporal segregation was undetectable
  regardless of sample size.

**Translation for Bosque Pehuén:** gross native/invasive segregation is detectable at current
n; subtle station-level avoidance is very likely not. Cite this to state that quantitatively
instead of hand-waving.

### C2. Time-to-encounter / time-between-detections at shared stations

- **Canonical papers:**
  - Karanth, K.U., Srivathsa, A., Vasudev, D., Puri, M., Parameshwaran, R. & Kumar, N.S.
    (2017) Spatio-temporal interactions facilitate large carnivore sympatry across a resource
    gradient. *Proceedings of the Royal Society B* 284(1848): 20161860.
    doi:10.1098/rspb.2016.1860
  - Cusack, J.J., Dickman, A.J., Kalyahe, M., Rowcliffe, J.M., Carbone, C., MacDonald, D.W. &
    Coulson, T. (2017) Revealing kleptoparasitic and predatory tendencies in an African mammal
    community using camera traps: a comparison of spatiotemporal approaches. *Oikos* 126(6):
    812–822.
  - Harmsen, B.J., Foster, R.J., Silver, S.C., Ostro, L.E.T. & Doncaster, C.P. (2009) Spatial
    and temporal interactions of sympatric jaguars (*Panthera onca*) and pumas (*Puma
    concolor*) in a Neotropical forest. *Journal of Mammalogy* 90(3): 612–620. — origin of the
    AB/BA time-interval notation, and a direct puma precedent.
- **What it answers:** does a native species arrive at a shared station sooner or later than
  expected after an invasive one has passed? This is behavioural avoidance that neither
  occupancy nor diel overlap can detect.
- **Critical caveat from Karanth's own results:** in low-density reserves they estimated
  spatio-temporal overlap of **zero**, and attributed this to low photo-encounter rates rather
  than genuine complete segregation. **Sparse cells manufacture apparent segregation.** Any
  zero or near-zero overlap at Bosque Pehuén must be interpreted against this.

### C3. Avoidance–attraction ratios (AARs) — AVOID; CITE THE REFUTATION

- **Refutation:** Dymit, E.M. (2025) Avoidance–attraction ratios incorrectly characterize
  behavioral interactions with camera trap data. *Ecology* e70134. doi:10.1002/ecy.70134
- Finding: AARs based on time intervals between detections **incorrectly characterise
  behavioural interactions**, and comparison of avoidance strength among species pairs is
  **confounded by artifacts driven by the relative encounter rate** of the species in the pair
  rather than by avoidance itself.
- **Why this matters acutely here:** the six focal species have wildly divergent detection
  rates (perro/liebre vs puma/güiña). AAR comparisons across those pairs would be measuring
  exactly the confound Dymit identifies.
- **Verdict:** do not build native-vs-invasive comparisons on AAR ratios. Cite Dymit if a
  reviewer or collaborator proposes them.

### C4. Multivariate Hawkes process — state of the art, probably out of reach

- **Canonical paper:** Nicvert, L., Donnet, S., Keith, M., Peel, M., Somers, M.J., Swanepoel,
  L.H., Venter, J., Fritz, H. & Dray, S. (2024) Using the multivariate Hawkes process to study
  interactions between multiple species from camera trap data. *Ecology* 105(4): e4237.
  doi:10.1002/ecy.4237
- **Implementation:** R package `camtrapHawkes` — https://github.com/LisaNicvert/camtrapHawkes
  (code and data archived on figshare, doi:10.6084/m9.figshare.24552157)
- **What it does:** models how detections of one species raise or lower the *intensity* of
  another species' detections in continuous time, with flexible pairwise interaction functions
  allowing asymmetry and time-varying effects. A principled successor to time-interval
  heuristics, and handles more than two species simultaneously.
- **Feasible here:** Probably not for a first paper — event-hungry and higher modelling
  overhead. **But:** the authors note the MHP can serve as a benchmark for other interaction
  methods, and Dymit (2025) cites it as a sounder alternative to AARs. Name it in the
  Discussion as the future direction; consider it seriously once campaigns accumulate.

---

## D. Feasibility triage

### Defensible now — single site, ~26 stations, current n

| Method | Bucket | Condition |
|---|---|---|
| Activity level + `compareAct` | A1 | None beyond independence filter |
| Δ1 (not Δ4) overlap with CIs | A2 | Re-run sparse pairs |
| Seasonal activity shifts | A3 | Exploratory framing only |
| Single-season occupancy + altitude | B1 | **Requires effort matrix**; common species only |
| Detection-rate GLMM | B6 | **Requires effort matrix**; index framing |
| Niedballa randomization test | C1 | With explicit power caveats |
| Time-to-encounter at shared stations | C2 | With low-n segregation caveat |

### Blocked — needs more data, more sites, or a new field protocol

| Method | Bucket | What it needs first |
|---|---|---|
| Multi-species co-occurrence (Rota) | B3 | Hundreds of sites — effectively out of reach |
| REM / REST / CT-DS density | B4 | Distance-calibrated deployment protocol |
| Güiña SECR | B5 | **Zenith camera redeployment** (see correction) |
| Dynamic occupancy | B2 | More primary periods; regular seasonal intervals |
| Multivariate Hawkes | C4 | Substantially more events |

---

## E. Ranked shortlist

The additions that would most strengthen a native-vs-invasive interaction study at a single
Andean site, in priority order.

**1. Effort matrix → single-season occupancy with altitude and guild covariates**
`unmarked::occu()` / `spOccupancy::PGOcc()` · MacKenzie et al. (2002)
*The enabling move. Converts the elevation stratifier into a tested gradient, separates
occupancy from detectability, and unblocks the entire spatial bucket.*

**2. Niedballa time-interval randomization framework**
R function supplied with paper · Niedballa et al. (2019), *Remote Sens Ecol Conserv* 5(3):272–285
*Directly answers whether a native avoids stations recently used by an invasive, and is
quantitatively honest about what this sample size can and cannot detect.*

**3. Activity-level estimation with `compareAct()`**
`activity` · Rowcliffe et al. (2014), *MEE* 5(11):1170–1179
*Cheap temporal upgrade giving overlap a real effect-size companion, plus intraspecific
seasonal comparison. Supplies the defensible null that Δ4 structurally lacks.*

**4. Corrected small-sample overlap reporting (Δ1)**
`overlap` · Ridout & Linkie (2009); Lashley et al. (2018)
*Lowest-effort, highest-credibility fix in the document. May change existing conclusions.*

### Methods to name as *deliberately excluded*, with citation

Stating these explicitly in Methods or Discussion is a strength, not an omission:

- **Rota multi-species co-occupancy** — Rota et al. (2016) + sample-size critique (≥400 sites)
- **Avoidance–attraction ratios** — Dymit (2025), *Ecology* e70134
- **REM / REST / CT-DS density** — Howe et al. (2017); Nakashima et al. (2018) — effort not
  distance-calibrated
- **Watson U² and Δ4 randomization** — already dropped; the justification is that they test
  curve difference rather than ecological meaningfulness of overlap. A1 is the constructive
  replacement.

---

## F. Regional literature — Chilean temperate forest carnivores

Essential for framing, and for the Introduction/Discussion of any Bosque Pehuén output. Note
that **Nicolás Gálvez's group (Fauna Australis / Wildlife Ecology and Coexistence Laboratory,
PUC Villarrica, CEDEL)** works in the Andean piedmont of La Araucanía — the closest active
research group to Bosque Pehuén, and the most obvious collaboration or peer-review target.

**Güiña occupancy and landscape ecology (Araucanía):**

- Gálvez, N., Hernández, F., Laker, J., Gilabert, H., Petitpas, R., Bonacic, C., Gimona, A.,
  Hester, A. & Macdonald, D.W. (2013) Forest cover outside protected areas plays an important
  role in the conservation of the Vulnerable guiña *Leopardus guigna*. *Oryx* 47(2): 251–258.
  doi:10.1017/S0030605312000099 — **first güiña occupancy estimates, Andean piedmont Araucanía.**
- Gálvez, N., Guillera-Arroita, G., St. John, F.A.V., Schüttler, E., Macdonald, D.W. &
  Davies, Z.G. (2018) A spatially integrated framework for assessing socioecological drivers
  of carnivore decline. *Journal of Applied Ecology* 55(3): 1393–1405.
  doi:10.1111/1365-2664.13072 — *of particular relevance given the sociology/conservation bridge.*
- Gálvez, N., Infante, J., Fernandez, A., Díaz, J. & Petracca, L. (2021) Land use
  intensification coupled with free-roaming dogs as potential defaunation drivers of
  mesocarnivores in agricultural landscapes. *Journal of Applied Ecology* 58: 2962–2974.
  doi:10.1111/1365-2664.14026 — **the closest Chilean precedent for the native-vs-dog question.**
- Gálvez, N., Infante-Varela, J., de Oliveira, T.G., Cepeda-Duque, J.C., Fox-Rosales, L.A.,
  Moreira, D. et al. (2023) Small wild felids of South America: a review of studies,
  conservation threats, and research needs. In: Mandujano, S., Naranjo, E.J. &
  Andrade-Ponce, G.P. (eds) *Neotropical Mammals: Hierarchical Analysis of Occupancy and
  Abundance*, pp. 13–41. Springer, Cham. doi:10.1007/978-3-031-39566-6_2

**Güiña activity patterns:**

- Delibes-Mateos, M., Díaz-Ruiz, F., Caro, J. & Ferreras, P. (2014) Activity patterns of the
  vulnerable guiña (*Leopardus guigna*) and its main prey in the Valdivian rainforest of
  southern Chile. *Mammalian Biology* 79: 393–397. doi:10.1016/j.mambio.2014.04.006
- Hernández, F., Gálvez, N., Gimona, A., Laker, J. & Bonacic, C. (2015) Activity patterns by
  two colour morphs of the vulnerable güiña, *Leopardus guigna* (Molina 1782), in temperate
  forests of southern Chile. *Gayana Zoología* 79(1): 102–105. — **Araucanía site, Nothofagus
  at lower elevations grading into Araucaria above ~900 m — closely comparable to Bosque Pehuén.**

**Mesocarnivores, dogs and invasives:**

- Moreira-Arce, D., Vergara, P.M., Boutin, S., Carrasco, G., Briones, R., Soto, G.E. &
  Jiménez, J.E. (2016) Mesocarnivores respond to fine-grain habitat structure in a mosaic
  landscape comprised by commercial forest plantations in southern Chile. *Forest Ecology and
  Management* 369: 135–143. doi:10.1016/j.foreco.2016.03.024
- Beltrami, E., Gálvez, N., Osorio, C., Kelly, M.J., Morales-Moraga, D. & Bonacic, C. (2023)
  Ravines as conservation strongholds for small wildcats under pressure from free-ranging dogs
  and cats in Mediterranean landscapes of Chile. *Studies on Neotropical Fauna and Environment*
  58(1): 138–154. doi:10.1080/01650521.2021.1933691 — uses co-occurrence modelling + KDE;
  a useful template *and* a useful object of methodological critique given B3 above.
- García, C.B., Svensson, G.L., Bravo, C., Undurraga, M.I., Díaz-Forestier, J., Godoy, K.
  et al. (2021) Remnants of native forests support carnivore diversity in the vineyard
  landscapes of central Chile. *Oryx* 55(2): 227–234. doi:10.1017/S0030605319000152
- Guzmán-Aguayo, L., Magni-Pérez, F., González, B.A., Estades, C.F., Medel, R. &
  Hernández, H.J. (2023) Occupancy patterns of two contrasting carnivores in an industrial
  forest mosaic. *Forest Ecology and Management* 544: 121170.
  doi:10.1016/j.foreco.2023.121170

**Conservation status:**

- Gálvez, N., Napolitano, C., Ibacache, F., Agostini, I. & Pliscoff, P. (2025)
  *Leopardus guigna*. The IUCN Red List of Threatened Species 2025.
  doi:10.2305/IUCN.UK.2015-2.RLTS.T15311A50657245.en
- Napolitano, C., Díaz, D., Sanderson, J., Johnson, W.E., Ritland, K., Ritland, C.E. &
  Poulin, E. (2015) Reduced genetic diversity and increased dispersal in guigna (*Leopardus
  guigna*) in Chilean fragmented landscapes. *Journal of Heredity* 106: 522–536.

---

## G. General references and best practice

- Burton, A.C., Neilson, E., Moreira, D., Ladle, A., Steenweg, R., Fisher, J.T. et al. (2015)
  Wildlife camera trapping: a review and recommendations for linking surveys to ecological
  processes. *Journal of Applied Ecology* 52: 675–685. doi:10.1111/1365-2664.12432
- Wearn, O.R. & Glover-Kapfer, P. (2017) *Camera-Trapping for Conservation: A Guide to
  Best-Practices.* WWF-UK, Woking.
- Rovero, F. & Zimmermann, F. (eds) (2016) *Camera Trapping for Wildlife Research.* Pelagic
  Publishing.
- Kéry, M. & Royle, J.A. (2016, 2021) *Applied Hierarchical Modeling in Ecology*, Volumes 1
  and 2. Academic Press.
- O'Brien, T.G., Kinnaird, M.F. & Wibisono, H.T. (2003) Crouching tigers, hidden prey:
  Sumatran tiger and prey populations in a tropical forest landscape. *Animal Conservation*
  6: 131–139. — origin of the 30-minute independence convention.
- Fiske, I. & Chandler, R. (2011) unmarked: an R package for fitting hierarchical models of
  wildlife occurrence and abundance. *Journal of Statistical Software* 43(10): 1–23.
  doi:10.18637/jss.v043.i10
- Doser, J.W., Finley, A.O., Kéry, M. & Zipkin, E.F. (2022) spOccupancy: an R package for
  single-species, multi-species, and integrated spatial occupancy models. *Methods in Ecology
  and Evolution* 13(8): 1670–1678. doi:10.1111/2041-210X.13897
- Kellner, K.F., Fowler, N.L., Petroelje, T.R., Kautz, T.M., Beyer, D.E. & Belant, J.L. (2022)
  ubms: an R package for fitting hierarchical occupancy and N-mixture abundance models in a
  Bayesian framework. *Methods in Ecology and Evolution* 13(3): 577–584.
- Niedballa, J., Sollmann, R., Courtiol, A. & Wilting, A. (2016) camtrapR: an R package for
  efficient camera trap data management. *Methods in Ecology and Evolution* 7: 1457–1462.
  doi:10.1111/2041-210X.12600

---

## H. R package summary

| Package | Purpose | Bucket |
|---|---|---|
| `camtrapR` | Data management, effort/operation matrix, detection histories | B0 |
| `overlap` | Coefficient of overlapping Δ1/Δ4, bootstrap CIs | A2 |
| `activity` | Activity level, solar anchoring, `compareAct`/`compareCkern` | A1, A3 |
| `unmarked` | `occu`, `colext`, `occuMulti` (frequentist) | B1, B2, B3 |
| `spOccupancy` | Bayesian single/multi-species/spatial occupancy, `tPGOcc` | B1, B2 |
| `ubms` | Bayesian occupancy via Stan | B1 |
| `glmmTMB` | Negative-binomial detection-rate GLMMs with effort offset | B6 |
| `secr` | Spatial capture–recapture (future, güiña) | B5 |
| `Distance` | Camera-trap distance sampling (future) | B4 |
| `camtrapHawkes` | Multivariate Hawkes process (future) | C4 |

---

## I. Open items

- [ ] Compile true per-station deployment start/end dates → build effort matrix.
      **Deferred 2026-07-28**: raw installation/maintenance file exists but needs
      cleanup with a field collaborator. Design decisions locked in (see
      Changelog): upstream Python script `camera-traps/build_camera_operation.py`
      writes per-campaign `camera_operation.csv` at
      `camera-traps/data/campaigns/<name>/camera_operation.csv`; malfunction
      verification uses a hybrid threshold rule (`gap > max(3 × p95_of_intervals, 7 days)`)
      applied to both end-of-deployment and mid-deployment gaps; flagged
      candidates written to `camera_operation_flags.csv` for human review;
      downstream `R/00_camera_operation.R` consumes the reviewed CSV. Awaits
      clean input file.
- [x] Apply 30-minute independence filter and document the choice
      **(2026-07-28: applied in `R/01_load_data.R` via `MIN_DELTA_TIME_MIN <- 30`
      and `filter_independent_events()`; `record_table.rds` is now event-filtered
      per (station × species × campaign); `records_all.rds` untouched for
      date-based analyses.)**
- [x] Re-run sparse-pair overlap with Δ1; compare against existing Δ4 results
      **(2026-07-28: `R/04_temporal_overlap.R` now dispatches per pair via
      `estimate_overlap()` — Δ4 when smaller sample ≥ 50, Δ1 otherwise; 1000
      bootstrap resamples; `data/overlap_stats.csv` carries new `estimator`
      column; comparison against previous Δ4-only results implicit in the
      regenerated table.)**
- [ ] Compute per-species-per-campaign n table against the A2 thresholds (100 / 20 / 10)
- [ ] Fit single-season occupancy for common species with altitude covariate
      (blocked on effort matrix above)
- [ ] Obtain and read Niedballa et al. (2019) supplementary R function
- [ ] Decide whether zenith güiña redeployment enters the next campaign design
- [ ] Consider contacting Gálvez (PUC Villarrica) — nearest comparable dataset and methods group

---

## Changelog

- **2026-07-28** — Applied the two § 0 prerequisites in code: 30-min
  independence filter on `record_table.rds`, and per-pair Δ1/Δ4 estimator
  dispatch in `04_temporal_overlap.R` (crossover at min-sample < 50, Ridout
  & Linkie 2009). Camera-operation matrix designed but deferred pending
  cleanup of the raw installation file (upstream location and hybrid
  malfunction check rule decided; see Open items).
- **2026-07-23** — Initial compilation. **B5 corrected**: güiña SECR reclassified from
  "feasible, high priority" to "blocked pending zenith redeployment", on the basis of
  Gálvez et al. (2026, *Oryx*), which establishes that güiña flanks lack sufficient individual
  variation and that melanistic morphs are unidentifiable.
