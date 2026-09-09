# WS-01 — Audit of the record and of what documents it

**Estación Meteorológica Bosque Pehuén · Campbell CR800 s/n 42107**

| | |
|---|---|
| Compiled | 2026-09-09 |
| Occasion | External request to standardise the station for inclusion in an international meteorological registry |
| Scope | Read-only audit. No file in any project was modified. |
| Author | Claude Opus 5, session `2026-09-09-estacion-meteorologica-timeline-y-metadatos-wmo` |
| Companion | Metadata sheet (58 WIGOS elements) — published artifact, see §9 |

> **How to read this.** Every claim below is either traceable to a file on this machine or computed during the audit from the raw records. Computations are labelled. Where I am inferring, the sentence says so. Where I do not know, it says that instead. Nothing here is filled in from plausibility.

---

## 1. Executive summary

The **data** is in excellent shape. The **metadata** essentially does not exist. Three findings inside the data matter more to an international submission than anything in the metadata folders.

- Seven and a half years of 15-minute records, 2018-09-21 to 2026-04-13, with exactly **one** genuine gap in the whole series (12 hours).
- The logger clock runs on a **fixed UTC−03:00 with no daylight saving**. The ingestion pipeline assumes Chilean civil time, which does observe DST, so the database copy carries timestamps one hour late in UTC for roughly half of every year.
- An undocumented **~2-hour clock excursion lasting about three months** in 2023.
- Five of sixteen instrument channels are non-operational, two of them **from day one**, and none of the failures is flagged in the data — they present as zeros and impossible constants, never as nulls.
- **No calibration has ever been performed.** No sensor is identified by make, model or serial. No sensor height was ever recorded. The directory created to hold the station protocol is empty.
- The station has been **offline since 2026-04-13** — 149 days. There is a recoverable-data deadline attached to this, around **2027-08-22**.

---

## 2. The complete timeline

### 2.1 Authoritative source

Not the `.dat` files. The authoritative series is:

```
data-pipeline/data/recovery/weather_station/{2018..2026}.parquet
```

| | |
|---|---|
| Records | 264,943 |
| Span | 2018-09-21 16:45 (−03) → 2026-04-13 13:00 (−04) |
| Interval | 15 min, timestamps at **interval end** |
| `station_id` | `bosque_pehuen` |
| Duplicate timestamps | 0 |

The seven `.dat` files in `Linea de tiempo/` stop at **2025-07-23 12:45**. The final nine months exist only in the pipeline copy.

### 2.2 A trap in the `.dat` set

**The seven files are not seven periods.** Each is a complete dump of the CR800's 47,600-record ring buffer at the moment of download, so they overlap heavily. Every file is exactly 47,604 lines (4 header rows + 47,600 records).

| File | First record | Last record | Counter range |
|---|---|---|---|
| `CR800Series_Table 21092018_30012020.dat` | 2018-09-21 16:45 | 2020-01-30 12:30 | 11138–58737 |
| `CR800Series_Table1_13092019_21012021.dat` | 2019-09-13 23:30 | 2021-01-21 19:15 | 45437–93036 |
| `CR800Series_Table 01122020_11042022.dat` | 2020-12-01 18:00 | 2022-04-11 13:45 | 88135–135734 |
| `CR800Series_Table 03082021_12122022.dat` | 2021-08-03 21:15 | 2022-12-12 17:00 | 111668–159267 |
| `CR800Series_Table 08112022_18032024.dat` | 2022-11-08 21:00 | 2024-03-18 16:45 | 156019–203618 |
| `CR800Series_Table 14012023_24052024.dat` | 2023-01-14 18:15 | 2024-05-24 14:00 | 162440–210039 |
| `CR800Series_Table 14032024_23072025.dat` | 2024-03-14 17:00 | 2025-07-23 12:45 | 203235–250834 |

333,200 raw rows collapse to **239,650 unique timestamps**. Anyone handed "seven files covering 2018–2025" without this warning will double-count.

The Synology copy `CR800Series BP_Table123-07-2025.dat` is byte-identical in content to `CR800Series_Table 14032024_23072025.dat` — verified by comparing sorted data rows; the only difference is CRLF versus LF line endings.

### 2.3 Coverage and gaps

Computed over the deduplicated series: **99.98 % complete** against the 239,697 fifteen-minute slots the span implies.

| Period | Extent | What happened |
|---|---|---|
| ≈2018-05-28 → 2018-09-21 16:45 | 116 d | Records 0–11,137 overwritten before the first download. Start date back-extrapolated from the counter, not documented. |
| **2023-07-12 13:45 → 2023-07-13 01:45** | **12 h · 47 rec** | The only true gap in the series. Cause unrecorded. |
| 2023-07-13 → ≈2023-10-10 | ≈89 d · ≈8,500 rec | Clock ~2 h behind (§4.2). Values fine, timestamps unreliable. |
| 2026-04-13 13:00 → present | 149 d, open | Telemetry failure — antenna. Possibly recoverable (§2.4). |
| Every year, ≈Apr → Sep | systematic | Timestamps 1 h late in UTC in the database copy only (§4.1). |

Records per year: 2018 — 9,725 · 2019 — 35,040 · 2020 — 35,136 · 2021 — 35,040 · 2022 — 35,040 · 2023 — 34,993 · 2024 — 35,136 · 2025 — 19,540 (to 23 July).

The eight further "gaps" of 1:15–2:15 h visible in the parquet all fall on the April DST-end date each year. They are artefacts of §4.1, not outages.

### 2.4 The recoverable-data deadline

Table1 holds **47,600 records**. At 15 minutes that is **714,000 minutes = 495.8 days** of on-board memory.

If the logger is still powered and recording, everything since the 2026-04-13 telemetry loss survives in the buffer until roughly **2027-08-22**, after which it begins overwriting. A site visit before then recovers the whole outage; after it, the data is gone.

Two unknowns, both material: whether the logger is still running, and whether the table size is unchanged. Nobody has been on site since the failure.

---

## 3. Sensor health

### 3.1 Method

Per-year non-null and zero fractions across all 40 columns of the 239,650 deduplicated raw records, plus targeted physical-plausibility checks (inter-channel correlation, variance, diurnal structure).

### 3.2 Verdict by channel

| Channel | Variable | Verdict |
|---|---|---|
| `AirTC_Max/Avg/Min` | Air temperature | **Operational, full record.** Annual means 7.0–8.2 °C; extremes −11.7 to +33.6 °C |
| `RH_Max/Avg/Min` | Relative humidity | **Operational.** Annual means 70–76 %. Anomalies Jun and Aug 2021 per the 2022 report, self-resolved, cause never determined |
| `WS_ms_Max/Avg/Min` | Wind speed | **Operational.** Annual means 1.58–1.89 m s⁻¹ |
| `WindDir_Max/Avg/Min/Std` | Wind direction | **Operational.** Dominant W, S, SE, SW; northerlies essentially absent |
| `Rain_mm_Tot` | Precipitation | **Operational.** See §3.3 for annual totals |
| `T107_10cm_*` | Soil temperature 10 cm | **Operational, full record** |
| `T107_50cm_*` | Soil temperature 50 cm | **Operational.** The 2022 report's May–Nov 2021 "anomaly" was an Excel decimal-locale artefact in that report's own processing; the raw values are correct |
| `incomingSW_Avg` | Downwelling shortwave | **Operational.** Annual peaks 1,086–1,308 W m⁻². Uncalibrated — treat as relative |
| `PtoRocio_Avg` | Dew point | **Computed on-logger, not measured.** Undocumented formula |
| `PTemp_C_Avg`, `BattV_Min` | Panel temp, battery | Housekeeping. Battery minimum never below 12.5 V in seven years |
| `BP_mbar_Avg` | Station pressure | **Never operational** (§3.4) |
| `incomingLW_Avg` | Downwelling longwave | **Not longwave** (§3.5) |
| `outgoingLW_Avg` | Upwelling longwave | **Identically zero, all 264,943 records.** Never worked |
| `outgoingSW_Avg` | Upwelling shortwave | **Sign-inverted.** Median −48 W m⁻², range −299 to +53. Possibly recoverable |
| `albedo_Avg` | Albedo | Computed on-logger from the shortwave pair; inherits their problems. 19 % zeros in 2025, first on 2025-01-15 |
| `DT_*`, `Q_*`, `TCDT_*` | Surface distance / snow depth | **Failed 2021** (§3.6) |

### 3.3 Precipitation totals

2019 — 2,126 · 2020 — 1,717 · 2021 — 1,579 · 2022 — 1,924 · 2023 — 2,861 · 2024 — 2,242 mm. Plausible for Andean Araucanía at this altitude.

**Caveat to declare:** the gauge is presumed unheated (inference from Campbell convention, unverified), which implies winter snow undercatch at a site that clearly receives snow. Better to declare this than to let a comparative analysis discover it.

### 3.4 The barometer never worked

| Statistic | Value |
|---|---|
| Mean | 636.68 mbar |
| **Standard deviation** | **0.16 mbar** |
| Range, seven years | 636.4257 – 638.1898 |

Real station pressure varies by tens of hPa on a weekly basis. A standard deviation of 0.16 mbar across seven years is physically impossible for an atmospheric measurement. Separately, 636 hPa would imply an altitude near 3,800 m — the site is near 1,220 m.

This channel is not a pressure measurement. It is a disconnected input, an unscaled default, or a bias voltage. The 2022 internal report never examined it.

*Consequence for §7: this is why the elevation could not simply be derived by inverting station pressure.*

### 3.5 The longwave channels

- `outgoingLW_Avg` is **identically zero** across all 264,943 records. The channel never functioned.
- `incomingLW_Avg` correlates with `incomingSW_Avg` at **r = 0.981**, ranges −346 to +1,199 W m⁻², and goes negative at night. Real downwelling longwave sits around 200–400 W m⁻² and does not track shortwave. This channel is either miswired or reporting an uncorrected thermopile output with no body-temperature term applied. Unusable as delivered.

### 3.6 The snow sensor

Non-zero `DT_Avg` counts by month show a clean failure signature:

- 2020: ~2,900 non-zero readings every month — fully operational
- 2021-01: 2,962 · 2021-02: 2,414 · 2021-03: 2,023 · 2021-04: 598 · 2021-05: 83 · 2021-06: 52 · 2021-07: 349 · 2021-08: 18 · 2021-09: 3
- 2021-10 onward: zero, with only isolated blips since

Operational 2018-09 to ~2021-03, degrading through mid-2021, effectively dead from August 2021.

The 2022 report reached the same conclusion and recommended inspection. **That was five years ago and it was never inspected.**

---

## 4. The clock — the most consequential finding

### 4.1 The logger runs on fixed UTC−03:00; the pipeline assumes otherwise

**Evidence 1 — no DST discontinuities.** Across seven years of raw naive logger timestamps there is not one duplicated timestamp at a DST-end transition, and not one missing hour at a DST-start transition. A clock that followed Chilean civil time could not produce that.

**Evidence 2 — solar noon.** Taking solar noon as the midpoint between the first and last quarter-hour each day with `incomingSW_Avg > 20 W m⁻²`, and correcting −0.125 h for interval-end labelling, the median across the record is **13.88 h**.

At longitude −71.733 the prediction is:

| Clock hypothesis | Predicted solar noon |
|---|---|
| UTC−03:00 | 13.78 h |
| UTC−04:00 | 12.78 h |

Observed 13.88 h. Monthly values swing 13.62 h (November) to 14.25 h (February), consistent with the equation of time. **The clock is UTC−03:00, fixed, all year, for the entire record.**

The 0.125 h correction is itself confirmation of the timestamp convention: without it the estimate sits exactly half an interval late, which is what interval-end labelling produces.

**The consequence.** `data-pipeline/src/tz_utils.py::localize_santiago_to_utc` localises the naive timestamps to `America/Santiago`, which observes DST. During the window Chile is on UTC−04:00 (roughly April to September), the pipeline reads a local `14:00` as `18:00 UTC` when the true instant is `17:00 UTC`.

**Every record in the database copy is one hour late in UTC for roughly half of every year.** The raw `.dat` files are unaffected. This also manufactures the eight spurious April gaps noted in §2.3.

### 4.2 The 2023 excursion

Weekly median solar noon around the July 2023 gap:

| Week ending | Solar noon (logger clock) |
|---|---|
| 2023-07-02 | 13.62 |
| 2023-07-09 | 14.88 |
| **2023-07-16** | **12.62** |
| 2023-07-23 | 11.75 |
| 2023-07-30 → 2023-10-08 | 11.69 – 11.75 |
| **2023-10-15** | **13.62** |
| 2023-10-22 onward | 13.62 – 14.12 |

Beginning immediately after the 12-hour gap of 2023-07-12/13 and persisting until about 2023-10-10, the clock ran **approximately 2 hours behind**. Roughly 8,500 records carry unreliable timestamps.

This appears in no log, no report and no session note. The 2022 report predates it; nothing since mentions it. It was found in this audit.

---

## 5. Metadata — what exists

| Source | Content | Trustworthiness |
|---|---|---|
| **TOA5 headers**, every `.dat` | CR800, serial **42107**, OS `CR800.Std.31.03`, program `estacion_tres_hermanas.CR8`, signature `10101`, table `Table1`, plus per-column names, units and aggregation methods | **High.** Machine-written by the logger. Serial and signature constant 2018–2025, so the program was never modified. *Caveat:* the station-name field drifts — see §6.1 |
| `plataforma-territorial/data/stations.yaml` | WS-01, lat −39.453642, lon −71.733092, model, Tailscale endpoint | **High.** Field-verified, and the file documents its own 2026-04-24 correction (the earlier −39.4417/−71.7420 was the map centre, never the logger). No altitude |
| `Informe Datos Estación Meteorológica.docx`<br>Synology · created 2022-09-13 · creator field "dell", author unnamed | Instrument-operability review, Sep 2018 – Apr 2022 | **Medium on failure dates, low on causes.** It independently found the same 2021 radiometer, albedometer and snow-sensor failures this audit found — real corroboration. But the author writes *"desconozco qué mide esta variable"* for PTemp, dew point, DT and Q; attributes an Excel decimal-locale artefact to possible instrument fault; never examines the barometer; and names no instrument model anywhere |
| `ProformaInvoice.pdf`<br>Campbell Scientific Centro Caribe · 166-2025-PA · 2025-03-31 | 20 W solar panel + 16×18″ enclosure, USD 1,563.50 | **High but nearly useless.** A 2025 repair order, not the original build. No sensors on it |
| `piso_vegetacional.geojson` | Biotope **Bosque Semidenso**, district **Ondulado**, species code `NP-AA`, 11.6 ha unit — extracted by point-in-polygon at WS-01 | **Medium.** The layer is trustworthy; the `NP-AA` expansion to *Nothofagus pumilio – Araucaria araucana* is **my inference** — confirm against the layer's legend. The classification scheme is FMA's own, not a standard one |
| `boundary.geojson` | WS-01 confirmed inside the reserve. 868.87 ha, Área de Protección Privada, Fundación Mar Adentro | **High.** Containment computed. *Note the boundary polygon is itself flagged "en revisión final" in the vault* |
| `camera-traps/data/campaigns/estaciones.csv` | 27 stations with `elevation_m`, used for the neighbourhood and relief analysis | **Medium.** Values originate in the field-notes `Altitud` column — handheld GPS readings (§7.3) |

### 5.1 Two things not to send

- **`Instrumentos Monitoreo Nasampulli CR2.kmz`** — a fluviometer on the Río Trafampulli and two canopy/open nodes at −39.016/−71.688 and −39.027/−71.674, 1,250–1,450 m. This is **Reserva Nasampulli**, a GEO Mountains partner site. Not ours.
- **`Fire risk dashboard/README.md` §2.1** — "Easting 263221, Northing 5630634 (≈ lat −39.61°, lon −71.71°)". The UTM pair is correct and converts to −39.4413/−71.7514. The decimal degrees printed beside it are **wrong by roughly 19 km in latitude**. The canonical coordinates are in `stations.yaml`.

---

## 6. Metadata — what does not exist

Absent everywhere on this machine:

- **Sensor makes, models and serial numbers** — for every sensor. No original purchase order exists.
- **Sensor heights above ground**, and the real (as opposed to nominal) soil-probe depths. Without wind sensor height the wind record cannot be reduced to a 10 m reference height for comparison.
- **Any calibration record, ever.** No schedule, no results, no certificates, no traceability to a reference standard.
- **The CRBasic program `estacion_tres_hermanas.CR8`.** Named in every file header; the file is nowhere. It holds the scan rate, the multipliers and offsets, the wiring, and the dew-point and albedo formulas — it is the single document that would explain both the dead barometer and the miswired longwave channels.
- **Installation / commissioning date.** Best available: first record 2018-09-21, back-extrapolation to ≈2018-05-28.
- **Maintenance or visit log.** One fragment survives: the 2025 proforma, with no record of when or why it was installed.
- **Siting and exposure description** — no photographs, no obstacle survey, no horizon sketch. No WMO siting class can be assigned for any variable.
- **Any registry identifier.** No DGA, DMC or WMO number anywhere in repository or vault.
- **Data licence, DOI or citation statement.**
- **The station protocol.** `SynologyDrive/1. Estacion Meteorológica/Protocolo estación meteorológica/` is an **empty directory**, created January 2025.

### 6.1 One active conflict

The station has **four names** in the record and none is canonical:

- `CR800Series`, `CR800Series_2`, `CR800Series BP` — the station-name field of different TOA5 dumps
- `estacion_tres_hermanas` — the logger program name, constant throughout

This is the Bosque Pehuén / Tres Hermanas label drift surfacing inside the instrument record itself. A registry name is permanent and public; **choose one before submitting**.

### 6.2 Sensor models — explicitly guessed

Inferred from Campbell channel-naming convention. **None verified for this build. Do not submit as fact.**

| Channels | Likely sensor | Confidence |
|---|---|---|
| `T107_10cm`, `T107_50cm` | Campbell Model 107 thermistors | Near-certain — the channel name states it |
| `DT`, `Q`, `TCDT` | SR50 / SR50A sonic ranging sensor | Near-certain — that triple is the sensor's signature |
| `AirTC`, `RH` | HMP60 or HC2S3 class probe | Moderate |
| `WS_ms`, `WindDir` | 03002 Wind Sentry cup-and-vane | Moderate |
| `Rain_mm_Tot` | TE525-family tipping bucket, unheated | Moderate |
| `BP_mbar` | CS106 class barometer | Moderate |
| Radiation quartet + albedo | Four-component net radiometer, likely CNR4 | Moderate |

---

## 7. Elevation

### 7.1 Current value

**1,223 m — provisional.** Open-Meteo elevation API, queried 2026-09-09.

Not yet submittable: the underlying DEM is unconfirmed, and an elevation without a named source is the entry that gets flagged later.

### 7.2 A coincidence that should not be read as corroboration

1,223 m is exactly CT02's recorded field altitude. CT02 is 230 m away and its figure is a handheld-GPS reading. Two different methods at two different points landing on the same integer says nothing — it may even indicate the DEM cell spans both.

### 7.3 Neighbourhood and relief

| Station | Field altitude (m) | Distance from mast (m) |
|---|---|---|
| CT02 | 1223 | 229 |
| CT05 | 1270 | 244 |
| CT01 | 1263 | 556 |
| CT17 | 1062 | 618 |
| CT07 | 1232 | 723 |
| CT14 | 1048 | 858 |
| CT27 | 1408 | 985 |

Roughly **346 m of relief within 1 km** of the mast. A 30 m DEM cell is averaging real topography here, so any two methods will disagree by more than their nominal errors suggest. Do not submit an interpolated figure.

These field altitudes come from the `Altitud` column of the camera-trap field notes — handheld GPS. Handheld vertical error typically runs 1.5–3× the horizontal: ±10–20 m under canopy, worse on a slope. They are **not** ground truth.

### 7.4 Open for the next session

1. **Name the dataset.** Open-Meteo does not report which DEM it served. My belief is Copernicus DEM GLO-90; *I am not confident*, and citability was the entire reason for preferring this over Google Earth. Confirm from Open-Meteo's documentation.
2. **Get GLO-30 citably.** The public Open-Topo-Data instance does not carry it — `copernicus30` returns `Dataset not in config`. Three routes:
   - `curl -s "https://api.opentopodata.org/datasets"` to see what it does host. `srtm30m` would give a citable cross-check — NASA SRTM v3, 1-arcsec — though SRTM was flown in 2000 and degrades in steep terrain, so it is a second opinion, not a replacement.
   - **OpenTopography's API** — free, serves GLO-30, needs an account key.
   - **Download the GLO-30 tile** covering S40/W072 from the Copernicus Data Space and read the pixel locally. One-off, and it leaves a file archivable beside the station record — which is what makes a metadata value auditable in five years. *Recommended.*

   *I am confident the latter two serve GLO-30; I have not verified current signup or tile-naming details.*
3. **Run the calibration.** Query the seven camera traps above through the same API and compare to their field altitudes. The residuals give an empirical uncertainty for this hillside instead of a nominal error bar. CT17 and CT27 are the informative ones — they sit on the steep parts. Residuals inside ±15 m make 1,223 m defensible as provisional; a 50 m miss on CT27 means only a GNSS fix will do.
4. **Record the datum.** A DEM returns orthometric height; a GNSS receiver returns ellipsoidal by default. In Chile those differ by tens of metres. Whichever number lands in the record must say which it is.

### 7.5 What would close it permanently

A **static GNSS log** — a survey-grade receiver on a tripod over the point recording raw carrier-phase observations for an hour or two, post-processed against a reference station or through a free service such as NRCan's CSRS-PPP. Centimetre-level. This is a different thing from averaging waypoints on a handheld, which reaches perhaps ±5 m.

Or **RTK** — the same physics with corrections arriving live from a nearby base. Needs a local base over a known point or a network subscription plus connectivity; with no permanent internet at Bosque Pehuén, network RTK is likely out.

FMA almost certainly owns neither. UACh and the LNAS partners do, and Antonio Lara's group is already the GEO Mountains counterpart. A borrowed receiver on the mast for two hours, during a visit already required for other reasons. Under canopy, log longer. Convert through a geoid model (EGM2008) before reporting metres above sea level.

---

## 8. What the requester probably means by "metadata"

No record exists in the vault of this specific request — nothing after the **2026-06-16 GEO Mountains** coordination meeting. That meeting is the likely origin, and it also explains why a Nasampulli KMZ sits in this directory.

Relevant context from `Meetings/2026-06-16-geo-mountains-coordinacion-inicial.md`: Swiss-funded, 12–14 months, three Andean sites (Bosque Pehuén, PN Villarrica, Reserva Nasampulli), feeding CONDESAN and the Atlas de los Andes del Sur. Participants: Patricio Contreras and Carla Marchant (LNAS), Antonio Lara (UACh), Felipe Ortega (UACh). Climate variables were identified as the only data currently common to all three sites. Agreement ~10M CLP, contingent on a methodological summary and a platform demo.

"World meteorological observatory database" is most likely one of two targets, and **which one changes the required schema**:

1. **WMO OSCAR/Surface** — the WIGOS station-metadata catalogue. Requires a WIGOS Station Identifier, issued through Chile's Permanent Representative to WMO, i.e. the **Dirección Meteorológica de Chile**. FMA cannot self-register. *I am confident OSCAR/Surface is WMO's station-metadata system and that WSIs route through the national met service; I have not verified DMC's current procedure.*
2. **A mountain-observatory inventory** run by GEO Mountains or CONDESAN — looser requirements, no national gatekeeper.

Either way, "metadata" means the **station description**, not the readings: identity and operator, exact position **including altitude**, siting and exposure, per-variable instrument make/model/serial, sensor height or depth, measurement and reporting interval, aggregation method, units, **the time reference and its offset**, calibration and maintenance history, known outages and quality flags, and data policy and licence with a contact.

Measured against FMA's own §3.5 spec in `Resources/estandares-datos-socios-plataforma-territorial.md`, we can supply `id`, `name`, `lat`, `lon`, `model` and `time_resolution`. We cannot supply installation date, calibration observations, sensor models or sensor heights.

---

## 9. Companion document

The full 58-element WIGOS checklist — every category, filled where the record allows, with blanks left blank — is published as an artifact:

**https://claude.ai/code/artifact/f66a5dd4-5e57-412e-898e-9da6a0572998**

Current tally: **21** in the record · **11** derived in this audit · **2** assumed · **23** absent · **1** conflicting.

> The ten-category structure there is my reconstruction of the WIGOS Metadata Standard (WMO-No. 1192) from recollection. It has **not** been checked against the current publication or the live OSCAR/Surface entry form. Treat the structure as a checklist to verify; the fills are the reliable part.

To render this analysis for circulation: `pandoc ANALISIS-ESTACION-WS01.md -o ANALISIS-ESTACION-WS01.docx`

---

## 10. Recommendations

### 10.1 Before replying to the requester

1. **Ask which registry.** OSCAR/WIGOS means routing through DMC; nothing should be drafted until that is settled.
2. **Start the WSI request in parallel** if OSCAR/Surface is the target. It is the one item on the list FMA does not control.
3. **Choose the station name** (§6.1). Permanent and public once submitted.
4. **Decide the data policy and licence** (§6). This is a decision, not a lookup. Note that the fire-risk dashboard in this same directory carries CC BY-NC 4.0 — that covers the dashboard, not the station data, and a non-commercial clause may be incompatible with what the registry expects. *Verify against the registry's stated policy rather than my recollection.*

### 10.2 What to submit

Submit: air temperature, relative humidity, wind speed and direction, precipitation, soil temperature at both depths, downwelling shortwave.

Withhold, declaring each as non-operational with dates: station pressure, both longwave channels, upwelling shortwave, albedo, snow depth.

Qualify: dew point, as computed rather than observed.

**Do not let the barometer through.** A constant 636 hPa in an international database is worse than an absent field.

### 10.3 One field visit closes almost everything

The ring-buffer deadline of ≈2027-08-22 gives this a date.

- [ ] Download Table1 — recovers the outage since 2026-04-13, and settles whether the logger kept running
- [ ] Retrieve `estacion_tres_hermanas.CR8` — fills four absent metadata elements at once
- [ ] Read the logger clock against a known-correct time; **record both raw readings, not a conclusion**. Confirms or refutes §4.1, which currently rests entirely on computation. (Same lesson the camera-trap programme learned when terreno supplied a verdict and the observation behind it was lost.)
- [ ] Photograph and record every sensor: make, model, serial
- [ ] Measure sensor heights above ground and real soil-probe depths
- [ ] Static GNSS log for the elevation (§7.5)
- [ ] Four photographs from the mast, one per cardinal direction, plus a horizon sketch
- [ ] Note whether the rain gauge is heated — one look, decides the snow-undercatch caveat

### 10.4 Fixes on this side

- [ ] Correct the pipeline timezone policy to a fixed UTC−03:00 offset (§4.1), and backfill the database copy
- [ ] Flag 2023-07-13 → ≈2023-10-10 as a clock-unreliable period in the warehouse (§4.2)
- [ ] Add quality flags — every channel is currently 100 % non-null including the dead ones, so a naive consumer reads failures as valid data
- [ ] Correct or remove the wrong decimal degrees in `Fire risk dashboard/README.md` §2.1
- [ ] Write the station protocol into the empty directory that was created for it

---

## Appendix · What was examined

**Repository:** the 7 `.dat` dumps in `Estacion meteorologica/Linea de tiempo/` · `merged_timeline.csv` and `unified timeline.py` · both notebooks · `data-pipeline/data/recovery/weather_station/*.parquet` (264,943 records) · `data-pipeline/data/cr800_state.json` · `data-pipeline/src/fetchers/cr800.py` · `plataforma-territorial/data/stations.yaml`, `boundary.geojson`, `piso_vegetacional.geojson` · `camera-traps/data/campaigns/estaciones.csv` · `Fire risk dashboard/README.md` · `Instrumentos Monitoreo Nasampulli CR2.kmz`

**Synology:** `SynologyDrive/Datos/1. Estacion Meteorológica/` in full, including `Informe Datos Estación Meteorológica.docx` and `ProformaInvoice.pdf` · `SynologyDrive/1. Estacion Meteorológica/Protocolo estación meteorológica/` (empty)

**Vault:** `Topics/` — Estacion-Meteorologica, CR800, Bosque-Pehuen, LNAS, Expediente-Reserva-Natural-Bosque-Pehuen · `Meetings/2026-06-16-geo-mountains-coordinacion-inicial.md` · `Resources/estandares-datos-socios-plataforma-territorial.md` · `Sessions/2026-05-06-data-pipeline-session-a-cr800-dst.md`

**Searched and not found:** any CRBasic `.CR8` or `.dld` file · any LoggerNet or Campbell installation directory · any DEM or elevation raster · any document naming a sensor model · any calibration record · any station identifier in a national or international registry

**Session log:** `SecondBrain/Sessions/2026-09-09-estacion-meteorologica-timeline-y-metadatos-wmo.md`
