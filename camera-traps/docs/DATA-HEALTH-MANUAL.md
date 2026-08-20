# Camera-Trap Data Health

### An end-to-end manual: from the field to the figure

**Programme:** Reserva Natural Bosque Pehuén · Fundación Mar Adentro
**Version:** 1.0 · 2026-08-20
**Scope:** one camera-trap campaign, from the decision to install a camera to the number printed in a report.

---

## Part 0 — How to use this manual

### 0.1 What this is

This is a **protocol with its reasoning attached**. Every rule in it exists because
skipping it destroys a specific measurement, and each rule is stated together with the
measurement it protects. It is not a tutorial for the software — the repository README is
the command-by-command guide, and this manual points at it rather than repeating it.

Read it in order the first time. After that, use Part 8, which is the whole manual
compressed into checklists.

### 0.2 The repeating unit

Every rule below is stated in four parts. They are always in this order:

> **Rule** — the invariant to protect.
> **Break point** — what becomes impossible if it is not protected. This is the part that
> matters: it converts "we should do this" into "if we don't, we lose X".
> **Recovery** — whether the damage can be undone afterwards, by what, and at what cost.
> *Observed* — where relevant, one line of evidence that the failure is real and not
> hypothetical.

**Break point** is the column to argue with. If a rule's break point is vague, the rule is
probably ceremony and should be dropped. Every rule in this manual has a concrete one.

### 0.3 The single principle everything follows from

**A data error does not stay where it was made.**

It travels downstream and arrives at the far end looking like a *result*. A camera at the
wrong coordinates does not announce itself as a typing mistake; it announces itself as a
species occurring somewhere it does not occur. A clock reading eight years wrong does not
announce itself as a hardware fault; it announces itself as an absence of activity. By the
time an error is visible as a strange finding, it has usually been copied into three
projects and two documents.

Three consequences, and they shape every design decision in this manual:

1. **Refuse early.** A defect is cheap at the boundary where the data enters and expensive
   everywhere after. Validation happens once, at the edge, and the interior is then trusted.
2. **Refuse loudly, and never guess quietly.** A pipeline that silently drops a bad row
   produces a smaller dataset with no error message — the worst possible outcome, because
   it looks like success. Refusals are recorded as refusals.
3. **One canonical record, and every consumer reads it.** Each time a downstream project
   re-derives something the producer already decided, that derivation becomes a second
   place a correction has to reach — and it will not reach it.

Point 3 is the general form of nearly every incident in this manual, and it is worth
stating as a test you can apply to any proposed change: *if this rule is fixed later, how
many files have to change?* If the answer is more than one, the knowledge is in the wrong
place.

### 0.4 Recoverable versus permanent

Not all mistakes are equal, and the difference is not how serious they look. Some cost a
week of re-processing; a few cost the measurement forever. Before reading on, it is worth
knowing which is which, because the permanent ones are all in Parts 1 and 2 — in the
field and on the way to storage — and they are all cheap to prevent.

**Permanent — no later process can recover these:**

| Lost thing | Why it cannot be recovered |
|---|---|
| The camera's aim and detection distance at a visit | Redefined the moment the camera is touched again |
| A clock reading, at a known wall-clock time | The clock has moved on; nothing else witnessed it |
| The date a camera stopped working | If nobody looked, the last frame is a lower bound and nothing more |
| Whether a camera was moved between visits | Two locations are pooled and cannot be separated afterwards |
| The camera-created folder structure, once flattened by hand with no log | Capture order is gone, and with it the only independent check on the clock |

**Recoverable, at a cost:**

| Problem | Recovered by |
|---|---|
| Wrong or missing station coordinates | The station registry, if one row is authoritative |
| Clock offset, constant | One anchor pairing a real clock to the camera clock |
| Clock offset, per segment | One anchor *inside each segment* |
| Capture order after a flatten | The move log, or the folder manifest sidecar |
| A wrong species label | Re-review of the frame, which still exists |
| A wrong number in a report | Re-run the chain — provided the inputs are still trustworthy |

The asymmetry is the argument for Parts 1 and 2. The engineering in Part 4 can repair a
great deal, but it can only repair things that were *witnessed*. Nothing recovers an
observation nobody made.

### 0.5 Glossary

| Term | Meaning here |
|---|---|
| **Station** | A physical monitoring site. Identity is permanent: `CT01`–`CT27`. |
| **Camera unit** | The physical device. Moves between stations; has its own ID. |
| **Deployment** | One station over one campaign — the unit that has an operating period. |
| **Campaign** | One cycle of the whole array, from installation/servicing to card retrieval. Named for the season it is **retrieved** in. |
| **Visit** | One physical trip to one station. Closes one campaign and opens the next. |
| **Segment** | A stretch of frames within one deployment over which the clock behaved consistently. A camera that resets its clock three times has four segments. |
| **Anchor** | A *pair* of readings — the true time and what the camera's clock said at that same instant. |
| **Sweep** | The human pass in which every image in a campaign is assigned a category. |
| **Episode** | An independent detection event: one animal presence, not one photograph. |
| **Effort** | Trap-nights. The denominator of any rate. |
| **Still / frame** | One photograph. Video is handled separately and excluded from the labelled export. |

---
## Part 1 — Decisions made once, before any fieldwork

These are conventions for the whole programme. They are in this manual, first, because
changing any of them later invalidates comparison between campaigns — and because two of
them have already cost months.

### 1.1 Station identity

> **Rule.** One canonical identifier per physical site, fixed forever: `CT01`–`CT27`.
> Zero-padded so it sorts. No prefix variants, no underscores, no suffixes.
>
> **Break point.** Every campaign is joined to every other, and to the coordinates, by
> this string. If the spelling drifts between campaigns, the joins silently return fewer
> rows — never an error, just a smaller dataset.
>
> **Recovery.** A translation table of historical spellings, held as *data* rather than
> as code. Cheap but permanent debt: it must be carried forever and read on every load.
>
> *Observed: four historical spellings of the same 27 sites (`CT01`, `TC10_M3.2`, `CT_18`,
> `100EK113`) across four campaigns, requiring an ~80-row alias table that will never be
> deleted.*

**The monitoring-grid code must not be part of the station ID.** This looks like harmless
extra information and is not.

> **Rule.** The grid cell is a property of a *place*, recorded in the station registry. It
> never appears in the identifier.
>
> **Break point.** A grid cell can contain more than one camera. Fusing the grid into the
> identity makes the mapping many-to-one, so the identifier stops identifying — and there
> is no way to split it back apart from the string alone.
>
> *Observed: grid `M15.2` holds cameras 11 and 18; `M16.2` holds 13 and 19; `M17.2` holds
> 12 and 20. Identifiers of the form `TC11_M15.2` and `TC18_M15.2` therefore encode a
> grouping inside an identity.*

### 1.2 The station is not the camera unit

> **Rule.** The physical device gets its own identifier, distinct in form from the station
> identifier — a prefix such as `CAM-` is enough.
>
> **Break point.** Camera units are swapped between sites when one fails. If unit numbers
> and station numbers share a numbering space, a field note saying "18" is ambiguous, and
> a swap becomes indistinguishable from a relocation.
>
> **Recovery.** Only from a written record of which unit was at which station. Absent
> that, the two readings cannot be separated afterwards.
>
> *Observed: at one servicing round, station CT23 received unit 18 while station CT18
> received unit 28. Both readings existed in one spreadsheet on one day.*

### 1.3 One station registry, and only one

> **Rule.** Exactly one file is authoritative for standing site facts — coordinates,
> elevation, grid cell, mounting height, bearing. Every other copy is *generated* from it.
> A test asserts that all copies agree, on station count and on coordinates to five
> decimal places.
>
> **Break point.** With several hand-maintained copies, they diverge — and the divergence
> is invisible, because each file is internally consistent. A station present in one copy
> and absent from another ingests with no coordinates at all and drops out of every
> spatial product without any error being raised.
>
> **Recovery.** Straightforward, once one file is declared the owner. The difficulty is
> never technical; it is deciding which copy is right.
>
> *Observed, and still live at the time of writing: three registries disagree. One holds
> 26 stations and is missing CT27 entirely, while two hold 27. The 26-station file is the
> one documented in code as "the single source of truth", so CT27's 344 images ingest
> without coordinates.*

**Standing facts are never re-entered on a visit form.** If a value is a property of the
place rather than of the trip, asking for it again on every visit guarantees that two
answers eventually disagree, with no way to tell which is right. The form should *display*
it, read-only, and never collect it.

### 1.4 Coordinates

> **Rule.** One format, decimal degrees, signed. Bounds checked against **the reserve**,
> not against the country. Sign and format validated where the value is entered.
>
> **Break point.** A coordinate error is not detectable by inspection — a wrong coordinate
> is still a perfectly plausible number. It propagates into every map, every spatial
> analysis, and every downstream platform, and it comes back looking like a biological
> result: a species apparently occurring where it does not.
>
> **Recovery.** Only by re-measurement in the field, or from a GPS track or waypoint file
> if one survives. Expensive, and often impossible for a past campaign.
>
> *Observed: one station's coordinates were wrong for four months; the error reached a
> public platform and returned as a 19 km displacement. Separately, a field file recorded
> `39.45183 / 71.72707` unsigned — a value that is in China, and that a
> country-wide bounds check would have accepted for Chile only by luck.*

Two details that are load-bearing rather than fussy:

- **Bounds must be tight.** With a Chile-wide box, both readings of an ambiguous
  coordinate are plausible and no automatic test can decide between them. With reserve
  bounds, one of them is impossible.
- **Degrees-minutes-seconds must be caught by a rule, not by eye.** A value typed in DMS
  and read as decimal produces a plausible number a few kilometres away.

### 1.5 Campaign identity

> **Rule.** A campaign is named for the season in which its cards are **retrieved**, and it
> spans installation-or-servicing through to the next retrieval. A re-reading of cards
> that were already read is **not** a campaign, however long afterwards it happens and
> however much new labelling it contains.
>
> **Break point.** If a second review pass is treated as a campaign, the same photographs
> enter the analysis twice under two names. The sample size doubles, "new" detections
> appear at stations that were never revisited, and the dataset looks richer than it is.
> Cross-campaign de-duplication cannot fix it, because de-duplication assumes the two
> campaigns are genuinely different events and resolves conflicts by *recency* — so the
> later *review* silently outranks the correct campaign.
>
> **Recovery.** Full, but only once the field record is consulted: the record of physical
> visits is what settles whether two datasets are two events or one event read twice.
>
> *Observed: a review pass held 792 reviewed rows and, while it was ranked as a later
> campaign, its April labels silently replaced the campaign's own August re-review —
> returning 169 rows in place of 744, and reverting adjudicated species. Of its 186
> unique-looking rows, all 186 turned out to exist in the campaign's own export.*

**Station counts differing between campaigns is not necessarily an error.** At Bosque
Pehuén the array was built up over time: 21 stations existed at the first retrieval, 26 at
the second, 27 at the third. That is deployment history, and forcing the counts to agree
would fabricate stations. What must stay constant across campaigns is the *rule* — every
still in the export becomes a row — never the station count itself.

---

## Part 2 — In the field

Everything in this part is cheap to do and impossible to reconstruct later. It is the part
of the manual with the highest ratio of value to effort, and the part most likely to be
skipped because its cost lands months after the omission.

### 2.1 The visit record

> **Rule.** One row per station per visit, filled at the station, in one file that
> accumulates forever. Not one file per trip.
>
> **Break point.** Multiple files mean a copying step, and a copying step means the
> realistic outcome is a colleague duplicating a sheet by hand and diverging from the
> template. Removing the copy step beats policing it.
>
> **Recovery.** Rows can be reconciled from separate files if the headers match — which is
> the reason to keep the headers stable even if someone does duplicate the sheet.

A visit is a **physical event**, not a property of a campaign. At Bosque Pehuén every
servicing visit swaps the card, so one visit *closes* one campaign and *opens* the next.
Recording it that way means the closing campaign never has to be typed — it is derived
from the previous visit to the same station, and therefore cannot contradict the record.

### 2.2 Record readings, never conclusions

This is the most important rule in Part 2, and the least intuitive.

> **Rule.** The form asks for **two clock readings** — the true time (from a reliable
> source) and what the camera's screen says at that same moment. It offers no field for a
> diagnosis, no "clock state", no "offset in hours".
>
> **Break point.** A conclusion cannot be un-drawn. Asked for a verdict, a technician
> supplies one honestly, and the two raw readings behind it are gone — so a later analyst
> can neither check it nor use it for anything the technician did not anticipate. Worse,
> the verdict cannot be produced honestly in the first place: the phone the camera is
> compared against **adjusts itself** for daylight saving, so "the clock has a fixed
> offset and civil time moved" and "the clock reset" look identical at the tree. Only the
> two raw readings separate them.
>
> **Recovery.** None. If the readings were not written down, the state of that clock at
> that moment is unknowable.
>
> *Observed: asked for a judgement, a field team recorded `shifted, −1.0` for an entire
> campaign. The column that would have settled the question — the camera's own reading —
> was left empty on all 26 rows.*

The general form of this rule applies well beyond clocks: **collect observations, derive
conclusions.** Any form field that asks for an interpretation destroys the observation it
was interpreting.

### 2.3 The visit time is not optional

> **Rule.** Every visit records the time to the minute, not only the date.
>
> **Break point.** A date-only visit can only ever support an **approximate** anchor. The
> date can be recovered; the time of day cannot. For any camera whose clock failed, that
> means the date of each detection is recoverable but its hour is not — and every activity
> pattern, nocturnality measure and species-overlap estimate for that camera is lost
> permanently, while the presence and count data survive.
>
> **Recovery.** Only from a photograph of the technician at the station, if one exists
> (§2.4). That is the *only* other witness of the wall clock.
>
> *Observed: 27 of 27 opening visits in one campaign recorded no time, so every anchor
> derived from them decayed to date-only.*

### 2.4 Photograph the technician, at every visit

> **Rule.** At every visit, trigger the camera deliberately so that it photographs the
> person servicing it — and note the wall-clock time when you do.
>
> **Break point.** A frame showing a person, at a plausible working hour, is the only
> **witness** that ties a wall clock to a camera clock. Without one, a failed clock can at
> best be *bracketed* by the deployment window (the date is narrowed to a range), never
> repaired to an instant.
>
> **Recovery.** Partial, and only by luck: the camera may have photographed the technician
> anyway. This is worth checking before declaring a camera unrepairable, and it works in
> both directions — see the boundary-visit rule in §4E.6.
>
> *Observed: two such frames, found afterwards in a later campaign's images, recovered 33
> animal records that had been written off. Both were on the newly installed card, not the
> retrieved one.*

Two qualifications, because they are what make the evidence trustworthy:

- **A frame is a witness only if it shows a person AND sits at a plausible working hour.**
  A frame at 03:37 with nobody in it is a wildlife trigger. Treating it as an install
  photograph applies a correction to a clock that was never wrong.
- **The first frame on a fresh card is not automatically the install.** It is where to
  *look*, not what happened. See §4E.2 — this distinction has already prevented one
  incorrect repair.

### 2.5 Trigger a photo even when nothing seems wrong

> **Rule.** The trigger-and-note-the-time step happens at every visit, unconditionally —
> not only when a clock looks suspicious.
>
> **Break point.** Clock faults are discovered months later, during ingest, and the visit
> cannot be repeated. A camera that looked fine in the field is exactly the case where no
> anchor gets recorded and none can be reconstructed.
>
> **Recovery.** None; this is the insurance itself.

The cost is under a minute per station. The benefit is that no clock fault is ever
unrepairable for want of evidence.

### 2.6 Whether the camera was working — and whether it had stopped

> **Rule.** Record explicitly whether the camera was functioning when you arrived, and any
> evidence about when it stopped (dead batteries, full card, physical damage, moisture).
>
> **Break point.** This one is about the **denominator**, which is why it is easy to
> underrate. If a camera's death date is unknown, its operating period is unknown, so its
> trap-nights are unknown — and it must leave the effort denominator entirely, not only
> the detection counts. A station silently kept in the denominator with an
> unknown-but-shorter operating period biases every rate at every station downward.
>
> **Recovery.** Only weakly. The last frame gives a lower bound on the death date, but a
> camera can sit dead for months, so the bound may be far off. This is why the field
> observation matters more than the image record.
>
> *Observed: one camera produced 10 real frames and then four clock resets. Its retrieval
> date is known from 26 sibling cameras; what is not known — and what no image can supply
> — is the date it died.*

### 2.7 Bearing and detection distance

> **Rule.** Record the direction the camera faces and the distance at which it reliably
> detects, at every install and every re-aim.
>
> **Break point.** Together these define the **effective sampled area**. Without them, no
> density estimate, no detection-zone correction, and no comparison of detection rates
> between stations with different sight-lines is ever possible — for that campaign or any
> past one.
>
> **Recovery.** None. The moment the camera is touched again, the geometry it had is gone.
> This is the most consequential permanent loss in the whole manual, precisely because
> nothing downstream complains about its absence.

### 2.8 Moved, reinstalled, or re-aimed

> **Rule.** Any change of position, orientation or mounting is recorded as such, with new
> coordinates if the position changed at all.
>
> **Break point.** An unrecorded move pools two distinct locations into one station. Every
> spatial result then refers to a place that does not exist, and the two periods cannot be
> separated afterwards because nothing in the data marks the boundary.
>
> **Recovery.** Only if the images happen to show a changed scene, which is a judgement
> call and not a measurement.

### 2.9 Clock adjustments

> **Rule.** If the clock is adjusted, record both readings before the change and the new
> setting after it. Better still: **do not adjust it** — see Part 5, which explains why an
> unadjusted clock is more recoverable than a corrected one.
>
> **Break point.** An unrecorded adjustment converts a constant, exactly removable offset
> into a piecewise one that steps at an unknown date. The first is a solved problem; the
> second is not solvable from the data alone.
>
> **Recovery.** Only from the visit record. Nothing in the images marks the moment an hour
> was added or removed, because a one-hour change breaks no internal consistency check.

### 2.10 Field checklist

At each station, before leaving:

- [ ] Visit **date and time** written, to the minute
- [ ] **Camera clock reading** written, as the camera shows it — no interpretation
- [ ] Camera **triggered deliberately**, with the technician in frame, at that noted time
- [ ] **Working / not working**, plus any evidence of when it stopped
- [ ] **Bearing** and **detection distance**, if installed, moved or re-aimed
- [ ] **Moved / reinstalled** noted, with new coordinates if the position changed at all
- [ ] **Camera unit ID** noted, distinguishable from the station ID
- [ ] Card and battery change noted
- [ ] Nothing on the form is a conclusion — every entry is something you *observed*

---
## Part 3 — From the card to storage

The transfer from SD card to the NAS looks like a file-copying chore. It is in fact the
stage at which the most important piece of evidence in the whole dataset is either
preserved or destroyed: **capture order**.

### 3.1 Why capture order matters this much

A camera writes its images in the order it takes them. That ordering is recorded in two
places — the filename counter, and the folder the camera created to hold them.

Capture order is the **only independent check on the clock**. If the timestamps are
increasing and the capture order is increasing, they agree; if a timestamp goes backwards
while the capture order goes forwards, the clock reset, and the exact frame where it
happened is identifiable. Destroy the ordering and a clock fault becomes undetectable —
not harder to detect, *undetectable* — because there is nothing left to contradict the
timestamps.

Everything in this part follows from that.

### 3.2 The registration sheet

> **Rule.** Every returning card is logged: station, campaign, dates covered, camera unit,
> file count, and who transferred it.
>
> **Break point.** The file count is the only way to detect a partial transfer later. A
> transfer that silently drops files produces a smaller dataset with no error, and the
> gap looks like a period of no animal activity.
>
> **Recovery.** From the card, if it has not been reused. Cards are usually reused.

### 3.3 One folder per station, canonically named, at the top level

> **Rule.** Each station's images sit in exactly one folder, named with the canonical
> station identifier, at the top level of the campaign. No station folder is ever inside
> another.
>
> **Break point.** A station folder nested inside another station's folder is attributed
> to the **wrong camera at the wrong coordinates** — an entire station's images appear as
> if they came from somewhere else. This is the single most dangerous structural error in
> the whole chain, because **it passes every count-based check**: nothing is lost, nothing
> is duplicated, the totals reconcile perfectly. Only attribution is wrong, and nothing
> counts attribution.
>
> **Recovery.** Full, if caught. Undetectable by inspection if not.
>
> *Observed: 2,460 files — a whole station — nested inside another station's folder. The
> two cameras used different filename schemes, so there were zero filename collisions and
> the transfer reported `moved=2460 renamed=0 lost=0`. Every existing check passed.*

Because folder names are typed by hand in the field, the convention is enforced **at the
folder**, at transfer time, rather than corrected in software afterwards. Correcting it
downstream means every consumer must know about the correction; enforcing it here means
nobody downstream needs to know anything.

### 3.4 Never flatten, rename or reorganise the stored copy

> **Rule.** The stored tree keeps the structure the camera created. Reorganising for
> convenience — merging folders, sorting into subfolders by date, renaming for tidiness —
> is prohibited.
>
> **Break point.** The camera's folder structure *is* the record of capture order, because
> the filename counter **restarts inside each folder**. Merge two folders and many files
> called `xxxx0001` land in one directory, so the counter no longer orders anything. Once
> that is done by hand with no record, clock-fault detection is gone for that station.
>
> **Recovery.** Only from a **log of the reorganisation**. Flattening consumes the tree,
> but it does not consume *the record of* the tree — which is the difference between a
> temporary and a permanent loss.
>
> *Observed: a campaign's ordering evidence was declared "lost for good" and was not. The
> transfer log from three months earlier had recorded all 5,748 moves, and three of the
> five affected stations were fully re-ordered from it.*
>
> *Observed, the other way: a hand-made folder of 723 loose frames was recorded as though
> it were a camera folder. Sorting by it asserted that January preceded October — a
> phantom clock reset across 2,097 frames.*

The corollary is a rule about evidence: **only a camera-created folder is ordering
evidence.** A folder made by a person asserts an order nobody observed. Software that
consumes the structure must be able to tell the two apart by *shape*, and refuse to treat
a hand-made grouping as evidence.

### 3.5 Never deduplicate by filename

> **Rule.** Two files with the same name, in the same station, are both kept. They are
> distinguished by which camera folder they came from.
>
> **Break point.** Two files with the same name are exactly what a camera with a reset
> clock, or a wrapped counter, produces. Deleting one as a "duplicate" destroys the
> evidence of the fault it superficially resembles — and reduces the image count, so the
> loss is invisible against a smaller expected total.
>
> **Recovery.** From the card or the original tree, if either survives.

The related trap: the filename counter **wraps at 999** in each folder. A camera with more
than 999 images per folder legitimately re-emits the same counter. Read naively, every wrap
looks like a reset.

> *Observed: one station produced 987 apparent clock resets, all of them counter wraps.*

### 3.6 Conservation, checked and not assumed

> **Rule.** After any transfer or restructuring, file counts in must equal counts out,
> per station. The check aborts the operation, it does not warn.
>
> **Break point.** Silent loss. A transfer that drops files leaves a dataset that is
> internally consistent and simply smaller — the failure mode that no downstream analysis
> can detect.
>
> **Recovery.** From the source, if it still exists.

Conservation is necessary and **not sufficient**: it catches loss and duplication, and it
is blind to misattribution (§3.3). Both checks are needed, and they check different things.

### 3.7 Video

> **Rule.** Video is stored, and excluded from the labelling export.
>
> **Break point in both directions.** Video left *in* the labelling export inflates row
> counts and, if it reaches a rate calculation, corrupts the denominator — at some stations
> video is the majority of files. Video *deleted* to simplify the export throws away real
> observations that cannot be recovered.
>
> **Recovery.** Full, provided the files were kept. The export is regenerable; the video is
> not.
>
> *Observed: a file count of 19,522 was treated as a target for the labelled export for
> some time. It resolved exactly as 16,904 stills plus 1,663 `.mp4` plus 955 `.mov` —
> nothing had ever been missing.*

### 3.8 Establish the sync direction as a fact

> **Rule.** Before deleting anything locally, establish from evidence — logs, not setting
> names — which way the synchronisation runs.
>
> **Break point.** Two opposite mistakes. If the mirror is one-way *download*, then local
> deletions are safe but any local restructuring is re-downloaded forever, and the stored
> copy is **not a backup of local work**. If it is two-way, a local deletion propagates and
> destroys the original.
>
> **Recovery.** Depends entirely on which mistake was made — which is the argument for
> checking first.
>
> *Observed: the setting was an integer with no documented meaning. The daemon log stated
> the semantics in words, and that is what licensed a deletion of 36 MB of superseded data.
> Separately, five frames were found to exist only in the local copy — so the stored copy
> was not a complete backup either.*

### 3.9 Storage checklist

- [ ] Card logged: station, campaign, dates, unit, **file count**
- [ ] One folder per station, **canonical name**, at the top level
- [ ] **No station folder inside another** — checked explicitly, not assumed
- [ ] Camera folder structure **untouched**
- [ ] No renaming, no merging, no sorting into new subfolders
- [ ] No filename de-duplication
- [ ] File counts reconciled per station
- [ ] Video present and separate
- [ ] Sync direction known, from logs

---
## Part 4 — Preparation and ingest

This is the longest part of the manual, and its purpose is a single sentence:

> **No data defect may reach a downstream project silently.**

Not "no defects" — that is not achievable. The goal is that every defect either stops the
pipeline or is recorded on the row it affects, so that a downstream analysis can decide
what to do about it. A defect that arrives without a label is the only kind that produces a
wrong published number.

---

### 4A — The chain

**Order of operations.** Each stage owns one decision and refuses rather than repairs.

| # | Stage | Owns |
|---|---|---|
| 1 | **Transfer and flatten** | Structural integrity: conservation, capture order, attribution |
| 2 | **Automatic detection** | Where in each image something is, and of what broad kind |
| 3 | **Human sweep** | A category for **every** image |
| 4 | **Species review** | An identification, where one is possible |
| 5 | **Review resolution** | Reconciling the sweep with the reviewer's own written corrections |
| 6 | **Order and clock diagnosis** | Segments, and whether each one's timestamps can be trusted |
| 7 | **Anchoring and repair** | Applying field evidence to segments that need it |
| 8 | **Canonical write** | One table, one grammar, with validity flags |
| 9 | **Contract publication** | A verifiable statement of what was written |

Two properties of this ordering are deliberate:

- **Diagnosis (6) happens before repair (7), and repair never happens without evidence.**
  A pipeline that guesses an offset produces plausible timestamps that are wrong, which is
  strictly worse than refusing.
- **Each stage refuses rather than repairing what an earlier stage should have done.** If
  stage 3 was skipped, stage 6 does not compensate; it stops. The alternative is a chain
  where every stage partially covers for the previous one, and no stage can be trusted
  alone.

**The two deliberate exceptions**, both narrow, both documented where they occur: a
day-boundary filename artefact is forgiven rather than treated as a fault (§4D.3), and a
single-segment camera claims all of its rows including unparseable ones (§4D.4). Each is
justified by an argument that it *cannot* admit bad data, not merely that it is convenient.

---

### 4B — Preserving and reconstructing capture order

#### 4B.1 The sidecar manifest

Labelling software generally wants one flat folder per station. That requirement conflicts
directly with §3.4.

> **Rule.** Flatten for the labelling tool, and write a **sidecar manifest** recording
> which camera folder each file came from. Nothing is renamed that was not already being
> renamed.
>
> **Break point.** Without the sidecar, the flatten is the moment capture order dies. With
> it, order survives as data rather than as directory structure.
>
> **Recovery.** From the flatten log, if the manifest was not written at the time.

> **Rule.** A sidecar, **not** a renaming scheme. Do not prefix the folder name onto the
> filenames.
>
> **Break point.** The filename is the join key for every label already assigned. Renaming
> orphans all prior review work — potentially thousands of adjudicated identifications —
> and there is no automatic way to re-associate them.

#### 4B.2 Manifest coverage must be total, or it is refused

> **Rule.** Within a described deployment, the manifest covers **every** frame or it is not
> used at all.
>
> **Break point.** Partial coverage is worse than none. Frames with a folder sort against
> frames without one, which asserts an ordering between them that no evidence supports.
> A wrong order is a fabricated clock fault, or a concealed one.

#### 4B.3 Failing to establish order does not condemn a camera

This is the most important corollary in Part 4B, and getting it wrong would have discarded
thousands of good frames.

> **Rule.** If capture order cannot be established, the camera is **not** condemned. Order
> is needed to *attribute* a fault to a particular frame — never to rule one out.
>
> **Break point.** A camera whose every frame falls inside its known deployment window, and
> whose filenames agree with their own timestamps, demonstrably never reset — whether or
> not we can put its frames in order. Condemning it for a missing manifest discards clean
> data for a fault that provably did not occur.
>
> **Recovery.** Not needed; the data was never bad. What *is* missing is a check, and §7.2
> supplies one.
>
> *Observed: five stations with clean clocks and over 6,000 frames between them would have
> been rejected by a rule that required order unconditionally. Three station-campaigns
> currently sit in this category — 1,735, 999 and 873 frames, with 802, 166 and 88 colliding
> counters respectively, all with clean clocks.*

---

### 4C — Labelling, and proving it happened

#### 4C.1 Automatic detection and human review do different jobs

Automatic detection (a detector model) answers *is there an animal, a person or a vehicle
in this frame, and where*. Human review answers *what species*. Neither substitutes for the
other, and the distinction matters for a practical reason: the detector will find
technician frames that the human sweep never recorded, which makes it the fastest route to
clock anchors.

> *Observed: a detector found 595 person frames and 28 vehicle frames in a campaign whose
> human sweep had recorded none. That turned anchor-finding from a search into a
> confirmation task — 17 stations had an install-side candidate.*

#### 4C.2 The sweep must cover every image

> **Rule.** The human pass assigns a category to **every** image in the campaign, not only
> to the interesting ones. The categories are a fixed, controlled vocabulary.
>
> **Break point.** A partial sweep is **indistinguishable from a complete one by
> inspection**. If the labelling template uses one value for both "empty" and "not yet
> looked at", then a file containing only `{animal, unassigned}` *looks* labelled while
> nothing was actually decided — and every row that was never examined is silently treated
> as a confirmed empty frame.
>
> **Recovery.** Full, but it means doing the sweep. The cost is proportional to campaign
> size, which is why the gate below exists: catching it at export costs a re-export;
> catching it after analysis costs the analysis.

**Controlled vocabulary.** One value per category, fixed across campaigns, with a distinct
value for *unassigned* that is never confused with *empty*:
`animal` · `human` · `vehicle` · `blank` · `unknown` · `unclassified`.

The one that matters most is that `blank` (looked at, nothing there) and `unclassified`
(not looked at) are different values. If the tool cannot separate them, the sweep cannot be
verified at all.

#### 4C.3 Proof of sweep

> **Rule.** The export must contain at least one category that **only a human pass
> produces**. In practice: `human` or `vehicle`.
>
> **Break point.** Presence of categories cannot itself be the test, because an unswept
> file has categories. What can be tested is presence of a category the detector does not
> assign and the reviewer only records deliberately. Under the field protocol in Part 2,
> a person frame should *always* exist in a campaign — so its absence is itself a finding
> worth stopping for.
>
> **Recovery.** Re-export, or a signed override (§4C.4).

#### 4C.4 What may be waived, and what may not

> **Rule.** An override may excuse an **exception to a rule**. It may never excuse the
> **absence of the work the rule checks for**.
>
> In practice: a genuinely person-free campaign can be signed off. A campaign where nothing
> was ever assigned cannot — no signature turns unswept rows into a sweep.
>
> **Break point.** An overridable "did you do the work" check is not a check. It becomes a
> button that gets pressed when the work is inconvenient, and the pipeline's guarantees
> quietly become optional.

> **Rule.** The override is a **file** — carrying who verified it, the date, and the reason
> — refused if any of the three is missing. Not a command-line flag.
>
> **Break point.** A flag leaves no trace: six months later nobody can tell whether a
> campaign was signed off deliberately or bypassed in a hurry. A file carries a name and a
> date, and travels with the data.

#### 4C.5 Two files must never both claim to be the reviewed truth

> **Rule.** At any moment, exactly one file is the reviewed record for a campaign. If a
> corrected version is produced, the original is immutable and the corrected one is the
> only input downstream.
>
> **Break point.** With two candidates, each consumer picks one, and **which one it picked
> is invisible in the output**. Two projects then report different numbers from "the same
> data", and the difference is a filename.
>
> **Recovery.** Full, and the fix is deletion rather than documentation: a documented
> ambiguity is still an ambiguity.

#### 4C.6 Resolving the reviewer's own corrections

Reviewers write free-text notes. Those notes routinely *contradict* the coarse category
assigned in the sweep — which is correct behaviour, because the species review is the
later and closer look.

> **Rule.** Resolve the disagreement explicitly, with a stated precedence, and record on
> each row **where its verdict came from**.
>
> The precedence that applies here:
> - a **named species** outranks the coarse category (the reviewer identified something);
> - a **negating comment** outranks a coarse `animal` (the reviewer looked and there is no
>   animal — the coarse category is the false positive being corrected);
> - when several subjects are in frame: any identified animal, then vehicle, then human.
>
> **Break point.** Unresolved, the rows stay typed as animals that the reviewer had already
> said held none. Whether that reaches a result depends on accidental features of each
> downstream filter — which is to say it is not controlled.
>
> **Recovery.** Full, by re-resolving from the comments, provided the comments were kept.
>
> *Observed: 815 rows across three campaigns carried a written negation and remained typed
> `animal`. They did not reach the published figures — but only because a downstream filter
> also required a non-empty species name, and none of the 815 had one. A loaded gun rather
> than a wound; the resolution is now explicit and the ambiguous file is deleted.*

Two design notes that generalise:

- **Fail closed on unknown comments.** An unrecognised note stops ingest rather than being
  ignored. Ignoring it means the pipeline's behaviour depends on free text nobody has read.
- **Record the provenance of every verdict.** Rows resolved from a review, from a comment,
  and from the sweep alone must be distinguishable, because they do not deserve equal
  confidence.

---
### 4D — Datetime errors: the full taxonomy

"The date is wrong" is not one problem. It is at least nine, they have different signatures,
they block different analyses, and they differ in whether they are recoverable at all. This
section is the reference for telling them apart.

#### 4D.1 What a camera clock actually is

A camera clock is a free-running counter with no external reference. It is set once by hand,
it drifts, and it resets to a factory epoch whenever it loses power without a battery
backup. It has **no time zone and no authority** — the number it stamps is a reading, not a
fact. Every rule below follows from taking that seriously.

Recovering the true time therefore requires either (a) a witness pairing the camera's
reading with a real clock, or (b) an internal structure in the data that pins the reading
without any field evidence. Both are used; (a) is stronger.

#### 4D.2 Segments are the unit of diagnosis

> **Rule.** A deployment is divided into **segments** — stretches over which the clock
> behaved consistently. Diagnosis, repair and validity are decided **per segment**, never
> per station.
>
> **Break point.** A per-station correction is wrong the moment a camera resets more than
> once. Applying one offset to a camera with four resets makes three of its five segments
> worse, and the result looks plausible because every timestamp is now in a believable
> range.
>
> **Recovery.** Full, by re-diagnosing. The danger is a repaired dataset that nobody
> re-examines because the dates look fine.
>
> *Observed: a camera treated as one reset was in fact five segments — 10, 32, 40, 3 and
> 227 frames — with four separate resets to a factory epoch.*

#### 4D.3 The error classes

Each entry: **signature** (how it looks in the data) · **detection** · **what it blocks** ·
**recoverability**.

**1. Reset to a factory epoch.**
*Signature:* timestamps jump backwards to a fixed implausible date, typically years before
the programme began, while the capture order continues forwards.
*Detection:* discontinuity between timestamp order and capture order; frames outside the
known deployment window.
*Blocks:* everything time-based for that segment; presence is unaffected.
*Recoverable:* yes, with one anchor inside the segment. Without one, the date is bracketed
by the deployment window at best.

**2. A forward jump.**
*Signature:* timestamps leap forward and continue plausibly.
*Detection:* the same discontinuity test. **Note the trap:** a threshold test of the form
"the year is implausibly old" cannot see this at all — a forward jump produces perfectly
modern dates.
*Blocks:* the same as (1), and it is more dangerous because nothing looks wrong.
*Recoverable:* yes with an anchor. Detectable only against capture order or the deployment
window — never by inspecting the dates.

**3. Repeated resets within one deployment.**
*Signature:* several discontinuities; segment count above two.
*Detection:* as above, applied segment-wise.
*Blocks:* per segment. A camera can have three good segments and two unrepairable ones, and
must be treated that way rather than accepted or rejected whole.
*Recoverable:* per segment, and **each segment needs its own anchor**.

**4. Corrupt date registers — the clock does not tick coherently.**
*Signature:* the date is not merely offset but internally inconsistent. Filenames encoded
with a date disagree with the timestamp on the same frame by arbitrary amounts, at
arbitrary hours; impossible values appear.
*Detection:* compare the filename's own encoded date against the timestamp it carries. A
mismatch far from midnight is the signature.
*Blocks:* everything time-based, permanently.
*Recoverable:* **no.** An anchor corrects an offset; it cannot correct a clock that does
not advance consistently. There is nothing to offset.
*Observed: one camera emits month `00` and month `16`. No anchor can repair it, and saying
so plainly is the correct outcome. Another disagreed by 14 hours across 61 frames, another
by 11 hours, a third by 3 hours.*

**5. The day-boundary artefact — which looks like (4) and is not.**
*Signature:* the filename's encoded date is one day ahead of the timestamp, and the frame
sits within seconds of midnight.
*Detection:* distance from midnight. This separates cleanly from (4): the benign cases sit
inside a minute; genuine corruption is hours away.
*Blocks:* **nothing.** The clock is correct and ticking; it crossed a day boundary while
writing the file, so the filename was built from the new date and the stamp from the old.
*Recoverable:* not applicable — there is nothing wrong.
> **Break point of getting this wrong.** A working camera is declared faulty and its
> entire season is discarded.
> *Observed: three frames at 23:59:28–29 — a maximum of 32 seconds from midnight, against
> 318 frames that agreed — caused a camera to be refused entirely, costing 321 images
> including 7 puma records. The camera was very nearly scrapped in the field on the
> strength of the false positive. A 120-second tolerance now forgives this class, and
> because the tolerance only ever forgives, a station with a single mismatch away from
> midnight is still refused.*

**6. A constant offset.**
*Signature:* every timestamp wrong by the same amount; internally perfectly consistent.
*Detection:* only against external evidence — an anchor, or the deployment window. There is
no internal signature whatsoever.
*Blocks:* nothing, once known. Time of day is exactly recoverable.
*Recoverable:* **fully.** This is the benign case, and Part 5 explains why we deliberately
keep clocks in this state rather than "fixing" them.

**7. A piecewise offset.**
*Signature:* an offset that changes at an unknown moment mid-deployment — the result of
someone adjusting the clock.
*Detection:* only from the visit record. A one-hour change breaks no coherence test, no
ordering test, and no window test.
*Blocks:* time-of-day analysis across the change point, unless the change date and size are
both recorded.
*Recoverable:* **only from the field record.** Not from the data.

**8. A systematic shift affecting a whole period.**
*Signature:* none in the data at all.
*Detection:* impossible internally. The only route is the field record plus knowledge of
external events, such as a civil time change.
*Blocks:* time-of-day analysis for the affected span, at the scale of the shift.
*Recoverable:* yes if the event and its dates are known — the correction is arithmetic.
*Observed: one campaign's cameras were set back one hour at a servicing visit while the
country's clocks had changed weeks earlier, producing roughly 40 days of frames one hour
away from local time — invisible to every consistency check in the pipeline.*

**9. Missing or undecodable stamps.**
*Signature:* an unparseable or absent timestamp.
*Detection:* trivial.
*Blocks:* everything time-based for those rows.
*Recoverable:* only by interpolation from neighbouring frames, which is a guess. Current
policy is to carry such frames as an explicit category rather than dropping or guessing —
a small, known and accepted limitation is preferable to an invisible one.

**A note on metadata recovery.** It is often suggested that the embedded image metadata
holds the true date even when the visible stamp is wrong. For the cameras in use here this
was tested and is false: all three embedded date fields are corrupted identically, the file
system timestamp is the same wrong value shifted by time zone, and the GPS block carries no
time component. Do not plan a recovery around it without testing it on your own hardware.

#### 4D.4 Two preconditions, both fail-closed

Before any clock verdict is issued, two things must hold. If either fails, the pipeline
refuses rather than guessing.

- **P1 — capture order is established.** From the folder manifest and the filename counter.
- **P2 — the segment's clock is internally coherent.** The filenames' own encoded dates
  agree with their timestamps, allowing for the midnight artefact in §4D.3(5).

And the corollary already stated in §4B.3, repeated because it is the one people get
backwards: **failing P1 does not condemn a camera.** P1 failing means a fault could not be
*located*; P2 passing plus a clean deployment window means no fault *occurred*.

One deliberate exception: **a single-segment camera claims all of its rows**, including
videos and frames with unparseable stamps. A camera that never reset has no split to
attribute a frame to, so there is no way to place a frame wrongly. On a multi-segment
camera the rule is strict containment and an unplaceable row is refused, never guessed.

#### 4D.5 Why preconditions and not a quality score

There is a persistent temptation to replace these binary gates with a score — "clock
quality 0.87" — and rank the data. Resist it.

> **Rule.** Admission decisions are made by **deterministic preconditions**. Heuristics are
> permitted only as *audit diagnostics*, which describe and never decide.
>
> **Break point.** A score built on an assumption nobody can verify will eventually admit
> bad data with a confident number attached, and the number makes it harder to question,
> not easier. A precondition that fails is a question someone has to answer; a score that
> comes out at 0.87 is a question nobody asks.
>
> *Observed: a proposed "slack" score compared the deployment window against the sum of
> segment durations, and rested on cameras rebooting promptly after power loss — which
> cannot be established. It was rejected as a criterion and kept as a diagnostic. The rule
> that replaced it is a sentence: **a segment is repairable if and only if it is coherent
> and contains at least one anchor.***

The same reasoning produced a second rule worth stating separately, because it applies to
every gate in this manual: **a rule should be derived from a stated premise, not enumerated
from the cases we have seen.** A gate that lists the three spellings we have encountered
will not catch the fourth. A gate that says *one deployment has one capture story — two
filename grammars each forming their own counter run means two cameras* catches cases nobody
anticipated, including a folder innocuously named `Camara 23`.

> *Observed: the enumerating version of this check was replaced by the premise-based version,
> which was then validated across 28,178 files in four campaigns with zero false positives
> before being switched on.*

---

### 4E — Anchors: what can be repaired, and with what

#### 4E.1 An anchor is a pair, and a visit is not an anchor

> **Rule.** An anchor records **two readings at one instant**: the true time, and what the
> camera's clock said. A visit record on its own is not an anchor.
>
> **Break point.** A visit says when somebody *arrived*, not what the clock *read*. Forcing
> a visit date onto a camera as though it were an anchor applies an offset to a clock that
> may have been perfectly correct.
>
> *Observed: one station's visit record says 2025-11-24 while its frames run from 2025-11-26
> across a single coherent segment. Treating the visit as an anchor would have applied a
> two-day offset to a clock that was never wrong.*

The corollary: **anchors are only proposed where a segment would otherwise be refused.** A
camera with a clean clock gets no anchor and needs none. Anchoring a healthy camera is not
harmless — it is a correction with no error to correct.

#### 4E.2 Witness evidence versus navigational evidence

This distinction does more work than any other in this section.

| | Witness | Navigational |
|---|---|---|
| **What it is** | A frame showing a person or vehicle at the station | A first-frame-of-folder, a segment edge, a counter reset |
| **What it establishes** | Somebody was there, at this camera time | Where to look |
| **May date a visit?** | Yes | **No** |

> **Break point.** Treating navigational evidence as a witness applies a correction derived
> from a frame the technician was never in.
>
> *Observed twice. A segment-edge frame was paired with a visit for a −5 day offset on ten
> frames whose clock was correct. And a first-frame-on-a-new-card at 03:37, with nobody in
> it, was proposed as an install anchor — it is a wildlife trigger on an already-deployed
> camera, and gives an upper bound on the install date, nothing more.*

Hence the two-part test in §2.4: a boundary frame is a witness only if it is **at a
plausible working hour** *and* **shows a person**.

#### 4E.3 Anchor classes

| Class | Examples | What it restores |
|---|---|---|
| **Exact** | install, mid-deployment visit, retrieval — each with a recorded time | Date **and** time of day |
| **Approximate** | date-only visit; last-real-frame proxy | Date only — time of day stays untrustworthy |
| **Unrepairable** | explicitly recorded refusal | Nothing; it records that a decision was made |

> **Rule.** Refusals are **written down**, as explicit rows, not omitted.
>
> **Break point.** A station nobody has examined and a station known to be unanchorable
> look identical downstream — and only one of them represents a decision anybody made. The
> first invites someone to waste a week; the second closes the question.

#### 4E.4 Tolerances are measured, not chosen

> **Rule.** Every tolerance is derived from the data it must tolerate, and the derivation is
> recorded next to the number.
>
> **Break point.** A guessed tolerance is either too tight (rejecting good data, and it will
> be loosened under pressure with no analysis) or too loose (admitting the fault it exists
> to catch).
>
> *In use here: exact anchors are matched within **1 hour**, since they are recorded to the
> minute. Visit-derived windows use **3 days**, chosen against the 20 stations whose
> coherence could be established from capture order alone — their largest excursion past a
> recorded visit date was **+1.67 days**. Three days is therefore above the observed
> maximum, and still around a thousand times tighter than the fault it must catch, since
> the worst clock failure in the dataset is eight years out. The midnight tolerance is 120
> seconds against a maximum observed artefact of 32–60 seconds, and two orders of magnitude
> tighter than the smallest genuine corruption at 3 hours.*

#### 4E.5 The deployment window is a bracket, not a band

> **Rule.** A frame before the opening visit or after the closing visit is impossible; a
> quiet stretch *inside* the window is evidence of nothing. Only the two edges are ever
> tested.
>
> **Break point.** Treating a gap inside the window as suspicious flags healthy cameras.
> Cameras legitimately go weeks without a trigger.
>
> *Observed: two stations went 35 and 41 days to their first trigger; one died 91 days
> before retrieval. All three are ordinary.*

#### 4E.6 The boundary-visit rule

A servicing visit is photographed **twice** — by the card coming out, and by the card going
in. The fresh card is often the better witness, because the retrieved card may be full,
dead, or clock-broken.

> **Rule.** When a campaign's retrieval lacks a witness, look at the **first frames of the
> next campaign** at that station. The technician who installed the new card is the same
> person who removed the old one, at the same moment.
>
> **Break point.** Without this, a retrieval with no witness on the retrieved card is
> written off — and with it every frame in the final segment.
>
> *Observed: two confirmed boundary frames, 33 minutes apart at neighbouring stations
> (walking distance, so each corroborates the other), recovered 33 animal records from the
> previous campaign. Trips corroborate across stations too: four stations share one date.*

Two limits worth stating so this is not over-applied: it only works at campaign boundaries,
not at segment boundaries *inside* a campaign — a mid-campaign segment edge is
navigational, not a visit. And it requires the visit to actually be photographed, which is
§2.4 again.

#### 4E.7 Strict containment

> **Rule.** An anchor repairs the segment it falls **inside**, and nothing else. An anchor
> falling inside no segment, or inside several overlapping ones, repairs nothing.
>
> **Break point.** Without containment, an anchor from one segment is applied to another —
> which is precisely the per-station correction error of §4D.2, re-entering through the
> anchor table.
>
> *Observed: this is what produces the honest verdict for a camera whose recorded install
> anchor falls inside none of its five segments. The anchor exists, and it repairs nothing.*

An escape hatch exists — an explicit segment index on the anchor row — for the case where
a human knows which segment an anchor belongs to and the automatic assignment cannot tell.

#### 4E.8 The recovery matrix

For each error class, what each kind of evidence recovers. **D** = date recovered,
**T** = time of day recovered, **—** = nothing recovered.

| Error class | No evidence | Deployment window only | Date-only visit | Exact anchor in segment | Exact anchor in *every* segment |
|---|---|---|---|---|---|
| 1. Reset to factory epoch | — | bracketed date | D | D + T | D + T |
| 2. Forward jump | — | bracketed date | D | D + T | D + T |
| 3. Repeated resets | — | bracketed date | D per anchored segment | D + T for that segment only | D + T throughout |
| 4. Corrupt date registers | — | — | — | — | **— (never)** |
| 5. Day-boundary artefact | *nothing to recover* | | | | |
| 6. Constant offset | undetectable but harmless | detectable | D | D + T | D + T |
| 7. Piecewise offset | — | — | partial | needs field record of the change | needs field record of the change |
| 8. Systematic shift | — | — | — | recoverable **only** from the field record + known event | as left |
| 9. Missing stamps | — | — | — | — | — |

Read three things off this table:

1. **Class 4 is unrecoverable no matter what evidence exists.** Anchors correct offsets;
   they cannot correct incoherence. This is worth knowing early, because effort spent
   hunting anchors for such a camera is wasted.
2. **Classes 7 and 8 are unrecoverable from the images at all.** They are only ever fixed
   by the field record, which is the whole argument for Part 2.
3. **A date-only visit costs you the T column everywhere.** One missing entry on a form,
   and the activity analysis for that station is gone. §2.3, quantified.

---
### 4F — The canonical table

#### 4F.1 One table, one grammar, written once

> **Rule.** Ingest writes **one** table per campaign, in one canonical grammar, and every
> consumer reads that. Validation happens here, at the boundary; the interior is then
> trusted and nothing re-validates.
>
> **Break point.** The alternative — a compatibility layer that tolerates every historical
> grammar forever — is decay with a nice name. Each new campaign adds a variant, every
> consumer has to know about all of them, and the set only grows.
>
> **Recovery.** Full, at the cost of a re-ingest.

Two properties make this work across separate projects:

- **A file, not a shared library.** Consumers in different repositories, on different
  machines, in different languages (Python and R here) need no shared code to read a
  Parquet file. A shared *reader* would require either a cross-repository dependency or a
  third copy of the logic.
- **Row set pinned to the gated export.** Every still in the campaign becomes a row —
  including those with no animal in them.

> **Break point of that second property.** If only rows with detections are written, a
> station that recorded no animals is **absent** from the table, and a station absent from
> the table is indistinguishable from one that was never deployed. That is harmless for a
> detection numerator and wrong for an effort denominator.
>
> *Observed: seven station-campaigns were missing from the canonical tables for exactly
> this reason — between 6 and 21 frames each, all real deployments with real trap-nights.*

#### 4F.2 Keys, not attributes

> **Rule.** The table carries the species **key** (the scientific name) and nothing else
> about the species. Spanish name, taxonomic group, invasive status, priority status — all
> joined at point of use from a separate catalogue.
>
> **Break point.** Baking attributes into the table freezes a copy of the catalogue into
> every output file. A correction to the catalogue then does not propagate without a full
> re-ingest, and the copies disagree in the meantime.

#### 4F.3 The three validity axes

This is the heart of the design. A single usable/unusable flag would be simpler and would
throw away recoverable data, because different faults block different questions.

| Axis | Level | FALSE means | Blocks | Does **not** block |
|---|---|---|---|---|
| `valid_date` | row | the date cannot be placed in time | seasonality, anything date-filtered, event counting | presence at a station |
| `valid_time_of_day` | row | the hour is untrustworthy | activity patterns, nocturnality, species overlap | presence, counts, seasonality (if the date is good) |
| `valid_effort` | **station** | the operating period is unknown | any rate, any occupancy model, any trap-night denominator | presence |

> **Rule.** Three independent axes, not one.
>
> **Break point.** A pure year error preserves the time of day **exactly**. Collapsing to a
> single flag throws away a complete activity record for a camera whose only fault is that
> its calendar is wrong. Conversely, a camera with a perfect calendar and an unknown death
> date has trustworthy dates and untrustworthy effort — one flag cannot express that.

**`valid_effort` is station-level, and that is deliberate.** If any segment's dates are
unknown, the camera's operating period is unknown, so its trap-nights are unknowable —
including for the segments whose own dates are fine. Such a station leaves the denominator
entirely.

> **Break point.** Excluding a station from the numerator but leaving it in the denominator
> is worse than excluding it from both: it adds trap-nights during which detection was
> impossible, biasing every rate downward at every station.

**The flags must be flags, not deletions.** Writing an empty value in place of a bad date
looks tidier and is worse: a consumer's natural "drop the missing dates" filter would then
remove those rows from **presence** analyses too, where the record is perfectly valid.
Presence at a station is spatially true regardless of what the clock said.

#### 4F.4 Cross-campaign duplicates

The same photograph can appear in two datasets — most often because one card was read twice.

> **Rule.** De-duplicate on a **natural key derived from the image** — station, filename and
> timestamp — with an explicit, ordered precedence deciding which copy wins.
>
> **Break point.** Without de-duplication, the same animal is counted twice. With
> de-duplication but no explicit precedence, *which* copy wins is an accident of ordering.
>
> **The trap, and it is a subtle one:** if precedence is "most recent", then a later
> **review pass** outranks the campaign it reviewed — see §1.5. Precedence must be defined
> over campaigns, and a review pass must not be in the list at all.
>
> *Observed: 396 overlapping records between a campaign and its own review pass, 31 of them
> conflicting. While the review pass was ranked as a later campaign it silently won all 396.*

#### 4F.5 A published contract, verified on load

> **Rule.** Ingest publishes a small, committed statement of what it wrote: per campaign,
> the row count, a hash of the table, and when it was written. Consumers verify against it
> and refuse to proceed when it disagrees. A missing or unreadable statement means refuse,
> not proceed.
>
> **Break point.** Without it, the canonical table can change shape or size and **every
> consumer keeps running**, silently, on different data. There is no error, no warning, and
> no way to tell from an output whether it was produced before or after the change.
>
> **Recovery.** Full, but only if somebody notices — which is exactly what cannot be relied
> on.
>
> *Observed: the canonical tables went from 3,359 rows to 35,807 in one rebuild — a
> deliberate and correct change — and not one consumer noticed or needed to be told. The
> same silence would have followed an accidental change. A contract nobody verifies is a
> comment.*

> **Rule.** **Publishing the contract is a separate act from ingesting.**
>
> **Break point.** If ingest re-published the statement itself, the check would always agree
> with whatever was just written, and could never catch an unintended rebuild. The
> verification would be structurally incapable of failing.

#### 4F.6 What the canonical table looks like in practice

For orientation, the current state of the three live campaigns:

| Campaign | Stations | Rows (stills) |
|---|---:|---:|
| otoño 2025 | 21 | 8,997 |
| primavera 2025 | 26 | 16,904 |
| otoño 2026 | 27 | 9,906 |
| **total** | **27 distinct** | **35,807** |

Of those rows, 3,359 carry a species review and 32,448 carry only a sweep category. By
resolved type: `blank` 31,090 · `animal` 2,522 · `human` 1,424 · `unknown` 521 ·
`vehicle` 250. **Zero rows are typed `animal` with an empty species name** — that invariant
is asserted by a test, because it is the signature of the review-resolution defect in
§4C.6.

---

### 4G — The boundary to every other project

#### 4G.1 A consumer reads the canonical table and nothing else

> **Rule.** Downstream projects read the canonical table. They do not read exports, review
> files, detector output, or each other's intermediates.
>
> **Break point.** Every re-derivation downstream is a second place a repair has to reach —
> **and it will not reach it.** This is the general form of nearly every incident in this
> manual, and the reason the rule is stated as an absolute rather than a preference.
>
> **Recovery.** Delete the duplicate derivation. Not "keep it in sync" — deletion is the
> only fix that stays fixed.
>
> *Observed, three times over.*
> - *A downstream parser independently re-derived five decisions the producer owned:
>   station numbering, coordinates, species translation, time zone conversion, and the
>   review resolution. On the fifth it disagreed on 515 live rows — it knew four comment
>   strings, only ever demoted them to `blank`, and had no rule at all producing `human`,
>   `vehicle` or `unknown`. Had it run, it would have faithfully rebuilt the exact
>   815-row defect that had just been closed upstream. It was deleted rather than taught the
>   new rules, because teaching it would have created a second place the next repair must
>   reach.*
> - *An analysis project loaded a review pass instead of the campaign, and never read the
>   campaign at all. Of 606 shared image keys, 128 carried a different species. Correcting
>   the source moved spring hare detections from 230 to 161 and culpeo fox from 59 to 82 —
>   enough to change what the analysis says.*
> - *An analysis project re-parsed station labels in three grammars, re-validated SD-card
>   names, and re-filtered species strings. All three had owners upstream. Deleting them
>   removed a genuine cross-check — but what that check verified was that a *label* had been
>   parsed correctly, and no label is parsed there any more. The check lost its subject
>   rather than its value.*

#### 4G.2 What belongs upstream

If a downstream project is doing any of the following, the logic is in the wrong place:

- parsing or normalising station identifiers
- repairing, offsetting or reinterpreting timestamps
- translating species names, or deciding what counts as a species
- deciding whether a frame holds an animal
- filtering out placeholder or "unidentifiable" strings
- converting time zones

#### 4G.3 The direction is never reversed

> **Rule.** The producer publishes; consumers verify. The producer must not know that any
> particular consumer exists.
>
> **Break point.** If the producer knows about a consumer's database, its schema, or its
> paths, then the consumer's requirements start shaping the canonical table — and the table
> stops being canonical and becomes one consumer's input.

#### 4G.4 Ingest checklist

- [ ] Structure: conservation reconciled, no nested stations, capture order preserved or
      manifest written
- [ ] Sweep: every image categorised; export gate passes, or a signed override file exists
- [ ] Exactly one file is the reviewed record
- [ ] Review comments resolved; unknown comments stopped ingest rather than being ignored
- [ ] Clock diagnosis run per segment; verdicts recorded, refusals recorded as refusals
- [ ] Anchors: only where needed, witness-based, strictly contained
- [ ] Canonical table written: every still, one grammar, keys not attributes
- [ ] All three validity axes populated
- [ ] Contract published, and verified by a separate act
- [ ] Test suite green
- [ ] Every moved number in a downstream output attributed to a named cause

That last item is worth its own note. When a re-ingest changes a published figure, **each
change is attributed to a cause before the figure is accepted**. Otherwise two unrelated
effects — a genuine data correction and an unrelated bug fix — arrive together and neither
can be evaluated.

> *Observed: one re-ingest changed the annual report by exactly one record, and the cause was
> named down to the individual image and adjudication. That is the standard: not "the numbers
> moved a little", but "row X was added because of decision Y".*

---
## Part 5 — The time model

This part is placed before the analyses because every time-based product depends on it, and
because the most common way to corrupt a whole dataset silently is to "fix" a time zone.

### 5.1 What is actually stored

> **Rule.** The stored timestamp is the camera's reading, treated as **naive local time** —
> no zone, no offset, no conversion. The camera is not an authority on time; it is an
> instrument whose reading we record.
>
> **Break point.** Attaching a zone to a reading you have not verified asserts a precision
> you do not have, and the assertion is invisible afterwards.

### 5.2 A time zone label is not a conversion

> **Rule.** When loading timestamps in any analysis, the zone is attached as a **label**,
> explicitly, and never left to the environment's default.
>
> **Break point.** This one is a genuine trap because it is environment-dependent: the same
> script gives different answers on different machines. An empty or default zone string is
> a silent no-op on a system with no zone database, and a **real conversion by the local
> offset** on a system that has one. Every activity figure moves by three or four hours,
> with no error and no warning, depending on which computer ran it.
>
> **Recovery.** Full, once noticed. The danger is entirely in not noticing.
>
> *Observed: found latent in an analysis loader. Pinned to an explicit label — which for
> a camera clock reading is the honest description, since the reading is already local time.*

### 5.3 Standard time versus daylight saving — leave the camera alone

This is a decision that was deferred three times, and the first answer was reversed. The
final rule is counter-intuitive:

> **Rule.** **Do not adjust camera clocks for civil time changes.** Leave them running on
> whatever they were set to. If a correction to civil time is wanted, apply it in analysis.
>
> **Break point.** An unadjusted clock has a **constant** offset from the true instant, and
> a constant offset is exactly removable with a single anchor. An adjusted clock has a
> **piecewise** offset that steps at arbitrary visit dates — so adjusting the camera
> *destroys the very property* that made the record correctable. Worse, a one-hour step
> breaks no coherence test, no ordering test and no window test, so nothing in the pipeline
> can detect it: the only record that it happened is the field note.
>
> **Recovery.** Only from the field record, and only if the change date and size were both
> written down.

Three supporting points, because this rule reliably gets argued with:

1. **Animals do not use civil time.** Correcting a camera to match a legal convention
   optimises for a frame of reference that has no biological meaning. The defensible frame
   for an activity analysis is solar, and solar time is derived from the instant and the
   location — both of which survive a constant offset.
2. **The technician cannot make this judgement in the field anyway.** The phone they compare
   against adjusts itself, so a fixed camera offset plus a civil change looks identical to a
   clock reset. This is §2.2: record two readings, decide later.
3. **A single adjustment is recoverable; a habit of adjusting is not.** Because clocks here
   were adjusted only once, the older campaigns are constant-offset rather than ambiguous.
   That is luck, and the rule exists so as not to need it again.

### 5.4 What is derived, and what is deliberately not stored yet

Currently derived at point of use: date, hour, season. Currently **not** stored, and known
to be missing: the true instant plus a fixed per-deployment offset, which is what would let
a consumer convert to any frame — civil, solar or UTC — without knowing the camera's
history. Until that exists, any cross-campaign time-of-day comparison spanning a civil time
change carries a known error at the scale of that change.

This is recorded here as a limitation rather than buried, because a reader of an activity
figure has a right to know it.

---

## Part 6 — What all of this makes possible

The product of Parts 1 to 5. For each analysis: what it needs, what disqualifies a station,
and what an unguarded pipeline would have produced instead.

Read this part as the *reason* for the rest of the manual. Everything above is cost;
everything here is what the cost buys.

### 6.1 Species inventory and presence

**Needs:** a correct identification and a correct station. **Nothing else.**

Presence at a place is spatially true regardless of what the clock said. This is the most
robust product in the whole set, and the one most easily damaged by careless filtering.

> **Failure to avoid.** Presence inheriting a **time**-based filter. A record with a broken
> clock is still a valid observation of that species at that station, so filtering on
> timestamp validity silently removes stations from a distribution map.
>
> *Observed: a presence/absence map showed a species at 6 stations when the correct answer
> was 8 — the loader's "drop rows with no usable date" filter had propagated into a question
> that does not involve dates.*

**Requires:** a station registry that agrees with itself (§1.3), or a station drops out with
no coordinates and no error.

### 6.2 Sampling effort and trap-nights

**Needs:** `valid_effort == TRUE`, and a known operating period per deployment.

The denominator is harder than the numerator and gets a fraction of the attention. Two
distinct failures:

> **Failure to avoid, one.** Dividing by the stations that **exist** rather than the
> stations that were **deployed**. When the array grows over time, a campaign that ran 21
> cameras divided by 27 understates every rate by a quarter — and the number looks
> completely reasonable.
>
> *Observed, and still live at the time of writing: a public dashboard computes occupancy as
> a count divided by the total station list rather than by the stations deployed in that
> campaign.*

> **Failure to avoid, two.** Keeping a station in the denominator when its operating period
> is unknown. It contributes trap-nights during which detection was impossible.

### 6.3 Detection rates and relative abundance

**Needs:** both of the above — a clean numerator (§6.4) and a defensible denominator (§6.2).

There is no way to compute a rate correctly if either side is wrong, and the two errors can
cancel, producing a plausible number from two mistakes.

### 6.4 Independent events (episodes)

**Needs:** `valid_date`, and a species identification.

Almost every count in a camera-trap report should be a count of **episodes**, not
photographs. A camera firing three frames per trigger, on an animal that lingers, produces
dozens of images from one visit.

> **Rule.** The episode rule is: within a station and species, a detection starts a new
> episode only if it is at least 30 minutes after the **last retained** detection — not
> after the previous detection. Grouping includes the campaign.
>
> **Break point of using the wrong variant.** The "previous detection" variant chains: a
> sequence of detections 20 minutes apart never starts a new episode, however long it runs.
> The two rules give different counts on the same data, and the difference is invisible in
> the output.

> **Failure to avoid.** Counting **images** where episodes were meant. This does not
> rescale results — it **reorders** them, which is far worse, because a rescaling is
> obvious and a reordering looks like a finding.
>
> *Observed: by image count, two species stood at 84 against 22 — a ratio of nearly 4:1. By
> episode count the same two stood at 17 against 13. The apparent dominance was an artefact
> of one species triggering more frames per visit. Across all species the image-to-episode
> ratios ranged from 1.7× to 4.9×, so no single correction factor could have fixed it.*

> **Rule.** Every consumer uses the **same** episode rule, from one implementation.
>
> **Break point.** Two projects reporting different episode counts from the same canonical
> table, with the difference being a convention nobody documented.

### 6.5 Activity patterns and nocturnality

**Needs:** `valid_time_of_day == TRUE`, at least 10 records per species, and episodes rather
than images.

This is the **most fragile** product in the set, and the one destroyed by the cheapest
omission in the manual: a visit time not written down (§2.3). A camera whose clock failed
and whose visit was recorded date-only contributes presence, counts and seasonality — and
contributes nothing here, permanently.

### 6.6 Temporal overlap between species

**Needs:** everything §6.5 needs, for **both** species simultaneously, plus a sample-size
rule.

The estimator itself depends on sample size: the Δ4 statistic is appropriate when the
smaller of the two samples is at least 50, and Δ1 below that. Reporting a single overlap
number without saying which estimator produced it is not interpretable.

Overlap inherits every fragility of §6.5 and squares it — a station lost to a missing visit
time is lost from every pair involving both its species.

### 6.7 Seasonal and spatial patterns

**Seasonal** needs `valid_date` (a season is derived from the month, so a wrong year with a
right month is still wrong if the year determines the campaign). Current practice requires
at least 30 records per species for a seasonal figure.

**Spatial** needs only a correct station — see §6.1 — *unless* the figure counts detections
rather than presence, in which case it also needs whatever §6.4 needs.

> **Rule.** State, per figure, whether it asks *where* or *how often*. They have different
> admissibility requirements and mixing them is the §6.1 failure.

### 6.8 Occupancy modelling

**Needs:** a detection history per station per occasion, which means `valid_date` **and**
`valid_effort` — an occasion during which a camera was not operating is a structural
zero, not an absence.

This is the analysis with the strictest requirements, and it is the clearest illustration of
why the effort record matters: the model's entire inference rests on separating "not
detected" from "not sampled". Only §6.2 can make that distinction.

*Status: the requirements are stated here as requirements. The current dashboard
implementation does not yet satisfy them — see §6.2 and Part 9.*

### 6.9 What we cannot do yet, and what would unlock it

| Not possible today | Blocked by | What would unlock it |
|---|---|---|
| Density estimation | Missing bearing and detection distance for past campaigns | §2.7 recorded from now on; past campaigns are permanently excluded |
| Solar-time activity analysis | Instant + fixed offset not stored | §5.4 — a schema addition, no new fieldwork |
| Cross-campaign activity comparison spanning a civil time change | The unadjusted-shift period | Known dates and arithmetic; the data is sufficient |
| Group-size or count analysis | The count field is empty across all campaigns | A review convention; the frames still exist |
| Activity for the cameras with corrupt clocks | Class 4 of §4D.3 | **Nothing.** Permanently lost |
| Occupancy at full station coverage | Registry disagreement (§1.3) | One authoritative registry |

### 6.10 The admissibility matrix

The whole of Part 6 on one page. **R** = required · **—** = not required · **E** = must use
episodes, not images.

| Analysis | `valid_date` | `valid_time_of_day` | `valid_effort` | Unit | Min n | Notes |
|---|:--:|:--:|:--:|:--:|:--:|---|
| Species inventory | — | — | — | record | — | Presence is clock-independent |
| Presence / absence map | — | — | — | presence | — | Use **all** stations |
| Species richness per station | — | — | — | presence | — | Narrows to time-admissible if computed by external tools |
| Detection counts per station | R | — | — | **E** | — | Episodes, or the ranking is wrong |
| Detection rate | R | — | **R** | **E** | — | Both sides must be clean |
| Naive occupancy | R | — | **R** | **E** | — | Denominator = deployed stations |
| Occupancy model | R | — | **R** | **E** | — | Needs occasion-level effort |
| Activity pattern | R | **R** | — | **E** | 10 | Most fragile product |
| Temporal overlap | R | **R** | — | **E** | 50 / <50 | Δ4 above 50, Δ1 below |
| Seasonal distribution | R | — | — | **E** | 30 | Season from the date |
| Trap-night summary | R | — | **R** | deployment | — | Effort is station-level |

Two rows deserve a second look. **Presence requires nothing** — that is the point of keeping
flags rather than deleting bad dates. And **every count-based row requires episodes** —
there is no analysis in this table for which counting images is correct.

---
## Part 7 — Verification, as distinct from gating

### 7.1 A gate decides; a diagnostic describes

> **Rule.** A **gate** admits or refuses data and must be deterministic. A **diagnostic**
> strengthens or questions a verdict and must never admit what a gate refused.
>
> **Break point.** A diagnostic promoted to a gate imports its assumptions into the
> admission decision. Diagnostics are typically heuristic — that is what makes them useful
> and what disqualifies them from deciding.

The clearest case: a check that reconstructs capture order **from the timestamps** and then
uses it to judge those same timestamps. As a diagnostic this is valuable. As a gate it would
be circular, and making the circularity load-bearing in fault detection would be exactly
the heuristic precondition §4D.5 rejects. So it lives as a separate script, it is not
called during ingest, and it writes nothing into the anchor record.

> *Observed: an attempt to write its result into the anchor table as a new anchor type was
> refused by the anchor loader, which fails closed on unknown types. An anchor records a
> field observation; an audit result is not one. The validation working as intended.*

### 7.2 Checking order when no folder record survives

For the stations where capture order cannot be established (§4B.3), there is still a
falsifiable test:

Sort every frame by timestamp; cut a new folder each time a filename counter **repeats**;
then ask the question that can fail — **is the counter monotonically increasing inside every
reconstructed block?**

A clock whose timestamps disagree with true capture order cannot satisfy that: the counters
inside the blocks would jump around. So a pass is evidence, and a failure is a positive
finding.

**What it can and cannot show.** It rules out a backwards reset (breaks monotonicity) and a
factory reset (would show implausible dates). It **cannot** rule out a clock set forward by
a constant with no frames spanning the jump — that leaves no trace here or anywhere else.
"Checked against an independent constraint and consistent" is not "verified clean", and the
distinction is stated in the output rather than blurred.

> **Validated against ground truth.** Run on stations that *do* have a folder manifest, it
> recovered 9 folders where the manifest says 9 — with block sizes landing exactly on the
> 999-image folder cap — and 3 where the manifest says 3. One station gets the count right
> and shows a single backwards step, explained by its manifest: a *mixed* structure with
> files both loose in the card root and in camera subfolders, which breaks the
> one-folder-one-counter-run assumption. That is a limit of the method, correctly surfaced,
> and precisely why it stays a diagnostic.

### 7.3 Validate a rule before wiring it in

> **Rule.** A new gate is run across **all existing data** before it is switched on, and any
> false positives are folded into the rule rather than tuned away.
>
> **Break point.** A gate switched on untested either blocks legitimate work — and will then
> be disabled under pressure, permanently — or passes everything and provides false comfort.
>
> *Observed: the "one deployment, one capture story" rule was validated across 28,178 files
> in four campaigns before being enabled, with zero false positives. The one false positive
> it did find during development was our own renaming prefix, and the rule was changed to
> account for it rather than the case being excluded.*

There is a matching rule for the other direction: **a claim that survives only because
nobody tested it is not a finding.** Several conclusions in this project's history were
stated confidently and turned out to be wrong when measured — "the ordering evidence is lost
for good", "this review pass must be merged", "these stations are permanently unrecoverable",
"this camera is faulty". Every one of them was reversed by measurement. The lesson is not
that the conclusions were careless; it is that **a verdict on data should cite the
measurement that produced it**, so the next person can re-check it cheaply.

### 7.4 Tests as the record of agreed cases

> **Rule.** When a scenario is discussed and a decision reached, it becomes a **test
> fixture**, not a paragraph.
>
> **Break point.** A documented convention decays; an executable one does not. A rule
> written in a README is re-litigated every time somebody has a reason to; a rule with a test
> fails visibly the moment it is broken.

This also constrains how scenarios are encoded. A taxonomy of five field scenarios should
become **five fixtures**, not five code branches — five branches is a structure organised by
circumstance rather than by knowledge, and it grows a sixth branch every campaign.

*Current state: 209 tests, run with the standard library test runner. That choice was
deliberate — the alternative framework is not installed in one of the two environments, and
the fixtures must run on both machines without adding a dependency.*

### 7.5 What to re-run when anything changes

| If you change… | Re-run |
|---|---|
| A flatten or transfer | Conservation check, nested-station check, manifest write |
| A labelling export | The export gate, immediately, at export time |
| Any review or comment mapping | Ingest (it fails closed on unknown comments), then the full test suite |
| A clock rule or tolerance | The full test suite, then re-diagnose **all** campaigns — not only the one that prompted it |
| An anchor row | Diagnosis for that campaign; check the anchor lands inside a segment |
| The canonical schema | Ingest all campaigns, re-publish the contract, then every consumer |
| A downstream filter | The affected outputs, with each moved number attributed |

The row that gets skipped most often is the fourth. A clock rule changed for one camera
changes verdicts for every camera, and the only way to know the effect is to re-run
everything.

---

## Part 8 — Quick reference

### 8.1 Field card

*(also at §2.10)*

- [ ] Visit **date and time**, to the minute
- [ ] **Camera clock reading**, as shown — not interpreted
- [ ] Camera **triggered with the technician in frame**, at that noted time
- [ ] **Working / not working**, and any evidence of when it stopped
- [ ] **Bearing** and **detection distance** if installed, moved or re-aimed
- [ ] **Moved / reinstalled**, with new coordinates if the position changed
- [ ] **Camera unit ID**, distinguishable from the station ID
- [ ] Card and battery change
- [ ] Every entry is an observation, never a conclusion

### 8.2 Storage checklist

- [ ] Card logged with **file count**
- [ ] One folder per station, canonical name, **top level**
- [ ] **No station folder inside another**
- [ ] Camera folder structure untouched — no flattening, renaming, merging, re-sorting
- [ ] No filename de-duplication
- [ ] Counts reconciled per station
- [ ] Video present and separate
- [ ] Sync direction established from logs before any deletion

### 8.3 Ingest checklist

*(also at §4G.4)*

- [ ] Conservation · no nesting · order preserved or manifest written
- [ ] Every image categorised; gate passes or a signed override exists
- [ ] Exactly one reviewed record
- [ ] Comments resolved; unknown comments stopped ingest
- [ ] Clock diagnosis per segment; refusals recorded as refusals
- [ ] Anchors only where needed, witness-based, strictly contained
- [ ] Every still becomes a row; keys not attributes
- [ ] Three validity axes populated
- [ ] Contract published separately, and verified
- [ ] Tests green; every moved number attributed

### 8.4 When a gate refuses

| Refusal | Meaning | Action |
|---|---|---|
| Nested station folder | Two stations' images in one folder | Move it up, rename canonically. **Never override** |
| Categories never assigned | The sweep was not done | Do the sweep. Not overridable, by design |
| No human or vehicle in export | No proof a human swept it | Check; if genuinely person-free, sign an override file |
| Video rows in export | Video reached the labelling export | Re-export stills only |
| Unrecognised category value | Vocabulary drift | Fix the template; do not add a synonym downstream |
| Unknown review comment | A note nobody has mapped | Decide what it means and add the mapping |
| Reviewed row missing from export | Diagnosis ran on a different frame set | Re-export; do not proceed with a partial match |
| Unknown anchor type | Something not a field observation is being written as one | It is not an anchor. Record it elsewhere |
| Campaign not in precedence order | De-duplication cannot decide which copy wins | Add it in review order — or confirm it is a review pass, not a campaign |
| Contract mismatch | The canonical table is not what was published | Find out why **before** re-publishing |

That last row is the one to be careful with. The instinct on a contract mismatch is to
re-publish so the error goes away. Re-publishing is correct only *after* the difference has
been explained — otherwise the mechanism is being used to suppress exactly the signal it
exists to raise.

### 8.5 The rules that have no exceptions

Everything else in this manual is a judgement with a stated cost. These five are absolute:

1. A station folder is never inside another station folder.
2. Nothing that checks whether work was done may be overridden.
3. An anchor requires a witness. Navigational evidence is never a witness.
4. A consumer reads the canonical table and nothing else.
5. Refusals are written down.

---
## Part 9 — What is not yet closed

Audited 2026-08-20 against the working tree, not against the review document's own
checkboxes. This part exists because a manual that describes guarantees the pipeline does
not yet enforce is worse than no manual: it invites people to trust checks that are not
running.

**Read the tiers as: (A) the guarantee in this manual is not actually enforced yet; (B) a
known defect with a known fix; (C) housekeeping.**

### Tier A — guarantees in this manual that are not yet enforced

**A1. The station registry still disagrees with itself.** §1.3 states the rule and the test.
Neither exists yet. Three files are live: one with 26 stations, two with 27. The 26-station
file is the one documented in code as the single source of truth, so **CT27's 344 images
ingest with no coordinates**, and CT27 is absent from any product built from it. This is the
same class of defect as the coordinate error that produced the 19 km displacement, and it is
the highest-priority item in this list.
*Fix: declare one owner, generate the others, add the agreement test.*

**A2. The contract is published but nothing verifies it downstream.** §4F.5 is half built.
The producer writes and can verify its own statement; the consumer-side freshness check —
the piece that makes a stale downstream database impossible rather than merely detectable —
does not exist. Until it does, the guarantee in §4F.5 is a convention, not a mechanism.
*Fix: the verifying half, in the tool that builds the downstream database.*

**A3. The field form has no loader.** §2.1 describes a visit record that flows into the
pipeline. The form exists and is correct; the code that turns a filled workbook into visit
rows was never written. Today a completed form is a spreadsheet nobody can ingest, which
means the whole of Part 2 currently depends on a manual transcription step that is not
described anywhere.
*Fix: the loader. The schema module already defines its entry point.*

**A4. Effort denominators are wrong in the dashboard.** §6.2's named failure is live: a
count divided by the full station list rather than by the stations deployed in that
campaign. A campaign that ran 21 cameras is divided by 27.
*Fix: divide by deployed stations per campaign. Gated behind the downstream rebuild (B1).*

### Tier B — known defects with known fixes

**B1. The downstream database rebuild.** The camera-trap tables are retired rather than
repaired — the ingest path raises an explicit error rather than running, so nothing can
silently ingest the wrong thing. The replacement, reading from the canonical table, is not
built. Also outstanding: the irreplaceable weather and literature tables exist on only one
machine and must be exported to committed files rather than migrated as a binary.
*Key design requirement, already established: the rebuilt keys must be derived from the
image, never inherited from the labelling tool's identifiers, which are per-project rather
than per-image — two projects reading the same card share filenames and zero identifiers.*

**B2. Analysis paths are machine-bound.** The research project hardcodes absolute Windows
paths and therefore cannot run on the second machine. Flagged in code.

**B3. Figures and tables are not re-rendered** against the corrected data. Two campaigns'
worth of numbers have moved and the documents still show the old ones. When re-rendering:
one campaign is excluded from a spatial grid and unlabelled in several facet labellers, so
**fix and re-render that separately** from the data change — otherwise the un-breaking of a
join looks like an effect of the re-ingest.

**B4. The field record is audited for coordinates only.** One column was checked and
repaired; no other column has been checked for the same class of error. 57 of 106 rows carry
a data-quality flag.

**B5. The capture-story check has not been re-run** on the freshly re-ingested campaign —
data it has not seen.

**B6. Manifest coverage is not stated per campaign**, including the stations that
legitimately have none. Without that statement, "no manifest" and "manifest not looked for"
are indistinguishable — the §4E.3 refusal-recording rule, applied to structure instead of
anchors.

**B7. One recoverable install date is not recorded.** A GPS waypoint file dates a station's
install to the minute and resolves a day/month ambiguity. It should be recorded as evidence
reconciled against the field record — not written in silently.

**B8. Test fixtures missing** for three rules this manual states: the manifest rebuild from a
transfer log (§3.4), the size-matched deletion accounting (§3.8), and the registry agreement
check (§1.3 / A1).

### Tier C — housekeeping

**C1.** Two superseded data files remain on disk: a label-conflict table for a comparison
that no longer has two sides, and a manual-review document keyed entirely to the old station
convention and a retired campaign.
**C2.** One stale code comment describing precedence between a campaign and a review pass
that is no longer in the precedence list.
**C3.** Whether one campaign's video files exist in storage was never confirmed. The
campaign's images are all stills, and the counter gaps are consistent with video having been
taken and not transferred.
**C4.** The count field is empty across all campaigns — a real gap, deliberately deferred
(§6.9).
**C5.** One seasonal figure sits just under its sample-size threshold for a species that is
otherwise fully recovered; three more records would tip it.

### The two deliberate exclusions

Both are real, both larger than the current review, and neither blocks it:

- **Storing the true instant plus a fixed per-deployment offset** (§5.4). Until it exists,
  cross-campaign time-of-day comparisons spanning a civil time change carry a known error.
- **A solar-time sensitivity analysis** for the activity and overlap results.

### Honest summary

Of the guarantees described in this manual: **Parts 3, 4B, 4C, 4D, 4E and 4F are enforced in
code and covered by tests.** Part 2's form exists but its loader does not (A3). Part 1's
registry rule is stated and not enforced (A1). Part 4F's contract is published but not
verified downstream (A2). Part 6's effort denominators are correct in the analysis scripts
and wrong in the dashboard (A4).

The chain from card to canonical table is sound. The two ends — the field record going in,
and the consumers coming out — are where the remaining work is.

---

## Appendix A — The canonical schema

One row per still. Names are the contract; column order is not.

| Column | Type | Meaning |
|---|---|---|
| `campaign` | string | Campaign identifier, in precedence order |
| `camera_num` | int | Station number, 1–27 |
| `station_canonical` | string | `CT01`–`CT27` |
| `file_name` | string | Filename as written by the camera |
| `rel_path` | string | Path within the campaign |
| `datetime` | timestamp | Naive local reading, repaired where an anchor allowed it |
| `observation_type` | string | `animal` · `human` · `vehicle` · `blank` · `unknown` |
| `species_latin` | string | Scientific name; empty unless `animal` |
| `observation_comments` | string | The reviewer's note, retained verbatim |
| `classification_probability` | float | Where a classifier produced one |
| `review_resolution` | string | Where this row's verdict came from |
| `valid_date` | bool | Row-level |
| `valid_time_of_day` | bool | Row-level |
| `valid_effort` | bool | **Station-level** |
| `repair_method` | string | How the timestamp was repaired, if it was |
| `clock_segment` | int | Which segment this row belongs to |

**De-duplication key:** station, filename, timestamp — derived from the image, never
inherited from the labelling tool.

**Invariants asserted by tests:** no row is `animal` with an empty species; every still in
the gated export appears exactly once; no row carries a verdict from a source not named in
`review_resolution`.

## Appendix B — The chain, in order

1. Transfer from card, preserving structure — log the moves
2. Flatten for labelling, writing the folder manifest sidecar
3. Structural gates: conservation · no nested stations · one capture story per deployment
4. Automatic detection
5. Human category sweep — every image
6. Species review
7. Export gate — full-category sweep, proof of sweep, stills only
8. Review resolution — comments reconciled against categories
9. Order established; clock diagnosed per segment
10. Anchor candidates listed; anchors confirmed in the field or refused explicitly
11. Repair applied per segment; validity axes emitted
12. Canonical table written
13. Contract published — a separate act
14. Consumers verify the contract, then read the table

## Appendix C — Where these rules came from

Every rule in this manual was decided in a working session and is recorded with its
reasoning. The decision record lives in three places:

- `camera-traps/docs/V2-REVIEW.md` — the current review, item by item, with status
- `camera-traps/docs/HANDOFF-clock-repair.md` — the clock specification and its rejected
  alternatives
- `camera-traps/README.md` — the operational guide, and the campaign history table
- the session logs in the vault, dated, each with a Key Decisions section

**Rules that were reversed at least once** — kept visible here, because a reversed rule is
the one most likely to be re-argued by somebody who has not read this far:

| Was believed | Turned out |
|---|---|
| A campaign's ordering evidence was lost for good | The transfer log had recorded every move |
| A review pass held labels that had to be merged | All of its unique rows already existed in the campaign |
| Three stations were permanently unrecoverable | All three have clean clocks and pass an independent order check |
| One camera was faulty and should be scrapped | Three frames within 32 seconds of midnight; a false positive |
| Segment count, then a slack score, should decide repairability | One anchor per coherent segment |
| Camera clocks should be corrected to civil time | Correcting destroys the recoverable quantity |
| A quality score could rank clock trustworthiness | Deterministic preconditions; heuristics demoted to diagnostics |

The pattern is worth naming: **every one of these was reversed by a measurement, not by an
argument.** That is the working method this manual is trying to encode.
