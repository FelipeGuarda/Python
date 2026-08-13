# toolbox

Operational scripts that recur but belong to no single project — the things
asked for a few times a year, for a different event each time.

**Last Updated:** 2026-08-13
**What Changed:** Created the toolbox; migrated the bulk-mail scripts and the
video converter out of `Envio correos/` and `Transforma MOV a MP4/`.
**Integration Status:** `Ready`
**Blockers/Notes:** Three credentials committed to this repo's history before
the migration still need rotating — see *Credentials* below.

---

## What's here

| Script | Does | Run it |
|---|---|---|
| `merge_contacts.py` | Add contacts from event forms to the canonical master list, preserving its formatting | `python scripts/merge_contacts.py --review --master M.xlsx --source A.xlsx` |
| `excel_crosscheck.py` | Compare two contact workbooks: who's missing from each, and which fields the second one fills in or contradicts | `python scripts/excel_crosscheck.py a.xlsx b.xlsx` |
| `send_campaign.py` | Personalised bulk mail from a spreadsheet + an HTML template | `python scripts/send_campaign.py lista.xlsx --template X --subject "Y"` |
| `video_to_mp4.py` | Convert MOV/AVI/MTS to H.264 MP4 | `python scripts/video_to_mp4.py input.MOV` |

## Status

| Script | Verified | Not yet verified |
|---|---|---|
| `merge_contacts.py` | Full run on the real master (141 → 166 rows): highlights and existing rows preserved byte-for-byte, packed address cells split, `N` continued, autofilter grown, share copy generated, original untouched | Behaviour once the standardised form exists (columns will map 1:1, so the review pass should become a formality) |
| `excel_crosscheck.py` | Run against real FMA workbooks and synthetic ones: banner rows above the header, accented column names, `Name <addr>` cells, case/whitespace variance, enrichment and conflict detection | — |
| `send_campaign.py` | Template rendering (both branches), signature, preview, dry run, ledger skip | **Actual SMTP delivery.** Needs a filled-in `.env`. Send yourself a one-row test list before any real campaign. |
| `video_to_mp4.py` | AVI → MP4 via bundled ffmpeg | Batch/directory mode, `--force` |

### Not migrated

`data/legacy/` (gitignored) holds the originals. Two email bodies are still
only in notebook form and have no template yet:

- **Smart Forests** encuentro thank-you — `Correo más attachment.ipynb`.
  Uses an inline image and a `Genero` column.
- **IPCC puntos focales** — `Correo para puntos focales IPCC.ipynb`. Takes
  per-row `Subject` and `Message` columns rather than a fixed body.

Port these to `templates/` when a round of either comes up. The old
`MOV to mp4.py` imports `moviepy.editor`, removed in moviepy 2.x — it is
superseded by `video_to_mp4.py` and not worth fixing.

## Setup, once

```bash
conda env create -f environment.yml
conda activate toolbox
cp .env.example .env      # then fill in SMTP_APP_PASSWORD
```

Run everything from the `toolbox/` directory.

---

## Cross-checking two contact lists

The recurring case: an event's registration export against the master list, to
find who's new.

```bash
python scripts/excel_crosscheck.py registrations.xlsx master.xlsx -o new_contacts.xlsx
```

You do **not** need to tell it which column holds the email, which row is the
header, or which sheet to read. It finds the address column by name
(`Email`, `Contacto`, `Correo`, `Direct email`, …) and falls back to sniffing
for email-shaped values, so form exports with a title banner above the header
row work as-is. Pass `--key "Correo institucional"` only when a file has two
address columns and it picks the wrong one.

Addresses are matched lowercased and accent-normalised, and `Juan <j@x.cl>`
matches `j@x.cl`.

Output workbook, four sheets:

- **summary** — counts, plus which column and sheet were detected in each file.
  Read this first; it's how you confirm it looked at what you meant.
- **only_in_left** — full rows present in the first file, absent from the second.
- **only_in_right** — the reverse.
- **changes** — field-level differences for people in both files, tagged:
  - `new_in_right` — the second file fills a blank in the first. *This is the
    enrichment case: new phone numbers, organisations, addresses.*
  - `conflict` — both have a value and they disagree.
  - `missing_in_right` — the second file lost a value the first one had.

---

## Merging event contacts into the master list

Two passes, because event forms pack a person and their organisation into one
free-text field and some of those cannot be split by any rule. The script
proposes, you correct, then it appends.

```bash
# 1. propose — writes revision.xlsx, touches nothing else
python scripts/merge_contacts.py --review \
    --master "data/2026-08-encuentro/LISTADO_CONTACTOS_MAESTRO.xlsx" \
    --source "data/2026-08-encuentro/Contactos Encuentros.xlsx" \
    --source "data/2026-08-encuentro/3° Encuentro (Respuestas).xlsx" \
    --review-file "data/2026-08-encuentro/revision.xlsx"

# 2. open revision.xlsx, fix any split, set Añadir=NO to reject

# 3. append
python scripts/merge_contacts.py --apply \
    --master "data/2026-08-encuentro/LISTADO_CONTACTOS_MAESTRO.xlsx" \
    --review-file "data/2026-08-encuentro/revision.xlsx" \
    --origen "3° Encuentro Hablemos de Conservación" \
    --fecha 2026-07-28
```

### The review sheet

Sorted so the rows needing a decision come first. Colour-coded: **orange** =
probable duplicate, **amber** = the split is a guess.

- `Confianza` — `alta` (a spaced delimiter separated two plausible halves, or
  the cell is plainly just a name), `media` (the boundary was inferred from an
  organisation keyword), `baja` (no rule applied; check `Original`).
- `Posible duplicado` — someone already on the list whose name matches. This
  catches the person registering with a personal address when the list has
  their institutional one, which address matching alone cannot see. **These
  default to `Añadir=NO`:** adding a duplicate is a mistake that hides,
  skipping someone is one you notice immediately.
- `Añadir` — `SI` appends, anything else rejects.

### What the append guarantees

- **The master is never written to.** Output goes to `*_actualizado.xlsx`;
  you replace the original yourself once satisfied.
- **No reordering.** New rows go at the bottom, in review-sheet order.
- **Formatting survives**, highlights included — the append copies font,
  border and alignment from the last row but deliberately *not* fill, so a
  highlight is never invented on a row nobody marked.
- **Rows whose address cell holds a note rather than an address are left
  exactly alone**, and the script names them in its output.
- `Origen` and `Fecha` come from the flags, never from the file, and are
  written only on the rows being added — an existing row's provenance records
  when that person joined and is never rewritten.

One deliberate change during the first restructure: the `N` column's formulas
(`=A2+1`) become the numbers they already evaluate to. The values are
identical and it matches rows 120+, which were already literals. It is
necessary because openpyxl discards cached formula results on save, so a
formula left in place reads back empty for the shared copy.

### Column layout

Canonical (9): `N`, `Nombre`, `Organización`, `Email principal`,
`Email alternativo`, `Consentimiento`, `Origen`, `Fecha`, `Notas`.

Shared (5), regenerated as `*_COMPARTIR.xlsx`: `N`, `Nombre`, `Organización`,
`Email principal`, `Email alternativo`.

**Never edit the shared copy.** Fix the canonical file and regenerate, or the
two diverge and you are reconciling contact lists against each other again.

`Consentimiento` is blank for anyone who registered before the form asked —
blank means "never asked", not "declined". Only an explicit `No` should stop
a mailing.

### The form that feeds this

`templates/formulario_contactos.md` is the Google Forms template whose columns
map 1:1 onto the master. Duplicate it per event rather than building a new
form — a new form reinvents the columns, and reinvented columns are what make
the review pass necessary in the first place. Once every source comes from
this template, `--review` should return `confianza = alta` on everything and
become a formality.

## Sending a campaign

Bodies are templates in `templates/`, not Python. Editing next year's copy
means editing an `.html.j2` file.

**Always preview first.** This renders the first three messages to a browser
tab and sends nothing:

```bash
python scripts/send_campaign.py Resultados.xlsx \
    --template fondo_fma_resultado.html.j2 \
    --subject "Aviso proceso de selección Fondo FMA 2025-2026"
```

When the preview looks right, add `--send`:

```bash
python scripts/send_campaign.py Resultados.xlsx \
    --template fondo_fma_resultado.html.j2 \
    --subject "Aviso proceso de selección Fondo FMA 2025-2026" \
    --send
```

**Without `--send`, nothing is delivered.** The previous versions of these
scripts sent to every row the moment you ran them.

Every delivery is appended to `data/sent_<template>.csv`. Re-running skips
anyone already marked `sent`, so an interrupted run resumes instead of
double-mailing. To deliberately re-send a campaign, delete that ledger file.

### Writing a new template

Copy `fondo_fma_resultado.html.j2`. Templates extend `_base.html`, which
supplies the body styling and the signature — don't restate either.

Available variables:

| Variable | Is |
|---|---|
| `{{ Nombre }}` | any column, by its own header name |
| `{{ row['Correo electrónico'] }}` | any column, when the header has spaces or accents |
| `{{ email }}` | the normalised recipient address |
| `{{ saludo }}` | `Estimado` / `Estimada` from a `Genero` column; `Estimado/a` when there isn't one |
| `{{ firma.nombre }}` | signature fields, from `.env` |

Per-round values (dates, deadlines, the convocatoria name) go in a `{% set %}`
block at the top of the template, so next year's edit is one visible place.

A typo'd variable raises rather than silently rendering blank — that's
deliberate, because you can't unsend 400 emails with a hole in them.

Attachments and inline images:

```bash
--attach bases.pdf
--inline-image crono.png     # then reference <img src="cid:crono.png"> in the template
```

---

## Converting video

```bash
python scripts/video_to_mp4.py 11210204.MOV          # one file
python scripts/video_to_mp4.py "D:/Salidas/Marzo"    # a whole folder, recursive
```

Walks `.mov .avi .mts .m4v .mpg .mpeg .wmv`, writes `.mp4` beside each source
(or into `-o DIR`), and skips files already converted unless you pass
`--force`. ffmpeg is not on this machine's PATH; the bundled
`imageio-ffmpeg` binary is what does the work.

---

## Credentials

Nothing in this directory holds a password. `send_campaign.py` reads
`SMTP_USER` and `SMTP_APP_PASSWORD` from `toolbox/.env`, which is gitignored.

**Rotate these three.** They were committed in plaintext to this repo before
the migration and remain in its history; deleting the files did not remove
them:

| Account | Was in |
|---|---|
| convocatorias@fundacionmaradentro.cl | `Envio correos/correos Fondo FMA masivos.py` |
| felipe.guarda@fundacionmaradentro.cl | `Envio correos/Correo más attachment.ipynb` |
| xdelavega@minciencia.gob.cl | `Envio correos/Correo para puntos focales IPCC.ipynb` |

Purging git history is a separate decision — it rewrites every commit hash and
breaks existing clones. Rotating the passwords makes the leak harmless without
that.

---

## DESIGN_NOTES

Two modules, each owning one piece of knowledge:

- **`lib/rosters.py`** — *what makes two spreadsheet rows the same person.*
  Column detection, address normalisation, duplicate collapsing, field diffing.
  A workbook with new column conventions is a change here and nowhere else.
- **`lib/mailer.py`** — *how FMA sends personalised bulk mail.* Credentials,
  MIME assembly, throttling, the send ledger. Moving off Gmail is a change here
  and nowhere else.
- **`lib/namesplit.py`** — *how a person and their organisation get packed
  into one cell, and how much to trust the split.* A source file with yet
  another delimiter is a change here alone. Pure functions, no I/O.
- **`lib/master_list.py`** — *the structure of the canonical contact list.*
  Column meanings, multi-address cells, `N` continuation, autofilter growth,
  and the rule that the file is a colleague's working document whose
  formatting must survive. The master gaining a column is a change here alone.

`mailer` consumes a `Roster`, never a raw DataFrame, so it never learns what a
contact column is called. Scripts are thin CLIs — argument parsing and
printing, no domain logic.

Video conversion has no module. It's a codec constant and one moviepy call; a
`lib/video.py` would be a shallow wrapper.

**Adding a script here:** if it needs contact data, take a `Roster`. If it
needs to send mail, build a `Campaign`. If it's genuinely standalone, keep it
standalone rather than inventing a layer.

## What doesn't belong here

Analysis one-offs tied to a dataset — those live with their project. This env
stays light (pandas, jinja2, moviepy); the moment something needs geopandas or
torch, it's a project, not a tool.

## See also

- `../PROJECT_STATUS.md` — ecosystem-wide status
- `../GIT_WORKFLOW_GUIDE.md`
