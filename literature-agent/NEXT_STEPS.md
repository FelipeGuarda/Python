# Literature Agent — Next Steps

**Context:** Last run found ~110 papers, many irrelevant (medical papers especially). Goal is to reduce to ~15–25 highly pertinent results and build a feedback loop for continuous improvement.

---

## The Problem (diagnosed)

Two root causes:
1. **Bad source fit** — PMC (PubMed Central) is a *biomedical* database. Sending it ecology keywords reliably returns medical papers. It is probably responsible for most of the noise.
2. **No relevance gate** — every paper that passes keyword matching goes straight to the email, with no "is this actually about what FMA cares about?" check.

---

## Phase 1 — Immediate fixes (no new infrastructure)

### Step 1a: Read and fix the fetchers

Read these files to see how filtering is currently configured:
- `src/fetchers/pubmed.py` — likely no field/category filter; consider disabling or adding MeSH term constraints
- `src/fetchers/openalex.py` — OpenAlex supports `concepts` and `topics` field-of-study filtering (e.g., "Environmental Science", "Ecology", "Biodiversity"); this is almost certainly not being used yet
- `src/fetchers/arxiv.py` — check if category filters are set (should be `eess.IV`, `cs.CV`, `q-bio.*`)

**Decision to make:** Disable PMC entirely, or add strict MeSH/category constraints. Disabling is simpler and probably correct — SciELO and OpenAlex cover the Latin American ecology literature far better.

### Step 1b: Add Claude relevance scoring to `summarizer.py`

After generating the Spanish summary, add a second Claude call (same Haiku model) that rates each paper 1–5 on relevance to FMA's conservation mission:

```
FMA focuses on: fire ecology, native Chilean fauna, invasive species,
remote sensing for conservation, bioacoustics, and climate change in Patagonia.

Rate this paper's relevance to FMA's work on a scale of 1–5.
Return only a JSON object: {"score": N, "reason": "one sentence"}

Title: {title}
Abstract: {abstract}
```

Filter: only papers with score ≥ 3 make it into the email.

**Expected result:** 110 papers → ~15–25 after scoring.

This feature is already documented in `README.md` under "Ideas & Future Features" — implement it in `summarizer.py` and wire it into `run.py`.

### Step 1c: Reduce `max_results` in `config.yaml`

Currently set to 10 per source per topic. With 6 topics × 5 sources = up to 300 raw results before dedup. After fixing sources and adding scoring, reduce `max_results` to 5 or even 3 per source — the scoring gate replaces quantity-based coverage.

---

## Phase 2 — Feedback collection (no server needed)

**Workflow:**
1. Weekly run generates a self-contained HTML review page (saved to `data/review_YYYY-WW.html`)
2. Email is still sent, but also mentions: *"Calificar artículos de esta semana →"* with a file path or attached HTML
3. HTML shows each paper with **Pertinente ✓** / **No pertinente ✗** buttons
4. "Guardar feedback" button triggers a JSON download (`feedback_YYYY-WW.json`)
5. User drops the JSON file in `data/feedback/`
6. Next run picks up all files in `data/feedback/` automatically

**New file to create:** `src/feedback.py` — loads feedback JSONs, returns liked/disliked paper records.

**New DuckDB table** (add to `schema.sql` in the data-pipeline project if storing centrally, or use a local SQLite/JSON for standalone use):
```sql
CREATE TABLE IF NOT EXISTS paper_feedback (
    paper_id    TEXT PRIMARY KEY,
    title       TEXT,
    week_of     DATE,
    rating      INTEGER,  -- 1 = relevant, 0 = not relevant
    source      TEXT,
    topics      TEXT
);
```

---

## Phase 3 — Feedback loop closes

Feed accumulated feedback into the Claude relevance scoring prompt as few-shot examples:

```
Previously, FMA marked these papers as RELEVANT: [titles]
Previously, FMA marked these papers as NOT RELEVANT: [titles]

Use this context to rate the following paper for FMA's work.
```

After 3–4 weeks of feedback, the scoring becomes contextually calibrated to FMA's actual interests without any ML infrastructure.

**Optional extension:** Auto-generate keyword exclusions. If N papers sharing a term (e.g., "clinical trial", "patients", "randomized controlled") are repeatedly rejected, append that term to an exclusion list in `config.yaml`.

---

## Summary: what to build, in order

| # | What | File(s) | Effort |
|---|---|---|---|
| 1 | Read fetchers, diagnose filters | `src/fetchers/*.py` | 30 min |
| 2 | Disable/fix PMC; add OpenAlex field-of-study filter | `src/fetchers/pubmed.py`, `openalex.py` | 1–2h |
| 3 | Add Claude relevance scoring (1–5, filter ≥ 3) | `src/summarizer.py`, `run.py` | 1h |
| 4 | Reduce `max_results` in config | `config.yaml` | 5 min |
| 5 | Test dry run (fetch + score, no send) | `run.py` | — |
| 6 | Build HTML review page generator | `src/reviewer.py` (new) | 2–3h |
| 7 | Build feedback JSON loader | `src/feedback.py` (new) | 1h |
| 8 | Integrate feedback into scoring prompt | `src/summarizer.py` | 1h |

---

## Starting prompt for next session

> "We're working on `C:/Dev/Python/literature-agent`. Read NEXT_STEPS.md first, then read `src/fetchers/pubmed.py` and `src/fetchers/openalex.py`. The goal is Phase 1: fix the source filtering and add Claude relevance scoring to cut irrelevant papers."
