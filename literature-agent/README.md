# Agente de Literatura — FMA

**Owner:** Felipe Guarda — Fundación Mar Adentro
**Status:** Built and operational.
**Independence:** Completely standalone. No dependency on the Plataforma Territorial or the data pipeline.

---

## What This Project Does

A weekly cron script that automatically finds, filters, summarizes, and delivers scientific literature relevant to FMA's conservation work.

Every week it:
1. Queries multiple academic databases for new papers on configured topics
2. Deduplicates across sources (same paper may appear in multiple APIs)
3. Uses Claude to summarize each paper in Spanish (2–3 sentences, plain language)
4. Builds an HTML digest email grouped by topic
5. Sends it to one or more recipients

This is the reading assistant — it surfaces relevant science without requiring anyone to manually search databases.

---

## Current State

- **Fully built.** All fetchers, deduplication, summarization, email builder, and sender are implemented.
- `config.yaml` is configured with FMA's topics, keywords, sources, and recipients.
- Run manually with `python run.py`, or schedule via cron for weekly delivery.

---

## Data Sources (APIs)

| Source | What it covers | API type |
|---|---|---|
| **arXiv** | Ecology, remote sensing, ML for conservation | REST (no key needed) |
| **SciELO** | Latin American journals — critical for Chilean/regional literature | REST / OAI-PMH |
| **PubMed Central (PMC)** | Ecology, biology, wildlife | NCBI E-utilities REST (free) |
| **CORE** | Open-access aggregator, broad coverage | REST (free API key) |
| **OpenAlex** | Comprehensive scholarly graph, best for filtering by topic/region | REST (no key needed) |

---

## Architecture

```
Weekly cron trigger (every Monday, before work hours)
    ↓
For each configured topic/keyword:
    → Query arXiv API       → list of {title, abstract, doi, date, authors}
    → Query SciELO API      → "
    → Query PMC E-utils     → "
    → Query CORE API        → "
    → Query OpenAlex API    → "
    ↓
Deduplication by DOI (same paper from multiple sources → keep once)
Filter: published in last 7 days (or since last run)
    ↓
For each paper:
    → Claude API: summarize abstract in Spanish (2–3 sentences, plain language)
    → Classify into topic bucket (fire ecology, wildlife, remote sensing, etc.)
    ↓
Build HTML email (grouped by topic, each paper: title + authors + source + Spanish summary + link)
    ↓
Send via Gmail API (or SMTP)
    ↓
Optionally: store paper metadata in DuckDB (literature table) for the Plataforma Asistente
```

---

## Topics / Keywords (to configure)

Example topics for FMA — adjust in `config.yaml`:

- **Ecología del fuego** — fire ecology, wildfire Chile, fuel moisture, fire behavior Araucanía
- **Fauna nativa Chile** — Guiña, Puma, Zorro culpeo, Condor, biodiversity southern Chile
- **Especies invasoras** — Jabali, Visón americano, Liebre europea, invasive mammals Patagonia
- **Teledetección / Remote sensing** — NDVI, burn severity, land cover change, Sentinel-2 Chile
- **Bioacústica** — bird song, passive acoustic monitoring, soundscape ecology
- **Cambio climático Patagonia** — climate change southern Andes, drought, precipitation trends

---

## Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.11 |
| HTTP requests | `httpx` or `requests` |
| AI summaries | Anthropic Claude API (Haiku — cheap, fast) |
| Email | Gmail API or `smtplib` (SMTP) |
| Deduplication | pandas + DOI as unique key |
| Scheduling | cron |
| Config | YAML (`config.yaml`) |
| Optional storage | DuckDB (`literatura` table) |

---

## File Structure

```
literatura-agent/
├── .env                      ← ANTHROPIC_API_KEY, email credentials
├── config.yaml               ← topics/keywords, sources to query, recipients, schedule
├── requirements.txt
│
├── src/
│   ├── fetchers/
│   │   ├── arxiv.py          ← arXiv API query
│   │   ├── scielo.py         ← SciELO REST / OAI-PMH query
│   │   ├── pubmed.py         ← NCBI E-utilities query
│   │   ├── core_api.py       ← CORE API query
│   │   └── openalex.py       ← OpenAlex API query
│   ├── dedup.py              ← DOI-based deduplication + date filtering
│   ├── summarizer.py         ← Claude Haiku: abstract → Spanish summary
│   ├── email_builder.py      ← HTML digest template
│   └── sender.py             ← Gmail API or SMTP send
│
└── run.py                    ← entry point (run manually or via cron)
```

---

## Email Output Format

```
Subject: [FMA Literatura] Semana del 2 de marzo 2026

━━━ Ecología del fuego (3 artículos) ━━━

[1] "Fire severity and post-fire recovery in Araucanía..."
    González et al., 2026 | Forests | DOI: 10.xxx
    Resumen: Este estudio analiza la severidad de incendios en La Araucanía entre 2019–2024,
    encontrando que las áreas con mayor acumulación de combustibles mostraron recuperación
    más lenta. Se identificaron tres patrones distintos de regeneración post-fuego.
    → Ver artículo completo

[2] ...

━━━ Fauna nativa Chile (2 artículos) ━━━
...
```

---

## Ideas & Future Features

- **Relevance scoring**: Use Claude to rate each paper's relevance to FMA's work (1–5) and only include papers scoring ≥ 3
- **Citation tracking**: Alert when a paper cites a specific author or institution (e.g., FMA-affiliated researchers)
- **DuckDB storage**: Save all papers and summaries to the `literatura` table in `fma_data.duckdb` so the Plataforma Asistente can answer "what does recent literature say about X?"
- **Slack delivery**: Alternative or additional delivery channel beyond email

---

## Key Design Decisions

1. **Haiku for summaries**: Each summary call is cheap (short input = abstract, short output = 2–3 sentences). A batch of 20 papers per week costs pennies.
2. **DOI deduplication**: DOI is a stable unique identifier. Some papers appear in arXiv and OpenAlex — deduplicate before summarizing to avoid redundant API calls.
3. **Spanish output**: FMA's team communicates in Spanish. Summaries in Spanish are immediately usable in reports and communications without translation.
4. **SciELO as a priority source**: Most conservation science about Chile and Latin America is published in Spanish-language journals that aren't in arXiv or PMC. SciELO is essential for regional coverage.
5. **Completely independent**: This script has zero imports from or dependencies on the Plataforma Territorial. Run it anywhere with just an API key and email credentials.

---

## Context for AI Sessions

When starting a new Claude session on this project:

1. This is a **standalone cron script** — no web interface, no database dependency (DuckDB is optional bonus feature)
2. The entry point is `run.py` — it orchestrates fetch → dedup → summarize → email
3. Each fetcher returns a list of dicts: `{title, abstract, doi, authors, date, source, url}`
4. Deduplication is done on `doi` field after merging all fetcher results
5. Claude call in `summarizer.py`: input = abstract (English), output = 2–3 sentence summary in Spanish
6. Model to use: `claude-haiku-4-5` — fast and cheap for this use case
7. Topics and API credentials are all in `config.yaml` / `.env` — no hardcoded values in source
