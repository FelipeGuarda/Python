"""Summarize abstracts and score relevance using Claude Haiku."""

import json
import re

from anthropic import Anthropic

SCORE_SYSTEM = (
    "You evaluate scientific papers for relevance to a conservation foundation in southern Chile (Patagonia/Araucanía). "
    "The foundation's research focus areas are:\n"
    "- Fire ecology and post-fire recovery in Chile\n"
    "- Native Chilean fauna (guiña, puma, culpeo fox, condor, native birds)\n"
    "- Invasive species in Chile/Patagonia (wild boar Sus scrofa, American mink Neovison vison, European hare)\n"
    "- Remote sensing for conservation (NDVI, burn severity, Sentinel-2, land cover change in Chile)\n"
    "- Bioacoustics and passive acoustic monitoring for biodiversity\n"
    "- Climate change, drought, and precipitation trends in Patagonia/southern Andes\n\n"
    "Rate the paper's relevance on a scale of 1–5:\n"
    "5 = directly relevant (same species/region/methods as above)\n"
    "4 = relevant (same region OR same species/methods, transferable results)\n"
    "3 = marginally relevant (related topic, different region or broader scope)\n"
    "2 = weak connection (adjacent field, unlikely to be useful)\n"
    "1 = not relevant (different domain, medical/clinical, other region entirely)\n\n"
    "Respond ONLY with valid JSON, no other text: {\"score\": N, \"reason\": \"one sentence\"}"
)

SUMMARY_SYSTEM = (
    "Eres un asistente científico. Resumí el siguiente abstract en 2-3 oraciones "
    "en español, usando lenguaje claro accesible para no-especialistas. "
    "No incluyas preámbulos, solo el resumen directo."
)


def score_relevance(papers):
    """Add relevance_score (1–5) and relevance_reason to each paper. Returns all papers."""
    client = Anthropic()

    for i, paper in enumerate(papers):
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        if not title and not abstract:
            paper["relevance_score"] = 1
            paper["relevance_reason"] = "No title or abstract"
            continue

        content = f"Title: {title}\n\nAbstract: {abstract}" if abstract else f"Title: {title}"
        print(f"  Scoring [{i + 1}/{len(papers)}] {title[:60]}...")

        try:
            response = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=100,
                system=SCORE_SYSTEM,
                messages=[{"role": "user", "content": content}],
            )
            raw = response.content[0].text.strip()
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
            result = json.loads(raw)
            paper["relevance_score"] = int(result.get("score", 3))
            paper["relevance_reason"] = result.get("reason", "")
        except Exception as e:
            print(f"    Scoring failed: {e} — defaulting to 3")
            paper["relevance_score"] = 3
            paper["relevance_reason"] = ""

    return papers


def summarize(papers):
    """Add a 'summary' field to each paper dict."""
    client = Anthropic()

    for i, paper in enumerate(papers):
        if not paper.get("abstract"):
            paper["summary"] = "Sin resumen disponible."
            continue

        print(f"  Summarizing [{i + 1}/{len(papers)}] {paper['title'][:60]}...")

        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=250,
            system=SUMMARY_SYSTEM,
            messages=[{"role": "user", "content": paper["abstract"]}],
        )
        paper["summary"] = response.content[0].text

    return papers
