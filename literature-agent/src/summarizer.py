"""Summarize abstracts in Spanish using Claude Haiku."""

from anthropic import Anthropic

SYSTEM_PROMPT = (
    "Eres un asistente científico. Resumí el siguiente abstract en 2-3 oraciones "
    "en español, usando lenguaje claro accesible para no-especialistas. "
    "No incluyas preámbulos, solo el resumen directo."
)


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
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": paper["abstract"]}],
        )
        paper["summary"] = response.content[0].text

    return papers
