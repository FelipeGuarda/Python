"""OpenAlex API fetcher.

Queries the OpenAlex works endpoint. No API key required.
Set email in config for polite pool (higher rate limits).
"""

import httpx

API_URL = "https://api.openalex.org/works"


def _reconstruct_abstract(inverted_index):
    """OpenAlex stores abstracts as inverted indexes — reconstruct to plain text."""
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort()
    return " ".join(w for _, w in word_positions)


def fetch(topics, since, config):
    papers = []
    max_results = config.get("max_results", 10)
    email = config.get("email", "")
    since_str = since.strftime("%Y-%m-%d")

    for topic in topics:
        for keyword in topic["keywords"]:
            params = {
                "search": keyword,
                "filter": f"from_publication_date:{since_str}",
                "per_page": max_results,
                "sort": "publication_date:desc",
            }
            if email:
                params["mailto"] = email

            resp = httpx.get(API_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            for work in data.get("results", []):
                doi_raw = work.get("doi") or ""
                doi = doi_raw.replace("https://doi.org/", "") if doi_raw else None

                abstract = ""
                if work.get("abstract_inverted_index"):
                    abstract = _reconstruct_abstract(work["abstract_inverted_index"])

                landing = (work.get("primary_location") or {}).get(
                    "landing_page_url", ""
                )

                papers.append(
                    {
                        "title": work.get("title", "") or "",
                        "abstract": abstract,
                        "doi": doi,
                        "authors": ", ".join(
                            a.get("author", {}).get("display_name", "")
                            for a in (work.get("authorships") or [])[:5]
                        ),
                        "date": work.get("publication_date", ""),
                        "source": "OpenAlex",
                        "url": landing or work.get("id", ""),
                        "topic": topic["name"],
                    }
                )

    return papers
