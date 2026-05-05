"""Semantic Scholar Graph API v1 fetcher.

No key required. Set SEMANTIC_SCHOLAR_API_KEY in .env for higher rate limits
(1 req/sec with key vs ~100 req/5 min without).
"""

import os
import time

import httpx

API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,abstract,authors,publicationDate,externalIds,openAccessPdf"
FIELDS_OF_STUDY = "Environmental Science,Biology,Geography,Agricultural and Food Sciences"


def _get_with_retry(url, params, headers, timeout=30, max_retries=3):
    """GET with exponential backoff on 429."""
    resp = None
    for attempt in range(max_retries):
        resp = httpx.get(url, params=params, headers=headers, timeout=timeout)
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp
        wait = 2 ** (attempt + 1)
        print(f"    429 rate limit, retrying in {wait}s...")
        time.sleep(wait)
    resp.raise_for_status()
    return resp


def fetch(topics, since, config):
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    headers = {"x-api-key": api_key} if api_key else {}

    papers = []
    max_results = config.get("max_results", 5)
    since_str = since.strftime("%Y-%m-%d")

    for topic in topics:
        for keyword in topic["keywords"]:
            params = {
                "query": keyword,
                "fields": FIELDS,
                "fieldsOfStudy": FIELDS_OF_STUDY,
                "publicationDateOrYear": f"{since_str}:",
                "limit": max_results,
            }

            try:
                resp = _get_with_retry(API_URL, params, headers)
            except httpx.HTTPStatusError as e:
                print(f"    Semantic Scholar query failed for '{keyword}': {e}")
                time.sleep(1)
                continue

            data = resp.json()
            time.sleep(1)

            for work in data.get("data", []):
                doi = (work.get("externalIds") or {}).get("DOI")
                pdf_info = work.get("openAccessPdf") or {}
                pdf_url = pdf_info.get("url") or ""
                paper_id = work.get("paperId", "")

                papers.append({
                    "title": work.get("title", "") or "",
                    "abstract": work.get("abstract", "") or "",
                    "doi": doi,
                    "authors": ", ".join(
                        a.get("name", "") for a in (work.get("authors") or [])[:5]
                    ),
                    "date": work.get("publicationDate", "") or str(work.get("year", "")),
                    "source": "Semantic Scholar",
                    "url": pdf_url or f"https://www.semanticscholar.org/paper/{paper_id}",
                    "topic": topic["name"],
                })

    return papers
