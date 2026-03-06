"""CORE API v3 fetcher.

Requires a free API key from https://core.ac.uk/services/api
Set CORE_API_KEY in .env
"""

import os
import time

import httpx

API_URL = "https://api.core.ac.uk/v3/search/works/"


def fetch(topics, since, config):
    api_key = os.environ.get("CORE_API_KEY", "")
    if not api_key:
        print("    CORE_API_KEY not set, skipping")
        return []

    papers = []
    max_results = config.get("max_results", 10)
    since_str = since.strftime("%Y-%m-%d")
    headers = {"Authorization": f"Bearer {api_key}"}

    for topic in topics:
        for keyword in topic["keywords"]:
            query = f'("{keyword}") AND publishedDate>={since_str}'
            params = {"q": query, "limit": max_results}

            try:
                resp = httpx.get(API_URL, params=params, headers=headers, timeout=30)
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                print(f"    CORE query failed for '{keyword}': {e}")
                time.sleep(2)
                continue
            data = resp.json()
            time.sleep(1)  # respect rate limits

            for work in data.get("results", []):
                download_url = work.get("downloadUrl") or ""
                source_urls = work.get("sourceFulltextUrls") or []
                url = download_url or (source_urls[0] if source_urls else "")

                papers.append(
                    {
                        "title": work.get("title", "") or "",
                        "abstract": work.get("abstract", "") or "",
                        "doi": work.get("doi") or None,
                        "authors": ", ".join(
                            a.get("name", "")
                            for a in (work.get("authors") or [])[:5]
                        ),
                        "date": (work.get("publishedDate") or "")[:10],
                        "source": "CORE",
                        "url": url,
                        "topic": topic["name"],
                    }
                )

    return papers
