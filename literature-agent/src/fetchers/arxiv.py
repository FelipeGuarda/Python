"""arXiv API fetcher.

Queries the arXiv Atom feed API. No API key required.
Rate guideline: ~1 request per 3 seconds.
"""

import time

import httpx
from bs4 import BeautifulSoup

API_URL = "https://export.arxiv.org/api/query"


def fetch(topics, since, config):
    papers = []
    max_results = config.get("max_results", 10)
    since_str = since.strftime("%Y-%m-%d")

    for topic in topics:
        query = " OR ".join(f'all:"{kw}"' for kw in topic["keywords"])
        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        resp = httpx.get(API_URL, params=params, timeout=30)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "xml")
        for entry in soup.find_all("entry"):
            published = entry.find("published").text[:10]
            if published < since_str:
                continue

            # Extract DOI from <link title="doi"> or <arxiv:doi>
            doi = None
            for link in entry.find_all("link"):
                if link.get("title") == "doi":
                    doi = (
                        link["href"]
                        .replace("http://dx.doi.org/", "")
                        .replace("https://doi.org/", "")
                    )
                    break
            if not doi:
                doi_tag = entry.find("doi")
                if doi_tag:
                    doi = doi_tag.text

            papers.append(
                {
                    "title": entry.find("title").text.strip().replace("\n", " "),
                    "abstract": entry.find("summary").text.strip().replace("\n", " "),
                    "doi": doi,
                    "authors": ", ".join(
                        a.find("name").text for a in entry.find_all("author")
                    ),
                    "date": published,
                    "source": "arXiv",
                    "url": entry.find("id").text,
                    "topic": topic["name"],
                }
            )

        time.sleep(3)  # respect arXiv rate guideline

    return papers
