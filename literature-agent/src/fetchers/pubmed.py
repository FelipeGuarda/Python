"""PubMed Central fetcher via NCBI E-utilities.

Two-step process: ESearch (get IDs) → EFetch (get metadata).
No API key required (3 req/sec limit; 10/sec with key).
"""

import httpx
from bs4 import BeautifulSoup

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def fetch(topics, since, config):
    papers = []
    max_results = config.get("max_results", 10)
    since_str = since.strftime("%Y/%m/%d")

    for topic in topics:
        for keyword in topic["keywords"]:
            # Step 1 — search for matching IDs
            search_params = {
                "db": "pmc",
                "term": keyword,
                "retmax": max_results,
                "retmode": "json",
                "mindate": since_str,
                "maxdate": "3000",
                "datetype": "pdat",
            }
            resp = httpx.get(ESEARCH_URL, params=search_params, timeout=30)
            resp.raise_for_status()
            ids = resp.json().get("esearchresult", {}).get("idlist", [])
            if not ids:
                continue

            # Step 2 — fetch full metadata
            fetch_params = {"db": "pmc", "id": ",".join(ids), "retmode": "xml"}
            try:
                resp = httpx.get(EFETCH_URL, params=fetch_params, timeout=30)
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                print(f"    efetch failed for keyword '{keyword}': {e}")
                continue

            soup = BeautifulSoup(resp.text, "xml")
            for article in soup.find_all("article"):
                title_tag = article.find("article-title")
                title = title_tag.get_text(" ") if title_tag else ""

                abstract_tag = article.find("abstract")
                abstract = abstract_tag.get_text(" ") if abstract_tag else ""

                # DOI
                doi = None
                for aid in article.find_all("article-id"):
                    if aid.get("pub-id-type") == "doi":
                        doi = aid.text
                        break

                # Authors (up to 5)
                authors = []
                for contrib in article.find_all(
                    "contrib", attrs={"contrib-type": "author"}
                ):
                    surname = contrib.find("surname")
                    given = contrib.find("given-names")
                    if surname:
                        name = f"{given.text} {surname.text}" if given else surname.text
                        authors.append(name)
                    if len(authors) >= 5:
                        break

                # Publication date
                pub_date = article.find("pub-date")
                date_str = ""
                if pub_date:
                    y = pub_date.find("year")
                    m = pub_date.find("month")
                    d = pub_date.find("day")
                    if y:
                        date_str = y.text
                        if m:
                            date_str += f"-{m.text.zfill(2)}"
                            if d:
                                date_str += f"-{d.text.zfill(2)}"

                # PMC URL
                pmc_id = None
                for aid in article.find_all("article-id"):
                    if aid.get("pub-id-type") == "pmc":
                        pmc_id = aid.text
                        break
                url = (
                    f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/"
                    if pmc_id
                    else ""
                )

                papers.append(
                    {
                        "title": title,
                        "abstract": abstract,
                        "doi": doi,
                        "authors": ", ".join(authors),
                        "date": date_str,
                        "source": "PubMed Central",
                        "url": url,
                        "topic": topic["name"],
                    }
                )

    return papers
