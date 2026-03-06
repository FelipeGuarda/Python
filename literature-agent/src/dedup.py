"""Deduplication by DOI and normalized title."""

import pandas as pd


def deduplicate(papers):
    """Remove duplicate papers across sources and topics."""
    if not papers:
        return []

    df = pd.DataFrame(papers)

    # Normalize DOI to lowercase for consistent matching
    df["_doi_norm"] = df["doi"].fillna("").str.lower().str.strip()

    # Normalize title for fuzzy matching
    df["_title_norm"] = (
        df["title"]
        .str.lower()
        .str.strip()
        .str.replace(r"[^\w\s]", "", regex=True)  # remove punctuation
        .str.replace(r"\s+", " ", regex=True)      # collapse whitespace
    )

    # When duplicates exist, merge topic names so no info is lost
    # e.g. "Fauna nativa Chile / Especies invasoras"
    seen_doi = {}   # normalized DOI -> index in result
    seen_title = {} # normalized title -> index in result
    result = []

    for _, row in df.iterrows():
        paper = row.to_dict()
        doi_n = paper.pop("_doi_norm")
        title_n = paper.pop("_title_norm")

        # Check DOI match first
        if doi_n:
            if doi_n in seen_doi:
                _merge_topic(result[seen_doi[doi_n]], paper["topic"])
                continue
            # Also check if title already seen
            if title_n in seen_title:
                _merge_topic(result[seen_title[title_n]], paper["topic"])
                continue
            idx = len(result)
            seen_doi[doi_n] = idx
            seen_title[title_n] = idx
            result.append(paper)
        else:
            # No DOI — match by title only
            if title_n in seen_title:
                _merge_topic(result[seen_title[title_n]], paper["topic"])
                continue
            idx = len(result)
            seen_title[title_n] = idx
            result.append(paper)

    return result


def _merge_topic(paper, new_topic):
    """Append new_topic to paper's topic if not already there."""
    existing = paper["topic"]
    if new_topic not in existing:
        paper["topic"] = f"{existing} / {new_topic}"
