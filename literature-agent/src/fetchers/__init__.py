from src.fetchers import arxiv, openalex, pubmed, core_api, scielo

FETCHERS = {
    "arxiv": arxiv.fetch,
    "openalex": openalex.fetch,
    "pubmed": pubmed.fetch,
    "core": core_api.fetch,
    "scielo": scielo.fetch,
}


def fetch_all(topics, since, config):
    """Query all enabled sources and return a flat list of paper dicts."""
    papers = []
    for name, fetch_fn in FETCHERS.items():
        source_cfg = config["sources"].get(name, {})
        if not source_cfg.get("enabled", True):
            print(f"  [{name}] disabled, skipping")
            continue
        try:
            results = fetch_fn(topics, since, source_cfg)
            print(f"  [{name}] {len(results)} papers")
            papers.extend(results)
        except Exception as e:
            print(f"  [{name}] FAILED: {e}")
    return papers
