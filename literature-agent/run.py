#!/usr/bin/env python3
"""
Literature Agent — FMA
Entry point: fetch → dedup → summarize → email.
Run manually or via weekly cron.
"""

import yaml
from datetime import datetime, timedelta
from dotenv import load_dotenv

from src.fetchers import fetch_all
from src.dedup import deduplicate
from src.summarizer import summarize
from src.email_builder import build_html
from src.sender import send


def main():
    load_dotenv()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    lookback = config.get("lookback_days", 7)
    since = datetime.now() - timedelta(days=lookback)

    print(f"=== FMA Literature Agent ===")
    print(f"Looking back {lookback} days (since {since.strftime('%Y-%m-%d')})\n")

    # 1. Fetch from all sources
    print("Fetching papers...")
    papers = fetch_all(config["topics"], since, config)
    print(f"→ Total fetched: {len(papers)}\n")

    if not papers:
        print("No papers found. Nothing to send.")
        return

    # 2. Deduplicate
    print("Deduplicating...")
    papers = deduplicate(papers)
    print(f"→ After dedup: {len(papers)}\n")

    # 3. Summarize with Claude
    print("Summarizing with Claude Haiku...")
    papers = summarize(papers)
    print(f"→ Summarized: {len(papers)}\n")

    # 4. Build HTML email
    print("Building email...")
    subject, html = build_html(papers, config)
    print(f"→ Subject: {subject}\n")

    # 5. Send
    print("Sending...")
    send(subject, html, config)

    print("\nDone!")


if __name__ == "__main__":
    main()
