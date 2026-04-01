#!/usr/bin/env python3
"""
Literature Agent — FMA
Entry point: fetch → dedup → summarize → email.
Run manually or via weekly cron.

Flags:
  --dump        Save fetched+deduped papers to papers_dump.csv and exit
                (no summarization, no email — use this to review/refine keywords)
"""

import sys
import csv
import yaml
from datetime import datetime, timedelta
from dotenv import load_dotenv

from src.fetchers import fetch_all
from src.dedup import deduplicate
from src.summarizer import summarize
from src.email_builder import build_html
from src.sender import send

DUMP_FILE = "papers_dump.csv"
DUMP_FIELDS = ["#", "title", "topic", "source", "date", "doi", "url"]


def dump_papers(papers):
    with open(DUMP_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DUMP_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for i, p in enumerate(papers, 1):
            writer.writerow({"#": i, **p})
    print(f"→ Saved {len(papers)} papers to {DUMP_FILE}")


def main():
    load_dotenv()

    dump_mode = "--dump" in sys.argv

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    lookback = config.get("lookback_days", 7)
    since = datetime.now() - timedelta(days=lookback)

    print(f"=== FMA Literature Agent ===")
    if dump_mode:
        print("Mode: DUMP (fetch + dedup only, no summarization or email)")
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

    if dump_mode:
        dump_papers(papers)
        return

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
