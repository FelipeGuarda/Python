"""
Weekly Schedule Agent — runs every Monday at 15:00.

Flow:
1. Fetch open tasks from Google Tasks
2. Analyze each task with Claude (complexity, time estimate)
3. Find free calendar slots for Tue–Fri of this week
4. Ask Claude to build a balanced week schedule
5. Save proposal to data/pending_proposal.json
6. Send HTML proposal email
7. Open the local approval UI in the browser
8. Wait for the user to approve → calendar events are created automatically
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Always run from the project root so relative paths work in any OS scheduler
os.chdir(Path(__file__).parent)

import pytz
import yaml
from dotenv import load_dotenv

# Allow running from the project root
sys.path.insert(0, str(Path(__file__).parent))

from src.google_tasks import fetch_open_tasks
from src.google_calendar import find_free_slots
from src.task_analyzer import analyze_tasks
from src.schedule_builder import build_schedule
from src.change_detector import save_snapshot
from src.gmail_sender import send_email, build_proposal_email
from src.approval_server import run_approval_server, PENDING_FILE

load_dotenv()


def _load_config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def main() -> None:
    config = _load_config()
    tz = pytz.timezone(config["timezone"])
    now = datetime.now(tz)

    print(f"\n{'='*55}")
    print(f"  Schedule Agent — Weekly Run")
    print(f"  {now.strftime('%A %d %B %Y, %H:%M')}")
    print(f"{'='*55}\n")

    # ── Step 1: Fetch tasks ──────────────────────────────────
    print("→ Fetching tasks from Google Tasks...")
    tasks = fetch_open_tasks()
    if not tasks:
        print("  No open tasks found. Nothing to schedule.")
        return
    print(f"  Found {len(tasks)} open task(s).")

    # ── Step 2: Save snapshot for change detection ───────────
    save_snapshot(tasks)
    print("  Snapshot saved for daily change detection.")

    # ── Step 3: Analyze tasks with Claude ────────────────────
    print("\n→ Analyzing tasks with Claude...")
    analyzed = analyze_tasks(tasks)
    for t in analyzed:
        print(f"  [{t['complexity']}/5] {t['title']} — {t['estimated_hours']}h ({t['time_of_day']})")

    # ── Step 4: Find free calendar slots ─────────────────────
    print("\n→ Scanning calendar for free slots (Tue–Fri)...")
    free_slots = find_free_slots()
    if not free_slots:
        print("  No free slots found this week. Is your calendar fully booked?")
        return
    total_available = sum(s["duration_minutes"] for s in free_slots) / 60
    print(f"  Found {len(free_slots)} free slot(s) — {total_available:.1f}h available.")

    # ── Step 5: Build schedule with Claude ───────────────────
    print("\n→ Building weekly schedule with Claude...")
    # Compute Tue–Fri dates
    today = now.date()
    tuesday = today + timedelta(days=(1 - today.weekday()) % 7)
    friday = tuesday + timedelta(days=3)
    week_start = tuesday.strftime("%a %d %b")
    week_end = friday.strftime("%a %d %b %Y")

    proposal = build_schedule(
        analyzed_tasks=analyzed,
        free_slots=free_slots,
        week_start=week_start,
        week_end=week_end,
    )

    scheduled_count = len(proposal["scheduled_tasks"])
    unscheduled_count = len(proposal.get("unscheduled_tasks", []))
    print(f"  Scheduled {scheduled_count} task(s).")
    if unscheduled_count:
        print(f"  Could not fit {unscheduled_count} task(s) this week.")

    # ── Step 6: Save pending proposal ────────────────────────
    PENDING_FILE.parent.mkdir(parents=True, exist_ok=True)
    PENDING_FILE.write_text(
        json.dumps(proposal, indent=2, default=str), encoding="utf-8"
    )
    print(f"\n  Proposal saved to {PENDING_FILE}")

    # ── Step 7: Send email ────────────────────────────────────
    port = int(os.getenv("APPROVAL_SERVER_PORT", "5555"))
    approval_url = f"http://localhost:{port}"

    print("\n→ Sending proposal email...")
    try:
        email_html = build_proposal_email(proposal, approval_url)
        send_email(
            subject=f"[Schedule Agent] Weekly proposal — {week_start} to {week_end}",
            html_body=email_html,
        )
        print("  Email sent.")
    except Exception as e:
        print(f"  Warning: Could not send email ({e}). Continuing to approval UI.")

    # ── Step 8: Open approval UI ──────────────────────────────
    print(f"\n→ Opening approval UI at {approval_url} ...")
    print("  Review the schedule and click 'Approve' to create calendar events.\n")
    run_approval_server(open_browser=True)


if __name__ == "__main__":
    main()
