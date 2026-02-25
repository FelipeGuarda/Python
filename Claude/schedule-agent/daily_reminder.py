"""
Daily Reminder Agent — runs every weekday at 08:00.

Flow:
1. Pull today's task events from Google Calendar
2. Compare current Google Tasks against saved snapshot
3. If changes found → include alert in the email
4. Send morning digest email with today's schedule
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import pytz
import yaml
from dotenv import load_dotenv

# Always run from the project root so relative paths work in any OS scheduler
os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path(__file__).parent))

from src.google_tasks import fetch_open_tasks
from src.auth import get_calendar_service
from src.change_detector import detect_changes, save_snapshot
from src.gmail_sender import send_email, build_daily_digest_email

load_dotenv()


def _load_config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def get_todays_scheduled_tasks(config: dict) -> list[dict]:
    """
    Fetches calendar events for today that were created by the schedule agent
    (description starts with '[schedule-agent]').
    Returns a list of simplified task dicts for the email template.
    """
    tz = pytz.timezone(config["timezone"])
    now = datetime.now(tz)
    today_start = tz.localize(datetime(now.year, now.month, now.day, 0, 0))
    today_end = tz.localize(datetime(now.year, now.month, now.day, 23, 59))

    service = get_calendar_service()
    result = (
        service.events()
        .list(
            calendarId=config["google_calendar"]["calendar_id"],
            timeMin=today_start.isoformat(),
            timeMax=today_end.isoformat(),
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )

    tasks = []
    for event in result.get("items", []):
        desc = event.get("description", "") or ""
        if not desc.startswith("[schedule-agent]"):
            continue

        summary = event.get("summary", "Task")
        # Strip the "[Task] " prefix we add during creation
        title = summary.removeprefix("[Task] ")

        start_raw = event["start"].get("dateTime", "")
        end_raw = event["end"].get("dateTime", "")

        start_dt = datetime.fromisoformat(start_raw).astimezone(tz) if start_raw else None
        end_dt = datetime.fromisoformat(end_raw).astimezone(tz) if end_raw else None

        # Extract complexity from description
        complexity = 3
        for line in desc.splitlines():
            if line.startswith("Complexity:"):
                try:
                    complexity = int(line.split(":")[1].split("/")[0].strip())
                except ValueError:
                    pass

        tasks.append({
            "title": title,
            "day_name": now.strftime("%A"),
            "start_time": start_dt.strftime("%H:%M") if start_dt else "?",
            "end_time": end_dt.strftime("%H:%M") if end_dt else "?",
            "complexity": complexity,
        })

    return tasks


def main() -> None:
    config = _load_config()
    tz = pytz.timezone(config["timezone"])
    now = datetime.now(tz)

    # Skip weekends
    if now.weekday() >= 5:
        print("Weekend — daily reminder skipped.")
        return

    print(f"\n{'='*55}")
    print(f"  Daily Reminder Agent")
    print(f"  {now.strftime('%A %d %B %Y, %H:%M')}")
    print(f"{'='*55}\n")

    # ── Today's scheduled tasks ──────────────────────────────
    print("→ Fetching today's scheduled tasks from calendar...")
    today_tasks = get_todays_scheduled_tasks(config)
    print(f"  Found {len(today_tasks)} task(s) for today.")

    # ── Change detection ─────────────────────────────────────
    print("→ Checking for task changes since last week's schedule...")
    current_tasks = fetch_open_tasks()
    changes = detect_changes(current_tasks)

    if changes:
        print(f"  ⚠  {len(changes)} change(s) detected:")
        for c in changes:
            print(f"     - {c['description']}")
        # Update snapshot so we don't re-report the same changes tomorrow
        save_snapshot(current_tasks)
    else:
        print("  No changes detected.")

    # ── Send email ────────────────────────────────────────────
    print("\n→ Sending daily digest email...")
    html_body = build_daily_digest_email(
        today_tasks=today_tasks,
        changes=changes if changes else None,
    )
    subject = f"[Daily Plan] {now.strftime('%A %d %b')} — {len(today_tasks)} task(s)"
    send_email(subject=subject, html_body=html_body)
    print("  Email sent.\n")


if __name__ == "__main__":
    main()
