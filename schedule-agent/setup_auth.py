"""
One-time Google OAuth setup.
Run this once to authorize the app and save the token.

Usage:
  python setup_auth.py                # Run the OAuth flow
  python setup_auth.py --list-tasklists   # Show your Google Task list IDs
  python setup_auth.py --list-calendars   # Show your Google Calendar IDs
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.auth import get_credentials, get_calendar_service
from src.google_tasks import fetch_all_tasklists


def main() -> None:
    print("\n── Google OAuth Setup ─────────────────────────────────")
    print("This will open your browser to authorize the Schedule Agent.")
    print("Permissions requested:")
    print("  • Google Tasks (read-only)")
    print("  • Google Calendar (read + write events)")
    print("  • Gmail (send emails only)\n")

    try:
        creds = get_credentials()
        print("✓ Authorization successful! Token saved.\n")
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nSteps to fix:")
        print("  1. Go to https://console.cloud.google.com")
        print("  2. Create a project (or select existing)")
        print("  3. Enable: Google Tasks API, Google Calendar API, Gmail API")
        print("  4. Go to APIs & Services → Credentials")
        print("  5. Create OAuth 2.0 Client ID → Desktop app")
        print("  6. Download the JSON and save it as 'credentials.json' in this folder")
        sys.exit(1)

    if "--list-tasklists" in sys.argv:
        print("── Your Google Task Lists ─────────────────────────────")
        lists = fetch_all_tasklists()
        for lst in lists:
            print(f"  ID: {lst['id']}")
            print(f"  Name: {lst['title']}\n")
        print("Copy the ID you want to use into config.yaml → google_tasks.tasklist_id")

    if "--list-calendars" in sys.argv:
        print("── Your Google Calendars ──────────────────────────────")
        service = get_calendar_service()
        result = service.calendarList().list().execute()
        for cal in result.get("items", []):
            cal_id = cal["id"]
            name = cal.get("summary", "(no name)")
            primary = " ← primary" if cal.get("primary") else ""
            print(f"  ID: {cal_id}")
            print(f"  Name: {name}{primary}\n")
        print("Add the IDs you want to config.yaml → google_calendar.busy_calendars")


if __name__ == "__main__":
    main()
