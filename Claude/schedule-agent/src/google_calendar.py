"""
Google Calendar API wrapper.
- find_free_slots(): returns available time blocks for the current week (Tue–Fri)
- create_event(): creates a private task event visible only to you as details
"""

from __future__ import annotations

import yaml
import pytz
from datetime import datetime, timedelta, date
from typing import Any

from .auth import get_calendar_service


def _load_config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def _parse_time(time_str: str, ref_date: date, tz: pytz.BaseTzInfo) -> datetime:
    """Parse 'HH:MM' into a timezone-aware datetime on the given date."""
    h, m = map(int, time_str.split(":"))
    return tz.localize(datetime(ref_date.year, ref_date.month, ref_date.day, h, m))


def _compute_free_slots_for_range(
    busy_by_day: dict,
    start_date: date,
    end_date: date,
    day_start_str: str,
    day_end_str: str,
    lunch_start_str: str | None,
    lunch_end_str: str | None,
    buffer_min: int,
    tz: pytz.BaseTzInfo,
    slot_type: str,
) -> list[dict[str, Any]]:
    """
    Generic helper: compute free slots between day_start and day_end for each date
    in [start_date, end_date], subtracting busy intervals and optional lunch.
    slot_type: "work" or "personal"
    """
    day_names = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday"}
    slots = []
    delta = (end_date - start_date).days
    for offset in range(delta + 1):
        current_date = start_date + timedelta(days=offset)
        day_name = day_names.get(current_date.weekday(), "")

        busy = sorted(busy_by_day.get(current_date, []))

        if lunch_start_str and lunch_end_str:
            lunch_s = _parse_time(lunch_start_str, current_date, tz)
            lunch_e = _parse_time(lunch_end_str, current_date, tz)
            busy.append((lunch_s, lunch_e))
            busy.sort()

        cursor = _parse_time(day_start_str, current_date, tz)
        day_end = _parse_time(day_end_str, current_date, tz)

        for busy_start, busy_end in busy:
            if cursor < busy_start:
                slot_end = busy_start - timedelta(minutes=buffer_min)
                if slot_end > cursor and (slot_end - cursor).total_seconds() >= 1800:
                    slots.append({
                        "date": current_date,
                        "day_name": day_name,
                        "start": cursor,
                        "end": slot_end,
                        "duration_minutes": int((slot_end - cursor).total_seconds() / 60),
                        "slot_type": slot_type,
                    })
            cursor = max(cursor, busy_end + timedelta(minutes=buffer_min))

        if cursor < day_end and (day_end - cursor).total_seconds() >= 1800:
            slots.append({
                "date": current_date,
                "day_name": day_name,
                "start": cursor,
                "end": day_end,
                "duration_minutes": int((day_end - cursor).total_seconds() / 60),
                "slot_type": slot_type,
            })
    return slots


def find_free_slots() -> list[dict[str, Any]]:
    """
    Returns available time slots for Tuesday–Friday of the current week.
    Each slot has slot_type: "work" or "personal".

    Work slots  → Tue–Thu 09:00–18:00, Fri 09:00–14:00 (excluding meetings/lunch)
    Personal slots → Tue–Thu 18:00–21:00, Fri 14:00–21:00 (excluding personal events)
    """
    config = _load_config()
    tz = pytz.timezone(config["timezone"])
    work_start     = config["work_hours"]["start"]
    work_end       = config["work_hours"]["end"]
    friday_end     = config["work_hours"].get("friday_end", work_end)
    lunch_start    = config["work_hours"]["lunch_start"]
    lunch_end      = config["work_hours"]["lunch_end"]
    buffer_min     = config["buffer_minutes"]
    skip_allday    = config["google_calendar"].get("skip_allday_events", True)
    busy_calendars = config["google_calendar"].get(
        "busy_calendars", [config["google_calendar"]["calendar_id"]]
    )
    personal_cfg   = config.get("personal_hours", {})
    personal_wday_start = personal_cfg.get("weekday_start", work_end)
    personal_wday_end   = personal_cfg.get("weekday_end", "21:00")
    personal_fri_start  = personal_cfg.get("friday_start", friday_end)
    personal_fri_end    = personal_cfg.get("friday_end", "21:00")

    today = datetime.now(tz).date()
    days_until_tuesday = (1 - today.weekday()) % 7
    tuesday = today + timedelta(days=days_until_tuesday)
    thursday = tuesday + timedelta(days=2)
    friday = tuesday + timedelta(days=3)

    time_min = tz.localize(datetime(tuesday.year, tuesday.month, tuesday.day, 0, 0))
    time_max = tz.localize(datetime(friday.year, friday.month, friday.day, 23, 59))

    service = get_calendar_service()

    # ── Collect busy intervals ────────────────────────────────
    busy_by_day: dict[date, list[tuple[datetime, datetime]]] = {}
    for cal_id in busy_calendars:
        events_result = (
            service.events()
            .list(
                calendarId=cal_id,
                timeMin=time_min.isoformat(),
                timeMax=time_max.isoformat(),
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
        for event in events_result.get("items", []):
            if "dateTime" not in event["start"]:
                if skip_allday:
                    continue
                start_raw = event["start"].get("date", "")
                if not start_raw:
                    continue
                try:
                    d = date.fromisoformat(start_raw)
                    ev_start = _parse_time(work_start, d, tz)
                    ev_end = _parse_time(work_end, d, tz)
                except ValueError:
                    continue
            else:
                try:
                    ev_start = datetime.fromisoformat(event["start"]["dateTime"]).astimezone(tz)
                    ev_end = datetime.fromisoformat(event["end"]["dateTime"]).astimezone(tz)
                except ValueError:
                    continue
            busy_by_day.setdefault(ev_start.date(), []).append((ev_start, ev_end))

    # ── Work slots: Tue–Thu normal hours, Fri until 14:00 ────
    free_slots = _compute_free_slots_for_range(
        busy_by_day, tuesday, thursday,
        work_start, work_end, lunch_start, lunch_end,
        buffer_min, tz, slot_type="work",
    )
    free_slots += _compute_free_slots_for_range(
        busy_by_day, friday, friday,
        work_start, friday_end, lunch_start, lunch_end,
        buffer_min, tz, slot_type="work",
    )

    # ── Personal slots: Tue–Thu after work, Fri after 14:00 ──
    free_slots += _compute_free_slots_for_range(
        busy_by_day, tuesday, thursday,
        personal_wday_start, personal_wday_end, None, None,
        buffer_min, tz, slot_type="personal",
    )
    free_slots += _compute_free_slots_for_range(
        busy_by_day, friday, friday,
        personal_fri_start, personal_fri_end, None, None,
        buffer_min, tz, slot_type="personal",
    )

    # Sort chronologically
    free_slots.sort(key=lambda s: s["start"])
    return free_slots


def create_task_event(
    title: str,
    start: datetime,
    end: datetime,
    description: str = "",
) -> str:
    """
    Creates a calendar event that:
    - Shows your task title and description when YOU view it
    - Shows only "Busy" to everyone else (private visibility)
    Returns the created event's HTML link.
    """
    config = _load_config()
    tz_name = config["timezone"]
    calendar_id = config["google_calendar"]["calendar_id"]
    color_id = str(config["google_calendar"]["event_color_id"])
    visibility = config["google_calendar"]["event_visibility"]

    service = get_calendar_service()

    event_body = {
        "summary": title,
        "description": description,
        "start": {"dateTime": start.isoformat(), "timeZone": tz_name},
        "end": {"dateTime": end.isoformat(), "timeZone": tz_name},
        "colorId": color_id,
        "visibility": visibility,   # "private" → team sees "Busy", not the title
        "transparency": "opaque",   # Marks you as busy during this time
    }

    created = (
        service.events()
        .insert(calendarId=calendar_id, body=event_body)
        .execute()
    )
    return created.get("htmlLink", "")


def delete_agent_events_for_week() -> int:
    """
    Deletes all agent-created task events for Tue–Fri of this week.
    Used when re-scheduling after task changes.
    Events created by the agent have description starting with '[schedule-agent]'.
    Returns count of deleted events.
    """
    config = _load_config()
    tz = pytz.timezone(config["timezone"])
    calendar_id = config["google_calendar"]["calendar_id"]

    today = datetime.now(tz).date()
    days_until_tuesday = (1 - today.weekday()) % 7
    tuesday = today + timedelta(days=days_until_tuesday)
    friday = tuesday + timedelta(days=3)

    time_min = tz.localize(datetime(tuesday.year, tuesday.month, tuesday.day, 0, 0))
    time_max = tz.localize(datetime(friday.year, friday.month, friday.day, 23, 59))

    service = get_calendar_service()
    events_result = (
        service.events()
        .list(
            calendarId=calendar_id,
            timeMin=time_min.isoformat(),
            timeMax=time_max.isoformat(),
            singleEvents=True,
        )
        .execute()
    )

    deleted = 0
    for event in events_result.get("items", []):
        desc = event.get("description", "") or ""
        if desc.startswith("[schedule-agent]"):
            service.events().delete(calendarId=calendar_id, eventId=event["id"]).execute()
            deleted += 1

    return deleted
