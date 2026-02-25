"""
Claude-powered weekly schedule builder.
Takes analyzed tasks + free calendar slots → structured week proposal.
"""

from __future__ import annotations

import json
import os
import yaml
from datetime import datetime
from typing import Any

import anthropic
from dotenv import load_dotenv

load_dotenv()


def _load_config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


SCHEDULE_PROMPT = """You are a productivity coach building a realistic work schedule for the week.

## Slot types
Each free slot has a slot_type:
- "work"     → regular working hours (Mon–Thu until 18:00, Fri until 14:00)
- "personal" → after-work hours (Mon–Thu 18:00–21:00, Fri 14:00–21:00)

## Task types
Each task has a list_name field:
- Tasks from these personal lists: {personal_tasklists}
  → MUST be scheduled in "personal" slots ONLY. Never during work hours.
- All other tasks → MUST be scheduled in "work" slots ONLY. Never in personal hours.

## Work task rules (slot_type = "work")
- Schedule tasks with complexity 3–5 ("heavy") in MORNING slots (before {morning_cutoff})
- Schedule tasks with complexity 1–2 ("light") in AFTERNOON slots (after {afternoon_start})
- Tasks marked time_of_day="morning" MUST go in morning work slots
- Tasks marked time_of_day="afternoon" MUST go in afternoon work slots
- Do NOT exceed {max_deep_work} hours of deep work (complexity 4–5) per day
- Friday work slots end at 14:00 — do not schedule work tasks past that time

## Personal task rules (slot_type = "personal")
- Personal tasks can be scheduled in any personal slot regardless of complexity
- Friday has the most personal time (14:00–21:00) — use it well for personal tasks

## General rules
- Leave at least a {buffer}‑minute gap between consecutive tasks
- If a task has a due date that falls within the week, prioritize it
- If you cannot fit all tasks, leave the lowest‑priority/easiest ones in "unscheduled_tasks"
- Each scheduled task must fit entirely within a single free slot — do NOT split tasks across slots

## Available free slots (this week, Tue–Fri)
{slots_json}

## Tasks to schedule (already analyzed)
{tasks_json}

## Output format
Respond ONLY with valid JSON — no explanation, no markdown fences.
Use this exact structure:

{{
  "scheduled_tasks": [
    {{
      "task_id": "string",
      "title": "string",
      "list_name": "string",
      "day_name": "Tuesday",
      "date": "YYYY-MM-DD",
      "start_time": "HH:MM",
      "end_time": "HH:MM",
      "complexity": 1,
      "estimated_hours": 2.0,
      "slot_type": "work",
      "rationale": "One sentence explaining why this slot was chosen"
    }}
  ],
  "unscheduled_tasks": [
    {{
      "task_id": "string",
      "title": "string",
      "list_name": "string",
      "complexity": 1,
      "estimated_hours": 1.0,
      "reason": "Why it could not be scheduled"
    }}
  ]
}}
"""


def _format_slots(slots: list[dict[str, Any]]) -> list[dict]:
    """Convert datetime objects to strings for the prompt."""
    return [
        {
            "day_name": s["day_name"],
            "date": s["date"].isoformat(),
            "start": s["start"].strftime("%H:%M"),
            "end": s["end"].strftime("%H:%M"),
            "duration_minutes": s["duration_minutes"],
            "slot_type": s.get("slot_type", "work"),
        }
        for s in slots
    ]


def build_schedule(
    analyzed_tasks: list[dict[str, Any]],
    free_slots: list[dict[str, Any]],
    week_start: str,
    week_end: str,
) -> dict[str, Any]:
    """
    Uses Claude to assign tasks to free calendar slots.
    Returns a proposal dict ready to be saved and shown in the approval UI.
    """
    if not analyzed_tasks:
        return {
            "generated_at": datetime.now().isoformat(),
            "week_start": week_start,
            "week_end": week_end,
            "scheduled_tasks": [],
            "unscheduled_tasks": [],
        }

    config = _load_config()
    model = config["claude"]["scheduler_model"]
    morning_cutoff = config["morning_cutoff"]
    afternoon_start = config["afternoon_start"]
    max_deep_work = config["max_deep_work_hours_per_day"]
    buffer = config["buffer_minutes"]
    personal_tasklists = config.get("personal_tasklists", [])

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    prompt = SCHEDULE_PROMPT.format(
        morning_cutoff=morning_cutoff,
        afternoon_start=afternoon_start,
        max_deep_work=max_deep_work,
        buffer=buffer,
        personal_tasklists=json.dumps(personal_tasklists),
        slots_json=json.dumps(_format_slots(free_slots), indent=2),
        tasks_json=json.dumps(analyzed_tasks, indent=2, ensure_ascii=False),
    )

    message = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    result = json.loads(raw)

    # Add day_name to unscheduled tasks if missing
    for ut in result.get("unscheduled_tasks", []):
        ut.setdefault("task_id", ut.get("task_id", ""))

    return {
        "generated_at": datetime.now().isoformat(),
        "week_start": week_start,
        "week_end": week_end,
        "scheduled_tasks": result.get("scheduled_tasks", []),
        "unscheduled_tasks": result.get("unscheduled_tasks", []),
    }
