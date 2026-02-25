"""
Claude-powered task analyzer.
Reads raw Google Tasks and returns structured complexity + time estimates.
"""

from __future__ import annotations

import json
import os
import yaml
from typing import Any

import anthropic
from dotenv import load_dotenv

load_dotenv()


def _load_config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


ANALYSIS_PROMPT = """You are a productivity assistant analyzing a task list to help schedule a work week.

For each task, evaluate:

1. **complexity** (integer 1–5):
   - 1: Trivial, almost no thinking needed (quick reply, click through a form)
   - 2: Light work, 30–60 min max (short email draft, update a doc)
   - 3: Moderate focus needed, 1–3 hours (write a report section, prepare a presentation)
   - 4: Heavy focus, sustained thinking, 2–4 hours (complex analysis, writing from scratch)
   - 5: Deep work, hardest type, 4+ hours (architecture decisions, critical strategic documents)

2. **estimated_hours** (float): Realistic hours to complete, from 0.5 to 6.0

3. **time_of_day** (string): When it is best tackled:
   - "morning" — requires high energy and focus (complexity 3+, creative work, deep analysis)
   - "afternoon" — fine when energy is lower (admin, reviews, routine tasks)
   - "any" — flexible

4. **category** (string): One of: deep_work | admin | communication | creative | review

Important: If a task has notes, use them as context for your estimate.
If the title is vague (e.g. "Meeting prep"), assume moderate complexity unless notes say otherwise.

Respond ONLY with a valid JSON array. No explanation, no markdown, no extra text.
Each element must have exactly: id, title, complexity, estimated_hours, time_of_day, category

Example:
[
  {{
    "id": "abc123",
    "title": "Write Q1 strategy document",
    "complexity": 4,
    "estimated_hours": 3.0,
    "time_of_day": "morning",
    "category": "deep_work"
  }}
]

Tasks to analyze:
{tasks_json}
"""


def analyze_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Sends the task list to Claude and returns analyzed tasks with
    complexity, estimated_hours, time_of_day, and category.
    """
    if not tasks:
        return []

    config = _load_config()
    model = config["claude"]["analysis_model"]
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    tasks_for_prompt = [
        {
            "id": t["id"],
            "title": t["title"],
            "notes": t["notes"] or "",
            "due": t.get("due", ""),
        }
        for t in tasks
    ]

    prompt = ANALYSIS_PROMPT.format(tasks_json=json.dumps(tasks_for_prompt, indent=2, ensure_ascii=False))

    message = client.messages.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()

    # Strip markdown code fences if Claude wraps the JSON
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    analyzed: list[dict] = json.loads(raw)

    # Merge back any original fields not returned by Claude (e.g. notes, due)
    original_by_id = {t["id"]: t for t in tasks}
    for item in analyzed:
        original = original_by_id.get(item["id"], {})
        item.setdefault("notes", original.get("notes", ""))
        item.setdefault("due", original.get("due"))

    return analyzed
