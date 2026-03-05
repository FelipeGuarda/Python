"""
Task change detector.
Compares the current Google Tasks state against a saved snapshot.
Used by the daily agent to detect new, completed, or modified tasks.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

SNAPSHOT_FILE = Path("data/tasks_snapshot.json")


def save_snapshot(tasks: list[dict[str, Any]]) -> None:
    """Persist current task list as the reference snapshot."""
    SNAPSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "saved_at": datetime.now().isoformat(),
        "tasks": {t["id"]: t for t in tasks},
    }
    SNAPSHOT_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_snapshot() -> dict[str, dict] | None:
    """Returns the previous snapshot as {task_id: task_dict}, or None if no snapshot exists."""
    if not SNAPSHOT_FILE.exists():
        return None
    data = json.loads(SNAPSHOT_FILE.read_text(encoding="utf-8"))
    return data.get("tasks", {})


def detect_changes(current_tasks: list[dict[str, Any]]) -> list[dict[str, str]]:
    """
    Compares current_tasks against the saved snapshot.
    Returns a list of change descriptions, e.g.:
      [{"type": "new", "description": "New task added: 'Review contract'"},
       {"type": "completed", "description": "Task completed: 'Send invoice'"},
       {"type": "modified", "description": "Task modified: 'Prepare slides'"}]
    """
    snapshot = load_snapshot()
    if snapshot is None:
        return []

    current_by_id = {t["id"]: t for t in current_tasks}
    changes: list[dict[str, str]] = []

    # Detect new tasks (in current but not in snapshot)
    for task_id, task in current_by_id.items():
        if task_id not in snapshot:
            changes.append({
                "type": "new",
                "description": f"New task added: \"{task['title']}\"",
            })

    # Detect completed or modified tasks (in snapshot but changed or missing in current)
    for task_id, old_task in snapshot.items():
        if task_id not in current_by_id:
            # Task disappeared — likely completed or deleted
            changes.append({
                "type": "completed",
                "description": f"Task completed/removed: \"{old_task['title']}\"",
            })
        else:
            new_task = current_by_id[task_id]
            if new_task["title"] != old_task["title"]:
                changes.append({
                    "type": "modified",
                    "description": (
                        f"Task renamed: \"{old_task['title']}\" → \"{new_task['title']}\""
                    ),
                })
            elif new_task.get("notes") != old_task.get("notes"):
                changes.append({
                    "type": "modified",
                    "description": f"Task notes updated: \"{new_task['title']}\"",
                })

    return changes
