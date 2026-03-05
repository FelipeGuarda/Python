"""
Google Tasks API wrapper.
Fetches open (incomplete) tasks from the configured task list.
"""

from __future__ import annotations

import yaml
from typing import Any

from .auth import get_tasks_service


def _load_config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def _fetch_tasks_from_list(service, tasklist_id: str, list_name: str, max_tasks: int) -> list[dict[str, Any]]:
    result = (
        service.tasks()
        .list(
            tasklist=tasklist_id,
            showCompleted=False,
            showHidden=False,
            maxResults=max_tasks,
        )
        .execute()
    )
    tasks = []
    for t in result.get("items", []):
        if t.get("status") == "completed":
            continue
        tasks.append(
            {
                "id": t["id"],
                "title": t.get("title", "(no title)").strip(),
                "notes": t.get("notes", "").strip(),
                "due": t.get("due"),
                "updated": t.get("updated"),
                "list_name": list_name,
            }
        )
    return tasks


def fetch_open_tasks() -> list[dict[str, Any]]:
    """
    Returns open tasks from the configured tasklist(s).
    If tasklist_id is "all", fetches from every list and combines them.
    Each task includes list_name so downstream logic can distinguish personal vs work tasks.
    """
    config = _load_config()
    tasklist_id = config["google_tasks"]["tasklist_id"]
    max_tasks = config["google_tasks"]["max_tasks"]

    service = get_tasks_service()
    all_lists = fetch_all_tasklists()

    if tasklist_id == "all":
        seen_ids: set[str] = set()
        tasks: list[dict[str, Any]] = []
        for lst in all_lists:
            for t in _fetch_tasks_from_list(service, lst["id"], lst["title"], max_tasks):
                if t["id"] not in seen_ids:
                    seen_ids.add(t["id"])
                    tasks.append(t)
        return tasks[:max_tasks]

    list_name = next((l["title"] for l in all_lists if l["id"] == tasklist_id), tasklist_id)
    return _fetch_tasks_from_list(service, tasklist_id, list_name, max_tasks)


def fetch_all_tasklists() -> list[dict[str, str]]:
    """Helper for setup — lists all tasklists so the user can find the right ID."""
    service = get_tasks_service()
    result = service.tasklists().list().execute()
    return [
        {"id": tl["id"], "title": tl["title"]}
        for tl in result.get("items", [])
    ]
