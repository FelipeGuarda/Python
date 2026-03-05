"""
Local Flask approval server.
Serves a web UI where the user can review the weekly proposal,
optionally remove tasks, drag them to different days, and click
"Approve & Create Events".

The server auto-shuts down after approval or after TIMEOUT_HOURS.
"""

from __future__ import annotations

import json
import os
import threading
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any

import pytz
import yaml
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template_string, request, redirect

from .google_calendar import create_task_event
from .gmail_sender import send_email

load_dotenv()

TIMEOUT_HOURS = 20  # Server shuts down automatically after this many hours
PENDING_FILE = Path("data/pending_proposal.json")
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

app = Flask(__name__)
_shutdown_event = threading.Event()


def _load_config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


# ─── HTML Template ────────────────────────────────────────────────────────────

APPROVAL_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Weekly Schedule — Approve</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #f0f2f5; color: #1a1a2e; }
  .container { max-width: 780px; margin: 32px auto; padding: 0 16px 48px; }
  .header { background: #2c3e50; color: white; border-radius: 10px 10px 0 0;
            padding: 24px 28px; }
  .header h1 { font-size: 22px; font-weight: 700; }
  .header p { margin-top: 4px; opacity: 0.7; font-size: 14px; }
  .card { background: white; border: 1px solid #dee2e6;
          border-radius: 0 0 10px 10px; margin-bottom: 24px; overflow: hidden; }

  /* Day sections */
  .day-section { border-top: 2px solid #bdc3c7; transition: background 0.15s; }
  .day-section:first-child { border-top: none; }
  .day-section.drag-over { background: #eaf4fb; outline: 2px dashed #3498db; outline-offset: -2px; }
  .day-section.drag-over .day-header { background: #d0e8f7; }
  .day-header { background: #ecf0f1; padding: 10px 20px;
                font-weight: 700; font-size: 13px; color: #2c3e50;
                display: flex; justify-content: space-between; align-items: center; }
  .day-hours { font-weight: 400; color: #7f8c8d; font-size: 12px; }
  .tasks-container { min-height: 44px; }
  .tasks-container.empty-hint { display: flex; align-items: center;
                                  padding: 10px 20px; color: #bdc3c7;
                                  font-size: 12px; font-style: italic; }

  /* Task rows */
  .task-row { display: flex; align-items: center; padding: 12px 20px;
              border-bottom: 1px solid #f0f2f5; gap: 12px;
              cursor: grab; user-select: none; background: white;
              transition: background 0.1s; }
  .task-row:last-child { border-bottom: none; }
  .task-row:hover { background: #f8f9fa; }
  .task-row.removed { opacity: 0.35; text-decoration: line-through; cursor: grab; }
  .task-row.dragging { opacity: 0.45; background: #eef; cursor: grabbing; }
  .task-row.moved { border-left: 3px solid #3498db; }

  .drag-handle { color: #bdc3c7; font-size: 15px; cursor: grab; flex-shrink: 0;
                 line-height: 1; letter-spacing: -1px; }
  .task-time { color: #6c757d; font-size: 13px; white-space: nowrap; min-width: 110px; }
  .task-title { flex: 1; font-weight: 500; font-size: 14px; }
  .badge { padding: 3px 10px; border-radius: 12px; font-size: 11px;
           font-weight: 700; color: white; white-space: nowrap; }
  .c1 { background: #27ae60; } .c2 { background: #2ecc71; }
  .c3 { background: #f39c12; } .c4 { background: #e67e22; }
  .c5 { background: #e74c3c; }
  .task-hours { color: #95a5a6; font-size: 12px; white-space: nowrap; }
  .remove-btn { background: none; border: 1px solid #e74c3c; color: #e74c3c;
                border-radius: 4px; padding: 3px 8px; cursor: pointer;
                font-size: 11px; white-space: nowrap; flex-shrink: 0; }
  .remove-btn:hover { background: #e74c3c; color: white; }
  .undo-btn { background: none; border: 1px solid #27ae60; color: #27ae60;
              border-radius: 4px; padding: 3px 8px; cursor: pointer;
              font-size: 11px; flex-shrink: 0; }
  .moved-label { font-size: 10px; color: #3498db; white-space: nowrap;
                 flex-shrink: 0; font-style: italic; }
  .rationale { font-size: 11px; color: #adb5bd; font-style: italic; margin-top: 2px; }

  .unscheduled { background: #fff3cd; border: 1px solid #ffc107;
                 border-radius: 8px; padding: 16px 20px; margin-bottom: 24px; }
  .unscheduled h3 { font-size: 14px; color: #856404; margin-bottom: 8px; }
  .unscheduled ul { padding-left: 18px; font-size: 13px; color: #6c757d; }
  .unscheduled li { margin-bottom: 4px; }

  .actions { display: flex; gap: 12px; flex-wrap: wrap; }
  .btn-approve { background: #27ae60; color: white; border: none;
                 padding: 14px 36px; border-radius: 8px; font-size: 16px;
                 font-weight: 700; cursor: pointer; flex: 1; min-width: 200px; }
  .btn-approve:hover { background: #219a52; }
  .btn-approve:disabled { background: #95a5a6; cursor: not-allowed; }
  .btn-cancel { background: white; color: #e74c3c; border: 2px solid #e74c3c;
                padding: 14px 24px; border-radius: 8px; font-size: 14px;
                font-weight: 600; cursor: pointer; }
  .success-msg { background: #d4edda; border: 1px solid #c3e6cb; color: #155724;
                 border-radius: 8px; padding: 16px 20px; text-align: center;
                 font-weight: 600; font-size: 16px; display: none;
                 margin-bottom: 16px; }
  .hint { font-size: 12px; color: #95a5a6; text-align: center;
          margin-bottom: 12px; }
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>Weekly Schedule Proposal</h1>
    <p>{{ week_start }} &rarr; {{ week_end }} &nbsp;·&nbsp; Review, adjust, then approve</p>
  </div>

  <div class="card">
    {% for day, tasks in by_day.items() %}
    <div class="day-section"
         data-day="{{ day }}"
         data-date="{{ tasks[0].date }}"
         ondragover="onDragOver(event)"
         ondragenter="onDragEnter(event, this)"
         ondragleave="onDragLeave(event, this)"
         ondrop="onDrop(event, this)">
      <div class="day-header">
        <span>{{ day }}</span>
        <span class="day-hours">{{ "%.1f"|format(tasks|sum(attribute='estimated_hours')) }}h scheduled</span>
      </div>
      <div class="tasks-container">
        {% for t in tasks %}
        <div class="task-row"
             data-task-id="{{ t.task_id }}"
             data-hours="{{ t.estimated_hours }}"
             data-orig-day="{{ day }}"
             draggable="true"
             ondragstart="onDragStart(event, this)"
             ondragend="onDragEnd(event, this)">
          <span class="drag-handle">&#8942;&#8942;</span>
          <span class="task-time">{{ t.start_time }} – {{ t.end_time }}</span>
          <div class="task-title">
            {{ t.title }}
            <div class="rationale">{{ t.rationale }}</div>
          </div>
          <span class="badge c{{ t.complexity }}">
            {{ ['','Easy','Light','Medium','Heavy','Deep'][t.complexity] }}
          </span>
          <span class="task-hours">{{ t.estimated_hours }}h</span>
          <span class="moved-label" style="display:none">moved</span>
          <button class="remove-btn"
                  onclick="toggleTask(this, '{{ t.task_id }}')">Remove</button>
        </div>
        {% endfor %}
      </div>
    </div>
    {% endfor %}
  </div>

  {% if unscheduled %}
  <div class="unscheduled">
    <h3>Could not fit this week:</h3>
    <ul>
      {% for t in unscheduled %}
      <li>{{ t.title }} — {{ t.reason }}</li>
      {% endfor %}
    </ul>
  </div>
  {% endif %}

  <p class="hint">Drag tasks between days to reschedule &nbsp;·&nbsp; Click Remove to exclude a task</p>

  <div id="success-msg" class="success-msg">
    ✓ Schedule approved! Calendar events are being created...
  </div>

  <div class="actions">
    <button class="btn-approve" id="approve-btn" onclick="approve()">
      Approve &amp; Create Calendar Events
    </button>
    <button class="btn-cancel" onclick="if(confirm('Cancel and discard this proposal?')) window.close()">
      Discard
    </button>
  </div>
</div>

<script>
  const removedIds = new Set();
  // Map: task_id → { new_day_name, new_date }
  const movedTasks = new Map();

  let draggedRow = null;

  // ── Hours counter ──────────────────────────────────────────────────────────
  function updateDayHours(section) {
    let total = 0;
    section.querySelectorAll('.task-row:not(.removed)').forEach(row => {
      total += parseFloat(row.dataset.hours || 0);
    });
    section.querySelector('.day-hours').textContent = total.toFixed(1) + 'h scheduled';
  }

  // ── Remove / Undo ──────────────────────────────────────────────────────────
  function toggleTask(btn, taskId) {
    const row = btn.closest('.task-row');
    const section = row.closest('.day-section');
    if (removedIds.has(taskId)) {
      removedIds.delete(taskId);
      row.classList.remove('removed');
      btn.textContent = 'Remove';
      btn.className = 'remove-btn';
    } else {
      removedIds.add(taskId);
      row.classList.add('removed');
      btn.textContent = 'Undo';
      btn.className = 'undo-btn';
    }
    updateDayHours(section);
  }

  // ── Drag & Drop ────────────────────────────────────────────────────────────
  function onDragStart(e, row) {
    draggedRow = row;
    e.dataTransfer.effectAllowed = 'move';
    // Delay so the element isn't already faded when ghost image is captured
    setTimeout(() => row.classList.add('dragging'), 0);
  }

  function onDragEnd(e, row) {
    row.classList.remove('dragging');
    draggedRow = null;
  }

  function onDragOver(e) {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
  }

  function onDragEnter(e, section) {
    if (draggedRow && !section.contains(draggedRow)) {
      section.classList.add('drag-over');
    }
  }

  function onDragLeave(e, section) {
    // Only remove class when truly leaving the section (not entering a child)
    if (!section.contains(e.relatedTarget)) {
      section.classList.remove('drag-over');
    }
  }

  function onDrop(e, section) {
    e.preventDefault();
    section.classList.remove('drag-over');
    if (!draggedRow) return;

    const taskId = draggedRow.dataset.taskId;
    const newDayName = section.dataset.day;
    const newDate = section.dataset.date;
    const origDay = draggedRow.dataset.origDay;

    const oldSection = draggedRow.closest('.day-section');
    if (oldSection === section) return; // dropped on same day

    section.querySelector('.tasks-container').appendChild(draggedRow);

    if (newDayName === origDay) {
      movedTasks.delete(taskId);
      draggedRow.classList.remove('moved');
      draggedRow.querySelector('.moved-label').style.display = 'none';
    } else {
      movedTasks.set(taskId, { new_day_name: newDayName, new_date: newDate });
      draggedRow.classList.add('moved');
      draggedRow.querySelector('.moved-label').style.display = '';
    }

    updateDayHours(oldSection);
    updateDayHours(section);
  }

  // ── Approve ────────────────────────────────────────────────────────────────
  function approve() {
    const btn = document.getElementById('approve-btn');
    btn.disabled = true;
    btn.textContent = 'Creating events…';

    const movedArray = Array.from(movedTasks.entries()).map(([task_id, info]) => ({
      task_id,
      new_day_name: info.new_day_name,
      new_date: info.new_date,
    }));

    fetch('/approve', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        removed_task_ids: Array.from(removedIds),
        moved_tasks: movedArray,
      })
    })
    .then(r => r.json())
    .then(data => {
      document.getElementById('success-msg').style.display = 'block';
      btn.textContent = `✓ ${data.created} events created`;
      setTimeout(() => window.close(), 3000);
    })
    .catch(err => {
      btn.disabled = false;
      btn.textContent = 'Approve & Create Calendar Events';
      alert('Error: ' + err.message);
    });
  }
</script>
</body>
</html>
"""


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    if not PENDING_FILE.exists():
        return "<h2>No pending proposal found. Run the weekly scheduler first.</h2>", 404

    proposal = json.loads(PENDING_FILE.read_text(encoding="utf-8"))

    by_day: dict[str, list] = {}
    for task in proposal["scheduled_tasks"]:
        by_day.setdefault(task["day_name"], []).append(task)

    # Sort days in calendar order (Tue → Wed → Thu → Fri)
    by_day = dict(
        sorted(by_day.items(), key=lambda x: DAY_ORDER.index(x[0]) if x[0] in DAY_ORDER else 99)
    )

    return render_template_string(
        APPROVAL_HTML,
        week_start=proposal["week_start"],
        week_end=proposal["week_end"],
        by_day=by_day,
        unscheduled=proposal.get("unscheduled_tasks", []),
    )


@app.route("/approve", methods=["POST"])
def approve():
    if not PENDING_FILE.exists():
        return jsonify({"error": "No pending proposal"}), 404

    proposal = json.loads(PENDING_FILE.read_text(encoding="utf-8"))
    data = request.get_json(silent=True) or {}
    removed_ids: set[str] = set(data.get("removed_task_ids", []))

    # Map: task_id → {new_day_name, new_date} for moved tasks
    moved_tasks_map: dict[str, dict] = {
        m["task_id"]: m for m in data.get("moved_tasks", [])
    }

    config = _load_config()
    tz = pytz.timezone(config["timezone"])

    tasks_to_create = []
    for t in proposal["scheduled_tasks"]:
        if t["task_id"] in removed_ids:
            continue
        t = dict(t)  # copy before mutating
        if t["task_id"] in moved_tasks_map:
            mv = moved_tasks_map[t["task_id"]]
            t["day_name"] = mv["new_day_name"]
            t["date"] = mv["new_date"]
        tasks_to_create.append(t)

    created_count = 0
    for task in tasks_to_create:
        start_dt = tz.localize(
            datetime.strptime(f"{task['date']} {task['start_time']}", "%Y-%m-%d %H:%M")
        )
        end_dt = tz.localize(
            datetime.strptime(f"{task['date']} {task['end_time']}", "%Y-%m-%d %H:%M")
        )
        description = (
            f"[schedule-agent]\n"
            f"Complexity: {task['complexity']}/5\n"
            f"Estimated: {task['estimated_hours']}h\n"
            f"{task.get('rationale', '')}"
        )
        create_task_event(
            title=f"[Task] {task['title']}",
            start=start_dt,
            end=end_dt,
            description=description,
        )
        created_count += 1

    # Save approved schedule to history
    history_dir = Path("data/approved_schedules")
    history_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_file = history_dir / f"schedule_{timestamp}.json"
    approved_proposal = {
        **proposal,
        "approved_at": datetime.now().isoformat(),
        "removed_task_ids": list(removed_ids),
        "moved_tasks": data.get("moved_tasks", []),
    }
    history_file.write_text(json.dumps(approved_proposal, indent=2), encoding="utf-8")

    # Remove pending file
    PENDING_FILE.unlink(missing_ok=True)

    # Send confirmation email
    try:
        send_email(
            subject=f"✓ Schedule approved — {created_count} events created",
            html_body=f"""
            <div style="font-family:Arial;max-width:500px;margin:0 auto;padding:24px">
              <h2 style="color:#27ae60">Schedule confirmed!</h2>
              <p style="color:#666">{created_count} calendar events were created for
              {proposal['week_start']} → {proposal['week_end']}.</p>
              <p style="color:#666;margin-top:8px">
                All events are marked as <strong>private</strong> — your team sees "Busy".
              </p>
            </div>""",
        )
    except Exception:
        pass  # Confirmation email is best-effort

    # Signal the server to shut down shortly after responding
    threading.Timer(2.0, _shutdown_event.set).start()

    return jsonify({"created": created_count})


# ─── Entry point ──────────────────────────────────────────────────────────────

def run_approval_server(open_browser: bool = True) -> None:
    """
    Starts the Flask server and optionally opens the browser.
    Blocks until the approval is submitted or TIMEOUT_HOURS elapses.
    """
    port = int(os.getenv("APPROVAL_SERVER_PORT", "5555"))
    url = f"http://localhost:{port}"

    if open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    server_thread = threading.Thread(
        target=lambda: app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False),
        daemon=True,
    )
    server_thread.start()

    print(f"  Approval UI running at {url}")
    print(f"  Waiting for your approval (auto-closes in {TIMEOUT_HOURS}h)...")

    _shutdown_event.wait(timeout=TIMEOUT_HOURS * 3600)
    print("  Approval server shut down.")
