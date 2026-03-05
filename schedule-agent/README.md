# Schedule Agent

**Owner:** Felipe Guarda — Fundación Mar Adentro
**Source project:** `/home/fguarda/Dev/Python/schedule-agent/` (self-contained here)
**Status:** Feature-complete. Deployed on Linux via cron. Originally built for Windows Task Scheduler.

---

## What This Project Does

An AI-powered personal scheduling agent that runs every Monday at 15:00. It:

1. Reads all open tasks from **Google Tasks** (across all task lists)
2. Uses **Claude Haiku** to score each task for complexity (1–5), time estimate, and best time of day
3. Scans **Google Calendar** for free slots Tuesday–Friday of the current week
4. Uses **Claude Sonnet** to build a balanced weekly schedule (heavy tasks in the morning, light in the afternoon)
5. Saves the proposal as JSON, sends an **HTML preview email** via Gmail API
6. Opens a **local Flask UI** at `http://localhost:5555` for review and approval
7. On approval, creates **private calendar events** — task titles visible to you, team sees only "Busy"

A second agent (`daily_reminder.py`) runs every weekday at 08:00: sends a morning digest + detects if the task list changed since Monday.

---

## Current State (as of March 2026)

### What works
- Full end-to-end pipeline: fetch → analyze → find slots → build schedule → email → approve → calendar events
- Personal vs. work task separation: tasks from `personal_tasklists` are scheduled in personal hours (after-work), not during 09:00–18:00
- Config-driven: all parameters in `config.yaml` (hours, thresholds, calendar IDs, Claude models)
- Change detection: Monday snapshot is diffed daily to detect new/completed/modified tasks
- Flask approval server with per-task "remove" controls before final approval

### Models used
- Task analysis: `claude-haiku-4-5` (fast, cheap, structured JSON output)
- Schedule building: `claude-sonnet-4-5` (needs reasoning for slot fitting)

### Deployment
- **Linux (current):** cron jobs — `python weekly_scheduler.py` at Mon 15:00, `python daily_reminder.py` at weekday 08:00
- **Windows (original):** Windows Task Scheduler via `setup_windows_tasks.py`

---

## Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.11 |
| AI (task analysis + scheduling) | Anthropic Claude API (`anthropic` SDK) |
| Google APIs | Google Tasks, Calendar, Gmail (OAuth 2.0) |
| Local approval UI | Flask (localhost:5555) |
| Email | Gmail API (HTML emails) |
| Config | YAML (`config.yaml`) |
| Auth | OAuth 2.0 — `credentials.json` + `token.json` |
| Secrets | `.env` file (ANTHROPIC_API_KEY, YOUR_EMAIL) |

---

## File Structure

```
schedule-agent/
├── .env                       ← ANTHROPIC_API_KEY, YOUR_EMAIL (never commit)
├── config.yaml                ← all tunable parameters
├── credentials.json           ← Google OAuth client secret (never commit)
├── token.json                 ← auto-generated auth token (never commit)
├── requirements.txt
│
├── src/
│   ├── auth.py                ← Google OAuth handler (token refresh)
│   ├── google_tasks.py        ← fetch open tasks from all/specific lists
│   ├── google_calendar.py     ← find free slots + create calendar events
│   ├── gmail_sender.py        ← build HTML email + send via Gmail API
│   ├── task_analyzer.py       ← Claude Haiku: complexity + time + time_of_day
│   ├── schedule_builder.py    ← Claude Sonnet: assign tasks to free slots
│   ├── approval_server.py     ← Flask: review/approve UI + creates calendar events
│   └── change_detector.py     ← diff Monday snapshot vs. current task list
│
├── weekly_scheduler.py        ← entry point: Monday 15:00 agent
├── daily_reminder.py          ← entry point: daily 08:00 digest + change alert
├── setup_auth.py              ← one-time Google OAuth flow
└── setup_scheduler.py         ← register cron (Linux) or Task Scheduler (Windows)
```

---

## Data Flow

```
Google Tasks API
    → fetch_open_tasks() → [{title, notes, due, tasklist}]
    →
Claude Haiku: analyze_tasks()
    → [{title, complexity:1-5, estimated_hours, time_of_day:morning|afternoon|any}]
    →
Google Calendar API: find_free_slots()
    → [{start, end, duration_minutes, is_morning}] for Tue–Fri
    →
Claude Sonnet: build_schedule()
    → {scheduled_tasks:[{task, slot, start, end}], unscheduled_tasks:[...]}
    →
data/pending_proposal.json (saved)
    → build_proposal_email() → Gmail API → HTML email sent
    → Flask server opens at localhost:5555
    →
User reviews, optionally removes tasks, clicks Approve
    →
Google Calendar API: create events (visibility: private, transparency: opaque)
```

---

## Configuration Reference (`config.yaml`)

| Setting | Current value | Description |
|---|---|---|
| `timezone` | `America/Santiago` | Local timezone |
| `work_hours.start/end` | `09:00–18:00` | Work window Mon–Thu |
| `work_hours.friday_end` | `14:00` | Friday ends early |
| `personal_hours.weekday_start` | `18:00` | Personal tasks after work |
| `personal_tasklists` | `["Personales"]` | Task lists treated as personal |
| `heavy_task_threshold` | `3` | Complexity ≥ 3 → morning slot |
| `max_deep_work_hours_per_day` | `5` | Caps deep work before overflow |
| `buffer_minutes` | `15` | Gap between scheduled events |
| `google_tasks.max_tasks` | `15` | Max tasks per weekly run |
| `claude.analysis_model` | `claude-haiku-4-5` | For task scoring |
| `claude.scheduler_model` | `claude-sonnet-4-5` | For schedule building |

---

## Ideas & Future Features

### High priority
- **Carry-over detection**: If a scheduled task wasn't completed (event still on calendar), auto-reschedule to next week
- **Task priority tagging**: Allow `[P1]` or `[urgent]` in task notes to force early morning slots
- **Approval via email link**: Approve the schedule directly from the email, without opening a browser

### Medium priority
- **Duration feedback**: After a week, ask if time estimates were accurate — use feedback to calibrate future estimates
- **Recurring task awareness**: Detect weekly recurring tasks (same title every week) and give them a reserved slot
- **Slack/Teams notification**: Post schedule summary to a Slack DM instead of / in addition to email

### Future / research
- **Calendar learning**: Analyze 4+ weeks of created events vs. completed tasks to improve scheduling heuristics automatically
- **Energy model**: Integrate user's own focus pattern (e.g., "I'm sharper on Tue/Wed mornings") to weight slot selection
- **Multi-person mode**: Share the proposal with a manager or assistant for collaborative approval

---

## Key Design Decisions (rationale for future devs)

1. **Two-model strategy**: Haiku for analysis (cheap, fast, many tasks), Sonnet for scheduling (needs reasoning to fit tasks into a constrained week).
2. **Local approval UI**: Approval happens on `localhost` intentionally — no cloud hosting needed. The email drives the user to open their own browser.
3. **Private + opaque events**: `visibility: private` hides task details from calendar shares. `transparency: opaque` marks the user as Busy. Both are set explicitly.
4. **Personal hours separation**: Work tasks and personal tasks never share time blocks. Enforced at the slot-finding stage.
5. **Structured JSON from Claude**: Both Claude calls request strict JSON with defined schemas. The calling code validates and falls back gracefully on parse errors.
6. **Flask for approval**: Minimal footprint — only Flask needed. Starts on a free port, opens the browser, exits cleanly after approval.

---

## Setup (quick reference)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure
cp .env.example .env     # add ANTHROPIC_API_KEY and YOUR_EMAIL
# Edit config.yaml for timezone, work hours, calendar IDs

# 3. Google auth (one-time, opens browser)
python setup_auth.py

# 4. Run manually to test
python weekly_scheduler.py    # full Monday flow
python daily_reminder.py      # daily digest

# 5. Register scheduled job
python setup_scheduler.py
```

---

## Context for AI Sessions

When starting a new Claude session on this project:

1. Two entry points: `weekly_scheduler.py` (Monday agent) and `daily_reminder.py` (daily digest)
2. All behavior parameters are in `config.yaml` — modify there, not in source code
3. Claude calls are in `src/task_analyzer.py` (Haiku, batch analysis) and `src/schedule_builder.py` (Sonnet, slot assignment) — both use `anthropic` SDK with structured JSON prompts
4. The approval server is `src/approval_server.py` — Flask app that creates calendar events on POST to `/approve`
5. Secrets: `ANTHROPIC_API_KEY` and `YOUR_EMAIL` in `.env`. Google credentials in `credentials.json` + `token.json`
6. The pending proposal is always at `data/pending_proposal.json` — what the Flask UI reads
7. `src/change_detector.py` compares the current task list against `data/tasks_snapshot.json` (saved every Monday)
