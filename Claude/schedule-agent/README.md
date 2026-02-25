# Schedule Agent

An AI-powered personal scheduler that:
- Reads your **Google Tasks** every Monday after your coordination meeting
- Uses **Claude** to analyze task complexity and estimate time
- Finds free slots in your **Google Calendar** (Tue–Fri of the current week)
- Proposes a balanced week: heavy tasks in the morning, light tasks in the afternoon
- Opens a **local approval UI** for you to review and confirm
- Creates **private calendar events** — you see the task title, your team sees "Busy"
- Sends a **daily morning digest** and alerts you if your task list changes

---

## Setup Guide

### 1. Install Python dependencies

```bash
cd C:\Users\USUARIO\Dev\schedule-agent
pip install -r requirements.txt
```

### 2. Set up Google Cloud credentials

You need a Google Cloud project with three APIs enabled.

**A. Create a project:**
1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Click **Select a project** → **New Project**
3. Name it `schedule-agent`, click **Create**

**B. Enable the three APIs:**
1. Go to **APIs & Services → Library**
2. Search and enable each:
   - `Google Tasks API`
   - `Google Calendar API`
   - `Gmail API`

**C. Create OAuth credentials:**
1. Go to **APIs & Services → Credentials**
2. Click **+ Create Credentials → OAuth client ID**
3. If prompted, configure the **consent screen** first:
   - User type: **External**
   - App name: `Schedule Agent`
   - Add your Gmail as a test user
4. Back in Credentials → Create OAuth client ID:
   - Application type: **Desktop app**
   - Name: `schedule-agent`
5. Click **Download JSON**
6. Rename the file to `credentials.json` and place it in this folder

### 3. Configure your settings

```bash
cp .env.example .env
```

Edit `.env`:
```
ANTHROPIC_API_KEY=sk-ant-...    # Get from console.anthropic.com
YOUR_EMAIL=you@gmail.com
```

Edit `config.yaml`:
- Set your `timezone` (e.g. `America/Bogota`, `America/Mexico_City`, `Europe/Madrid`)
- Adjust `work_hours` if needed

### 4. Authorize Google (one-time)

```bash
python setup_auth.py
```

This opens your browser. Log in with your Google account and grant the requested permissions.
A `token.json` file is saved — you won't need to log in again.

**Optional:** To see your task list IDs (if you want to schedule from a specific list):
```bash
python setup_auth.py --list-tasklists
```
Then update `config.yaml → google_tasks.tasklist_id`.

### 5. Register the Windows scheduled tasks

```bash
python setup_windows_tasks.py
```

This registers two background tasks:
| Task | Schedule |
|---|---|
| `ScheduleAgent-Weekly` | Every Monday at 15:00 |
| `ScheduleAgent-Daily` | Every weekday at 08:00 |

To remove them later: `python setup_windows_tasks.py --remove`

---

## Usage

### Run the weekly scheduler manually (for testing)
```bash
python weekly_scheduler.py
```
1. Claude analyzes your tasks
2. Your calendar is scanned for free slots
3. A proposal is generated and emailed to you
4. Your browser opens `http://localhost:5555` with the schedule
5. Review, optionally remove tasks, click **Approve**
6. Events appear in your calendar immediately

### Run the daily reminder manually
```bash
python daily_reminder.py
```

---

## How calendar events work

Events are created with:
- **`visibility: private`** — you see the full task title and details
- **`transparency: opaque`** — marks you as "Busy" in shared views
- **Title format:** `[Task] Your task name here`
- **Description:** includes complexity score, time estimate, and Claude's rationale

Your team's calendar view shows only your busy blocks — no task details are exposed.

---

## File structure

```
schedule-agent/
├── .env                     # Your API keys (never commit this)
├── config.yaml              # Work hours, preferences, calendar/task list IDs
├── credentials.json         # Google OAuth client secret (never commit this)
├── token.json               # Auto-generated auth token (never commit this)
├── requirements.txt
│
├── src/
│   ├── auth.py              # Google OAuth handler
│   ├── google_tasks.py      # Google Tasks API
│   ├── google_calendar.py   # Google Calendar API (free slots + event creation)
│   ├── gmail_sender.py      # HTML email builder + Gmail API sender
│   ├── task_analyzer.py     # Claude: complexity scoring + time estimates
│   ├── schedule_builder.py  # Claude: weekly schedule from tasks + free slots
│   ├── approval_server.py   # Flask: local review/approve UI
│   └── change_detector.py   # Task diff: detect new/completed/modified tasks
│
├── weekly_scheduler.py      # Entry point: Monday 15:00 agent
├── daily_reminder.py        # Entry point: daily 8:00 agent
├── setup_auth.py            # One-time Google auth setup
└── setup_windows_tasks.py   # Windows Task Scheduler registration
```

---

## Troubleshooting

**"No module named 'anthropic'"**
→ Run `pip install -r requirements.txt`

**"Google credentials file not found"**
→ Download `credentials.json` from Google Cloud Console (see Step 2C above)

**"Token has been expired or revoked"**
→ Delete `token.json` and re-run `python setup_auth.py`

**Approval UI shows "No pending proposal"**
→ Run `python weekly_scheduler.py` first to generate a proposal

**Events created but not showing as private**
→ Check `config.yaml → google_calendar.event_visibility` is set to `"private"`
→ Verify your calendar sharing settings in Google Calendar settings
