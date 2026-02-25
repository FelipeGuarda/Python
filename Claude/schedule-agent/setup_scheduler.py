"""
Cross-platform scheduler setup.
Registers weekly_scheduler.py and daily_reminder.py as scheduled tasks.

Usage:
  python setup_scheduler.py           # Register scheduled tasks
  python setup_scheduler.py --remove  # Remove scheduled tasks

Supports: Windows (Task Scheduler) and Linux/macOS (crontab)
"""

import subprocess
import sys
from pathlib import Path


def get_python() -> str:
    return sys.executable


def get_project() -> Path:
    return Path(__file__).parent.resolve()


# ─── Windows ──────────────────────────────────────────────────────────────────

def _win_create(task_name: str, script: str, schedule_args: list[str]) -> None:
    project = get_project()
    python = get_python()
    script_path = project / script
    log_path = project / "data" / script.replace(".py", ".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "schtasks", "/create",
        "/tn", task_name,
        "/tr", f'cmd /c "cd /d "{project}" && "{python}" "{script_path}" >> "{log_path}" 2>&1"',
        "/f",
        "/ru", "INTERACTIVE",
    ] + schedule_args

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  ✓ '{task_name}' registered.")
    else:
        print(f"  ✗ Failed to register '{task_name}': {result.stderr.strip()}")


def _win_remove(task_name: str) -> None:
    result = subprocess.run(
        ["schtasks", "/delete", "/tn", task_name, "/f"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"  ✓ '{task_name}' removed.")
    else:
        print(f"  ✗ Could not remove '{task_name}': {result.stderr.strip()}")


def setup_windows(remove: bool = False) -> None:
    if remove:
        print("\n── Removing tasks from Windows Task Scheduler ──")
        _win_remove("ScheduleAgent-Weekly")
        _win_remove("ScheduleAgent-Daily")
        return

    print("\n── Registering tasks in Windows Task Scheduler ──\n")
    _win_create(
        "ScheduleAgent-Weekly", "weekly_scheduler.py",
        ["/sc", "WEEKLY", "/d", "MON", "/st", "15:00"],
    )
    _win_create(
        "ScheduleAgent-Daily", "daily_reminder.py",
        ["/sc", "WEEKLY", "/d", "MON,TUE,WED,THU,FRI", "/st", "08:00"],
    )
    project = get_project()
    print(f"""
  Weekly scheduler : Mondays at 15:00
  Daily reminder   : Weekdays at 08:00
  Logs → {project / 'data'}

  To test: schtasks /run /tn "ScheduleAgent-Weekly"
""")


# ─── Linux / macOS ────────────────────────────────────────────────────────────

def setup_unix(remove: bool = False) -> None:
    project = get_project()
    python = get_python()
    log_dir = project / "data"
    log_dir.mkdir(parents=True, exist_ok=True)

    marker = "# schedule-agent"

    weekly_line = (
        f"0 15 * * 1 cd \"{project}\" && \"{python}\" \"{project / 'weekly_scheduler.py'}\" "
        f">> \"{log_dir / 'weekly_scheduler.log'}\" 2>&1  {marker}-weekly"
    )
    daily_line = (
        f"0 8 * * 1-5 cd \"{project}\" && \"{python}\" \"{project / 'daily_reminder.py'}\" "
        f">> \"{log_dir / 'daily_reminder.log'}\" 2>&1  {marker}-daily"
    )

    # Read current crontab (may be empty if none exists)
    result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    current_lines = result.stdout.splitlines() if result.returncode == 0 else []

    # Remove existing schedule-agent entries
    filtered = [l for l in current_lines if marker not in l]

    if remove:
        new_crontab = "\n".join(filtered) + ("\n" if filtered else "")
        subprocess.run(["crontab", "-"], input=new_crontab, text=True)
        print("  ✓ schedule-agent cron jobs removed.")
        return

    print("\n── Registering cron jobs ──\n")
    filtered += [weekly_line, daily_line]
    new_crontab = "\n".join(filtered) + "\n"

    proc = subprocess.run(["crontab", "-"], input=new_crontab, text=True)
    if proc.returncode == 0:
        print("  ✓ ScheduleAgent-Weekly  → Mondays at 15:00")
        print("  ✓ ScheduleAgent-Daily   → Weekdays at 08:00")
        print(f"\n  Logs → {log_dir}")
        print("\n  Verify with: crontab -l")
    else:
        print("  ✗ Failed to update crontab.")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    remove = "--remove" in sys.argv
    if sys.platform == "win32":
        setup_windows(remove=remove)
    else:
        setup_unix(remove=remove)


if __name__ == "__main__":
    main()
