"""
Registers both agents in Windows Task Scheduler.

Run once (as administrator is NOT required for user-level tasks):
  python setup_windows_tasks.py

To remove the scheduled tasks later:
  python setup_windows_tasks.py --remove
"""

import subprocess
import sys
from pathlib import Path


def get_python_path() -> str:
    return sys.executable


def get_project_path() -> Path:
    return Path(__file__).parent.resolve()


def create_task(task_name: str, script: str, schedule_args: list[str]) -> None:
    project = get_project_path()
    python = get_python_path()
    script_path = project / script

    # Use a wrapper so the working directory is correct
    cmd = [
        "schtasks", "/create",
        "/tn", task_name,
        "/tr", f'cmd /c "cd /d "{project}" && "{python}" "{script_path}" >> "{project}\\data\\{script.replace(".py","")}.log" 2>&1"',
        "/f",           # Force overwrite if exists
        "/ru", "INTERACTIVE",  # Run as the current logged-in user
    ] + schedule_args

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  ✓ '{task_name}' registered.")
    else:
        print(f"  ✗ Failed to register '{task_name}':")
        print(f"    {result.stderr.strip()}")


def remove_task(task_name: str) -> None:
    result = subprocess.run(
        ["schtasks", "/delete", "/tn", task_name, "/f"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"  ✓ '{task_name}' removed.")
    else:
        print(f"  ✗ Could not remove '{task_name}': {result.stderr.strip()}")


def main() -> None:
    if "--remove" in sys.argv:
        print("\n── Removing Schedule Agent tasks from Windows Task Scheduler ──")
        remove_task("ScheduleAgent-Weekly")
        remove_task("ScheduleAgent-Daily")
        print("\nDone.")
        return

    print("\n── Registering Schedule Agent in Windows Task Scheduler ──\n")

    # Weekly: every Monday at 15:00
    create_task(
        task_name="ScheduleAgent-Weekly",
        script="weekly_scheduler.py",
        schedule_args=[
            "/sc", "WEEKLY",
            "/d", "MON",
            "/st", "15:00",
        ],
    )

    # Daily: every weekday at 08:00
    create_task(
        task_name="ScheduleAgent-Daily",
        script="daily_reminder.py",
        schedule_args=[
            "/sc", "WEEKLY",
            "/d", "MON,TUE,WED,THU,FRI",
            "/st", "08:00",
        ],
    )

    project = get_project_path()
    print(f"""
Tasks registered successfully!

  Weekly scheduler : Mondays at 15:00
  Daily reminder   : Weekdays at 08:00

Logs will be saved to:
  {project}\\data\\weekly_scheduler.log
  {project}\\data\\daily_reminder.log

To view in Task Scheduler: Win+R → taskschd.msc → Task Scheduler Library
To test immediately: schtasks /run /tn "ScheduleAgent-Weekly"
""")


if __name__ == "__main__":
    main()
