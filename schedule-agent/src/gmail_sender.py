"""
Gmail API email sender.
Builds and sends HTML emails for the weekly proposal and daily digest.
"""

from __future__ import annotations

import base64
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

from dotenv import load_dotenv

from .auth import get_gmail_service

load_dotenv()


def _build_message(to: str, subject: str, html_body: str) -> dict:
    msg = MIMEMultipart("alternative")
    msg["To"] = to
    msg["From"] = to  # Sending to yourself
    msg["Subject"] = subject
    msg.attach(MIMEText(html_body, "html"))
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    return {"raw": raw}


def send_email(subject: str, html_body: str) -> None:
    recipient = os.getenv("YOUR_EMAIL")
    if not recipient:
        raise ValueError("YOUR_EMAIL not set in .env")
    service = get_gmail_service()
    message = _build_message(recipient, subject, html_body)
    service.users().messages().send(userId="me", body=message).execute()


# ─── Email Templates ──────────────────────────────────────────────────────────

COMPLEXITY_LABELS = {1: "Easy", 2: "Light", 3: "Medium", 4: "Heavy", 5: "Deep"}
COMPLEXITY_COLORS = {
    1: "#27ae60", 2: "#2ecc71", 3: "#f39c12", 4: "#e67e22", 5: "#e74c3c"
}


def _task_badge(complexity: int) -> str:
    label = COMPLEXITY_LABELS.get(complexity, "?")
    color = COMPLEXITY_COLORS.get(complexity, "#999")
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:12px;font-size:11px;font-weight:600">{label}</span>'
    )


def build_proposal_email(proposal: dict[str, Any], approval_url: str) -> str:
    """
    Builds the HTML body for the weekly schedule proposal email.
    """
    week_start = proposal["week_start"]
    week_end = proposal["week_end"]
    by_day: dict[str, list] = {}
    for task in proposal["scheduled_tasks"]:
        by_day.setdefault(task["day_name"], []).append(task)

    rows = ""
    for day, tasks in by_day.items():
        total_h = sum(t["estimated_hours"] for t in tasks)
        rows += f"""
        <tr style="background:#f8f9fa">
          <td colspan="4" style="padding:10px 16px;font-weight:700;
              font-size:14px;color:#2c3e50;border-top:2px solid #dee2e6">
            {day} &nbsp;<span style="font-weight:400;color:#6c757d;font-size:12px">
            ({total_h:.1f}h scheduled)</span>
          </td>
        </tr>"""
        for t in tasks:
            rows += f"""
        <tr>
          <td style="padding:8px 16px;color:#495057">{t['start_time']} – {t['end_time']}</td>
          <td style="padding:8px 16px;font-weight:500">{t['title']}</td>
          <td style="padding:8px 16px">{_task_badge(t['complexity'])}</td>
          <td style="padding:8px 16px;color:#6c757d;font-size:13px">{t['estimated_hours']}h</td>
        </tr>"""

    unscheduled = ""
    if proposal.get("unscheduled_tasks"):
        items = "".join(
            f'<li style="margin-bottom:4px">{t["title"]} '
            f'({_task_badge(t["complexity"])})</li>'
            for t in proposal["unscheduled_tasks"]
        )
        unscheduled = f"""
        <div style="margin-top:24px;padding:16px;background:#fff3cd;border-radius:8px;
                    border-left:4px solid #ffc107">
          <strong>Could not fit this week:</strong>
          <ul style="margin:8px 0 0 0;padding-left:20px">{items}</ul>
        </div>"""

    return f"""
    <div style="font-family:Arial,sans-serif;max-width:680px;margin:0 auto">
      <div style="background:#2c3e50;color:white;padding:24px;border-radius:8px 8px 0 0">
        <h1 style="margin:0;font-size:22px">Weekly Schedule Proposal</h1>
        <p style="margin:4px 0 0;opacity:0.8">{week_start} &rarr; {week_end}</p>
      </div>
      <div style="border:1px solid #dee2e6;border-top:none;border-radius:0 0 8px 8px;
                  overflow:hidden">
        <table width="100%" cellpadding="0" cellspacing="0"
               style="border-collapse:collapse;font-size:14px">
          <thead>
            <tr style="background:#ecf0f1">
              <th style="padding:10px 16px;text-align:left;color:#6c757d;
                         font-weight:600;font-size:12px;text-transform:uppercase">Time</th>
              <th style="padding:10px 16px;text-align:left;color:#6c757d;
                         font-weight:600;font-size:12px;text-transform:uppercase">Task</th>
              <th style="padding:10px 16px;text-align:left;color:#6c757d;
                         font-weight:600;font-size:12px;text-transform:uppercase">Level</th>
              <th style="padding:10px 16px;text-align:left;color:#6c757d;
                         font-weight:600;font-size:12px;text-transform:uppercase">Est.</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>
      </div>
      {unscheduled}
      <div style="margin-top:24px;text-align:center">
        <a href="{approval_url}"
           style="background:#27ae60;color:white;padding:14px 36px;
                  text-decoration:none;border-radius:6px;font-weight:700;
                  font-size:16px;display:inline-block">
          Review &amp; Approve Schedule
        </a>
        <p style="margin-top:12px;color:#6c757d;font-size:12px">
          Opens on your computer at <code>{approval_url}</code>
        </p>
      </div>
    </div>
    """


def build_daily_digest_email(
    today_tasks: list[dict],
    changes: list[dict] | None = None,
) -> str:
    """
    HTML body for the morning daily digest.
    """
    day_label = today_tasks[0]["day_name"] if today_tasks else "Today"

    task_rows = ""
    for t in today_tasks:
        task_rows += f"""
        <tr>
          <td style="padding:10px 16px;color:#495057;white-space:nowrap">
            {t['start_time']} – {t['end_time']}</td>
          <td style="padding:10px 16px;font-weight:500">{t['title']}</td>
          <td style="padding:10px 16px">{_task_badge(t['complexity'])}</td>
        </tr>"""

    if not task_rows:
        task_rows = """
        <tr><td colspan="3" style="padding:16px;color:#6c757d;text-align:center">
          No tasks scheduled for today.</td></tr>"""

    changes_block = ""
    if changes:
        items = "".join(
            f'<li style="margin-bottom:4px">{c["description"]}</li>' for c in changes
        )
        changes_block = f"""
        <div style="margin-top:20px;padding:16px;background:#fde8e8;border-radius:8px;
                    border-left:4px solid #e74c3c">
          <strong style="color:#c0392b">Task changes detected — schedule may need adjustment:</strong>
          <ul style="margin:8px 0 0;padding-left:20px">{items}</ul>
          <p style="margin:8px 0 0;font-size:13px;color:#6c757d">
            Run the weekly scheduler to rebuild the schedule, or wait until next Monday.
          </p>
        </div>"""

    return f"""
    <div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto">
      <div style="background:#2980b9;color:white;padding:20px 24px;border-radius:8px 8px 0 0">
        <h1 style="margin:0;font-size:20px">Good morning — {day_label}'s Plan</h1>
      </div>
      <div style="border:1px solid #dee2e6;border-top:none;border-radius:0 0 8px 8px">
        <table width="100%" cellpadding="0" cellspacing="0"
               style="border-collapse:collapse;font-size:14px">
          <thead>
            <tr style="background:#ecf0f1">
              <th style="padding:8px 16px;text-align:left;color:#6c757d;
                         font-weight:600;font-size:12px;text-transform:uppercase">Time</th>
              <th style="padding:8px 16px;text-align:left;color:#6c757d;
                         font-weight:600;font-size:12px;text-transform:uppercase">Task</th>
              <th style="padding:8px 16px;text-align:left;color:#6c757d;
                         font-weight:600;font-size:12px;text-transform:uppercase">Level</th>
            </tr>
          </thead>
          <tbody>{task_rows}</tbody>
        </table>
      </div>
      {changes_block}
    </div>
    """
