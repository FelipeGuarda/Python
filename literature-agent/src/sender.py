"""Send HTML email via SMTP (Gmail app password)."""

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def send(subject, html, config):
    """Send the digest email to all configured recipients."""
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_pass = os.environ.get("SMTP_PASSWORD", "")
    if not smtp_user or not smtp_pass:
        print("[WARN] SMTP_USER / SMTP_PASSWORD not set — saving HTML to file instead")
        with open("last_digest.html", "w") as f:
            f.write(html)
        print("  → Saved to last_digest.html")
        return

    email_cfg = config["email"]
    recipients = email_cfg["recipients"]
    host = email_cfg.get("smtp_host", "smtp.gmail.com")
    port = email_cfg.get("smtp_port", 587)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = ", ".join(recipients)
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP(host, port) as server:
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(smtp_user, recipients, msg.as_string())

    print(f"  Email sent to {', '.join(recipients)}")
