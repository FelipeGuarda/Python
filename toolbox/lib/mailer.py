"""How FMA sends personalised bulk mail.

This module owns the sending contract: where credentials come from, how a
recipient row becomes an HTML message, the throttle between sends, and the
ledger that makes a re-run resume instead of double-sending.

Callers supply a contact workbook and a template name. They never touch
smtplib, MIME, or column names — recipients arrive via `rosters.Roster`, so
a workbook that labels its address column differently is that module's
problem, not this one's.
"""

from __future__ import annotations

import os
import smtplib
import time
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .rosters import KEY, Roster, load_roster

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = ROOT / "templates"
DATA_DIR = ROOT / "data"

# Gmail drops connections that send too fast. Two seconds matches what the
# 2025 Fondo FMA run used without being throttled.
THROTTLE_SECONDS = 2.0

# Rows whose gender cell starts with this letter are addressed "Estimado".
_MASCULINE = "m"
_GENDER_HINTS = ("genero", "género", "gender", "sexo")

# Signature block fields. Overridable in .env so a campaign sent from
# convocatorias@ does not carry a personal name, without editing templates.
_FIRMA_DEFAULTS = {
    "nombre": "Equipo FMA",
    "cargo": "Fundación Mar Adentro",
    "telefono": "+56 2 2322 4286",
    "direccion": "Don Carlos 3171 of. C, Las Condes",
}


def signature_fields() -> dict:
    """Signature values, from .env where set. Needs no SMTP credentials, so
    previews render identically to what will actually be sent."""
    load_dotenv(ROOT / ".env")
    return {
        key: os.getenv(f"FIRMA_{key.upper()}", "").strip() or default
        for key, default in _FIRMA_DEFAULTS.items()
    }


def _template_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=True,
        undefined=StrictUndefined,  # a typo'd variable must fail loudly, not send blank
    )


@dataclass(frozen=True)
class Credentials:
    user: str
    password: str
    host: str
    port: int
    sender_name: str

    @classmethod
    def from_env(cls) -> "Credentials":
        """Read SMTP settings from toolbox/.env.

        Raises RuntimeError when unset: an unauthenticated run would fail on
        every recipient, so it must abort before the first send.
        """
        load_dotenv(ROOT / ".env")
        user = os.getenv("SMTP_USER", "").strip()
        password = os.getenv("SMTP_APP_PASSWORD", "").strip()
        if not user or not password:
            raise RuntimeError(
                "SMTP_USER and SMTP_APP_PASSWORD must be set in toolbox/.env "
                "(copy .env.example and fill it in)."
            )
        return cls(
            user=user,
            password=password,
            host=os.getenv("SMTP_HOST", "smtp.gmail.com").strip(),
            port=int(os.getenv("SMTP_PORT", "465")),
            sender_name=os.getenv("SENDER_NAME", "").strip() or user,
        )


@dataclass
class SendLog:
    """Outcome of a send, one row per recipient."""

    campaign: str
    entries: list[dict] = field(default_factory=list)

    @property
    def sent(self) -> int:
        return sum(1 for e in self.entries if e["status"] == "sent")

    @property
    def failed(self) -> list[dict]:
        return [e for e in self.entries if e["status"] == "failed"]

    @property
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.entries, columns=["email", "status", "detail", "timestamp"])


def _ledger_path(campaign: str) -> Path:
    return DATA_DIR / f"sent_{campaign}.csv"


def _already_sent(campaign: str) -> set[str]:
    path = _ledger_path(campaign)
    if not path.exists():
        return set()
    ledger = pd.read_csv(path)
    return set(ledger.loc[ledger["status"] == "sent", "email"].astype(str))


def _append_ledger(campaign: str, entry: dict) -> None:
    path = _ledger_path(campaign)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([entry]).to_csv(path, mode="a", header=not path.exists(), index=False)


@dataclass
class Campaign:
    """A template plus the people it will be rendered for."""

    roster: Roster
    template: str
    subject: str
    attachments: list[Path] = field(default_factory=list)
    inline_images: list[Path] = field(default_factory=list)

    @property
    def name(self) -> str:
        """Ledger key. Derived from the template so re-runs of the same
        campaign resume, while a different template starts a fresh ledger."""
        return Path(self.template).name.split(".")[0]

    @classmethod
    def from_excel(
        cls,
        path: str | Path,
        template: str,
        subject: str,
        key: str | None = None,
        sheet: str | int | None = None,
        attachments: list[str | Path] | None = None,
        inline_images: list[str | Path] | None = None,
    ) -> "Campaign":
        return cls(
            roster=load_roster(path, key=key, sheet=sheet),
            template=template,
            subject=subject,
            attachments=[Path(p) for p in (attachments or [])],
            inline_images=[Path(p) for p in (inline_images or [])],
        )

    def _context(self, row: pd.Series) -> dict:
        """Template variables for one recipient.

        Columns are exposed by their own names so `{{ Nombre }}` works, and
        also under `row` so headers with spaces or accents stay reachable as
        `{{ row['Correo electrónico'] }}`.
        """
        cells = {k: ("" if pd.isna(v) else v) for k, v in row.items() if k != KEY}
        context = dict(cells)
        context["row"] = cells
        context["email"] = row[KEY]
        context["saludo"] = self._saludo(cells)
        context["images"] = [p.name for p in self.inline_images]
        context["firma"] = signature_fields()
        return context

    @staticmethod
    def _saludo(cells: dict) -> str:
        for column, value in cells.items():
            if str(column).strip().lower() in _GENDER_HINTS:
                text = str(value).strip().lower()
                if text.startswith(_MASCULINE):
                    return "Estimado"
                if text:
                    return "Estimada"
        return "Estimado/a"

    def render(self, row: pd.Series) -> tuple[str, str]:
        """Returns (subject, html_body) for one recipient."""
        env = _template_env()
        context = self._context(row)
        subject = env.from_string(self.subject).render(**context)
        body = env.get_template(self.template).render(**context)
        return subject, body

    def _build(self, row: pd.Series, credentials: Credentials) -> tuple[str, MIMEMultipart]:
        recipient = row[KEY]
        subject, body = self.render(row)

        message = MIMEMultipart("related")
        message["From"] = formataddr((credentials.sender_name, credentials.user))
        message["To"] = recipient
        message["Subject"] = subject

        alternative = MIMEMultipart("alternative")
        alternative.attach(MIMEText(body, "html", "utf-8"))
        message.attach(alternative)

        for image in self.inline_images:
            part = MIMEImage(image.read_bytes())
            # Templates reference these as <img src="cid:filename.png">.
            part.add_header("Content-ID", f"<{image.name}>")
            part.add_header("Content-Disposition", "inline", filename=image.name)
            message.attach(part)

        for attachment in self.attachments:
            part = MIMEApplication(attachment.read_bytes())
            part.add_header("Content-Disposition", "attachment", filename=attachment.name)
            message.attach(part)

        return recipient, message

    def preview(self, n: int = 3) -> Path:
        """Render the first `n` recipients to one HTML file and open it.

        Sends nothing and needs no credentials.
        """
        rows = self.roster.unique.head(n)
        blocks = []
        for _, row in rows.iterrows():
            subject, body = self.render(row)
            blocks.append(
                f'<section style="border:1px solid #ccc;margin:24px;padding:16px">'
                f'<p style="font:13px monospace;color:#555">'
                f'To: {row[KEY]}<br>Subject: {subject}</p><hr>{body}</section>'
            )
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = DATA_DIR / f"preview_{self.name}.html"
        path.write_text(
            f"<!doctype html><meta charset='utf-8'>"
            f"<title>Preview — {self.name}</title>"
            f"<p style='font:14px sans-serif'>Previewing {len(rows)} of "
            f"{len(self.roster)} recipients. Nothing has been sent.</p>"
            + "".join(blocks),
            encoding="utf-8",
        )
        webbrowser.open(path.as_uri())
        return path

    def send(self, confirm: bool = False, throttle: float = THROTTLE_SECONDS) -> SendLog:
        """Send to every recipient not already in the ledger.

        Without `confirm=True` this is a dry run: it renders everything,
        reports what would go out, and opens no connection. Recipients that
        fail individually are recorded and the run continues; a failed login
        aborts before anything is sent.
        """
        log = SendLog(campaign=self.name)
        done = _already_sent(self.name)
        pending = self.roster.unique[~self.roster.unique[KEY].isin(done)]

        if done:
            print(f"Ledger: {len(done)} already sent, {len(pending)} remaining.")

        if not confirm:
            for _, row in pending.iterrows():
                subject, _ = self.render(row)
                log.entries.append({
                    "email": row[KEY],
                    "status": "dry-run",
                    "detail": subject,
                    "timestamp": "",
                })
            print(f"DRY RUN — {len(pending)} message(s) would be sent. "
                  f"Pass confirm=True (or --send) to deliver.")
            return log

        credentials = Credentials.from_env()
        with smtplib.SMTP_SSL(credentials.host, credentials.port) as server:
            server.login(credentials.user, credentials.password)
            for position, (_, row) in enumerate(pending.iterrows(), start=1):
                recipient, message = self._build(row, credentials)
                try:
                    server.send_message(message)
                    entry = {"email": recipient, "status": "sent", "detail": message["Subject"]}
                except smtplib.SMTPException as error:
                    entry = {"email": recipient, "status": "failed", "detail": str(error)}
                entry["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                _append_ledger(self.name, entry)
                log.entries.append(entry)
                print(f"[{position}/{len(pending)}] {entry['status']}: {recipient}")
                if position < len(pending):
                    time.sleep(throttle)

        print(f"Done — {log.sent} sent, {len(log.failed)} failed. "
              f"Ledger: {_ledger_path(self.name)}")
        return log
