"""
Google OAuth handler — shared by Tasks, Calendar, and Gmail APIs.
Token is cached in GOOGLE_TOKEN_FILE and refreshed automatically.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

load_dotenv()

# All scopes needed across the full agent suite
SCOPES = [
    "https://www.googleapis.com/auth/tasks.readonly",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/gmail.send",
]


def get_credentials() -> Credentials:
    """
    Load cached credentials or run the OAuth browser flow if needed.
    Saves the refreshed token back to disk automatically.
    """
    credentials_file = os.getenv("GOOGLE_CREDENTIALS_FILE", "credentials.json")
    token_file = os.getenv("GOOGLE_TOKEN_FILE", "token.json")

    creds = None

    if Path(token_file).exists():
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not Path(credentials_file).exists():
                raise FileNotFoundError(
                    f"Google credentials file not found: {credentials_file}\n"
                    "Download it from Google Cloud Console → APIs & Services → Credentials."
                )
            flow = InstalledAppFlow.from_client_secrets_file(credentials_file, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(token_file, "w") as f:
            f.write(creds.to_json())

    return creds


def get_tasks_service():
    return build("tasks", "v1", credentials=get_credentials())


def get_calendar_service():
    return build("calendar", "v3", credentials=get_credentials())


def get_gmail_service():
    return build("gmail", "v1", credentials=get_credentials())
