"""
Contact form service.

Replicates the original `save_to_gsheet()` function from
`Deployment/app.py` — same scopes, same target spreadsheet, same
append-row behavior — but reads credentials from application config
(environment variable / file path) instead of Streamlit's `st.secrets`,
so it's portable outside of Streamlit.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import gspread
from oauth2client.service_account import ServiceAccountCredentials

from app.core.config import Settings
from app.core.logger import get_logger

logger = get_logger(__name__)

GOOGLE_SHEETS_SCOPE = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive",
]


class ContactServiceError(RuntimeError):
    """Raised when the message could not be saved to Google Sheets."""


class ContactService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def _load_credentials_dict(self) -> dict:
        raw = self._settings.GOOGLE_SERVICE_ACCOUNT_JSON
        if not raw:
            raise ContactServiceError(
                "GOOGLE_SERVICE_ACCOUNT_JSON is not configured. "
                "Set it to a service-account JSON file path or raw JSON string."
            )

        candidate_path = Path(raw)
        if candidate_path.exists():
            return json.loads(candidate_path.read_text())

        # Fall back to treating the env var as the raw JSON content itself.
        return json.loads(raw)

    def save_message(self, name: str, email: str, message: str) -> None:
        """
        Append a row to the configured Google Sheet:
        [timestamp, name, email, message] — identical column order to the
        original implementation.
        """
        try:
            creds_dict = self._load_credentials_dict()
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, GOOGLE_SHEETS_SCOPE)
            client = gspread.authorize(creds)

            sheet = client.open_by_key(self._settings.CONTACT_SHEET_KEY).worksheet(
                self._settings.CONTACT_SHEET_WORKSHEET
            )
            sheet.append_row(
                [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    name,
                    email,
                    message,
                ]
            )
        except Exception as e:  # noqa: BLE001 - surfaced as a service-level error to the route
            logger.exception("Failed to save contact message to Google Sheets")
            raise ContactServiceError("Something went wrong. Please try again.") from e
