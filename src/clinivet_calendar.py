import json
import os
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import pytz
from googleapiclient.discovery import build
from google.oauth2 import service_account

TIMEZONE = pytz.timezone("America/Sao_Paulo")
SLOT_DURATION_MINUTES = 30
WORK_DAY_START_HOUR = 8
WORK_DAY_END_HOUR = 18
DAY_FORMAT = "%Y-%m-%d"
SLOT_FORMAT = "%H:%M"

_calendar_service: Optional["GoogleCalendarService"] = None


def _load_service_account_info() -> dict:
    payload = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not payload:
        raise EnvironmentError("GOOGLE_SERVICE_ACCOUNT_JSON nao definido.")

    payload = payload.strip()

    if os.path.isfile(payload):
        with open(payload, "r", encoding="utf-8") as file:
            return json.load(file)

    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise EnvironmentError(
            "GOOGLE_SERVICE_ACCOUNT_JSON precisa ser um JSON valido ou caminho para ele."
        ) from exc


def _round_up_to_slot(dt: datetime) -> datetime:
    dt = dt.replace(second=0, microsecond=0)
    remainder = dt.minute % SLOT_DURATION_MINUTES
    if remainder == 0:
        return dt
    return dt + timedelta(minutes=(SLOT_DURATION_MINUTES - remainder))


def build_slot_datetime(day: str, slot: str) -> datetime:
    naive = datetime.strptime(f"{day} {slot}", f"{DAY_FORMAT} {SLOT_FORMAT}")
    return TIMEZONE.localize(naive)


class GoogleCalendarService:
    def __init__(self):
        credentials_info = _load_service_account_info()
        credentials = service_account.Credentials.from_service_account_info(
            credentials_info,
            scopes=["https://www.googleapis.com/auth/calendar"],
        )
        self.client = build("calendar", "v3", credentials=credentials)

        calendar_id = os.getenv("GOOGLE_CALENDAR_ID")
        if not calendar_id:
            raise EnvironmentError("GOOGLE_CALENDAR_ID nao definido.")

        self.calendar_id = calendar_id

    def get_free_slots(self, day: str) -> List[str]:
        try:
            day_start = TIMEZONE.localize(datetime.strptime(day, DAY_FORMAT))
        except ValueError as exc:
            raise ValueError(f"Data invalida para geracao de horarios: {day}") from exc

        window_start = day_start + timedelta(hours=WORK_DAY_START_HOUR)
        window_end = day_start + timedelta(hours=WORK_DAY_END_HOUR)
        payload = {
            "timeMin": window_start.isoformat(),
            "timeMax": window_end.isoformat(),
            "timeZone": str(TIMEZONE),
            "items": [{"id": self.calendar_id}],
        }

        try:
            response = self.client.freebusy().query(body=payload).execute()
        except Exception as exc:
            raise RuntimeError("Nao foi possivel consultar o calendario do Google.") from exc

        busy_periods = response.get("calendars", {}).get(self.calendar_id, {}).get("busy", [])

        slot_duration = timedelta(minutes=SLOT_DURATION_MINUTES)
        pointer = window_start
        available: List[str] = []

        busy_intervals: List[Tuple[datetime, datetime]] = []
        for period in busy_periods:
            try:
                start = datetime.fromisoformat(period["start"]).astimezone(TIMEZONE)
                end = datetime.fromisoformat(period["end"]).astimezone(TIMEZONE)
            except (KeyError, ValueError):
                continue
            if end <= window_start or start >= window_end:
                continue
            busy_intervals.append((max(start, window_start), min(end, window_end)))

        busy_intervals.sort(key=lambda interval: interval[0])

        for busy_start, busy_end in busy_intervals:
            while pointer + slot_duration <= busy_start:
                available.append(pointer.strftime(SLOT_FORMAT))
                pointer += slot_duration
            pointer = max(pointer, _round_up_to_slot(busy_end))
            if pointer > window_end:
                break

        while pointer + slot_duration <= window_end:
            available.append(pointer.strftime(SLOT_FORMAT))
            pointer += slot_duration

        return available

    def create_event(self, summary: str, start_time: datetime, duration_minutes: int) -> str:
        start_time = start_time.astimezone(TIMEZONE)
        end_time = start_time + timedelta(minutes=duration_minutes)

        event = {
            "summary": summary,
            "start": {"dateTime": start_time.isoformat(), "timeZone": str(TIMEZONE)},
            "end": {"dateTime": end_time.isoformat(), "timeZone": str(TIMEZONE)},
        }

        try:
            created = self.client.events().insert(
                calendarId=self.calendar_id,
                body=event,
            ).execute()
        except Exception as exc:
            raise RuntimeError("Falha ao criar evento no Google Calendar.") from exc

        return created.get("id", "")


def get_calendar_service() -> "GoogleCalendarService":
    global _calendar_service
    if _calendar_service is None:
        _calendar_service = GoogleCalendarService()
    return _calendar_service
