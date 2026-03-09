import re
from datetime import datetime, timedelta
from typing import Iterable, List, Optional

from langchain_core.messages import HumanMessage

from src.clinivet_calendar import TIMEZONE

SCHEDULE_KEYWORDS = ("agendar", "marcar", "consulta", "retorno", "vacina", "vacinacao")
CANCEL_KEYWORDS = ("cancelar", "desmarcar")
RESCHEDULE_KEYWORDS = ("remarcar", "reagendar", "mudar horario", "trocar horario")
DATE_CHANGE_KEYWORDS = ("mudar o dia", "trocar o dia", "outro dia", "novo dia")
CHECK_KEYWORDS = (
    "meu agendamento",
    "meus agendamentos",
    "tenho consulta",
    "qual horario",
    "ver agendamento",
)
HISTORY_KEYWORDS = ("historico", "prontuario", "vacinas aplicadas", "consultas anteriores")
SLOT_SUGGESTION_KEYWORDS = ("horario", "horarios", "disponivel", "disponiveis")
CLOSING_KEYWORDS = ("obrigado", "obrigada", "valeu", "tchau", "ate logo", "até logo")


def get_latest_human_message(messages: Iterable[object]) -> str:
    for message in reversed(list(messages or [])):
        if isinstance(message, HumanMessage):
            return str(message.content).strip()
    return ""


def detect_intent(message: str, pending_action: Optional[str] = None) -> str:
    text = (message or "").strip().lower()

    if pending_action == "confirm_slot":
        return "schedule"
    if pending_action == "cancel_appointment":
        return "cancel"
    if pending_action == "reschedule_appointment":
        return "reschedule"
    if pending_action == "awaiting_time_preference":
        return "schedule"
    if pending_action == "awaiting_reschedule_date":
        return "reschedule"

    if any(keyword in text for keyword in CANCEL_KEYWORDS):
        return "cancel"
    if any(keyword in text for keyword in RESCHEDULE_KEYWORDS + DATE_CHANGE_KEYWORDS):
        return "reschedule"
    if any(keyword in text for keyword in CHECK_KEYWORDS):
        return "check_appointment"
    if any(keyword in text for keyword in HISTORY_KEYWORDS):
        return "load_pet_history"
    if any(keyword in text for keyword in SCHEDULE_KEYWORDS):
        return "schedule"
    return "triage"


def is_conversation_closing(text: str) -> bool:
    normalized = (text or "").strip().lower()
    return any(keyword in normalized for keyword in CLOSING_KEYWORDS)


def extract_time_preference(message: str) -> Optional[str]:
    text = (message or "").lower()
    if "manha" in text or "manha" in text:
        return "morning"
    if "tarde" in text:
        return "afternoon"
    if "qualquer horario" in text or "qualquer horario" in text or "qualquer" in text:
        return "any"
    return None


def normalize_time_input(raw_value: str) -> Optional[str]:
    text = (raw_value or "").strip().lower()
    if not text:
        return None

    hour_only_match = re.fullmatch(r"([01]?\d|2[0-3])\s*h?", text)
    if hour_only_match:
        hour = int(hour_only_match.group(1))
        return f"{hour:02d}:00"

    separated_match = re.search(r"\b([01]?\d|2[0-3])\s*[:h\.]\s*(\d{1,2})\b", text)
    if separated_match:
        hour = int(separated_match.group(1))
        minute_text = separated_match.group(2)
        minute = int(minute_text) * 10 if len(minute_text) == 1 else int(minute_text)
        if 0 <= minute <= 59:
            return f"{hour:02d}:{minute:02d}"
        return None

    compact_match = re.search(r"\b(\d{3,4})\b", text)
    if compact_match:
        compact = compact_match.group(1)
        if len(compact) == 3:
            hour = int(compact[0])
            minute = int(compact[1:])
        else:
            hour = int(compact[:2])
            minute = int(compact[2:])
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return f"{hour:02d}:{minute:02d}"

    return None


def extract_time_choice(message: str) -> Optional[str]:
    return normalize_time_input(message)


def parse_natural_date(text: str, reference: Optional[datetime] = None) -> Optional[str]:
    raw_text = (text or "").strip().lower()
    if not raw_text:
        return None

    now = reference or datetime.now(TIMEZONE)
    normalized = (
        raw_text.replace("amanha", "amanha")
        .replace("manha", "manha")
        .replace("fevereiro", "fevereiro")
        .replace("marco", "marco")
        .replace("terça", "terca")
        .replace("terça-feira", "terca")
        .replace("quarta-feira", "quarta")
        .replace("quinta-feira", "quinta")
        .replace("sexta-feira", "sexta")
    )

    if "amanha" in normalized:
        return (now + timedelta(days=1)).strftime("%Y-%m-%d")

    weekday_map = {
        "segunda": 0,
        "terca": 1,
        "quarta": 2,
        "quinta": 3,
        "sexta": 4,
        "sabado": 5,
        "domingo": 6,
    }
    for label, weekday in weekday_map.items():
        if label in normalized:
            delta = (weekday - now.weekday()) % 7
            delta = 7 if delta == 0 else delta
            return (now + timedelta(days=delta)).strftime("%Y-%m-%d")

    numeric_match = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", normalized)
    if numeric_match:
        day = int(numeric_match.group(1))
        month = int(numeric_match.group(2))
        year = int(numeric_match.group(3))
        try:
            return TIMEZONE.localize(datetime(year, month, day)).strftime("%Y-%m-%d")
        except ValueError:
            return None

    month_map = {
        "janeiro": 1,
        "fevereiro": 2,
        "marco": 3,
        "abril": 4,
        "maio": 5,
        "junho": 6,
        "julho": 7,
        "agosto": 8,
        "setembro": 9,
        "outubro": 10,
        "novembro": 11,
        "dezembro": 12,
    }
    text_month_match = re.search(r"\b(\d{1,2})\s+de\s+([a-zç]+)\b", normalized)
    if text_month_match:
        day = int(text_month_match.group(1))
        month_name = text_month_match.group(2).replace("ç", "c")
        month = month_map.get(month_name)
        if month is None:
            return None
        year = now.year
        try:
            candidate = TIMEZONE.localize(datetime(year, month, day))
        except ValueError:
            return None
        if candidate.date() < now.date():
            candidate = TIMEZONE.localize(datetime(year + 1, month, day))
        return candidate.strftime("%Y-%m-%d")

    day_only_match = re.search(r"\bdia\s+(\d{1,2})\b", normalized)
    if day_only_match:
        day = int(day_only_match.group(1))
        month = now.month
        year = now.year
        try:
            candidate = TIMEZONE.localize(datetime(year, month, day))
        except ValueError:
            return None
        if candidate.date() < now.date():
            month += 1
            if month > 12:
                month = 1
                year += 1
            try:
                candidate = TIMEZONE.localize(datetime(year, month, day))
            except ValueError:
                return None
        return candidate.strftime("%Y-%m-%d")

    return None


def extract_appointment_id(message: str) -> Optional[int]:
    text = (message or "").lower()
    normalized_time = extract_time_choice(text)
    explicit_time_marker = any(marker in text for marker in (":", "h", "."))
    compact_time_digits = bool(re.fullmatch(r"\s*\d{3,4}\s*", text))
    if normalized_time and (explicit_time_marker or compact_time_digits):
        return None
    match = re.search(r"\b(?:id\s*)?(\d{1,10})\b", text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def wants_slot_suggestions(message: str) -> bool:
    text = (message or "").lower()
    return any(keyword in text for keyword in SLOT_SUGGESTION_KEYWORDS)


def format_slots_message(slots: List[str]) -> str:
    listed = "\n".join(slots)
    return f"Encontrei estes horarios disponiveis:\n{listed}\nQual deles voce prefere?"


def format_appointments_message(appointments: List[dict]) -> str:
    if not appointments:
        return "Nao encontrei agendamentos ativos para este numero."

    lines = []
    for appointment in appointments:
        lines.append(
            f"ID {appointment['id']}: {appointment['service_name']} em "
            f"{appointment['appointment_br']}"
        )
    return "Encontrei estes agendamentos:\n" + "\n".join(lines)


def format_pet_history_message(history: dict) -> str:
    pets = history.get("pets") or []
    consultations = history.get("consultations") or []
    vaccines = history.get("vaccines") or []

    if not pets:
        return "Nao encontrei historico veterinario para este numero."

    pet_names = ", ".join(pet.get("name", "Pet sem nome") for pet in pets)
    message_parts = [f"Pets encontrados: {pet_names}."]

    if consultations:
        message_parts.append(f"Consultas registradas: {len(consultations)}.")
    else:
        message_parts.append("Ainda nao ha consultas registradas.")

    if vaccines:
        message_parts.append(f"Vacinas registradas: {len(vaccines)}.")
    else:
        message_parts.append("Ainda nao ha vacinas registradas.")

    return " ".join(message_parts)
