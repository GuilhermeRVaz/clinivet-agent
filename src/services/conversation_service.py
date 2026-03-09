import re
from typing import Iterable, List, Optional

from langchain_core.messages import HumanMessage

SCHEDULE_KEYWORDS = ("agendar", "marcar", "consulta", "retorno", "vacina", "vacinacao")
CANCEL_KEYWORDS = ("cancelar", "desmarcar")
RESCHEDULE_KEYWORDS = ("remarcar", "reagendar", "mudar horario", "trocar horario")
CHECK_KEYWORDS = (
    "meu agendamento",
    "meus agendamentos",
    "tenho consulta",
    "qual horario",
    "ver agendamento",
)
HISTORY_KEYWORDS = ("historico", "prontuario", "vacinas aplicadas", "consultas anteriores")
SLOT_SUGGESTION_KEYWORDS = ("horario", "horarios", "disponivel", "disponiveis")


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

    if any(keyword in text for keyword in CANCEL_KEYWORDS):
        return "cancel"
    if any(keyword in text for keyword in RESCHEDULE_KEYWORDS):
        return "reschedule"
    if any(keyword in text for keyword in CHECK_KEYWORDS):
        return "check_appointment"
    if any(keyword in text for keyword in HISTORY_KEYWORDS):
        return "load_pet_history"
    if any(keyword in text for keyword in SCHEDULE_KEYWORDS):
        return "schedule"
    return "triage"


def extract_time_preference(message: str) -> Optional[str]:
    text = (message or "").lower()
    if "manha" in text or "manhã" in text:
        return "morning"
    if "tarde" in text:
        return "afternoon"
    if "qualquer horario" in text or "qualquer horário" in text or "qualquer" in text:
        return "any"
    return None


def extract_time_choice(message: str) -> Optional[str]:
    text = (message or "").lower().replace("h", ":")
    match = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", text)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2))
    return f"{hour:02d}:{minute:02d}"


def extract_appointment_id(message: str) -> Optional[int]:
    text = (message or "").lower()
    if extract_time_choice(text):
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
    return f"Tenho os seguintes horarios disponiveis:\n{listed}\nQual prefere?"


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
