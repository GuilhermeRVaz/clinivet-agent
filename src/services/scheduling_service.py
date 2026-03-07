from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from src.clinivet_calendar import TIMEZONE, get_calendar_service


def build_next_business_day(reference: Optional[datetime] = None) -> str:
    base = reference or datetime.now(TIMEZONE)
    return (base + timedelta(days=1)).strftime("%Y-%m-%d")


def detect_service_type(service_name: Optional[str]) -> str:
    return service_name or "Consulta"


def get_available_slots_for_service(service_name: str, day: str) -> List[str]:
    calendar_service = get_calendar_service(service_name)
    return calendar_service.get_free_slots(day)


def get_next_available_slot(service_name: str, day: str) -> Optional[str]:
    slots = get_available_slots_for_service(service_name, day)
    if not slots:
        return None
    return slots[0]


def resolve_scheduling_context(
    service_name: Optional[str], preferred_day: Optional[str] = None
) -> Tuple[str, str, List[str]]:
    resolved_service = detect_service_type(service_name)
    target_day = preferred_day or build_next_business_day()
    slots = get_available_slots_for_service(resolved_service, target_day)
    return resolved_service, target_day, slots
