import re
from typing import List, Optional

from src.models.triage_model import TriageOutput

DEFAULT_SERVICE = "Consulta"

REQUIRED_FIELDS_FOR_APPOINTMENT = ["pet_name", "tutor_name", "phone"]
MISSING_FIELD_QUESTIONS = {
    "pet_name": "Entendi. Qual e o nome do seu pet?",
    "tutor_name": "Perfeito. Qual e o seu nome?",
    "phone": "Voce poderia me informar um telefone para contato?",
}


def clean_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned if cleaned else None


def extract_phone_candidate(raw_value: Optional[str]) -> Optional[str]:
    cleaned = clean_text(raw_value)
    if not cleaned:
        return None
    digits = re.sub(r"\D", "", cleaned)
    if len(digits) < 10:
        return None
    return digits


def normalize_triage_result(triage: TriageOutput) -> TriageOutput:
    triage.tutor_name = clean_text(triage.tutor_name)
    triage.pet_name = clean_text(triage.pet_name)
    triage.symptoms_summary = clean_text(triage.symptoms_summary)
    triage.service_suggested = clean_text(triage.service_suggested) or DEFAULT_SERVICE
    triage.phone = extract_phone_candidate(triage.phone)
    return triage


def is_species_missing(species: Optional[str]) -> bool:
    return species in (None, "Desconhecido")


def merge_triage_data(previous: Optional[TriageOutput], current: TriageOutput) -> TriageOutput:
    if not previous:
        return current

    merged_species = current.pet_species
    if is_species_missing(merged_species) and not is_species_missing(previous.pet_species):
        merged_species = previous.pet_species

    merged = TriageOutput(
        tutor_name=current.tutor_name or previous.tutor_name,
        pet_name=current.pet_name or previous.pet_name,
        pet_species=merged_species,
        urgency_level=(
            "emergency"
            if "emergency" in (current.urgency_level, previous.urgency_level)
            else "routine"
        ),
        service_suggested=(
            clean_text(current.service_suggested)
            or clean_text(previous.service_suggested)
            or DEFAULT_SERVICE
        ),
        symptoms_summary=current.symptoms_summary or previous.symptoms_summary,
        phone=current.phone or previous.phone,
    )

    return normalize_triage_result(merged)


def get_missing_required_fields(triage: TriageOutput, thread_id: Optional[str]) -> List[str]:
    missing: List[str] = []
    effective_phone = triage.phone or extract_phone_candidate(thread_id)

    if not triage.pet_name:
        missing.append("pet_name")

    if not triage.tutor_name:
        missing.append("tutor_name")

    if not effective_phone:
        missing.append("phone")

    return missing


def build_missing_data_message(missing_fields: List[str]) -> str:
    if not missing_fields:
        return "Perfeito. Vamos continuar."

    return MISSING_FIELD_QUESTIONS[missing_fields[0]]
