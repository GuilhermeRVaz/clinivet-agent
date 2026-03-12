import re
from typing import List, Optional

from src.models.triage_model import TriageOutput
from src.services.conversation_service import detect_species_from_message

DEFAULT_SERVICE = "Consulta"
UNKNOWN_PET_SIZE = "unknown"

REQUIRED_FIELDS_FOR_APPOINTMENT = ["pet_name", "tutor_name", "phone"]
MISSING_FIELD_QUESTIONS = {
    "pet_name": "Entendi. Qual e o nome do seu pet?",
    "tutor_name": "Perfeito. Qual e o seu nome?",
    "tutor_cpf": "Para continuar, voce pode me informar o CPF do tutor?",
    "pet_species": "Qual e a especie do seu pet? Pode me dizer se e cao ou gato?",
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


def normalize_cpf(raw_value: Optional[str]) -> Optional[str]:
    cleaned = clean_text(raw_value)
    if not cleaned:
        return None
    digits = re.sub(r"\D", "", cleaned)
    if len(digits) != 11:
        return None
    return digits


def classify_pet_size(weight: Optional[float]) -> str:
    if weight is None:
        return UNKNOWN_PET_SIZE
    if weight <= 10:
        return "pequeno"
    if weight <= 25:
        return "medio"
    return "grande"


def normalize_triage_result(triage: TriageOutput) -> TriageOutput:
    triage.tutor_name = clean_text(triage.tutor_name)
    triage.tutor_cpf = normalize_cpf(triage.tutor_cpf)
    triage.pet_name = clean_text(triage.pet_name)
    triage.symptoms_summary = clean_text(triage.symptoms_summary)
    triage.service_suggested = clean_text(triage.service_suggested) or DEFAULT_SERVICE
    triage.phone = extract_phone_candidate(triage.phone)
    triage.pet_breed = clean_text(triage.pet_breed)
    triage.pet_age = clean_text(triage.pet_age)

    if triage.pet_weight is not None:
        try:
            triage.pet_weight = float(triage.pet_weight)
        except (TypeError, ValueError):
            triage.pet_weight = None

    triage.pet_size = classify_pet_size(triage.pet_weight)
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
        tutor_cpf=current.tutor_cpf or previous.tutor_cpf,
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
        pet_weight=current.pet_weight if current.pet_weight is not None else previous.pet_weight,
        pet_breed=current.pet_breed or previous.pet_breed,
        pet_age=current.pet_age or previous.pet_age,
        pet_size=current.pet_size or previous.pet_size,
    )

    return normalize_triage_result(merged)


def get_missing_required_fields(
    triage: TriageOutput,
    thread_id: Optional[str],
    require_onboarding_fields: bool = False,
) -> List[str]:
    missing: List[str] = []
    effective_phone = triage.phone or extract_phone_candidate(thread_id)

    if require_onboarding_fields:
        if not triage.tutor_name:
            missing.append("tutor_name")
        if not triage.tutor_cpf:
            missing.append("tutor_cpf")
        if not triage.pet_name:
            missing.append("pet_name")
        if is_species_missing(triage.pet_species):
            missing.append("pet_species")
    else:
        if not triage.pet_name:
            missing.append("pet_name")
        if not triage.tutor_name:
            missing.append("tutor_name")

    if not effective_phone:
        missing.append("phone")

    return missing


def build_missing_data_message(
    missing_fields: List[str],
    *,
    guided_onboarding: bool = False,
) -> str:
    if not missing_fields:
        return "Perfeito. Vamos continuar."

    question = MISSING_FIELD_QUESTIONS[missing_fields[0]]
    if not guided_onboarding:
        return question

    return (
        "Para agilizar seu atendimento, vou precisar de alguns dados essenciais. "
        "Se puder, me responda de forma direta. "
        f"{question}"
    )


def has_plausible_field_answer(field_name: str, message: Optional[str]) -> bool:
    cleaned = clean_text(message)
    if not cleaned:
        return False

    if field_name == "tutor_cpf":
        return normalize_cpf(cleaned) is not None

    if field_name == "phone":
        return extract_phone_candidate(cleaned) is not None

    if field_name == "pet_species":
        return detect_species_from_message(cleaned) is not None

    letters = re.findall(r"[A-Za-zÀ-ÿ]+", cleaned)
    return bool(letters)



