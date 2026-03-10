import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from dotenv import load_dotenv
from supabase import Client, create_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ClinivetDB")

BASE_DIR = Path(__file__).resolve().parents[1]
DOTENV_PATH = BASE_DIR / ".env"
ENV_SUPABASE_URL_BEFORE_DOTENV = os.getenv("SUPABASE_URL")

dotenv_loaded = load_dotenv(dotenv_path=DOTENV_PATH, override=True)
if not dotenv_loaded:
    logger.warning(f".env nao carregado automaticamente em: {DOTENV_PATH}")
else:
    logger.info(f".env carregado de: {DOTENV_PATH}")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Optional[Client] = None

if ENV_SUPABASE_URL_BEFORE_DOTENV and ENV_SUPABASE_URL_BEFORE_DOTENV != SUPABASE_URL:
    logger.warning(
        "SUPABASE_URL preexistente no ambiente foi sobrescrita pelo valor do .env."
    )

logger.info(f"SUPABASE_URL lida do ambiente: {SUPABASE_URL}")


def get_supabase_client() -> Client:
    global supabase
    if supabase is not None:
        return supabase

    if not SUPABASE_URL or not SUPABASE_KEY:
        raise EnvironmentError(
            f"SUPABASE_URL ou SUPABASE_KEY nao definidos. Verifique o arquivo .env em: {DOTENV_PATH}"
        )

    if not SUPABASE_URL.startswith("https://"):
        raise ValueError(
            f"SUPABASE_URL invalida: '{SUPABASE_URL}'. Ela deve comecar com 'https://'."
        )

    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as exc:
        raise ConnectionError(
            "Falha ao inicializar cliente Supabase. "
            f"SUPABASE_URL usada: '{SUPABASE_URL}'. Erro original: {exc}"
        ) from exc

    logger.info("Cliente Supabase inicializado com sucesso.")
    return supabase


def register_lead(
    tutor_name: str,
    tutor_cpf: Optional[str],
    pet_name: str,
    pet_species: str,
    phone: str,
    pet_weight: Optional[float] = None,
    pet_breed: Optional[str] = None,
    pet_age: Optional[str] = None,
    pet_size: Optional[str] = None,
) -> int:
    data = {
        "tutor_name": tutor_name,
        "tutor_cpf": tutor_cpf,
        "pet_name": pet_name,
        "pet_species": pet_species,
        "phone": phone,
        "status": "Interessado",
        "pet_weight": pet_weight,
        "pet_breed": pet_breed,
        "pet_age": pet_age,
        "pet_size": pet_size,
    }

    response = get_supabase_client().table("leads").insert(data).execute()

    if not response.data:
        logger.error(f"Erro ao registrar lead: {response}")
        raise Exception("Falha ao registrar lead.")

    lead_id = response.data[0]["id"]
    logger.info(f"LEAD CREATED: {lead_id}")
    return lead_id


def get_lead_by_phone(phone: str) -> Optional[dict]:
    response = (
        get_supabase_client()
        .table("leads")
        .select("*")
        .eq("phone", phone)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    if not response.data:
        return None
    return response.data[0]


def upsert_pet_profile(
    tutor_phone: str,
    name: str,
    species: Optional[str] = None,
    breed: Optional[str] = None,
    age: Optional[str] = None,
    weight: Optional[float] = None,
    size: Optional[str] = None,
) -> Optional[dict]:
    if not tutor_phone or not name:
        return None

    client = get_supabase_client()
    response = (
        client.table("pets")
        .select("*")
        .eq("tutor_phone", tutor_phone)
        .eq("name", name)
        .limit(1)
        .execute()
    )

    payload = {
        "tutor_phone": tutor_phone,
        "name": name,
        "species": species,
        "breed": breed,
        "age": age,
        "weight": weight,
        "size": size,
    }
    payload = {key: value for key, value in payload.items() if value is not None}

    if response.data:
        pet = response.data[0]
        updated = client.table("pets").update(payload).eq("id", pet["id"]).execute()
        return updated.data[0] if updated.data else pet

    created = client.table("pets").insert(payload).execute()
    return created.data[0] if created.data else None


def update_lead_status(lead_id: int, new_status: str):
    response = (
        get_supabase_client()
        .table("leads")
        .update({"status": new_status})
        .eq("id", lead_id)
        .execute()
    )

    if not response.data:
        logger.error(f"Erro ao atualizar status do lead {lead_id}")
        raise Exception("Falha ao atualizar status.")

    logger.info(f"Lead {lead_id} atualizado para {new_status}")
    return response.data


def get_service_id_by_name(service_name: str) -> int:
    response = (
        get_supabase_client()
        .table("services")
        .select("id")
        .eq("name", service_name)
        .limit(1)
        .execute()
    )

    if not response.data:
        response = (
            get_supabase_client()
            .table("services")
            .select("id")
            .eq("name", "Consulta")
            .limit(1)
            .execute()
        )

        if not response.data:
            raise Exception("Servico padrao 'Consulta' nao encontrado.")

    return response.data[0]["id"]


def _normalize_candidate_slot(slot: str) -> str:
    return slot


def is_slot_available(service_id: int, appointment_time: str) -> bool:
    return _is_slot_available_for_service(service_id, appointment_time)


def _is_slot_available_for_service(
    service_id: int,
    appointment_time: str,
    ignore_appointment_id: Optional[int] = None,
) -> bool:
    response = (
        get_supabase_client()
        .table("appointments")
        .select("id")
        .eq("service_id", service_id)
        .eq("appointment_time", appointment_time)
        .neq("status", "Cancelado")
        .limit(1)
    )
    if ignore_appointment_id is not None:
        response = response.neq("id", ignore_appointment_id)
    response = response.execute()
    return not bool(response.data)


def find_next_available_slot(
    service_id: int,
    candidate_slots: Optional[Iterable[str]] = None,
) -> Optional[str]:
    if not candidate_slots:
        return None

    for slot in candidate_slots:
        normalized = _normalize_candidate_slot(slot)
        if is_slot_available(service_id, normalized):
            return normalized

    return None


def confirm_appointment(
    lead_id: int,
    service_id: int,
    appointment_time: str,
    duration_minutes: int,
    google_event_id: Optional[str] = None,
    pet_weight: Optional[float] = None,
    pet_breed: Optional[str] = None,
    pet_age: Optional[str] = None,
    pet_size: Optional[str] = None,
    pet_id: Optional[int] = None,
    candidate_slots: Optional[Iterable[str]] = None,
):
    if not is_slot_available(service_id, appointment_time):
        next_slot = find_next_available_slot(service_id, candidate_slots)
        logger.info(
            "Conflito de agenda detectado antes da insercao. service_id=%s appointment_time=%s next=%s",
            service_id,
            appointment_time,
            next_slot,
        )
        return {"status": "conflict", "next_available_time": next_slot}

    data = {
        "lead_id": lead_id,
        "service_id": service_id,
        "appointment_time": appointment_time,
        "duration_minutes": duration_minutes,
        "status": "Confirmado",
        "google_event_id": google_event_id,
        "pet_weight": pet_weight,
        "pet_breed": pet_breed,
        "pet_age": pet_age,
        "pet_size": pet_size,
        "pet_id": pet_id,
    }
    try:
        response = get_supabase_client().table("appointments").insert(data).execute()
    except Exception as exc:
        message = str(exc).lower()
        duplicate_violation = "duplicate key" in message or "23505" in message
        if duplicate_violation:
            next_slot = find_next_available_slot(service_id, candidate_slots)
            logger.info(
                "Conflito de agenda por concorrencia. service_id=%s appointment_time=%s next=%s",
                service_id,
                appointment_time,
                next_slot,
            )
            return {"status": "conflict", "next_available_time": next_slot}
        raise

    if not response.data:
        logger.error("Erro ao inserir agendamento.")
        raise Exception("Falha ao confirmar agendamento.")

    logger.info(f"APPOINTMENT CREATED: lead_id={lead_id}")
    return response.data[0]


def has_appointment_for_lead(lead_id: int) -> bool:
    response = (
        get_supabase_client()
        .table("appointments")
        .select("id")
        .eq("lead_id", lead_id)
        .neq("status", "Cancelado")
        .limit(1)
        .execute()
    )
    return bool(response.data)


def set_appointment_google_event_id(appointment_id: int, google_event_id: str):
    response = (
        get_supabase_client()
        .table("appointments")
        .update({"google_event_id": google_event_id})
        .eq("id", appointment_id)
        .execute()
    )
    return response.data


def get_appointment_by_id(appointment_id: int) -> Optional[dict]:
    response = (
        get_supabase_client()
        .table("appointments")
        .select("*")
        .eq("id", appointment_id)
        .limit(1)
        .execute()
    )
    if not response.data:
        return None
    appointment = response.data[0]

    service_response = (
        get_supabase_client()
        .table("services")
        .select("id, name")
        .eq("id", appointment["service_id"])
        .limit(1)
        .execute()
    )
    if service_response.data:
        appointment["service_name"] = service_response.data[0]["name"]

    lead_response = (
        get_supabase_client()
        .table("leads")
        .select("id, tutor_name, pet_name, phone")
        .eq("id", appointment["lead_id"])
        .limit(1)
        .execute()
    )
    if lead_response.data:
        appointment.update(lead_response.data[0])

    return appointment


def get_user_appointments(phone: str) -> List[dict]:
    leads_response = (
        get_supabase_client()
        .table("leads")
        .select("id, tutor_name, pet_name, phone")
        .eq("phone", phone)
        .execute()
    )
    if not leads_response.data:
        return []

    leads = leads_response.data
    lead_ids = [lead["id"] for lead in leads]
    lead_map = {lead["id"]: lead for lead in leads}

    query = (
        get_supabase_client()
        .table("appointments")
        .select("*")
        .neq("status", "Cancelado")
        .order("appointment_time")
    )
    if len(lead_ids) == 1:
        query = query.eq("lead_id", lead_ids[0])
    else:
        query = query.in_("lead_id", lead_ids)
    appointments_response = query.execute()

    appointments = appointments_response.data or []
    if not appointments:
        return []

    service_ids = list({appointment["service_id"] for appointment in appointments})
    services_query = get_supabase_client().table("services").select("id, name")
    if len(service_ids) == 1:
        services_query = services_query.eq("id", service_ids[0])
    else:
        services_query = services_query.in_("id", service_ids)
    services_response = services_query.execute()
    service_map = {service["id"]: service["name"] for service in (services_response.data or [])}

    enriched: List[dict] = []
    for appointment in appointments:
        lead = lead_map.get(appointment["lead_id"], {})
        enriched.append(
            {
                **appointment,
                "service_name": service_map.get(appointment["service_id"], "Consulta"),
                "tutor_name": lead.get("tutor_name"),
                "pet_name": lead.get("pet_name"),
                "phone": lead.get("phone"),
            }
        )

    return enriched


def get_active_appointment_by_phone(phone: str) -> Optional[dict]:
    appointments = get_user_appointments(phone)
    if not appointments:
        return None

    ordered_appointments = sorted(
        appointments,
        key=lambda appointment: appointment["appointment_time"],
        reverse=True,
    )
    return ordered_appointments[0]


def cancel_appointment(appointment_id: int) -> Optional[dict]:
    response = (
        get_supabase_client()
        .table("appointments")
        .update({"status": "Cancelado"})
        .eq("id", appointment_id)
        .execute()
    )
    if not response.data:
        return None
    return response.data[0]


def reschedule_appointment(appointment_id: int, new_time: str) -> Optional[dict]:
    current = get_appointment_by_id(appointment_id)
    if not current:
        return None

    if not _is_slot_available_for_service(
        current["service_id"], new_time, ignore_appointment_id=appointment_id
    ):
        raise ValueError("Horario indisponivel para remarcacao.")

    response = (
        get_supabase_client()
        .table("appointments")
        .update({"appointment_time": new_time, "status": "Confirmado"})
        .eq("id", appointment_id)
        .execute()
    )
    if not response.data:
        return None
    return response.data[0]


def get_pet_history(phone: str) -> Dict[str, List[dict]]:
    client = get_supabase_client()
    pets_response = client.table("pets").select("*").eq("tutor_phone", phone).execute()
    pets = pets_response.data or []
    if not pets:
        return {"pets": [], "consultations": [], "vaccines": []}

    pet_ids = [pet["id"] for pet in pets]

    consultations_query = client.table("consultations").select("*").order("created_at", desc=True)
    vaccines_query = client.table("vaccines").select("*").order("date_applied", desc=True)

    if len(pet_ids) == 1:
        consultations_query = consultations_query.eq("pet_id", pet_ids[0])
        vaccines_query = vaccines_query.eq("pet_id", pet_ids[0])
    else:
        consultations_query = consultations_query.in_("pet_id", pet_ids)
        vaccines_query = vaccines_query.in_("pet_id", pet_ids)

    consultations_response = consultations_query.execute()
    vaccines_response = vaccines_query.execute()

    return {
        "pets": pets,
        "consultations": consultations_response.data or [],
        "vaccines": vaccines_response.data or [],
    }
