import logging
import os
from pathlib import Path
from typing import Optional

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


def register_lead(tutor_name: str, pet_name: str, pet_species: str, phone: str) -> int:
    data = {
        "tutor_name": tutor_name,
        "pet_name": pet_name,
        "pet_species": pet_species,
        "phone": phone,
        "status": "Interessado",
    }

    response = get_supabase_client().table("leads").insert(data).execute()

    if not response.data:
        logger.error(f"Erro ao registrar lead: {response}")
        raise Exception("Falha ao registrar lead.")

    lead_id = response.data[0]["id"]
    logger.info(f"LEAD CREATED: {lead_id}")
    return lead_id


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


def confirm_appointment(
    lead_id: int,
    service_id: int,
    appointment_time: str,
    duration_minutes: int,
    google_event_id: Optional[str] = None,
):
    data = {
        "lead_id": lead_id,
        "service_id": service_id,
        "appointment_time": appointment_time,
        "duration_minutes": duration_minutes,
        "status": "Confirmado",
        "google_event_id": google_event_id,
    }

    response = get_supabase_client().table("appointments").insert(data).execute()

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
        .limit(1)
        .execute()
    )
    return bool(response.data)
