import re
from typing import Annotated, TypedDict, List, Literal, Optional
from datetime import datetime, timedelta

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage
from pydantic import BaseModel

from src.clinivet_db import (
    register_lead,
    confirm_appointment,
    update_lead_status,
    get_service_id_by_name,
)

from src.clinivet_calendar import (
    TIMEZONE,
    get_calendar_service,
    build_slot_datetime,
)

# =================================
# MODELO DE TRIAGEM
# =================================

class TriageOutput(BaseModel):
    tutor_name: Optional[str] = None
    pet_name: Optional[str] = None
    pet_species: Literal["Cão", "Gato", "Desconhecido"] = "Desconhecido"
    urgency_level: Literal["emergency", "routine"] = "routine"
    service_suggested: str = "Consulta"
    symptoms_summary: Optional[str] = None
    phone: Optional[str] = None


# =================================
# ESTADO DO AGENTE
# =================================

class ClinivetState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], "Histórico da conversa"]
    lead_id: Optional[int]
    triage_data: Optional[TriageOutput]
    urgency_level: Optional[str]
    next_step: Optional[str]
    available_slots: Optional[List[str]]
    appointment_date: Optional[str]
    thread_id: Optional[str]
    missing_fields: Optional[List[str]]
    assistant_message: Optional[str]


# =================================
# LLM
# =================================

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

structured_llm = llm.with_structured_output(TriageOutput)


# =================================
# PROMPT
# =================================

TRIAGE_PROMPT = """
Você é o assistente da Clinivet Lins da Dra. Daniely.

Extraia da conversa:

- nome do tutor
- nome do pet
- espécie (cão, gato ou desconhecido)
- sintomas
- nível de urgência
- serviço sugerido
- telefone (somente se for informado na conversa)

Urgências incluem:
- convulsão
- hemorragia
- dificuldade respiratória
- trauma grave

Regras obrigatórias:
- Se houver risco imediato marque urgency_level = "emergency".
- Caso contrário marque urgency_level = "routine".
- service_suggested nunca pode ser null ou vazio. Se não tiver clareza, use "Consulta".
- phone não deve ser inventado. Se não houver telefone explícito, retorne null.
- symptoms_summary deve ser um resumo curto em português.
"""

DEFAULT_SERVICE = "Consulta"
UNKNOWN_PHONE = "unknown"
UNKNOWN_PET = "Paciente"

REQUIRED_FIELDS_FOR_APPOINTMENT = ["tutor_name", "pet_name", "pet_species"]
MISSING_FIELD_QUESTIONS = {
    "tutor_name": "Antes de continuar, posso saber seu nome?",
    "pet_name": "Qual é o nome do seu pet?",
    "pet_species": "Seu pet é um cão ou gato?",
}


def _clean_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned if cleaned else None


def _extract_phone_candidate(raw_value: Optional[str]) -> Optional[str]:
    cleaned = _clean_text(raw_value)
    if not cleaned:
        return None
    digits = re.sub(r"\D", "", cleaned)
    if len(digits) < 10:
        return None
    return digits


def _normalize_triage_result(triage: TriageOutput) -> TriageOutput:
    triage.tutor_name = _clean_text(triage.tutor_name)
    triage.pet_name = _clean_text(triage.pet_name)
    triage.symptoms_summary = _clean_text(triage.symptoms_summary)
    triage.service_suggested = _clean_text(triage.service_suggested) or DEFAULT_SERVICE
    triage.phone = _extract_phone_candidate(triage.phone)
    return triage


def _is_species_missing(species: Optional[str]) -> bool:
    return species in (None, "Desconhecido")


def _merge_triage_data(previous: Optional[TriageOutput], current: TriageOutput) -> TriageOutput:
    if not previous:
        return current

    merged_species = current.pet_species
    if _is_species_missing(merged_species) and not _is_species_missing(previous.pet_species):
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
            _clean_text(current.service_suggested)
            or _clean_text(previous.service_suggested)
            or DEFAULT_SERVICE
        ),
        symptoms_summary=current.symptoms_summary or previous.symptoms_summary,
        phone=current.phone or previous.phone,
    )

    return _normalize_triage_result(merged)


def _get_missing_required_fields(triage: TriageOutput) -> List[str]:
    missing: List[str] = []

    if not triage.tutor_name:
        missing.append("tutor_name")

    if not triage.pet_name:
        missing.append("pet_name")

    if _is_species_missing(triage.pet_species):
        missing.append("pet_species")

    return missing


def _build_missing_data_message(missing_fields: List[str]) -> str:
    if not missing_fields:
        return "Perfeito. Vamos continuar."

    if len(missing_fields) == 1:
        return MISSING_FIELD_QUESTIONS[missing_fields[0]]

    questions = [MISSING_FIELD_QUESTIONS[field] for field in missing_fields]
    bullet_list = "\n".join([f"- {question}" for question in questions])

    return (
        "Sinto muito que seu pet nao esteja bem. "
        "Antes de continuar, preciso de algumas informacoes:\n"
        f"{bullet_list}"
    )


# =================================
# TRIAGEM
# =================================

def triage_node(state: ClinivetState):

    system_msg = SystemMessage(content=TRIAGE_PROMPT)

    # Usa historico do thread quando houver checkpoint.
    conversation: List[BaseMessage] = [system_msg] + list(state.get("messages") or [])

    try:
        triage_result: TriageOutput = structured_llm.invoke(conversation)
    except Exception as exc:
        print("Erro ao estruturar triagem com LLM:", exc)
        triage_result = TriageOutput(
            service_suggested=DEFAULT_SERVICE,
            urgency_level="routine",
        )

    triage_result = _normalize_triage_result(triage_result)
    triage_result = _merge_triage_data(state.get("triage_data"), triage_result)

    print("TRIAGE RESULT:", triage_result)

    missing_fields = _get_missing_required_fields(triage_result)
    if missing_fields:
        return {
            "triage_data": triage_result,
            "missing_fields": missing_fields,
            "urgency_level": triage_result.urgency_level,
            "next_step": "ask_missing_data",
            "assistant_message": None,
        }

    lead_id = state.get("lead_id")

    # Telefone pode vir do conteudo da conversa ou do thread_id (ex.: WhatsApp).
    phone_number = (
        triage_result.phone
        or _extract_phone_candidate(state.get("thread_id"))
        or UNKNOWN_PHONE
    )

    # Se ainda nao existe lead e ja temos os campos minimos obrigatorios.
    if not lead_id:
        try:
            lead_id = register_lead(
                tutor_name=triage_result.tutor_name,
                pet_name=triage_result.pet_name,
                pet_species=triage_result.pet_species,
                phone=phone_number,
            )
            print("Lead criado:", lead_id)
        except Exception as e:
            print("Erro ao criar lead:", e)

    next_step = "scheduling" if triage_result.urgency_level == "routine" else "end"

    emergency_message = None
    if triage_result.urgency_level == "emergency":
        emergency_message = (
            "Identifiquei sinais de urgencia. "
            "Leve seu pet imediatamente para atendimento veterinario de emergencia."
        )

    return {
        "triage_data": triage_result,
        "lead_id": lead_id,
        "urgency_level": triage_result.urgency_level,
        "missing_fields": [],
        "assistant_message": emergency_message,
        "next_step": next_step,
    }


# =================================
# PEDIR DADOS FALTANTES
# =================================

def ask_missing_data_node(state: ClinivetState):
    missing_fields = state.get("missing_fields") or []
    message = _build_missing_data_message(missing_fields)

    return {
        "assistant_message": message,
        "next_step": "end",
    }


# =================================
# BUSCAR HORARIOS
# =================================

def scheduling_node(state: ClinivetState):

    calendar_service = get_calendar_service()

    tomorrow = (datetime.now(TIMEZONE) + timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        available_slots = calendar_service.get_free_slots(tomorrow)
    except Exception as e:
        print("Erro ao consultar agenda:", e)
        raise

    print("SLOTS DISPONIVEIS:", available_slots)

    if not available_slots:
        raise RuntimeError("Nenhum horario disponivel.")

    return {
        "next_step": "conversion",
        "available_slots": available_slots,
        "appointment_date": tomorrow,
    }


# =================================
# CONVERSAO (AGENDAMENTO)
# =================================

def conversion_node(state: ClinivetState):

    lead_id = state.get("lead_id")
    triage_data = state.get("triage_data")
    available_slots = state.get("available_slots")
    appointment_date = state.get("appointment_date")

    if not lead_id:
        raise ValueError("Lead nao encontrado.")

    if not triage_data:
        raise ValueError("Dados de triagem ausentes.")

    selected_slot = available_slots[0]

    appointment_datetime = build_slot_datetime(
        appointment_date,
        selected_slot
    )

    calendar_service = get_calendar_service()

    service_name = _clean_text(triage_data.service_suggested) or DEFAULT_SERVICE
    pet_display_name = _clean_text(triage_data.pet_name) or UNKNOWN_PET

    try:
        event_id = calendar_service.create_event(
            summary=f"{service_name} - {pet_display_name}",
            start_time=appointment_datetime,
            duration_minutes=30,
        )
        print("Evento criado:", event_id)
    except Exception as e:
        print("Erro ao criar evento no Google Calendar:", e)
        event_id = None

    service_id = get_service_id_by_name(service_name)

    confirm_appointment(
        lead_id=lead_id,
        service_id=service_id,
        appointment_time=appointment_datetime.isoformat(),
        duration_minutes=30,
        google_event_id=event_id,
    )

    update_lead_status(lead_id, "Agendado")

    print("Agendamento finalizado.")

    confirmation_message = (
        f"Agendamento confirmado para {pet_display_name}. "
        f"Servico: {service_name}. "
        f"Data e horario: {appointment_datetime.strftime('%d/%m/%Y %H:%M')}."
    )

    return {
        "assistant_message": confirmation_message,
        "next_step": "end",
    }


# =================================
# ROTEADOR
# =================================

def router(state: ClinivetState) -> Literal["ask_missing_data", "scheduling", "conversion", "end"]:

    if state.get("urgency_level") == "emergency":
        return "end"

    return state.get("next_step", "end")


# =================================
# GRAFO
# =================================

workflow = StateGraph(ClinivetState)

workflow.add_node("triage", triage_node)
workflow.add_node("ask_missing_data", ask_missing_data_node)
workflow.add_node("scheduling", scheduling_node)
workflow.add_node("conversion", conversion_node)

workflow.set_entry_point("triage")

workflow.add_conditional_edges(
    "triage",
    router,
    {
        "ask_missing_data": "ask_missing_data",
        "scheduling": "scheduling",
        "end": END,
    },
)

workflow.add_edge("ask_missing_data", END)

workflow.add_conditional_edges(
    "scheduling",
    router,
    {
        "conversion": "conversion",
        "end": END,
    },
)

workflow.add_edge("conversion", END)

checkpointer = MemorySaver()
clinivet_agent = workflow.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    print("Clinivet Brain carregado.")
