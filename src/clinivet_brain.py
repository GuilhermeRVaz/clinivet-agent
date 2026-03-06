import logging
from typing import Annotated, List, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from src.clinivet_calendar import build_slot_datetime, get_calendar_service
from src.clinivet_db import (
    confirm_appointment,
    get_service_id_by_name,
    has_appointment_for_lead,
    register_lead,
    update_lead_status,
)
from src.models.triage_model import TriageOutput
from src.services.scheduling_service import build_next_business_day
from src.services.triage_service import (
    DEFAULT_SERVICE,
    build_missing_data_message,
    extract_phone_candidate,
    get_missing_required_fields,
    merge_triage_data,
    normalize_triage_result,
)

logger = logging.getLogger("ClinivetBrain")

TRIAGE_PROMPT = """
Voce e o assistente da Clinivet Lins da Dra. Daniely.

Extract from conversation:
- tutor_name
- pet_name
- pet_species (Cao, Gato, or Desconhecido)
- urgency_level
- service_suggested
- symptoms_summary
- phone (only when explicitly informed)

Service detection allowed values:
- Consulta
- Vacinacao
- Banho e Tosa
- Retorno
- Emergencia

Urgent signs include:
- convulsao
- hemorragia
- dificuldade respiratoria
- trauma grave

Rules:
- If there is immediate risk, set urgency_level = "emergency".
- Otherwise set urgency_level = "routine".
- service_suggested cannot be empty. If unclear, use "Consulta".
- Never fabricate phone numbers. If the user did not provide a phone, return null.
- symptoms_summary must be short and in Portuguese.
"""

UNKNOWN_PHONE = "unknown"
UNKNOWN_PET = "Paciente"
APPOINTMENT_DURATION_MINUTES = 30

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
structured_llm = llm.with_structured_output(TriageOutput)


class ClinivetState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    thread_id: Optional[str]
    lead_id: Optional[int]
    triage_data: Optional[TriageOutput]
    urgency_level: Optional[str]
    next_step: Optional[str]
    available_slots: Optional[List[str]]
    appointment_date: Optional[str]
    missing_fields: Optional[List[str]]
    assistant_message: Optional[str]


def triage_node(state: ClinivetState):
    system_msg = SystemMessage(content=TRIAGE_PROMPT)
    conversation: List[BaseMessage] = [system_msg] + list(state.get("messages") or [])

    try:
        triage_result: TriageOutput = structured_llm.invoke(conversation)
    except Exception as exc:
        logger.exception("Failed to run triage LLM: %s", exc)
        triage_result = TriageOutput(service_suggested=DEFAULT_SERVICE, urgency_level="routine")

    triage_result = normalize_triage_result(triage_result)
    triage_result = merge_triage_data(state.get("triage_data"), triage_result)

    if not triage_result.phone:
        triage_result.phone = extract_phone_candidate(state.get("thread_id"))

    logger.info("TRIAGE RESULT: %s", triage_result.model_dump())

    missing_fields = get_missing_required_fields(triage_result, state.get("thread_id"))
    if missing_fields:
        return {
            "triage_data": triage_result,
            "missing_fields": missing_fields,
            "urgency_level": triage_result.urgency_level,
            "next_step": "ask_missing_data",
            "assistant_message": None,
        }

    lead_id = state.get("lead_id")
    if not lead_id:
        try:
            lead_id = register_lead(
                tutor_name=triage_result.tutor_name,
                pet_name=triage_result.pet_name,
                pet_species=triage_result.pet_species,
                phone=triage_result.phone or UNKNOWN_PHONE,
            )
        except Exception as exc:
            logger.exception("Failed to register lead: %s", exc)
            error_message = "Nao consegui registrar seu atendimento agora. Tente novamente em instantes."
            return {
                "triage_data": triage_result,
                "assistant_message": error_message,
                "messages": [AIMessage(content=error_message)],
                "next_step": "end",
            }

    next_step = "scheduling" if triage_result.urgency_level == "routine" else "end"
    emergency_message = None
    emergency_messages = None

    if triage_result.urgency_level == "emergency":
        emergency_message = (
            "Identifiquei sinais de urgencia. "
            "Leve seu pet imediatamente para atendimento veterinario de emergencia."
        )
        emergency_messages = [AIMessage(content=emergency_message)]

    return {
        "triage_data": triage_result,
        "lead_id": lead_id,
        "thread_id": state.get("thread_id"),
        "urgency_level": triage_result.urgency_level,
        "missing_fields": [],
        "assistant_message": emergency_message,
        "messages": emergency_messages,
        "next_step": next_step,
    }


def ask_missing_data_node(state: ClinivetState):
    missing_fields = state.get("missing_fields") or []
    message = build_missing_data_message(missing_fields)
    return {
        "assistant_message": message,
        "messages": [AIMessage(content=message)],
        "next_step": "end",
    }


def scheduling_node(state: ClinivetState):
    calendar_service = get_calendar_service()
    target_day = build_next_business_day()

    try:
        available_slots = calendar_service.get_free_slots(target_day)
    except Exception as exc:
        logger.exception("Failed to load slots: %s", exc)
        raise

    logger.info("AVAILABLE SLOTS: %s", available_slots)

    if not available_slots:
        raise RuntimeError("Nenhum horario disponivel.")

    return {
        "next_step": "conversion",
        "available_slots": available_slots,
        "appointment_date": target_day,
    }


def conversion_node(state: ClinivetState):
    lead_id = state.get("lead_id")
    triage_data = state.get("triage_data")
    available_slots = state.get("available_slots")
    appointment_date = state.get("appointment_date")

    if not lead_id:
        raise ValueError("Lead nao encontrado.")
    if not triage_data:
        raise ValueError("Dados de triagem ausentes.")
    if not available_slots or not appointment_date:
        raise ValueError("Dados de agendamento ausentes.")

    if has_appointment_for_lead(lead_id):
        duplicate_message = "Voce ja possui um agendamento registrado."
        return {
            "assistant_message": duplicate_message,
            "messages": [AIMessage(content=duplicate_message)],
            "next_step": "end",
        }

    selected_slot = available_slots[0]
    appointment_datetime = build_slot_datetime(appointment_date, selected_slot)
    calendar_service = get_calendar_service()

    service_name = triage_data.service_suggested or DEFAULT_SERVICE
    pet_display_name = triage_data.pet_name or UNKNOWN_PET

    try:
        event_id = calendar_service.create_event(
            summary=f"{service_name} - {pet_display_name}",
            start_time=appointment_datetime,
            duration_minutes=APPOINTMENT_DURATION_MINUTES,
        )
    except Exception as exc:
        logger.exception("Failed to create Google Calendar event: %s", exc)
        event_id = None

    service_id = get_service_id_by_name(service_name)
    confirm_appointment(
        lead_id=lead_id,
        service_id=service_id,
        appointment_time=appointment_datetime.isoformat(),
        duration_minutes=APPOINTMENT_DURATION_MINUTES,
        google_event_id=event_id,
    )
    update_lead_status(lead_id, "Agendado")

    confirmation_message = (
        f"Agendamento confirmado para {pet_display_name}. "
        f"Servico: {service_name}. "
        f"Data e horario: {appointment_datetime.strftime('%d/%m/%Y %H:%M')}."
    )

    logger.info("APPOINTMENT CREATED: lead_id=%s date=%s", lead_id, appointment_datetime)

    return {
        "assistant_message": confirmation_message,
        "messages": [AIMessage(content=confirmation_message)],
        "next_step": "end",
    }


def router(state: ClinivetState) -> Literal["ask_missing_data", "scheduling", "conversion", "end"]:
    if state.get("urgency_level") == "emergency":
        return "end"
    return state.get("next_step", "end")


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
    logger.info("Clinivet Brain loaded.")
