import logging
from datetime import datetime
from typing import Annotated, List, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from src.clinivet_calendar import TIMEZONE, build_slot_datetime, get_calendar_service
from src.clinivet_db import (
    confirm_appointment,
    get_service_id_by_name,
    has_appointment_for_lead,
    register_lead,
    set_appointment_google_event_id,
    update_lead_status,
)
from src.models.triage_model import TriageOutput
from src.services.scheduling_service import resolve_scheduling_context
from src.services.triage_service import (
    DEFAULT_SERVICE,
    build_missing_data_message,
    extract_phone_candidate,
    get_missing_required_fields,
    merge_triage_data,
    normalize_triage_result,
)

logger = logging.getLogger("ClinivetBrain")

try:
    from langchain_openai import ChatOpenAI
except ModuleNotFoundError:
    ChatOpenAI = None

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
- pet_weight (kg)
- pet_breed
- pet_age

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

if ChatOpenAI is not None:
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    structured_llm = llm.with_structured_output(TriageOutput)
else:
    llm = None
    structured_llm = None


class ClinivetState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    thread_id: Optional[str]
    lead_id: Optional[int]
    triage_data: Optional[TriageOutput]
    urgency_level: Optional[str]
    next_step: Optional[str]
    available_slots: Optional[List[str]]
    appointment_date: Optional[str]
    service_name: Optional[str]
    missing_fields: Optional[List[str]]
    assistant_message: Optional[str]


def triage_node(state: ClinivetState):
    if structured_llm is None:
        raise RuntimeError(
            "langchain_openai is required at runtime. Install dependencies with: pip install -r requirements.txt"
        )

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
                pet_weight=triage_result.pet_weight,
                pet_breed=triage_result.pet_breed,
                pet_age=triage_result.pet_age,
                pet_size=triage_result.pet_size,
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

    response = {
        "triage_data": triage_result,
        "lead_id": lead_id,
        "thread_id": state.get("thread_id"),
        "urgency_level": triage_result.urgency_level,
        "missing_fields": [],
        "assistant_message": emergency_message,
        "next_step": next_step,
    }
    if emergency_messages:
        response["messages"] = emergency_messages
    return response


def ask_missing_data_node(state: ClinivetState):
    missing_fields = state.get("missing_fields") or []
    message = build_missing_data_message(missing_fields)
    return {
        "assistant_message": message,
        "messages": [AIMessage(content=message)],
        "next_step": "end",
    }


def scheduling_node(state: ClinivetState):
    triage_data = state.get("triage_data")
    service_name = (triage_data.service_suggested if triage_data else None) or DEFAULT_SERVICE

    try:
        resolved_service, target_day, available_slots = resolve_scheduling_context(service_name)
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
        "service_name": resolved_service,
    }


def conversion_node(state: ClinivetState):
    lead_id = state.get("lead_id")
    triage_data = state.get("triage_data")
    available_slots = state.get("available_slots")
    appointment_date = state.get("appointment_date")
    service_name = state.get("service_name") or (
        triage_data.service_suggested if triage_data else DEFAULT_SERVICE
    )

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

    calendar_service = get_calendar_service(service_name)
    pet_display_name = triage_data.pet_name or UNKNOWN_PET

    service_id = get_service_id_by_name(service_name)

    candidate_times = [
        build_slot_datetime(appointment_date, slot).isoformat() for slot in available_slots
    ]
    appointment_record = None
    chosen_time_iso: Optional[str] = None
    attempt_index = 0

    while attempt_index < len(candidate_times):
        candidate_time = candidate_times[attempt_index]
        result = confirm_appointment(
            lead_id=lead_id,
            service_id=service_id,
            appointment_time=candidate_time,
            duration_minutes=APPOINTMENT_DURATION_MINUTES,
            google_event_id=None,
            pet_weight=triage_data.pet_weight,
            pet_breed=triage_data.pet_breed,
            pet_age=triage_data.pet_age,
            pet_size=triage_data.pet_size,
            candidate_slots=candidate_times[attempt_index + 1 :],
        )

        if isinstance(result, dict) and result.get("status") == "conflict":
            next_time = result.get("next_available_time")
            if next_time and next_time in candidate_times:
                attempt_index = candidate_times.index(next_time)
                continue
            if next_time and next_time not in candidate_times:
                candidate_times.append(next_time)
                attempt_index = len(candidate_times) - 1
                continue
            attempt_index += 1
            continue

        appointment_record = result
        chosen_time_iso = candidate_time
        break

    if appointment_record is None or not chosen_time_iso:
        no_slots_message = (
            "Nao consegui confirmar esse horario porque ele acabou de ser ocupado. "
            "Vou te apresentar novas opcoes em seguida."
        )
        return {
            "assistant_message": no_slots_message,
            "messages": [AIMessage(content=no_slots_message)],
            "next_step": "end",
        }

    appointment_datetime = datetime.fromisoformat(chosen_time_iso).astimezone(TIMEZONE)

    try:
        event_id = calendar_service.create_event(
            summary=f"{service_name} - {pet_display_name}",
            start_time=appointment_datetime,
            duration_minutes=APPOINTMENT_DURATION_MINUTES,
        )
    except Exception as exc:
        logger.exception("Failed to create Google Calendar event: %s", exc)
        event_id = None

    if event_id and isinstance(appointment_record, dict) and appointment_record.get("id"):
        try:
            set_appointment_google_event_id(appointment_record["id"], event_id)
        except Exception as exc:
            logger.exception("Failed to persist google_event_id: %s", exc)

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
