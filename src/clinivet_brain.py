import logging
from datetime import datetime
from typing import Annotated, List, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from src.clinivet_calendar import TIMEZONE, build_slot_datetime, get_calendar_service
from src.clinivet_db import (
    cancel_appointment as cancel_appointment_record,
    confirm_appointment,
    get_appointment_by_id,
    get_user_appointments,
    get_service_id_by_name,
    has_appointment_for_lead,
    register_lead,
    reschedule_appointment as reschedule_appointment_record,
    set_appointment_google_event_id,
    upsert_pet_profile,
    update_lead_status,
)
from src.models.triage_model import TriageOutput
from src.services.conversation_service import (
    detect_intent,
    extract_appointment_id,
    extract_time_choice,
    extract_time_preference,
    format_appointments_message,
    format_pet_history_message,
    format_slots_message,
    get_latest_human_message,
    wants_slot_suggestions,
)
from src.services.pet_history_service import load_pet_history
from src.services.scheduling_service import build_next_business_day, find_available_slots, resolve_scheduling_context
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
    pet_id: Optional[int]
    triage_data: Optional[TriageOutput]
    intent: Optional[str]
    urgency_level: Optional[str]
    next_step: Optional[str]
    available_slots: Optional[List[str]]
    appointment_date: Optional[str]
    service_name: Optional[str]
    missing_fields: Optional[List[str]]
    time_preference: Optional[str]
    selected_slot: Optional[str]
    selected_appointment_id: Optional[int]
    pending_action: Optional[str]
    user_appointments: Optional[List[dict]]
    pet_history: Optional[dict]
    assistant_message: Optional[str]


def _resolve_phone_from_state(state: ClinivetState) -> Optional[str]:
    triage_data = state.get("triage_data")
    return (
        (triage_data.phone if triage_data else None)
        or extract_phone_candidate(state.get("thread_id"))
    )


def _build_event_summary(service_name: str, pet_name: Optional[str]) -> str:
    return f"{service_name} - {pet_name or UNKNOWN_PET}"


def _format_appointment_datetime(appointment_time_iso: str) -> str:
    appointment_datetime = datetime.fromisoformat(appointment_time_iso).astimezone(TIMEZONE)
    return appointment_datetime.strftime("%d/%m/%Y %H:%M")


def _persist_confirmed_appointment(
    *,
    lead_id: int,
    pet_id: Optional[int],
    triage_data: TriageOutput,
    service_name: str,
    appointment_time_iso: str,
    appointment_record: Optional[dict] = None,
) -> str:
    if appointment_record is None:
        service_id = get_service_id_by_name(service_name)
        appointment_record = confirm_appointment(
            lead_id=lead_id,
            pet_id=pet_id,
            service_id=service_id,
            appointment_time=appointment_time_iso,
            duration_minutes=APPOINTMENT_DURATION_MINUTES,
            google_event_id=None,
            pet_weight=triage_data.pet_weight,
            pet_breed=triage_data.pet_breed,
            pet_age=triage_data.pet_age,
            pet_size=triage_data.pet_size,
        )

    if isinstance(appointment_record, dict) and appointment_record.get("status") == "conflict":
        raise RuntimeError("Horario indisponivel para confirmacao.")

    appointment_datetime = datetime.fromisoformat(appointment_time_iso).astimezone(TIMEZONE)
    calendar_service = get_calendar_service(service_name)
    event_id = None

    try:
        event_id = calendar_service.create_event(
            summary=_build_event_summary(service_name, triage_data.pet_name),
            start_time=appointment_datetime,
            duration_minutes=APPOINTMENT_DURATION_MINUTES,
        )
    except Exception as exc:
        logger.exception("Failed to create Google Calendar event: %s", exc)

    if event_id and isinstance(appointment_record, dict) and appointment_record.get("id"):
        try:
            set_appointment_google_event_id(appointment_record["id"], event_id)
        except Exception as exc:
            logger.exception("Failed to persist google_event_id: %s", exc)

    update_lead_status(lead_id, "Agendado")

    return (
        f"Agendamento confirmado para {triage_data.pet_name or UNKNOWN_PET}. "
        f"Servico: {service_name}. "
        f"Data e horario: {appointment_datetime.strftime('%d/%m/%Y %H:%M')}."
    )


def triage_node(state: ClinivetState):
    latest_message = get_latest_human_message(state.get("messages") or [])
    pending_action = state.get("pending_action")
    intent = detect_intent(latest_message, pending_action)
    time_preference = extract_time_preference(latest_message) or state.get("time_preference")
    selected_slot = extract_time_choice(latest_message)
    selected_appointment_id = extract_appointment_id(latest_message) or state.get(
        "selected_appointment_id"
    )

    if intent in {"cancel", "reschedule", "check_appointment", "load_pet_history"}:
        next_step = {
            "cancel": "cancel_appointment",
            "reschedule": "reschedule_appointment",
            "check_appointment": "check_appointment",
            "load_pet_history": "load_pet_history",
        }[intent]
        return {
            "intent": intent,
            "next_step": next_step,
            "time_preference": time_preference,
            "selected_slot": selected_slot,
            "selected_appointment_id": selected_appointment_id,
            "pending_action": pending_action,
        }

    if pending_action == "confirm_slot" and selected_slot:
        return {
            "intent": "schedule",
            "next_step": "confirm_slot",
            "selected_slot": selected_slot,
            "time_preference": time_preference,
            "selected_appointment_id": selected_appointment_id,
        }

    if pending_action == "awaiting_time_preference":
        if time_preference:
            return {
                "intent": "schedule",
                "next_step": "suggest_slots",
                "time_preference": time_preference,
            }
        return {
            "intent": "schedule",
            "next_step": "ask_time_preference",
        }

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
            "intent": intent,
            "time_preference": time_preference,
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
                "intent": intent,
            }

    pet_record = None
    try:
        pet_record = upsert_pet_profile(
            tutor_phone=triage_result.phone or UNKNOWN_PHONE,
            name=triage_result.pet_name or UNKNOWN_PET,
            species=triage_result.pet_species,
            breed=triage_result.pet_breed,
            age=triage_result.pet_age,
            weight=triage_result.pet_weight,
            size=triage_result.pet_size,
        )
    except Exception as exc:
        logger.exception("Failed to upsert pet profile: %s", exc)

    next_step = "scheduling" if triage_result.urgency_level == "routine" else "end"
    emergency_message = None
    emergency_messages = None

    if triage_result.urgency_level == "emergency":
        emergency_message = (
            "Identifiquei sinais de urgencia. "
            "Leve seu pet imediatamente para atendimento veterinario de emergencia."
        )
        emergency_messages = [AIMessage(content=emergency_message)]

    if triage_result.urgency_level == "routine":
        if time_preference:
            next_step = "suggest_slots"
        elif wants_slot_suggestions(latest_message):
            next_step = "ask_time_preference"

    response = {
        "triage_data": triage_result,
        "lead_id": lead_id,
        "pet_id": pet_record["id"] if pet_record else state.get("pet_id"),
        "thread_id": state.get("thread_id"),
        "urgency_level": triage_result.urgency_level,
        "intent": intent,
        "missing_fields": [],
        "assistant_message": emergency_message,
        "time_preference": time_preference,
        "selected_slot": selected_slot,
        "selected_appointment_id": selected_appointment_id,
        "pending_action": None,
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


def ask_time_preference_node(state: ClinivetState):
    message = "Voce prefere manha, tarde ou qualquer horario?"
    return {
        "assistant_message": message,
        "messages": [AIMessage(content=message)],
        "pending_action": "awaiting_time_preference",
        "next_step": "end",
    }


def suggest_slots_node(state: ClinivetState):
    triage_data = state.get("triage_data")
    service_name = state.get("service_name") or (
        triage_data.service_suggested if triage_data else DEFAULT_SERVICE
    )
    time_preference = state.get("time_preference") or "any"
    appointment_date = state.get("appointment_date") or build_next_business_day()
    slots = find_available_slots(
        date=appointment_date,
        period=time_preference,
        service_name=service_name,
    )

    if not slots:
        message = "Nao encontrei horarios disponiveis para essa preferencia agora."
        return {
            "assistant_message": message,
            "messages": [AIMessage(content=message)],
            "next_step": "end",
            "pending_action": None,
        }

    message = format_slots_message(slots)
    return {
        "assistant_message": message,
        "messages": [AIMessage(content=message)],
        "available_slots": slots,
        "appointment_date": appointment_date,
        "service_name": service_name,
        "pending_action": "confirm_slot",
        "next_step": "end",
    }


def confirm_slot_node(state: ClinivetState):
    lead_id = state.get("lead_id")
    triage_data = state.get("triage_data")
    selected_slot = state.get("selected_slot")
    appointment_date = state.get("appointment_date") or build_next_business_day()
    service_name = state.get("service_name") or (
        triage_data.service_suggested if triage_data else DEFAULT_SERVICE
    )
    available_slots = state.get("available_slots") or []

    if not lead_id or not triage_data:
        raise ValueError("Dados insuficientes para confirmar horario.")

    if not selected_slot:
        message = "Qual horario voce prefere entre as opcoes enviadas?"
        return {
            "assistant_message": message,
            "messages": [AIMessage(content=message)],
            "pending_action": "confirm_slot",
            "next_step": "end",
        }

    if available_slots and selected_slot not in available_slots:
        message = "Esse horario nao esta entre as opcoes sugeridas. Escolha um dos horarios informados."
        return {
            "assistant_message": message,
            "messages": [AIMessage(content=message)],
            "pending_action": "confirm_slot",
            "next_step": "end",
        }

    if has_appointment_for_lead(lead_id):
        duplicate_message = "Voce ja possui um agendamento registrado."
        return {
            "assistant_message": duplicate_message,
            "messages": [AIMessage(content=duplicate_message)],
            "pending_action": None,
            "next_step": "end",
        }

    appointment_time_iso = build_slot_datetime(appointment_date, selected_slot).isoformat()
    confirmation_message = _persist_confirmed_appointment(
        lead_id=lead_id,
        pet_id=state.get("pet_id"),
        triage_data=triage_data,
        service_name=service_name,
        appointment_time_iso=appointment_time_iso,
    )

    return {
        "assistant_message": confirmation_message,
        "messages": [AIMessage(content=confirmation_message)],
        "pending_action": None,
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

    pet_display_name = triage_data.pet_name or UNKNOWN_PET
    confirmation_message = _persist_confirmed_appointment(
        lead_id=lead_id,
        pet_id=state.get("pet_id"),
        triage_data=triage_data,
        service_name=service_name,
        appointment_time_iso=chosen_time_iso,
        appointment_record=appointment_record,
    )
    appointment_datetime = datetime.fromisoformat(chosen_time_iso).astimezone(TIMEZONE)

    logger.info("APPOINTMENT CREATED: lead_id=%s date=%s", lead_id, appointment_datetime)

    return {
        "assistant_message": confirmation_message,
        "messages": [AIMessage(content=confirmation_message)],
        "pending_action": None,
        "next_step": "end",
    }


def check_appointment_node(state: ClinivetState):
    phone = _resolve_phone_from_state(state)
    if not phone:
        message = "Nao consegui identificar o telefone para consultar seus agendamentos."
        return {
            "assistant_message": message,
            "messages": [AIMessage(content=message)],
            "next_step": "end",
        }

    appointments = get_user_appointments(phone)
    for appointment in appointments:
        appointment["appointment_br"] = _format_appointment_datetime(appointment["appointment_time"])

    message = format_appointments_message(appointments)
    return {
        "assistant_message": message,
        "messages": [AIMessage(content=message)],
        "user_appointments": appointments,
        "pending_action": None,
        "next_step": "end",
    }


def cancel_appointment_node(state: ClinivetState):
    phone = _resolve_phone_from_state(state)
    if not phone:
        message = "Nao consegui identificar o telefone para cancelar o agendamento."
        return {
            "assistant_message": message,
            "messages": [AIMessage(content=message)],
            "next_step": "end",
        }

    appointments = get_user_appointments(phone)
    if not appointments:
        message = "Nao encontrei agendamentos ativos para cancelar."
        return {
            "assistant_message": message,
            "messages": [AIMessage(content=message)],
            "pending_action": None,
            "next_step": "end",
        }

    appointment_id = state.get("selected_appointment_id")
    if appointment_id is None:
        if len(appointments) == 1:
            appointment_id = appointments[0]["id"]
        else:
            for appointment in appointments:
                appointment["appointment_br"] = _format_appointment_datetime(
                    appointment["appointment_time"]
                )
            message = format_appointments_message(appointments) + "\nInforme o ID que deseja cancelar."
            return {
                "assistant_message": message,
                "messages": [AIMessage(content=message)],
                "user_appointments": appointments,
                "pending_action": "cancel_appointment",
                "next_step": "end",
            }

    appointment = get_appointment_by_id(appointment_id)
    if not appointment:
        message = "Nao encontrei o agendamento informado."
        return {
            "assistant_message": message,
            "messages": [AIMessage(content=message)],
            "pending_action": None,
            "next_step": "end",
        }

    cancel_appointment_record(appointment_id)
    if appointment.get("google_event_id"):
        try:
            get_calendar_service(appointment.get("service_name")).delete_event(appointment["google_event_id"])
        except Exception as exc:
            logger.exception("Failed to delete Google Calendar event: %s", exc)

    message = f"Agendamento {appointment_id} cancelado com sucesso."
    return {
        "assistant_message": message,
        "messages": [AIMessage(content=message)],
        "pending_action": None,
        "selected_appointment_id": None,
        "next_step": "end",
    }


def reschedule_appointment_node(state: ClinivetState):
    phone = _resolve_phone_from_state(state)
    if not phone:
        message = "Nao consegui identificar o telefone para remarcar o agendamento."
        return {
            "assistant_message": message,
            "messages": [AIMessage(content=message)],
            "next_step": "end",
        }

    appointments = get_user_appointments(phone)
    if not appointments:
        message = "Nao encontrei agendamentos ativos para remarcar."
        return {
            "assistant_message": message,
            "messages": [AIMessage(content=message)],
            "pending_action": None,
            "next_step": "end",
        }

    appointment_id = state.get("selected_appointment_id")
    if appointment_id is None:
        if len(appointments) == 1:
            appointment_id = appointments[0]["id"]
        else:
            for appointment in appointments:
                appointment["appointment_br"] = _format_appointment_datetime(
                    appointment["appointment_time"]
                )
            message = format_appointments_message(appointments) + "\nInforme o ID que deseja remarcar."
            return {
                "assistant_message": message,
                "messages": [AIMessage(content=message)],
                "user_appointments": appointments,
                "pending_action": "reschedule_appointment",
                "next_step": "end",
            }

    appointment = get_appointment_by_id(appointment_id)
    if not appointment:
        message = "Nao encontrei o agendamento informado."
        return {
            "assistant_message": message,
            "messages": [AIMessage(content=message)],
            "pending_action": None,
            "next_step": "end",
        }

    selected_slot = state.get("selected_slot")
    time_preference = state.get("time_preference")
    appointment_date = state.get("appointment_date") or build_next_business_day()
    service_name = appointment.get("service_name") or DEFAULT_SERVICE
    available_slots = state.get("available_slots") or []

    if not selected_slot:
        if not time_preference:
            message = "Para remarcar, voce prefere manha, tarde ou qualquer horario?"
            return {
                "assistant_message": message,
                "messages": [AIMessage(content=message)],
                "pending_action": "reschedule_appointment",
                "selected_appointment_id": appointment_id,
                "next_step": "end",
            }

        available_slots = find_available_slots(
            date=appointment_date,
            period=time_preference,
            service_name=service_name,
        )
        if not available_slots:
            message = "Nao encontrei horarios disponiveis para essa preferencia."
            return {
                "assistant_message": message,
                "messages": [AIMessage(content=message)],
                "pending_action": None,
                "next_step": "end",
            }

        message = format_slots_message(available_slots)
        return {
            "assistant_message": message,
            "messages": [AIMessage(content=message)],
            "available_slots": available_slots,
            "appointment_date": appointment_date,
            "service_name": service_name,
            "selected_appointment_id": appointment_id,
            "pending_action": "reschedule_appointment",
            "next_step": "end",
        }

    if available_slots and selected_slot not in available_slots:
        message = "Escolha um dos horarios sugeridos para remarcar."
        return {
            "assistant_message": message,
            "messages": [AIMessage(content=message)],
            "pending_action": "reschedule_appointment",
            "selected_appointment_id": appointment_id,
            "next_step": "end",
        }

    new_time_iso = build_slot_datetime(appointment_date, selected_slot).isoformat()
    try:
        reschedule_appointment_record(appointment_id, new_time_iso)
    except ValueError:
        message = "Esse horario nao esta mais disponivel. Posso te sugerir outras opcoes."
        return {
            "assistant_message": message,
            "messages": [AIMessage(content=message)],
            "pending_action": "reschedule_appointment",
            "selected_appointment_id": appointment_id,
            "next_step": "end",
        }

    if appointment.get("google_event_id"):
        try:
            get_calendar_service(service_name).update_event(
                event_id=appointment["google_event_id"],
                summary=_build_event_summary(service_name, appointment.get("pet_name")),
                start_time=build_slot_datetime(appointment_date, selected_slot),
                duration_minutes=appointment.get("duration_minutes", APPOINTMENT_DURATION_MINUTES),
            )
        except Exception as exc:
            logger.exception("Failed to update Google Calendar event: %s", exc)

    message = (
        f"Agendamento {appointment_id} remarcado para "
        f"{_format_appointment_datetime(new_time_iso)}."
    )
    return {
        "assistant_message": message,
        "messages": [AIMessage(content=message)],
        "pending_action": None,
        "selected_appointment_id": None,
        "next_step": "end",
    }


def load_pet_history_node(state: ClinivetState):
    phone = _resolve_phone_from_state(state)
    if not phone:
        message = "Nao consegui identificar o telefone para consultar o historico."
        return {
            "assistant_message": message,
            "messages": [AIMessage(content=message)],
            "next_step": "end",
        }

    history = load_pet_history(phone)
    message = format_pet_history_message(history)
    return {
        "assistant_message": message,
        "messages": [AIMessage(content=message)],
        "pet_history": history,
        "pending_action": None,
        "next_step": "end",
    }


def router(
    state: ClinivetState,
) -> Literal[
    "ask_missing_data",
    "ask_time_preference",
    "suggest_slots",
    "confirm_slot",
    "scheduling",
    "conversion",
    "cancel_appointment",
    "reschedule_appointment",
    "check_appointment",
    "load_pet_history",
    "end",
]:
    if state.get("urgency_level") == "emergency":
        return "end"
    return state.get("next_step", "end")


workflow = StateGraph(ClinivetState)
workflow.add_node("triage", triage_node)
workflow.add_node("ask_missing_data", ask_missing_data_node)
workflow.add_node("ask_time_preference", ask_time_preference_node)
workflow.add_node("suggest_slots", suggest_slots_node)
workflow.add_node("confirm_slot", confirm_slot_node)
workflow.add_node("scheduling", scheduling_node)
workflow.add_node("conversion", conversion_node)
workflow.add_node("cancel_appointment", cancel_appointment_node)
workflow.add_node("reschedule_appointment", reschedule_appointment_node)
workflow.add_node("check_appointment", check_appointment_node)
workflow.add_node("load_pet_history", load_pet_history_node)

workflow.set_entry_point("triage")

workflow.add_conditional_edges(
    "triage",
    router,
    {
        "ask_missing_data": "ask_missing_data",
        "ask_time_preference": "ask_time_preference",
        "suggest_slots": "suggest_slots",
        "confirm_slot": "confirm_slot",
        "scheduling": "scheduling",
        "cancel_appointment": "cancel_appointment",
        "reschedule_appointment": "reschedule_appointment",
        "check_appointment": "check_appointment",
        "load_pet_history": "load_pet_history",
        "end": END,
    },
)
workflow.add_edge("ask_missing_data", END)
workflow.add_edge("ask_time_preference", END)
workflow.add_edge("suggest_slots", END)
workflow.add_edge("confirm_slot", END)

workflow.add_conditional_edges(
    "scheduling",
    router,
    {
        "conversion": "conversion",
        "end": END,
    },
)
workflow.add_edge("conversion", END)
workflow.add_edge("cancel_appointment", END)
workflow.add_edge("reschedule_appointment", END)
workflow.add_edge("check_appointment", END)
workflow.add_edge("load_pet_history", END)

checkpointer = MemorySaver()
clinivet_agent = workflow.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    logger.info("Clinivet Brain loaded.")
