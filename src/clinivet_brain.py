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
    get_active_appointment_by_phone,
    get_appointment_by_id,
    get_lead_by_cpf,
    get_lead_by_phone,
    get_pets_by_phone,
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
    build_greeting_message,
    detect_intent,
    detect_requested_service,
    extract_multiple_pet_mentions,
    extract_single_pet_choice,
    extract_appointment_id,
    parse_natural_date,
    extract_time_choice,
    extract_time_preference,
    format_appointments_message,
    format_pet_history_message,
    format_slots_message,
    get_latest_human_message,
    is_explicit_new_schedule_request,
    is_conversation_closing,
    is_clean_time_choice,
    is_greeting_only,
    is_non_progress_message,
    is_user_frustrated,
    split_pet_names,
    wants_slot_suggestions,
)
from src.services.pet_history_service import load_pet_history
from src.services.response_service import generate_conversational_response
from src.services.scheduling_service import (
    build_next_business_day,
    find_available_slots,
    is_valid_schedule_date,
    resolve_scheduling_context,
)
from src.services.triage_service import (
    DEFAULT_SERVICE,
    build_missing_data_message,
    extract_phone_candidate,
    get_missing_required_fields,
    has_plausible_field_answer,
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
- tutor_cpf
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
- Never fabricate CPF numbers. If the user did not provide a CPF, return null.
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
    detected_date: Optional[str]
    service_name: Optional[str]
    missing_fields: Optional[List[str]]
    time_preference: Optional[str]
    selected_slot: Optional[str]
    detected_time: Optional[str]
    selected_appointment_id: Optional[int]
    pending_action: Optional[str]
    conversation_completed: Optional[bool]
    new_schedule_active: Optional[bool]
    onboarding_started: Optional[bool]
    onboarding_guidance_needed: Optional[bool]
    user_appointments: Optional[List[dict]]
    pet_history: Optional[dict]
    assistant_message: Optional[str]
    pending_pet_names: Optional[List[str]]


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


def _respond(action: str, base_message: str, **context) -> str:
    return generate_conversational_response(action=action, base_message=base_message, context=context)


def _calendar_failure_message(action: str = "schedule") -> str:
    return _respond(
        action,
        "Tive uma instabilidade ao consultar a agenda agora. Se quiser, posso tentar novamente em instantes.",
    )


def _build_consultation_details_invite(triage_data: Optional[TriageOutput]) -> str:
    if not triage_data:
        return ""
    if (triage_data.service_suggested or DEFAULT_SERVICE) != "Consulta":
        return ""

    missing_details = []
    if not triage_data.pet_breed:
        missing_details.append("raca")
    if triage_data.pet_weight is None:
        missing_details.append("peso")
    if not triage_data.pet_age:
        missing_details.append("idade")
    if not triage_data.symptoms_summary:
        missing_details.append("sintomas")

    if not missing_details:
        return ""

    return (
        " Se quiser agilizar a consulta, voce tambem pode me contar raca, peso, idade "
        "e os sintomas que estiver percebendo."
    )


def _services_match(requested_service: Optional[str], current_service: Optional[str]) -> bool:
    return (requested_service or DEFAULT_SERVICE).strip().lower() == (current_service or DEFAULT_SERVICE).strip().lower()


def _carry_detected_date(pending_action: Optional[str]) -> bool:
    return pending_action in {
        "awaiting_time_preference",
        "confirm_slot",
        "reschedule_appointment",
        "awaiting_reschedule_date",
    }


def _carry_selected_appointment_id(intent: Optional[str], pending_action: Optional[str]) -> bool:
    return intent in {"cancel", "reschedule", "check_appointment"} or pending_action in {
        "cancel_appointment",
        "reschedule_appointment",
        "awaiting_reschedule_date",
    }


def _load_tutor_memory(phone: Optional[str], tutor_cpf: Optional[str] = None) -> tuple[Optional[dict], List[dict]]:
    if not phone and not tutor_cpf:
        return None, []

    lead = None
    if phone:
        try:
            lead = get_lead_by_phone(phone)
        except Exception as exc:
            logger.warning("Failed to load lead memory: %s", exc)
            lead = None

    if lead is None and tutor_cpf:
        try:
            lead = get_lead_by_cpf(tutor_cpf)
        except Exception as exc:
            logger.warning("Failed to load lead memory by CPF: %s", exc)
            lead = None

    try:
        pets_phone = phone or (lead.get("phone") if lead else None)
        pets = get_pets_by_phone(pets_phone) if pets_phone else []
    except Exception as exc:
        logger.warning("Failed to load pet memory: %s", exc)
        pets = []

    return lead, pets


def _hydrate_triage_from_memory(
    triage_data: TriageOutput,
    *,
    phone: Optional[str],
) -> TriageOutput:
    lead, pets = _load_tutor_memory(phone, triage_data.tutor_cpf)
    if not lead and not pets:
        return triage_data

    known_pet = None
    pet_name = (triage_data.pet_name or "").strip().lower()
    if pet_name:
        for pet in pets:
            if (pet.get("name") or "").strip().lower() == pet_name:
                known_pet = pet
                break
    elif len(pets) == 1:
        known_pet = pets[0]

    hydrated = TriageOutput(
        tutor_name=triage_data.tutor_name or (lead.get("tutor_name") if lead else None),
        tutor_cpf=triage_data.tutor_cpf or (lead.get("tutor_cpf") if lead else None),
        pet_name=triage_data.pet_name or (known_pet.get("name") if known_pet else None),
        pet_species=(
            triage_data.pet_species
            if triage_data.pet_species != "Desconhecido"
            else (known_pet.get("species") if known_pet and known_pet.get("species") else "Desconhecido")
        ),
        urgency_level=triage_data.urgency_level,
        service_suggested=triage_data.service_suggested,
        symptoms_summary=triage_data.symptoms_summary,
        phone=triage_data.phone or phone,
        pet_weight=triage_data.pet_weight if triage_data.pet_weight is not None else (known_pet.get("weight") if known_pet else None),
        pet_breed=triage_data.pet_breed or (known_pet.get("breed") if known_pet else None),
        pet_age=triage_data.pet_age or (known_pet.get("age") if known_pet else None),
        pet_size=triage_data.pet_size or (known_pet.get("size") if known_pet else None),
    )
    return normalize_triage_result(hydrated)


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
    parsed_date = parse_natural_date(latest_message)
    detected_date = parsed_date or (
        state.get("detected_date") if _carry_detected_date(pending_action) else None
    )
    explicit_appointment_id = extract_appointment_id(latest_message)
    selected_appointment_id = (
        explicit_appointment_id
        if explicit_appointment_id is not None
        else (
            state.get("selected_appointment_id")
            if _carry_selected_appointment_id(intent, pending_action)
            else None
        )
    )
    conversation_completed = bool(state.get("conversation_completed"))
    new_schedule_active = bool(state.get("new_schedule_active"))
    onboarding_started = bool(state.get("onboarding_started"))
    onboarding_guidance_needed = bool(state.get("onboarding_guidance_needed"))
    pending_pet_names = state.get("pending_pet_names") or []
    reset_for_new_schedule = False
    current_lead_id = None if new_schedule_active else state.get("lead_id")
    current_pet_id = None if new_schedule_active else state.get("pet_id")
    phone_hint = extract_phone_candidate(state.get("thread_id"))
    known_lead, known_pets = _load_tutor_memory(phone_hint)
    require_guided_onboarding = (
        not new_schedule_active
        and (
            known_lead is None
            or not known_lead.get("tutor_name")
            or not known_lead.get("tutor_cpf")
        )
    )

    if (
        not state.get("triage_data")
        and not state.get("lead_id")
        and not pending_action
        and is_greeting_only(latest_message)
    ):
        greeting_message = _respond("greeting", build_greeting_message(latest_message))
        return {
            "assistant_message": greeting_message,
            "messages": [AIMessage(content=greeting_message)],
            "intent": "greeting",
            "next_step": "end",
            "pending_action": "awaiting_initial_request",
            "onboarding_started": True,
            "onboarding_guidance_needed": True,
        }

    if pending_action == "awaiting_initial_request" and is_non_progress_message(latest_message):
        reminder_message = _respond(
            "greeting",
            "Posso te ajudar com agendamento, remarcacao, cancelamento ou informacoes sobre o atendimento. Para comecarmos, me diga objetivamente o que voce precisa.",
        )
        return {
            "assistant_message": reminder_message,
            "messages": [AIMessage(content=reminder_message)],
            "intent": "greeting",
            "next_step": "end",
            "pending_action": "awaiting_initial_request",
            "onboarding_started": True,
            "onboarding_guidance_needed": True,
        }

    if is_user_frustrated(latest_message):
        frustration_message = _respond(
            "frustration",
            "Sinto muito pela experiencia. Se quiser, posso retomar seu atendimento de forma mais objetiva ou encerrar por aqui.",
        )
        return {
            "assistant_message": frustration_message,
            "messages": [AIMessage(content=frustration_message)],
            "intent": "frustration",
            "next_step": "end",
            "pending_action": None,
            "available_slots": [],
            "appointment_date": None,
            "selected_slot": None,
            "detected_time": None,
            "detected_date": None,
            "selected_appointment_id": None,
            "new_schedule_active": False,
        }

    if pending_action in {"awaiting_reschedule_date", "reschedule_appointment"} and is_explicit_new_schedule_request(
        latest_message
    ):
        intent = "schedule"
        conversation_completed = False
        new_schedule_active = True
        reset_for_new_schedule = True
        current_lead_id = None
        current_pet_id = None
        pending_action = None
        selected_appointment_id = None
        detected_date = parsed_date
        selected_slot = None

    if conversation_completed and intent not in {
        "cancel",
        "reschedule",
        "check_appointment",
        "load_pet_history",
    }:
        if is_explicit_new_schedule_request(latest_message):
            conversation_completed = False
            new_schedule_active = True
            reset_for_new_schedule = True
            current_lead_id = None
            current_pet_id = None
            pending_action = None
            selected_appointment_id = None
            detected_date = parsed_date
            selected_slot = None
        else:
            if is_conversation_closing(latest_message):
                closing_message = _respond(
                    "closing",
                    "Por nada! Se precisar de mais alguma coisa, estou por aqui. Ate logo.",
                )
                return {
                    "assistant_message": closing_message,
                    "messages": [AIMessage(content=closing_message)],
                    "intent": "closing",
                    "next_step": "end",
                    "pending_action": None,
                    "available_slots": [],
                    "appointment_date": None,
                    "selected_slot": None,
                    "detected_time": None,
                    "detected_date": None,
                    "selected_appointment_id": None,
                    "conversation_completed": True,
                    "new_schedule_active": False,
                }

            already_confirmed_message = _respond(
                "post_confirmation",
                "Seu agendamento ja esta confirmado. Se quiser, posso te ajudar a consultar, remarcar ou cancelar.",
            )
            return {
                "assistant_message": already_confirmed_message,
                "messages": [AIMessage(content=already_confirmed_message)],
                "intent": "post_confirmation",
                "next_step": "end",
                "pending_action": None,
                "available_slots": [],
                "appointment_date": None,
                "selected_slot": None,
                "detected_time": None,
                "detected_date": None,
                "selected_appointment_id": None,
                "conversation_completed": True,
                "new_schedule_active": False,
            }

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
            "detected_time": selected_slot,
            "detected_date": detected_date,
            "selected_appointment_id": selected_appointment_id,
            "pending_action": pending_action,
            "conversation_completed": conversation_completed,
            "new_schedule_active": new_schedule_active,
            "onboarding_started": onboarding_started,
        }

    if pending_action == "confirm_slot" and selected_slot:
        if is_clean_time_choice(latest_message):
            return {
                "intent": "schedule",
                "next_step": "confirm_slot",
                "selected_slot": selected_slot,
                "detected_time": selected_slot,
                "time_preference": time_preference,
                "detected_date": detected_date,
                "selected_appointment_id": selected_appointment_id,
                "conversation_completed": conversation_completed,
                "new_schedule_active": new_schedule_active,
                "onboarding_started": onboarding_started,
            }

    if pending_action == "confirm_slot":
        message = _respond("schedule", "Escolha apenas um dos horarios sugeridos para eu confirmar o agendamento.")
        return {
            "assistant_message": message,
            "messages": [AIMessage(content=message)],
            "intent": "schedule",
            "next_step": "end",
            "pending_action": "confirm_slot",
            "available_slots": state.get("available_slots") or [],
            "appointment_date": state.get("appointment_date"),
            "detected_date": detected_date,
            "selected_appointment_id": selected_appointment_id,
            "conversation_completed": conversation_completed,
            "new_schedule_active": new_schedule_active,
            "onboarding_started": onboarding_started,
        }

    if pending_action == "awaiting_time_preference":
        if time_preference:
            return {
                "intent": "schedule",
                "next_step": "suggest_slots",
                "time_preference": time_preference,
                "detected_date": detected_date,
                "conversation_completed": conversation_completed,
                "new_schedule_active": new_schedule_active,
                "onboarding_started": onboarding_started,
            }
        return {
            "intent": "schedule",
            "next_step": "ask_time_preference",
            "detected_date": detected_date,
            "conversation_completed": conversation_completed,
            "new_schedule_active": new_schedule_active,
            "onboarding_started": onboarding_started,
        }

    if pending_action == "awaiting_reschedule_date":
        if detected_date:
            return {
                "intent": "reschedule",
                "next_step": "reschedule_appointment",
                "detected_date": detected_date,
                "selected_appointment_id": selected_appointment_id,
                "pending_action": "reschedule_appointment",
                "conversation_completed": conversation_completed,
                "new_schedule_active": new_schedule_active,
                "onboarding_started": onboarding_started,
            }
        return {
            "intent": "reschedule",
            "next_step": "reschedule_appointment",
            "selected_appointment_id": selected_appointment_id,
            "pending_action": "awaiting_reschedule_date",
            "conversation_completed": conversation_completed,
            "new_schedule_active": new_schedule_active,
            "onboarding_started": onboarding_started,
        }

    if pending_action == "awaiting_reschedule_or_new_service":
        normalized_message = latest_message.lower()
        if "novo agendamento" in normalized_message or "novo servico" in normalized_message:
            return {
                "intent": "schedule",
                "next_step": "ask_time_preference",
                "conversation_completed": False,
                "new_schedule_active": True,
                "pending_action": None,
                "lead_id": None,
                "pet_id": None,
                "selected_appointment_id": None,
                "available_slots": [],
                "appointment_date": None,
                "selected_slot": None,
                "detected_time": None,
                "detected_date": detected_date,
                "service_name": state.get("service_name"),
                "triage_data": state.get("triage_data"),
                "onboarding_started": onboarding_started,
            }
        if "remarcar" in normalized_message or "consulta atual" in normalized_message or "mesma consulta" in normalized_message:
            return {
                "intent": "reschedule",
                "next_step": "reschedule_appointment",
                "selected_appointment_id": selected_appointment_id,
                "pending_action": "reschedule_appointment",
                "detected_date": detected_date,
                "conversation_completed": conversation_completed,
                "new_schedule_active": False,
                "onboarding_started": onboarding_started,
            }
        clarification = _respond(
            "reschedule",
            "Posso remarcar a consulta atual ou abrir um novo agendamento para o novo servico. O que voce prefere?",
        )
        return {
            "assistant_message": clarification,
            "messages": [AIMessage(content=clarification)],
            "intent": "reschedule",
            "next_step": "end",
            "pending_action": "awaiting_reschedule_or_new_service",
            "selected_appointment_id": selected_appointment_id,
            "conversation_completed": conversation_completed,
            "new_schedule_active": False,
            "onboarding_started": onboarding_started,
        }

    if pending_action == "retry_schedule_slots":
        return {
            "intent": "schedule",
            "next_step": "scheduling",
            "detected_date": detected_date,
            "conversation_completed": conversation_completed,
            "new_schedule_active": new_schedule_active,
            "onboarding_started": onboarding_started,
            "pending_action": None,
        }

    if pending_action == "retry_suggest_slots":
        return {
            "intent": "schedule",
            "next_step": "suggest_slots",
            "detected_date": detected_date,
            "time_preference": time_preference or state.get("time_preference"),
            "conversation_completed": conversation_completed,
            "new_schedule_active": new_schedule_active,
            "onboarding_started": onboarding_started,
            "pending_action": None,
        }

    if pending_action == "retry_reschedule_slots":
        return {
            "intent": "reschedule",
            "next_step": "reschedule_appointment",
            "detected_date": detected_date,
            "time_preference": time_preference or state.get("time_preference"),
            "selected_appointment_id": selected_appointment_id,
            "conversation_completed": conversation_completed,
            "new_schedule_active": False,
            "onboarding_started": onboarding_started,
            "pending_action": "reschedule_appointment",
        }

    if pending_action in {"awaiting_missing_data", "awaiting_initial_request"}:
        current_missing_fields = state.get("missing_fields") or []
        expected_field = current_missing_fields[0] if current_missing_fields else None
        if expected_field and (
            is_non_progress_message(latest_message)
            or (
                detect_intent(latest_message) != "triage"
                and not has_plausible_field_answer(expected_field, latest_message)
            )
            or (
                not has_plausible_field_answer(expected_field, latest_message)
                and expected_field in {"tutor_cpf", "pet_species", "phone"}
            )
        ):
            message = _respond(
                "triage",
                build_missing_data_message(
                    [expected_field],
                    guided_onboarding=bool(state.get("onboarding_guidance_needed") or onboarding_started),
                ),
            )
            return {
                "assistant_message": message,
                "messages": [AIMessage(content=message)],
                "intent": "triage",
                "next_step": "end",
                "pending_action": pending_action,
                "missing_fields": current_missing_fields,
                "triage_data": state.get("triage_data"),
                "lead_id": current_lead_id,
                "pet_id": current_pet_id,
                "conversation_completed": conversation_completed,
                "new_schedule_active": new_schedule_active,
                "available_slots": state.get("available_slots") or [],
                "appointment_date": state.get("appointment_date"),
                "detected_date": detected_date,
                "selected_appointment_id": selected_appointment_id,
                "onboarding_started": onboarding_started,
                "onboarding_guidance_needed": bool(state.get("onboarding_guidance_needed")),
                "pending_pet_names": pending_pet_names,
            }

    chosen_pet_name = None
    if pending_action == "awaiting_single_pet_choice":
        chosen_pet_name = extract_single_pet_choice(latest_message, pending_pet_names)
        if not chosen_pet_name:
            listed_pet_names = ", ".join(pending_pet_names)
            message = _respond(
                "multi_pet",
                f"Para evitar erro, vamos seguir com um pet por vez. Escolha um destes pets: {listed_pet_names}.",
                pet_names=pending_pet_names,
            )
            return {
                "assistant_message": message,
                "messages": [AIMessage(content=message)],
                "intent": "schedule",
                "next_step": "end",
                "pending_action": "awaiting_single_pet_choice",
                "pending_pet_names": pending_pet_names,
                "conversation_completed": False,
                "new_schedule_active": True,
                "available_slots": [],
                "appointment_date": None,
                "selected_slot": None,
                "detected_time": None,
                "selected_appointment_id": None,
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
    triage_result = _hydrate_triage_from_memory(
        triage_result,
        phone=triage_result.phone or extract_phone_candidate(state.get("thread_id")),
    )
    if chosen_pet_name:
        triage_result.pet_name = chosen_pet_name
        triage_result = normalize_triage_result(triage_result)

    explicit_pet_mentions = extract_multiple_pet_mentions(latest_message)
    if len(explicit_pet_mentions) > 1:
        multi_pet_message = _respond(
            "multi_pet",
            "Consigo ajudar com varios pets, mas para evitar erros preciso organizar um de cada vez. Qual pet voce quer agendar primeiro?",
            pet_names=explicit_pet_mentions,
        )
        return {
            "triage_data": triage_result,
            "lead_id": current_lead_id,
            "pet_id": current_pet_id,
            "assistant_message": multi_pet_message,
            "messages": [AIMessage(content=multi_pet_message)],
            "intent": intent,
            "next_step": "end",
            "pending_action": "awaiting_single_pet_choice",
            "pending_pet_names": explicit_pet_mentions,
            "conversation_completed": False,
            "new_schedule_active": True,
            "available_slots": [],
            "appointment_date": None,
            "selected_slot": None,
            "detected_time": None,
            "selected_appointment_id": None,
        }

    pet_names = split_pet_names(triage_result.pet_name)
    if len(pet_names) > 1:
        multi_pet_message = _respond(
            "multi_pet",
            "Consigo ajudar com varios pets, mas para evitar erros preciso organizar um de cada vez. Qual pet voce quer agendar primeiro?",
            pet_names=pet_names,
        )
        return {
            "triage_data": triage_result,
            "assistant_message": multi_pet_message,
            "messages": [AIMessage(content=multi_pet_message)],
            "intent": intent,
            "next_step": "end",
            "pending_action": "awaiting_single_pet_choice",
            "pending_pet_names": pet_names,
            "conversation_completed": False,
            "new_schedule_active": True,
            "available_slots": [],
            "appointment_date": None,
            "selected_slot": None,
            "detected_time": None,
            "selected_appointment_id": None,
        }

    logger.info("TRIAGE RESULT: %s", triage_result.model_dump())

    if not (known_lead and triage_result.phone == phone_hint):
        known_lead, loaded_pets = _load_tutor_memory(triage_result.phone, triage_result.tutor_cpf)
        if loaded_pets:
            known_pets = loaded_pets
    missing_fields = get_missing_required_fields(
        triage_result,
        state.get("thread_id"),
        require_onboarding_fields=(
            onboarding_started
            or require_guided_onboarding
            or bool(known_lead and known_lead.get("tutor_name") and not known_lead.get("tutor_cpf"))
        ),
    )
    if missing_fields:
        return {
            "triage_data": triage_result,
            "lead_id": current_lead_id,
            "pet_id": current_pet_id,
            "missing_fields": missing_fields,
            "urgency_level": triage_result.urgency_level,
            "next_step": "ask_missing_data",
            "intent": intent,
            "time_preference": time_preference,
            "detected_date": detected_date,
            "assistant_message": None,
            "onboarding_started": onboarding_started or require_guided_onboarding,
            "onboarding_guidance_needed": onboarding_guidance_needed or (
                require_guided_onboarding and not onboarding_started
            ),
            "conversation_completed": conversation_completed,
            "new_schedule_active": new_schedule_active,
            "available_slots": [] if reset_for_new_schedule else state.get("available_slots"),
            "appointment_date": None if reset_for_new_schedule else state.get("appointment_date"),
            "selected_slot": None,
            "detected_time": None,
            "selected_appointment_id": None,
            "pending_action": "awaiting_missing_data",
            "pending_pet_names": pending_pet_names if pending_action == "awaiting_single_pet_choice" else [],
        }

    lead_id = current_lead_id
    if not lead_id:
        try:
            lead_id = register_lead(
                tutor_name=triage_result.tutor_name,
                tutor_cpf=triage_result.tutor_cpf,
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
        "pet_id": pet_record["id"] if pet_record else current_pet_id,
        "thread_id": state.get("thread_id"),
        "urgency_level": triage_result.urgency_level,
        "intent": intent,
        "missing_fields": [],
        "assistant_message": emergency_message,
        "time_preference": time_preference,
        "detected_date": detected_date,
        "selected_slot": selected_slot,
        "detected_time": selected_slot,
        "selected_appointment_id": selected_appointment_id,
        "pending_action": None,
        "conversation_completed": conversation_completed,
        "new_schedule_active": False,
        "onboarding_started": False,
        "onboarding_guidance_needed": False,
        "pending_pet_names": [],
        "available_slots": [] if reset_for_new_schedule else state.get("available_slots"),
        "appointment_date": None if reset_for_new_schedule else state.get("appointment_date"),
        "next_step": next_step,
    }
    if emergency_messages:
        response["messages"] = emergency_messages
    return response


def ask_missing_data_node(state: ClinivetState):
    missing_fields = state.get("missing_fields") or []
    guided_onboarding = bool(state.get("onboarding_guidance_needed"))
    message = _respond(
        "triage",
        build_missing_data_message(missing_fields, guided_onboarding=guided_onboarding),
    )
    return {
        "assistant_message": message,
        "messages": [AIMessage(content=message)],
        "onboarding_guidance_needed": False,
        "next_step": "end",
    }


def ask_time_preference_node(state: ClinivetState):
    triage_data = state.get("triage_data")
    base_message = (
        "Claro! Vamos encontrar um bom horario para o seu pet. "
        "Voce prefere atendimento pela manha, a tarde, ou qualquer horario disponivel?"
    )
    base_message += _build_consultation_details_invite(triage_data)
    message = _respond("schedule", base_message, detected_date=state.get("detected_date"))
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
    detected_date = state.get("detected_date")
    appointment_date = detected_date or state.get("appointment_date") or build_next_business_day()
    if detected_date:
        is_valid_date, validation_message = is_valid_schedule_date(appointment_date)
    else:
        is_valid_date, validation_message = True, None
    if not is_valid_date:
        message = _respond("schedule", validation_message or "Nao consegui usar essa data para agendamento.")
        return {
            "assistant_message": message,
            "messages": [AIMessage(content=message)],
            "next_step": "end",
            "pending_action": None,
        }

    try:
        slots = find_available_slots(
            date=appointment_date,
            period=time_preference,
            service_name=service_name,
        )
    except RuntimeError as exc:
        logger.warning("Failed to load suggested slots: %s", exc)
        message = _calendar_failure_message("schedule")
        return {
            "assistant_message": message,
            "messages": [AIMessage(content=message)],
            "next_step": "end",
            "pending_action": "retry_suggest_slots",
            "appointment_date": appointment_date,
            "detected_date": appointment_date,
            "time_preference": time_preference,
            "service_name": service_name,
        }

    if not slots:
        message = _respond("schedule", "Nao encontrei horarios disponiveis para essa preferencia agora.")
        return {
            "assistant_message": message,
            "messages": [AIMessage(content=message)],
            "next_step": "end",
            "pending_action": None,
        }

    message = _respond("schedule", format_slots_message(slots), slots=slots, detected_date=appointment_date)
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

    if state.get("detected_date"):
        is_valid_date, validation_message = is_valid_schedule_date(appointment_date)
    else:
        is_valid_date, validation_message = True, None
    if not is_valid_date:
        message = _respond("schedule", validation_message or "Nao consegui usar essa data para agendamento.")
        return {
            "assistant_message": message,
            "messages": [AIMessage(content=message)],
            "pending_action": None,
            "next_step": "end",
        }

    if not selected_slot:
        message = _respond("schedule", "Qual horario voce prefere entre as opcoes enviadas?")
        return {
            "assistant_message": message,
            "messages": [AIMessage(content=message)],
            "pending_action": "confirm_slot",
            "next_step": "end",
        }

    if available_slots and selected_slot not in available_slots:
        message = _respond(
            "schedule",
            "Esse horario nao esta entre as opcoes sugeridas. Escolha um dos horarios informados.",
        )
        return {
            "assistant_message": message,
            "messages": [AIMessage(content=message)],
            "pending_action": "confirm_slot",
            "next_step": "end",
        }

    if has_appointment_for_lead(lead_id):
        duplicate_message = _respond("schedule", "Voce ja possui um agendamento registrado.")
        return {
            "assistant_message": duplicate_message,
            "messages": [AIMessage(content=duplicate_message)],
            "pending_action": None,
            "next_step": "end",
        }

    appointment_time_iso = build_slot_datetime(appointment_date, selected_slot).isoformat()
    try:
        confirmation_message = _persist_confirmed_appointment(
            lead_id=lead_id,
            pet_id=state.get("pet_id"),
            triage_data=triage_data,
            service_name=service_name,
            appointment_time_iso=appointment_time_iso,
        )
    except RuntimeError:
        conflict_message = _respond(
            "schedule_conflict", "Esse horario acabou de ser reservado. Vou buscar outro disponivel."
        )
        return {
            "assistant_message": conflict_message,
            "messages": [AIMessage(content=conflict_message)],
            "pending_action": None,
            "next_step": "end",
        }

    return {
        "assistant_message": confirmation_message,
        "messages": [AIMessage(content=confirmation_message)],
        "pending_action": None,
        "available_slots": [],
        "selected_slot": None,
        "detected_time": None,
        "selected_appointment_id": None,
        "conversation_completed": True,
        "next_step": "end",
    }


def scheduling_node(state: ClinivetState):
    triage_data = state.get("triage_data")
    service_name = (triage_data.service_suggested if triage_data else None) or DEFAULT_SERVICE

    try:
        resolved_service, target_day, available_slots = resolve_scheduling_context(
            service_name, preferred_day=state.get("detected_date")
        )
    except ValueError as exc:
        message = str(exc)
        rendered_message = _respond("schedule", message)
        return {
            "assistant_message": rendered_message,
            "messages": [AIMessage(content=rendered_message)],
            "next_step": "end",
        }
    except RuntimeError as exc:
        logger.warning("Failed to load slots: %s", exc)
        message = _calendar_failure_message("schedule")
        return {
            "assistant_message": message,
            "messages": [AIMessage(content=message)],
            "next_step": "end",
            "pending_action": "retry_schedule_slots",
            "detected_date": state.get("detected_date"),
        }
    except Exception as exc:
        logger.warning("Unexpected scheduling error while loading slots: %s", exc)
        message = _calendar_failure_message("schedule")
        return {
            "assistant_message": message,
            "messages": [AIMessage(content=message)],
            "next_step": "end",
            "pending_action": "retry_schedule_slots",
            "detected_date": state.get("detected_date"),
        }

    logger.info("AVAILABLE SLOTS: %s", available_slots)

    if not available_slots:
        raise RuntimeError("Nenhum horario disponivel.")

    return {
        "next_step": "conversion",
        "available_slots": available_slots,
        "appointment_date": target_day,
        "detected_date": target_day,
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
        duplicate_message = _respond("schedule", "Voce ja possui um agendamento registrado.")
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
        no_slots_message = _respond(
            "schedule_conflict", "Esse horario acabou de ser reservado. Vou buscar outro disponivel."
        )
        return {
            "assistant_message": no_slots_message,
            "messages": [AIMessage(content=no_slots_message)],
            "next_step": "end",
        }

    pet_display_name = triage_data.pet_name or UNKNOWN_PET
    confirmation_message = _respond(
        "schedule_confirmation",
        _persist_confirmed_appointment(
        lead_id=lead_id,
        pet_id=state.get("pet_id"),
        triage_data=triage_data,
        service_name=service_name,
        appointment_time_iso=chosen_time_iso,
        appointment_record=appointment_record,
        ),
        detected_date=appointment_date,
        detected_time=chosen_time_iso,
    )
    appointment_datetime = datetime.fromisoformat(chosen_time_iso).astimezone(TIMEZONE)

    logger.info("APPOINTMENT CREATED: lead_id=%s date=%s", lead_id, appointment_datetime)

    return {
        "assistant_message": confirmation_message,
        "messages": [AIMessage(content=confirmation_message)],
        "pending_action": None,
        "available_slots": [],
        "selected_slot": None,
        "detected_time": None,
        "selected_appointment_id": None,
        "conversation_completed": True,
        "next_step": "end",
    }


def check_appointment_node(state: ClinivetState):
    phone = _resolve_phone_from_state(state)
    if not phone:
        message = _respond("check_appointment", "Nao consegui identificar o telefone para consultar seus agendamentos.")
        return {
            "assistant_message": message,
            "messages": [AIMessage(content=message)],
            "next_step": "end",
        }

    appointments = get_user_appointments(phone)
    for appointment in appointments:
        appointment["appointment_br"] = _format_appointment_datetime(appointment["appointment_time"])

    message = _respond("check_appointment", format_appointments_message(appointments), appointments=appointments)
    return {
        "assistant_message": message,
        "messages": [AIMessage(content=message)],
        "user_appointments": appointments,
        "pending_action": None,
        "available_slots": [],
        "appointment_date": None,
        "selected_slot": None,
        "detected_time": None,
        "detected_date": None,
        "next_step": "end",
    }


def cancel_appointment_node(state: ClinivetState):
    phone = _resolve_phone_from_state(state)
    if not phone:
        message = _respond("cancel", "Nao consegui identificar o telefone para cancelar o agendamento.")
        return {
            "assistant_message": message,
            "messages": [AIMessage(content=message)],
            "next_step": "end",
        }

    appointments = get_user_appointments(phone)
    if not appointments:
        message = _respond("cancel", "Nao encontrei agendamentos ativos para cancelar.")
        return {
            "assistant_message": message,
            "messages": [AIMessage(content=message)],
            "pending_action": None,
            "next_step": "end",
        }

    appointment_id = state.get("selected_appointment_id")
    appointment = get_appointment_by_id(appointment_id) if appointment_id is not None else None

    if appointment is None:
        appointment = get_active_appointment_by_phone(phone)
        if appointment:
            appointment_id = appointment["id"]

    if not appointment:
        message = _respond("cancel", "Nao encontrei o agendamento informado.")
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

    message = _respond("cancel", f"Agendamento {appointment_id} cancelado com sucesso.")
    return {
        "assistant_message": message,
        "messages": [AIMessage(content=message)],
        "pending_action": None,
        "available_slots": [],
        "selected_slot": None,
        "detected_time": None,
        "detected_date": None,
        "appointment_date": None,
        "selected_appointment_id": None,
        "conversation_completed": False,
        "next_step": "end",
    }


def reschedule_appointment_node(state: ClinivetState):
    phone = _resolve_phone_from_state(state)
    if not phone:
        message = _respond("reschedule", "Nao consegui identificar o telefone para remarcar o agendamento.")
        return {
            "assistant_message": message,
            "messages": [AIMessage(content=message)],
            "next_step": "end",
        }

    appointments = get_user_appointments(phone)
    if not appointments:
        message = _respond("reschedule", "Nao encontrei agendamentos ativos para remarcar.")
        return {
            "assistant_message": message,
            "messages": [AIMessage(content=message)],
            "pending_action": None,
            "next_step": "end",
        }

    appointment_id = state.get("selected_appointment_id")
    appointment = get_appointment_by_id(appointment_id) if appointment_id is not None else None

    if appointment is None:
        appointment = get_active_appointment_by_phone(phone)
        if appointment:
            appointment_id = appointment["id"]

    if not appointment:
        message = _respond("reschedule", "Nao encontrei o agendamento informado.")
        return {
            "assistant_message": message,
            "messages": [AIMessage(content=message)],
            "pending_action": None,
            "next_step": "end",
        }

    selected_slot = state.get("selected_slot")
    time_preference = state.get("time_preference")
    detected_date = state.get("detected_date")
    appointment_date = detected_date or state.get("appointment_date") or build_next_business_day()
    service_name = appointment.get("service_name") or DEFAULT_SERVICE
    available_slots = state.get("available_slots") or []
    latest_message = get_latest_human_message(state.get("messages") or [])
    requested_service = detect_requested_service(latest_message)

    if requested_service and not _services_match(requested_service, service_name):
        triage_data = state.get("triage_data")
        if triage_data:
            triage_data = triage_data.model_copy(update={"service_suggested": requested_service})
        message = _respond(
            "reschedule",
            f"Posso remarcar a {service_name.lower()} atual ou abrir um novo agendamento para {requested_service}. O que voce prefere?",
        )
        return {
            "assistant_message": message,
            "messages": [AIMessage(content=message)],
            "pending_action": "awaiting_reschedule_or_new_service",
            "selected_appointment_id": appointment_id,
            "detected_date": detected_date,
            "service_name": requested_service,
            "triage_data": triage_data,
            "next_step": "end",
        }

    if not selected_slot:
        if not state.get("detected_date") and not time_preference:
            message = _respond(
                "reschedule",
                "Claro! Vamos remarcar a consulta do seu pet. Qual novo dia voce prefere?",
                appointment_id=appointment_id,
            )
            return {
                "assistant_message": message,
                "messages": [AIMessage(content=message)],
                "pending_action": "awaiting_reschedule_date",
                "selected_appointment_id": appointment_id,
                "next_step": "end",
            }

        if not time_preference:
            message = _respond(
                "reschedule",
                "Claro! Vamos remarcar a consulta do seu pet. "
                "Voce prefere atendimento pela manha, a tarde, ou qualquer horario disponivel?",
                detected_date=appointment_date,
            )
            return {
                "assistant_message": message,
                "messages": [AIMessage(content=message)],
                "pending_action": "reschedule_appointment",
                "selected_appointment_id": appointment_id,
                "next_step": "end",
            }

        if detected_date:
            is_valid_date, validation_message = is_valid_schedule_date(appointment_date)
        else:
            is_valid_date, validation_message = True, None
        if not is_valid_date:
            message = _respond("reschedule", validation_message or "Nao consegui usar essa data para remarcacao.")
            return {
                "assistant_message": message,
                "messages": [AIMessage(content=message)],
                "pending_action": None,
                "selected_appointment_id": appointment_id,
                "next_step": "end",
            }

        try:
            available_slots = find_available_slots(
                date=appointment_date,
                period=time_preference,
                service_name=service_name,
            )
        except RuntimeError as exc:
            logger.warning("Failed to load reschedule slots: %s", exc)
            message = _calendar_failure_message("reschedule")
            return {
                "assistant_message": message,
                "messages": [AIMessage(content=message)],
                "pending_action": "retry_reschedule_slots",
                "selected_appointment_id": appointment_id,
                "detected_date": appointment_date,
                "time_preference": time_preference,
                "next_step": "end",
            }
        if not available_slots:
            message = _respond("reschedule", "Nao encontrei horarios disponiveis para essa preferencia.")
            return {
                "assistant_message": message,
                "messages": [AIMessage(content=message)],
                "pending_action": None,
                "next_step": "end",
            }

        message = _respond(
            "reschedule",
            format_slots_message(available_slots),
            slots=available_slots,
            detected_date=appointment_date,
        )
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
        message = _respond("reschedule", "Escolha um dos horarios sugeridos para remarcar.")
        return {
            "assistant_message": message,
            "messages": [AIMessage(content=message)],
            "pending_action": "reschedule_appointment",
            "selected_appointment_id": appointment_id,
            "next_step": "end",
        }

    if detected_date:
        is_valid_date, validation_message = is_valid_schedule_date(appointment_date)
    else:
        is_valid_date, validation_message = True, None
    if not is_valid_date:
        message = _respond("reschedule", validation_message or "Nao consegui usar essa data para remarcacao.")
        return {
            "assistant_message": message,
            "messages": [AIMessage(content=message)],
            "pending_action": None,
            "selected_appointment_id": appointment_id,
            "next_step": "end",
        }

    new_time_iso = build_slot_datetime(appointment_date, selected_slot).isoformat()
    try:
        reschedule_appointment_record(appointment_id, new_time_iso)
    except ValueError:
        message = _respond(
            "schedule_conflict", "Esse horario acabou de ser reservado. Vou buscar outro disponivel."
        )
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

    message = _respond(
        "reschedule_confirmation",
        f"Agendamento {appointment_id} remarcado para "
        f"{_format_appointment_datetime(new_time_iso)}.",
        detected_date=appointment_date,
        detected_time=selected_slot,
    )
    return {
        "assistant_message": message,
        "messages": [AIMessage(content=message)],
        "pending_action": None,
        "available_slots": [],
        "selected_slot": None,
        "detected_time": None,
        "selected_appointment_id": None,
        "conversation_completed": True,
        "next_step": "end",
    }


def load_pet_history_node(state: ClinivetState):
    phone = _resolve_phone_from_state(state)
    if not phone:
        message = _respond("load_pet_history", "Nao consegui identificar o telefone para consultar o historico.")
        return {
            "assistant_message": message,
            "messages": [AIMessage(content=message)],
            "next_step": "end",
        }

    history = load_pet_history(phone)
    message = _respond("load_pet_history", format_pet_history_message(history), history=history)
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
