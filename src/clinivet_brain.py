from typing import Annotated, TypedDict, List, Literal, Optional
from datetime import datetime, timedelta

from langgraph.graph import StateGraph, END
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
    service_suggested: Optional[str] = "Consulta"
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

Extraia da mensagem:

- nome do tutor
- nome do pet
- espécie (cão ou gato)
- sintomas
- nível de urgência

Urgências incluem:
- convulsão
- hemorragia
- dificuldade respiratória
- trauma grave

Se houver risco imediato marque urgency_level = "emergency".
Caso contrário marque "routine".
"""


# =================================
# TRIAGEM
# =================================

def triage_node(state: ClinivetState):

    system_msg = SystemMessage(content=TRIAGE_PROMPT)

    conversation: List[BaseMessage] = [system_msg] + list(state.get("messages") or [])

    triage_result: TriageOutput = structured_llm.invoke(conversation)

    print("TRIAGE RESULT:", triage_result)

    lead_id = state.get("lead_id")

    # telefone normalmente vem do WhatsApp
    phone_number = (
        triage_result.phone
        or state.get("thread_id")
        or "unknown"
    )

    # se ainda não existe lead
    if not lead_id:

        if triage_result.tutor_name and triage_result.pet_name:

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

        else:

            print("Dados insuficientes para criar lead.")

    next_step = "scheduling" if triage_result.urgency_level == "routine" else "end"

    return {
        "triage_data": triage_result,
        "lead_id": lead_id,
        "urgency_level": triage_result.urgency_level,
        "next_step": next_step,
    }


# =================================
# BUSCAR HORÁRIOS
# =================================

def scheduling_node(state: ClinivetState):

    calendar_service = get_calendar_service()

    tomorrow = (datetime.now(TIMEZONE) + timedelta(days=1)).strftime("%Y-%m-%d")

    try:

        available_slots = calendar_service.get_free_slots(tomorrow)

    except Exception as e:

        print("Erro ao consultar agenda:", e)

        raise

    print("SLOTS DISPONÍVEIS:", available_slots)

    if not available_slots:

        raise RuntimeError("Nenhum horário disponível.")

    return {
        "next_step": "conversion",
        "available_slots": available_slots,
        "appointment_date": tomorrow,
    }


# =================================
# CONVERSÃO (AGENDAMENTO)
# =================================

def conversion_node(state: ClinivetState):

    lead_id = state.get("lead_id")
    triage_data = state.get("triage_data")
    available_slots = state.get("available_slots")
    appointment_date = state.get("appointment_date")

    if not lead_id:

        raise ValueError("Lead não encontrado.")

    if not triage_data:

        raise ValueError("Dados de triagem ausentes.")

    selected_slot = available_slots[0]

    appointment_datetime = build_slot_datetime(
        appointment_date,
        selected_slot
    )

    calendar_service = get_calendar_service()

    try:

        event_id = calendar_service.create_event(
            summary=f"{triage_data.service_suggested} - {triage_data.pet_name}",
            start_time=appointment_datetime,
            duration_minutes=30,
        )

        print("Evento criado:", event_id)

    except Exception as e:

        print("Erro ao criar evento no Google Calendar:", e)

        event_id = None

    service_id = get_service_id_by_name(
        triage_data.service_suggested or "Consulta"
    )

    confirm_appointment(
        lead_id=lead_id,
        service_id=service_id,
        appointment_time=appointment_datetime.isoformat(),
        duration_minutes=30,
        google_event_id=event_id,
    )

    update_lead_status(lead_id, "Agendado")

    print("Agendamento finalizado.")

    return {
        "next_step": "end",
    }


# =================================
# ROTEADOR
# =================================

def router(state: ClinivetState) -> Literal["scheduling", "conversion", "end"]:

    if state.get("urgency_level") == "emergency":

        return "end"

    return state.get("next_step", "end")


# =================================
# GRAFO
# =================================

workflow = StateGraph(ClinivetState)

workflow.add_node("triage", triage_node)
workflow.add_node("scheduling", scheduling_node)
workflow.add_node("conversion", conversion_node)

workflow.set_entry_point("triage")

workflow.add_conditional_edges(
    "triage",
    router,
    {
        "scheduling": "scheduling",
        "end": END,
    },
)

workflow.add_conditional_edges(
    "scheduling",
    router,
    {
        "conversion": "conversion",
        "end": END,
    },
)

workflow.add_edge("conversion", END)

clinivet_agent = workflow.compile()


if __name__ == "__main__":
    print("Clinivet Brain carregado.")