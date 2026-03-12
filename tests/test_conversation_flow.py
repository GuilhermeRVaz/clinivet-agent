from datetime import datetime

import pytz
from langchain_core.messages import HumanMessage

from src.clinivet_brain import TriageOutput, clinivet_agent


class SequenceStructuredLLM:
    def __init__(self, results):
        self.results = results
        self.index = 0

    def invoke(self, _conversation):
        result = self.results[self.index]
        self.index += 1
        return result


class FakeCalendarService:
    def get_free_slots(self, _day: str):
        return ["10:00"]

    def create_event(self, **_kwargs):
        return "evt_conversation"


def test_conversation_slot_filling_then_schedule(monkeypatch):
    thread_id = "5514999900000"
    llm_sequence = SequenceStructuredLLM(
        [
            TriageOutput(
                tutor_name=None,
                tutor_cpf=None,
                pet_name=None,
                pet_species="Cao",
                urgency_level="routine",
                service_suggested="Consulta",
                phone=None,
            ),
            TriageOutput(
                tutor_name="Joao",
                tutor_cpf=None,
                pet_name=None,
                pet_species="Cao",
                urgency_level="routine",
                service_suggested="Consulta",
                phone=None,
            ),
            TriageOutput(
                tutor_name=None,
                tutor_cpf="111.222.333-44",
                pet_name=None,
                pet_species="Cao",
                urgency_level="routine",
                service_suggested="Consulta",
                phone=None,
            ),
            TriageOutput(
                tutor_name=None,
                tutor_cpf=None,
                pet_name="Rex",
                pet_species="Cao",
                urgency_level="routine",
                service_suggested="Consulta",
                phone=None,
            ),
        ]
    )

    monkeypatch.setattr("src.clinivet_brain.structured_llm", llm_sequence)
    monkeypatch.setattr("src.clinivet_brain.get_lead_by_phone", lambda _phone: None)
    monkeypatch.setattr("src.clinivet_brain.get_pets_by_phone", lambda _phone: [])
    monkeypatch.setattr("src.clinivet_brain.register_lead", lambda **_kwargs: 77)
    monkeypatch.setattr(
        "src.clinivet_brain.resolve_scheduling_context",
        lambda _service_name, preferred_day=None: ("Consulta", preferred_day or "2030-01-01", ["10:00"]),
    )
    monkeypatch.setattr(
        "src.clinivet_brain.get_calendar_service", lambda *_args, **_kwargs: FakeCalendarService()
    )
    monkeypatch.setattr("src.clinivet_brain.get_service_id_by_name", lambda _name: 5)
    monkeypatch.setattr("src.clinivet_brain.has_appointment_for_lead", lambda _lead_id: False)
    monkeypatch.setattr("src.clinivet_brain.confirm_appointment", lambda **_kwargs: {"id": 500})
    monkeypatch.setattr("src.clinivet_brain.set_appointment_google_event_id", lambda *_args, **_kwargs: True)
    monkeypatch.setattr("src.clinivet_brain.update_lead_status", lambda _lead_id, _status: True)
    monkeypatch.setattr(
        "src.clinivet_brain.build_slot_datetime",
        lambda day, slot: pytz.timezone("America/Sao_Paulo").localize(
            datetime.strptime(f"{day} {slot}", "%Y-%m-%d %H:%M")
        ),
    )

    config = {"configurable": {"thread_id": thread_id}}

    turn_1 = clinivet_agent.invoke(
        {"messages": [HumanMessage(content="Meu cachorro esta vomitando.")], "thread_id": thread_id},
        config=config,
    )
    assert "dados essenciais" in turn_1["assistant_message"].lower()
    assert "qual e o seu nome" in turn_1["assistant_message"].lower()

    turn_2 = clinivet_agent.invoke(
        {"messages": [HumanMessage(content="Joao")], "thread_id": thread_id},
        config=config,
    )
    assert "cpf do tutor" in turn_2["assistant_message"].lower()

    turn_3 = clinivet_agent.invoke(
        {"messages": [HumanMessage(content="111.222.333-44")], "thread_id": thread_id},
        config=config,
    )
    assert "nome do seu pet" in turn_3["assistant_message"].lower()

    turn_4 = clinivet_agent.invoke(
        {"messages": [HumanMessage(content="Rex")], "thread_id": thread_id},
        config=config,
    )
    assert "Agendamento confirmado" in turn_4["assistant_message"]
    assert turn_4["lead_id"] == 77


def test_direct_schedule_starts_guided_onboarding(monkeypatch):
    thread_id = "conversation-thread-guided-onboarding"
    llm_sequence = SequenceStructuredLLM(
        [
            TriageOutput(
                tutor_name=None,
                tutor_cpf=None,
                pet_name=None,
                pet_species="Gato",
                urgency_level="routine",
                service_suggested="Consulta",
                phone=None,
            )
        ]
    )

    monkeypatch.setattr("src.clinivet_brain.structured_llm", llm_sequence)
    monkeypatch.setattr("src.clinivet_brain.get_lead_by_phone", lambda _phone: None)
    monkeypatch.setattr("src.clinivet_brain.get_pets_by_phone", lambda _phone: [])

    result = clinivet_agent.invoke(
        {
            "messages": [HumanMessage(content="quero marcar consulta para meu gato")],
            "thread_id": thread_id,
        },
        config={"configurable": {"thread_id": thread_id}},
    )

    assert "dados essenciais" in result["assistant_message"].lower()
    assert "responda de forma direta" in result["assistant_message"].lower()
    assert "qual e o seu nome" in result["assistant_message"].lower()


def test_memory_timeout_does_not_break_guided_onboarding(monkeypatch):
    thread_id = "conversation-thread-memory-timeout"
    llm_sequence = SequenceStructuredLLM(
        [
            TriageOutput(
                tutor_name=None,
                tutor_cpf=None,
                pet_name=None,
                pet_species="Desconhecido",
                urgency_level="routine",
                service_suggested="Consulta",
                phone=None,
            )
        ]
    )

    monkeypatch.setattr("src.clinivet_brain.structured_llm", llm_sequence)
    monkeypatch.setattr(
        "src.clinivet_brain.get_lead_by_phone",
        lambda _phone: (_ for _ in ()).throw(TimeoutError("supabase timeout")),
    )
    monkeypatch.setattr("src.clinivet_brain.get_pets_by_phone", lambda _phone: [])

    result = clinivet_agent.invoke(
        {
            "messages": [HumanMessage(content="quero marcar consulta")],
            "thread_id": thread_id,
        },
        config={"configurable": {"thread_id": thread_id}},
    )

    assert "dados essenciais" in result["assistant_message"].lower()
    assert "qual e o seu nome" in result["assistant_message"].lower()


def test_cpf_memory_hydrates_existing_tutor_from_other_phone(monkeypatch):
    thread_id = "conversation-thread-cpf-memory"
    llm_sequence = SequenceStructuredLLM(
        [
            TriageOutput(
                tutor_name="Marina Silva",
                tutor_cpf=None,
                pet_name=None,
                pet_species="Desconhecido",
                urgency_level="routine",
                service_suggested="Consulta",
                phone=None,
            ),
            TriageOutput(
                tutor_name=None,
                tutor_cpf="123.456.789-09",
                pet_name=None,
                pet_species="Desconhecido",
                urgency_level="routine",
                service_suggested="Consulta",
                phone=None,
            ),
        ]
    )

    monkeypatch.setattr("src.clinivet_brain.structured_llm", llm_sequence)
    monkeypatch.setattr("src.clinivet_brain.get_lead_by_phone", lambda _phone: None)
    monkeypatch.setattr(
        "src.clinivet_brain.get_lead_by_cpf",
        lambda _cpf: {
            "id": 900,
            "tutor_name": "Marina Silva",
            "tutor_cpf": "12345678909",
            "phone": "5514997000999",
            "pet_name": "Nino",
            "pet_species": "Gato",
        },
    )
    monkeypatch.setattr(
        "src.clinivet_brain.get_pets_by_phone",
        lambda phone: [{"id": 91, "name": "Nino", "species": "Gato"}] if phone == "5514997000999" else [],
    )

    config = {"configurable": {"thread_id": thread_id}}

    first = clinivet_agent.invoke(
        {
            "messages": [HumanMessage(content="quero marcar consulta")],
            "thread_id": thread_id,
        },
        config=config,
    )
    assert "cpf do tutor" in first["assistant_message"].lower()

    second = clinivet_agent.invoke(
        {
            "messages": [HumanMessage(content="123.456.789-09")],
            "thread_id": thread_id,
        },
        config=config,
    )
    assert "nome do seu pet" not in second["assistant_message"].lower()
    assert "nino" in second["assistant_message"].lower() or "especie" not in second["assistant_message"].lower()


def test_greeting_starts_with_introduction(monkeypatch):
    thread_id = "conversation-thread-greeting"

    monkeypatch.setattr(
        "src.clinivet_brain.structured_llm",
        SequenceStructuredLLM(
            [
                TriageOutput(
                    tutor_name=None,
                    tutor_cpf=None,
                    pet_name=None,
                    pet_species="Desconhecido",
                    urgency_level="routine",
                    service_suggested="Consulta",
                    phone=None,
                )
            ]
        ),
    )

    config = {"configurable": {"thread_id": thread_id}}

    greeting_turn = clinivet_agent.invoke(
        {"messages": [HumanMessage(content="bom dia")], "thread_id": thread_id},
        config=config,
    )

    assert "assistente virtual da Clinica Clinivet" in greeting_turn["assistant_message"]
    assert "Como posso te ajudar?" in greeting_turn["assistant_message"]


def test_compound_greeting_starts_with_introduction(monkeypatch):
    thread_id = "conversation-thread-greeting-compound"

    monkeypatch.setattr(
        "src.clinivet_brain.structured_llm",
        SequenceStructuredLLM(
            [
                TriageOutput(
                    tutor_name=None,
                    tutor_cpf=None,
                    pet_name=None,
                    pet_species="Desconhecido",
                    urgency_level="routine",
                    service_suggested="Consulta",
                    phone=None,
                )
            ]
        ),
    )

    greeting_turn = clinivet_agent.invoke(
        {"messages": [HumanMessage(content="ola bom dia")], "thread_id": thread_id},
        config={"configurable": {"thread_id": thread_id}},
    )

    assert "assistente virtual da Clinica Clinivet" in greeting_turn["assistant_message"]
    assert greeting_turn["intent"] == "greeting"


def test_awaiting_initial_request_repeated_greeting_does_not_run_free_triage(monkeypatch):
    thread_id = "conversation-thread-repeated-greeting"
    monkeypatch.setattr(
        "src.clinivet_brain.structured_llm",
        SequenceStructuredLLM(
            [
                TriageOutput(
                    tutor_name="Marina Silva",
                    tutor_cpf="12345678909",
                    pet_name="Felix",
                    pet_species="Gato",
                    urgency_level="routine",
                    service_suggested="Consulta",
                    phone=None,
                )
            ]
        ),
    )

    config = {"configurable": {"thread_id": thread_id}}

    clinivet_agent.invoke(
        {"messages": [HumanMessage(content="boa tarde")], "thread_id": thread_id},
        config=config,
    )

    second_turn = clinivet_agent.invoke(
        {"messages": [HumanMessage(content="boa tarde")], "thread_id": thread_id},
        config=config,
    )

    assert "me diga objetivamente" in second_turn["assistant_message"].lower()
    assert second_turn["pending_action"] == "awaiting_initial_request"


def test_greeting_onboarding_requests_tutor_name_first(monkeypatch):
    thread_id = "5514999900001"
    llm_sequence = SequenceStructuredLLM(
        [
            TriageOutput(
                tutor_name=None,
                tutor_cpf=None,
                pet_name=None,
                pet_species="Desconhecido",
                urgency_level="routine",
                service_suggested="Consulta",
                phone=None,
            )
        ]
    )

    monkeypatch.setattr("src.clinivet_brain.structured_llm", llm_sequence)

    config = {"configurable": {"thread_id": thread_id}}

    clinivet_agent.invoke(
        {"messages": [HumanMessage(content="boa tarde")], "thread_id": thread_id},
        config=config,
    )

    second_turn = clinivet_agent.invoke(
        {"messages": [HumanMessage(content="Quero marcar consulta")], "thread_id": thread_id},
        config=config,
    )

    assert "dados essenciais" in second_turn["assistant_message"].lower()
    assert "qual e o seu nome" in second_turn["assistant_message"].lower()


def test_greeting_onboarding_collects_essential_fields_in_order(monkeypatch):
    thread_id = "5514999900002"
    llm_sequence = SequenceStructuredLLM(
        [
            TriageOutput(
                tutor_name=None,
                tutor_cpf=None,
                pet_name=None,
                pet_species="Desconhecido",
                urgency_level="routine",
                service_suggested="Consulta",
                phone=None,
            ),
            TriageOutput(
                tutor_name="Padre Pedro",
                tutor_cpf=None,
                pet_name=None,
                pet_species="Desconhecido",
                urgency_level="routine",
                service_suggested="Consulta",
                phone=None,
            ),
            TriageOutput(
                tutor_name=None,
                tutor_cpf="444.232.543-22",
                pet_name=None,
                pet_species="Desconhecido",
                urgency_level="routine",
                service_suggested="Consulta",
                phone=None,
            ),
            TriageOutput(
                tutor_name=None,
                tutor_cpf=None,
                pet_name="Riuden",
                pet_species="Desconhecido",
                urgency_level="routine",
                service_suggested="Consulta",
                phone=None,
            ),
        ]
    )

    monkeypatch.setattr("src.clinivet_brain.structured_llm", llm_sequence)

    config = {"configurable": {"thread_id": thread_id}}

    clinivet_agent.invoke(
        {"messages": [HumanMessage(content="boa tarde")], "thread_id": thread_id},
        config=config,
    )

    turn_1 = clinivet_agent.invoke(
        {"messages": [HumanMessage(content="Quero marcar consulta")], "thread_id": thread_id},
        config=config,
    )
    assert "dados essenciais" in turn_1["assistant_message"].lower()
    assert "qual e o seu nome" in turn_1["assistant_message"].lower()

    turn_2 = clinivet_agent.invoke(
        {"messages": [HumanMessage(content="Padre Pedro")], "thread_id": thread_id},
        config=config,
    )
    assert turn_2["assistant_message"] == "Para continuar, voce pode me informar o CPF do tutor?"

    turn_3 = clinivet_agent.invoke(
        {"messages": [HumanMessage(content="444.232.543-22")], "thread_id": thread_id},
        config=config,
    )
    assert turn_3["assistant_message"] == "Entendi. Qual e o nome do seu pet?"

    turn_4 = clinivet_agent.invoke(
        {"messages": [HumanMessage(content="Riuden")], "thread_id": thread_id},
        config=config,
    )
    assert turn_4["assistant_message"] == "Qual e a especie do seu pet? Pode me dizer se e cao ou gato?"


def test_missing_cpf_repeats_question_for_non_progress_message(monkeypatch):
    thread_id = "55149999000021"
    llm_sequence = SequenceStructuredLLM(
        [
            TriageOutput(
                tutor_name=None,
                tutor_cpf=None,
                pet_name=None,
                pet_species="Desconhecido",
                urgency_level="routine",
                service_suggested="Consulta",
                phone=None,
            ),
            TriageOutput(
                tutor_name="Padre Pedro",
                tutor_cpf=None,
                pet_name=None,
                pet_species="Desconhecido",
                urgency_level="routine",
                service_suggested="Consulta",
                phone=None,
            ),
        ]
    )

    monkeypatch.setattr("src.clinivet_brain.structured_llm", llm_sequence)
    config = {"configurable": {"thread_id": thread_id}}

    clinivet_agent.invoke(
        {"messages": [HumanMessage(content="boa tarde")], "thread_id": thread_id},
        config=config,
    )
    clinivet_agent.invoke(
        {"messages": [HumanMessage(content="quero marcar consulta")], "thread_id": thread_id},
        config=config,
    )
    second_turn = clinivet_agent.invoke(
        {"messages": [HumanMessage(content="Padre Pedro")], "thread_id": thread_id},
        config=config,
    )
    assert "cpf do tutor" in second_turn["assistant_message"].lower()

    third_turn = clinivet_agent.invoke(
        {"messages": [HumanMessage(content="boa tarde")], "thread_id": thread_id},
        config=config,
    )
    assert "cpf do tutor" in third_turn["assistant_message"].lower()


def test_compound_species_reply_accepts_extra_pet_details(monkeypatch):
    thread_id = "5514999900099"

    monkeypatch.setattr(
        "src.clinivet_brain.structured_llm",
        SequenceStructuredLLM(
            [
                TriageOutput(
                    tutor_name="Mauricio Prata",
                    tutor_cpf="33311122200",
                    pet_name="Apogeu",
                    pet_species="Cao",
                    urgency_level="routine",
                    service_suggested="Consulta",
                    phone=thread_id,
                    pet_breed="Pug",
                    pet_weight=4.0,
                    pet_age="2 anos",
                )
            ]
        ),
    )
    monkeypatch.setattr("src.clinivet_brain.get_lead_by_phone", lambda _phone: None)
    monkeypatch.setattr("src.clinivet_brain.get_lead_by_cpf", lambda _cpf: None)
    monkeypatch.setattr("src.clinivet_brain.get_pets_by_phone", lambda _phone: [])
    monkeypatch.setattr("src.clinivet_brain.register_lead", lambda **_kwargs: 70)
    monkeypatch.setattr("src.clinivet_brain.upsert_pet_profile", lambda **_kwargs: {"id": 81})
    monkeypatch.setattr(
        "src.clinivet_brain.resolve_scheduling_context",
        lambda _service_name, preferred_day=None: ("Consulta", preferred_day or "2030-01-01", ["10:00"]),
    )
    monkeypatch.setattr("src.clinivet_brain.get_calendar_service", lambda *_args, **_kwargs: FakeCalendarService())
    monkeypatch.setattr("src.clinivet_brain.get_service_id_by_name", lambda _name: 5)
    monkeypatch.setattr("src.clinivet_brain.has_appointment_for_lead", lambda _lead_id: False)
    monkeypatch.setattr("src.clinivet_brain.confirm_appointment", lambda **_kwargs: {"id": 500})
    monkeypatch.setattr("src.clinivet_brain.set_appointment_google_event_id", lambda *_args, **_kwargs: True)
    monkeypatch.setattr("src.clinivet_brain.update_lead_status", lambda _lead_id, _status: True)
    monkeypatch.setattr(
        "src.clinivet_brain.build_slot_datetime",
        lambda day, slot: pytz.timezone("America/Sao_Paulo").localize(
            datetime.strptime(f"{day} {slot}", "%Y-%m-%d %H:%M")
        ),
    )

    result = clinivet_agent.invoke(
        {
            "messages": [HumanMessage(content="cachorrinho, raca Pug, 4 kilos de peso, tem 2 anos")],
            "thread_id": thread_id,
            "pending_action": "awaiting_missing_data",
            "missing_fields": ["pet_species"],
            "triage_data": TriageOutput(
                tutor_name="Mauricio Prata",
                tutor_cpf="33311122200",
                pet_name="Apogeu",
                pet_species="Desconhecido",
                urgency_level="routine",
                service_suggested="Consulta",
                phone=thread_id,
            ),
        },
        config={"configurable": {"thread_id": thread_id}},
    )

    assert "especie do seu pet" not in result["assistant_message"].lower()
    assert "Agendamento confirmado" in result["assistant_message"]


def test_no_lead_created_before_minimum_profile_is_complete(monkeypatch):
    thread_id = "5514999900003"
    llm_sequence = SequenceStructuredLLM(
        [
            TriageOutput(
                tutor_name=None,
                tutor_cpf=None,
                pet_name=None,
                pet_species="Gato",
                urgency_level="routine",
                service_suggested="Consulta",
                phone=None,
            ),
            TriageOutput(
                tutor_name="Antonio Jose",
                tutor_cpf=None,
                pet_name=None,
                pet_species="Gato",
                urgency_level="routine",
                service_suggested="Consulta",
                phone=None,
            ),
            TriageOutput(
                tutor_name=None,
                tutor_cpf="111.222.333-44",
                pet_name=None,
                pet_species="Gato",
                urgency_level="routine",
                service_suggested="Consulta",
                phone=None,
            ),
            TriageOutput(
                tutor_name=None,
                tutor_cpf=None,
                pet_name="Tigrinho",
                pet_species="Gato",
                urgency_level="routine",
                service_suggested="Consulta",
                phone=None,
            ),
        ]
    )
    register_calls = {"count": 0}

    monkeypatch.setattr("src.clinivet_brain.structured_llm", llm_sequence)
    monkeypatch.setattr("src.clinivet_brain.get_lead_by_phone", lambda _phone: None)
    monkeypatch.setattr("src.clinivet_brain.get_pets_by_phone", lambda _phone: [])
    monkeypatch.setattr(
        "src.clinivet_brain.register_lead",
        lambda **_kwargs: register_calls.__setitem__("count", register_calls["count"] + 1) or 88,
    )
    monkeypatch.setattr("src.clinivet_brain.upsert_pet_profile", lambda **_kwargs: {"id": 5})
    monkeypatch.setattr(
        "src.clinivet_brain.resolve_scheduling_context",
        lambda _service_name, preferred_day=None: ("Consulta", preferred_day or "2030-01-01", ["10:00"]),
    )
    monkeypatch.setattr(
        "src.clinivet_brain.get_calendar_service", lambda *_args, **_kwargs: FakeCalendarService()
    )
    monkeypatch.setattr("src.clinivet_brain.get_service_id_by_name", lambda _name: 5)
    monkeypatch.setattr("src.clinivet_brain.has_appointment_for_lead", lambda _lead_id: False)
    monkeypatch.setattr("src.clinivet_brain.confirm_appointment", lambda **_kwargs: {"id": 500})
    monkeypatch.setattr("src.clinivet_brain.set_appointment_google_event_id", lambda *_args, **_kwargs: True)
    monkeypatch.setattr("src.clinivet_brain.update_lead_status", lambda _lead_id, _status: True)
    monkeypatch.setattr(
        "src.clinivet_brain.build_slot_datetime",
        lambda day, slot: pytz.timezone("America/Sao_Paulo").localize(
            datetime.strptime(f"{day} {slot}", "%Y-%m-%d %H:%M")
        ),
    )

    config = {"configurable": {"thread_id": thread_id}}

    first = clinivet_agent.invoke(
        {
            "messages": [HumanMessage(content="quero marcar consulta para meu gato")],
            "thread_id": thread_id,
        },
        config=config,
    )
    assert register_calls["count"] == 0
    assert "qual e o seu nome" in first["assistant_message"].lower()

    second = clinivet_agent.invoke(
        {"messages": [HumanMessage(content="Antonio Jose")], "thread_id": thread_id},
        config=config,
    )
    assert register_calls["count"] == 0
    assert "cpf do tutor" in second["assistant_message"].lower()

    third = clinivet_agent.invoke(
        {"messages": [HumanMessage(content="111.222.333-44")], "thread_id": thread_id},
        config=config,
    )
    assert register_calls["count"] == 0
    assert "nome do seu pet" in third["assistant_message"].lower()

    fourth = clinivet_agent.invoke(
        {"messages": [HumanMessage(content="Tigrinho")], "thread_id": thread_id},
        config=config,
    )
    assert register_calls["count"] == 1
    assert "Agendamento confirmado" in fourth["assistant_message"]


def test_existing_tutor_memory_avoids_reasking_name(monkeypatch):
    thread_id = "5514999911111"
    monkeypatch.setattr(
        "src.clinivet_brain.structured_llm",
        SequenceStructuredLLM(
            [
                TriageOutput(
                    tutor_name=None,
                    tutor_cpf=None,
                    pet_name="Tigrinho",
                    pet_species="Gato",
                    urgency_level="routine",
                    service_suggested="Vacinacao",
                    phone=None,
                )
            ]
        ),
    )
    monkeypatch.setattr(
        "src.clinivet_brain.get_lead_by_phone",
        lambda _phone: {
            "id": 12,
            "tutor_name": "Antonio Jose",
            "tutor_cpf": "11122233344",
            "pet_name": "Tigrinho",
            "pet_species": "Gato",
            "phone": thread_id,
        },
    )
    monkeypatch.setattr(
        "src.clinivet_brain.get_pets_by_phone",
        lambda _phone: [{"id": 7, "name": "Tigrinho", "species": "Gato"}],
    )
    monkeypatch.setattr("src.clinivet_brain.register_lead", lambda **_kwargs: 90)
    monkeypatch.setattr("src.clinivet_brain.upsert_pet_profile", lambda **_kwargs: {"id": 7})
    monkeypatch.setattr(
        "src.clinivet_brain.resolve_scheduling_context",
        lambda _service_name, preferred_day=None: ("Vacinacao", preferred_day or "2030-01-01", ["10:00"]),
    )
    monkeypatch.setattr(
        "src.clinivet_brain.get_calendar_service", lambda *_args, **_kwargs: FakeCalendarService()
    )
    monkeypatch.setattr("src.clinivet_brain.get_service_id_by_name", lambda _name: 5)
    monkeypatch.setattr("src.clinivet_brain.has_appointment_for_lead", lambda _lead_id: False)
    monkeypatch.setattr("src.clinivet_brain.confirm_appointment", lambda **_kwargs: {"id": 501})
    monkeypatch.setattr("src.clinivet_brain.set_appointment_google_event_id", lambda *_args, **_kwargs: True)
    monkeypatch.setattr("src.clinivet_brain.update_lead_status", lambda _lead_id, _status: True)
    monkeypatch.setattr(
        "src.clinivet_brain.build_slot_datetime",
        lambda day, slot: pytz.timezone("America/Sao_Paulo").localize(
            datetime.strptime(f"{day} {slot}", "%Y-%m-%d %H:%M")
        ),
    )

    result = clinivet_agent.invoke(
        {"messages": [HumanMessage(content="quero agendar vacina pro tigrinho")], "thread_id": thread_id},
        config={"configurable": {"thread_id": thread_id}},
    )

    assert "Qual e o seu nome?" not in result["assistant_message"]
    assert "Agendamento confirmado" in result["assistant_message"]


def test_multi_pet_request_asks_to_handle_one_pet_at_a_time(monkeypatch):
    thread_id = "5514999922222"
    monkeypatch.setattr(
        "src.clinivet_brain.structured_llm",
        SequenceStructuredLLM(
            [
                TriageOutput(
                    tutor_name="Antonio Jose",
                    tutor_cpf="11122233344",
                    pet_name="Tigrinho, Felix, Meiudu",
                    pet_species="Desconhecido",
                    urgency_level="routine",
                    service_suggested="Vacinacao",
                    phone=None,
                )
            ]
        ),
    )
    monkeypatch.setattr("src.clinivet_brain.get_lead_by_phone", lambda _phone: None)
    monkeypatch.setattr("src.clinivet_brain.get_pets_by_phone", lambda _phone: [])

    result = clinivet_agent.invoke(
        {
            "messages": [HumanMessage(content="quero agendar a vacinacao do tigrinho, do felix e do meiudu")],
            "thread_id": thread_id,
        },
        config={"configurable": {"thread_id": thread_id}},
    )

    assert "varios pets" in result["assistant_message"]
    assert "qual pet voce quer agendar primeiro" in result["assistant_message"].lower()


def test_multi_pet_detection_uses_raw_message_mentions(monkeypatch):
    thread_id = "5514999922223"
    monkeypatch.setattr(
        "src.clinivet_brain.structured_llm",
        SequenceStructuredLLM(
            [
                TriageOutput(
                    tutor_name="Antonio Jose",
                    tutor_cpf="11122233344",
                    pet_name="Tigrinho",
                    pet_species="Gato",
                    urgency_level="routine",
                    service_suggested="Vacinacao",
                    phone=None,
                )
            ]
        ),
    )
    monkeypatch.setattr("src.clinivet_brain.get_lead_by_phone", lambda _phone: None)
    monkeypatch.setattr("src.clinivet_brain.get_pets_by_phone", lambda _phone: [])

    result = clinivet_agent.invoke(
        {
            "messages": [HumanMessage(content="quero marcar a vacinacao do tigrinho, do felix e do meiudu")],
            "thread_id": thread_id,
        },
        config={"configurable": {"thread_id": thread_id}},
    )

    assert "varios pets" in result["assistant_message"]


def test_multi_pet_requires_valid_choice_before_continuing(monkeypatch):
    thread_id = "5514999922224"
    monkeypatch.setattr(
        "src.clinivet_brain.structured_llm",
        SequenceStructuredLLM(
            [
                TriageOutput(
                    tutor_name="Antonio Jose",
                    tutor_cpf="11122233344",
                    pet_name="Tigrinho, Felix, Meiudu",
                    pet_species="Gato",
                    urgency_level="routine",
                    service_suggested="Vacinacao",
                    phone=None,
                )
            ]
        ),
    )
    monkeypatch.setattr("src.clinivet_brain.get_lead_by_phone", lambda _phone: None)
    monkeypatch.setattr("src.clinivet_brain.get_pets_by_phone", lambda _phone: [])

    config = {"configurable": {"thread_id": thread_id}}

    clinivet_agent.invoke(
        {
            "messages": [HumanMessage(content="quero marcar a vacinacao do tigrinho, do felix e do meiudu")],
            "thread_id": thread_id,
        },
        config=config,
    )

    follow_up = clinivet_agent.invoke(
        {"messages": [HumanMessage(content="todos")], "thread_id": thread_id},
        config=config,
    )

    assert "escolha um destes pets" in follow_up["assistant_message"].lower()
    assert follow_up["pending_action"] == "awaiting_single_pet_choice"


def test_multi_pet_choice_by_ordinal_continues_with_selected_pet(monkeypatch):
    thread_id = "5514999922225"
    llm_sequence = SequenceStructuredLLM(
        [
            TriageOutput(
                tutor_name="Antonio Jose",
                tutor_cpf="11122233344",
                pet_name="Tigrinho, Felix, Meiudu",
                pet_species="Gato",
                urgency_level="routine",
                service_suggested="Vacinacao",
                phone=None,
            ),
            TriageOutput(
                tutor_name="Antonio Jose",
                tutor_cpf="11122233344",
                pet_name=None,
                pet_species="Gato",
                urgency_level="routine",
                service_suggested="Vacinacao",
                phone=None,
            ),
        ]
    )

    monkeypatch.setattr("src.clinivet_brain.structured_llm", llm_sequence)
    monkeypatch.setattr("src.clinivet_brain.get_lead_by_phone", lambda _phone: None)
    monkeypatch.setattr("src.clinivet_brain.get_pets_by_phone", lambda _phone: [])
    monkeypatch.setattr("src.clinivet_brain.register_lead", lambda **_kwargs: 91)
    monkeypatch.setattr("src.clinivet_brain.upsert_pet_profile", lambda **_kwargs: {"id": 8})
    monkeypatch.setattr(
        "src.clinivet_brain.resolve_scheduling_context",
        lambda _service_name, preferred_day=None: ("Vacinacao", preferred_day or "2030-01-01", ["10:00"]),
    )
    monkeypatch.setattr(
        "src.clinivet_brain.get_calendar_service", lambda *_args, **_kwargs: FakeCalendarService()
    )
    monkeypatch.setattr("src.clinivet_brain.get_service_id_by_name", lambda _name: 5)
    monkeypatch.setattr("src.clinivet_brain.has_appointment_for_lead", lambda _lead_id: False)
    monkeypatch.setattr("src.clinivet_brain.confirm_appointment", lambda **_kwargs: {"id": 502})
    monkeypatch.setattr("src.clinivet_brain.set_appointment_google_event_id", lambda *_args, **_kwargs: True)
    monkeypatch.setattr("src.clinivet_brain.update_lead_status", lambda _lead_id, _status: True)
    monkeypatch.setattr(
        "src.clinivet_brain.build_slot_datetime",
        lambda day, slot: pytz.timezone("America/Sao_Paulo").localize(
            datetime.strptime(f"{day} {slot}", "%Y-%m-%d %H:%M")
        ),
    )

    config = {"configurable": {"thread_id": thread_id}}

    clinivet_agent.invoke(
        {
            "messages": [HumanMessage(content="quero marcar a vacinacao do tigrinho, do felix e do meiudu")],
            "thread_id": thread_id,
        },
        config=config,
    )

    follow_up = clinivet_agent.invoke(
        {"messages": [HumanMessage(content="o segundo")], "thread_id": thread_id},
        config=config,
    )

    assert "felix" in follow_up["assistant_message"].lower()
