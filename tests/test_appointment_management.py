from datetime import datetime

import pytz
from langchain_core.messages import HumanMessage

from src.clinivet_brain import TriageOutput, clinivet_agent
from src.services.conversation_service import (
    extract_appointment_id,
    is_conversation_closing,
    normalize_time_input,
    parse_natural_date,
)
from src.services.scheduling_service import is_valid_schedule_date


class DummyStructuredLLM:
    def __init__(self, result: TriageOutput):
        self.result = result

    def invoke(self, _conversation):
        return self.result


class FakeCalendarService:
    def __init__(self):
        self.deleted_event_id = None
        self.updated_event = None

    def create_event(self, **_kwargs):
        return "evt_slot"

    def delete_event(self, event_id: str):
        self.deleted_event_id = event_id
        return True

    def update_event(self, **kwargs):
        self.updated_event = kwargs
        return kwargs["event_id"]


def _slot_datetime(day: str, slot: str):
    return pytz.timezone("America/Sao_Paulo").localize(
        datetime.strptime(f"{day} {slot}", "%Y-%m-%d %H:%M")
    )


def test_schedule_with_time_preference(monkeypatch):
    thread_id = "5514999991000"
    fake_calendar = FakeCalendarService()

    monkeypatch.setattr(
        "src.clinivet_brain.structured_llm",
        DummyStructuredLLM(
            TriageOutput(
                tutor_name="Joao",
                pet_name="Rex",
                pet_species="Cao",
                urgency_level="routine",
                service_suggested="Consulta",
                phone="14999999999",
            )
        ),
    )
    monkeypatch.setattr("src.clinivet_brain.register_lead", lambda **_kwargs: 1)
    monkeypatch.setattr("src.clinivet_brain.upsert_pet_profile", lambda **_kwargs: {"id": 55})
    monkeypatch.setattr(
        "src.clinivet_brain.find_available_slots",
        lambda date, period, service_name="Consulta": ["09:00", "09:30", "10:30"],
    )
    monkeypatch.setattr("src.clinivet_brain.has_appointment_for_lead", lambda _lead_id: False)
    monkeypatch.setattr("src.clinivet_brain.get_service_id_by_name", lambda _name: 10)
    monkeypatch.setattr("src.clinivet_brain.confirm_appointment", lambda **_kwargs: {"id": 123})
    monkeypatch.setattr(
        "src.clinivet_brain.get_calendar_service", lambda *_args, **_kwargs: fake_calendar
    )
    monkeypatch.setattr("src.clinivet_brain.set_appointment_google_event_id", lambda *_args, **_kwargs: True)
    monkeypatch.setattr("src.clinivet_brain.update_lead_status", lambda _lead_id, _status: True)
    monkeypatch.setattr("src.clinivet_brain.build_slot_datetime", _slot_datetime)
    monkeypatch.setattr("src.clinivet_brain.build_next_business_day", lambda reference=None: "2030-01-01")

    config = {"configurable": {"thread_id": thread_id}}

    first_turn = clinivet_agent.invoke(
        {
            "messages": [HumanMessage(content="Quero consulta de manha para o Rex. Meu nome e Joao.")],
            "thread_id": thread_id,
        },
        config=config,
    )

    assert "09:00" in first_turn["assistant_message"]
    assert "09:30" in first_turn["assistant_message"]

    second_turn = clinivet_agent.invoke(
        {"messages": [HumanMessage(content="09:30")], "thread_id": thread_id},
        config=config,
    )

    assert "Agendamento confirmado" in second_turn["assistant_message"]


def test_cancel_appointment_flow(monkeypatch):
    thread_id = "5514999991001"
    fake_calendar = FakeCalendarService()
    calls = {"cancelled_id": None}

    appointment = {
        "id": 10,
        "service_name": "Consulta",
        "appointment_time": "2030-01-01T09:00:00-03:00",
        "google_event_id": "evt_cancel",
        "pet_name": "Rex",
    }

    monkeypatch.setattr("src.clinivet_brain.get_user_appointments", lambda _phone: [appointment])
    monkeypatch.setattr("src.clinivet_brain.get_appointment_by_id", lambda _appointment_id: appointment)
    monkeypatch.setattr("src.clinivet_brain.get_active_appointment_by_phone", lambda _phone: appointment)
    monkeypatch.setattr(
        "src.clinivet_brain.cancel_appointment_record",
        lambda appointment_id: calls.__setitem__("cancelled_id", appointment_id) or appointment,
    )
    monkeypatch.setattr(
        "src.clinivet_brain.get_calendar_service", lambda *_args, **_kwargs: fake_calendar
    )

    result = clinivet_agent.invoke(
        {
            "messages": [HumanMessage(content="Quero cancelar meu agendamento")],
            "thread_id": thread_id,
        },
        config={"configurable": {"thread_id": thread_id}},
    )

    assert result["assistant_message"] == "Agendamento 10 cancelado com sucesso."
    assert calls["cancelled_id"] == 10
    assert fake_calendar.deleted_event_id == "evt_cancel"


def test_reschedule_appointment_flow(monkeypatch):
    thread_id = "5514999991002"
    fake_calendar = FakeCalendarService()
    calls = {"new_time": None}

    appointment = {
        "id": 20,
        "service_name": "Consulta",
        "duration_minutes": 30,
        "appointment_time": "2030-01-01T09:00:00-03:00",
        "google_event_id": "evt_reschedule",
        "pet_name": "Luna",
    }

    monkeypatch.setattr("src.clinivet_brain.get_user_appointments", lambda _phone: [appointment])
    monkeypatch.setattr("src.clinivet_brain.get_appointment_by_id", lambda _appointment_id: appointment)
    monkeypatch.setattr("src.clinivet_brain.get_active_appointment_by_phone", lambda _phone: appointment)
    monkeypatch.setattr(
        "src.clinivet_brain.find_available_slots",
        lambda date, period, service_name="Consulta": ["14:00", "14:30", "15:00"],
    )
    monkeypatch.setattr(
        "src.clinivet_brain.reschedule_appointment_record",
        lambda appointment_id, new_time: calls.__setitem__("new_time", new_time) or appointment,
    )
    monkeypatch.setattr(
        "src.clinivet_brain.get_calendar_service", lambda *_args, **_kwargs: fake_calendar
    )
    monkeypatch.setattr("src.clinivet_brain.build_slot_datetime", _slot_datetime)
    monkeypatch.setattr("src.clinivet_brain.build_next_business_day", lambda reference=None: "2030-01-01")

    config = {"configurable": {"thread_id": thread_id}}

    first_turn = clinivet_agent.invoke(
        {
            "messages": [HumanMessage(content="Quero remarcar meu agendamento para tarde")],
            "thread_id": thread_id,
        },
        config=config,
    )

    assert "14:00" in first_turn["assistant_message"]
    assert "14:30" in first_turn["assistant_message"]

    second_turn = clinivet_agent.invoke(
        {"messages": [HumanMessage(content="14:30")], "thread_id": thread_id},
        config=config,
    )

    assert "Agendamento 20 remarcado" in second_turn["assistant_message"]
    assert calls["new_time"].startswith("2030-01-01T14:30:00")
    assert fake_calendar.updated_event["event_id"] == "evt_reschedule"


def test_load_pet_history_flow(monkeypatch):
    thread_id = "5514999991003"
    monkeypatch.setattr(
        "src.clinivet_brain.load_pet_history",
        lambda _phone: {
            "pets": [{"id": 1, "name": "Rex"}],
            "consultations": [{"id": 7, "pet_id": 1}],
            "vaccines": [{"id": 8, "pet_id": 1}],
        },
    )

    result = clinivet_agent.invoke(
        {
            "messages": [HumanMessage(content="Quero ver o historico do meu pet")],
            "thread_id": thread_id,
        },
        config={"configurable": {"thread_id": thread_id}},
    )

    assert "Pets encontrados: Rex." in result["assistant_message"]
    assert "Consultas registradas: 1." in result["assistant_message"]
    assert "Vacinas registradas: 1." in result["assistant_message"]


def test_normalize_time_input_accepts_multiple_formats():
    assert normalize_time_input("13h30") == "13:30"
    assert normalize_time_input("13:30") == "13:30"
    assert normalize_time_input("13.30") == "13:30"
    assert normalize_time_input("13:3") == "13:30"
    assert normalize_time_input("1330") == "13:30"
    assert normalize_time_input("14h") == "14:00"
    assert normalize_time_input("14") == "14:00"
    assert normalize_time_input("14:00") == "14:00"
    assert normalize_time_input("14.00") == "14:00"
    assert normalize_time_input("14h00") == "14:00"


def test_confirm_slot_returns_friendly_conflict_message(monkeypatch):
    thread_id = "5514999991004"

    monkeypatch.setattr(
        "src.clinivet_brain.structured_llm",
        DummyStructuredLLM(
            TriageOutput(
                tutor_name="Joao",
                pet_name="Rex",
                pet_species="Cao",
                urgency_level="routine",
                service_suggested="Consulta",
                phone="14999999999",
            )
        ),
    )
    monkeypatch.setattr("src.clinivet_brain.register_lead", lambda **_kwargs: 1)
    monkeypatch.setattr("src.clinivet_brain.upsert_pet_profile", lambda **_kwargs: {"id": 55})
    monkeypatch.setattr(
        "src.clinivet_brain.find_available_slots",
        lambda date, period, service_name="Consulta": ["09:00", "09:30", "10:30"],
    )
    monkeypatch.setattr("src.clinivet_brain.has_appointment_for_lead", lambda _lead_id: False)
    monkeypatch.setattr("src.clinivet_brain.get_service_id_by_name", lambda _name: 10)
    monkeypatch.setattr(
        "src.clinivet_brain.confirm_appointment",
        lambda **_kwargs: {"status": "conflict", "next_available_time": None},
    )
    monkeypatch.setattr("src.clinivet_brain.build_slot_datetime", _slot_datetime)
    monkeypatch.setattr("src.clinivet_brain.build_next_business_day", lambda reference=None: "2030-01-01")

    config = {"configurable": {"thread_id": thread_id}}

    clinivet_agent.invoke(
        {
            "messages": [HumanMessage(content="Quero consulta de manha para o Rex. Meu nome e Joao.")],
            "thread_id": thread_id,
        },
        config=config,
    )

    second_turn = clinivet_agent.invoke(
        {"messages": [HumanMessage(content="09:30")], "thread_id": thread_id},
        config=config,
    )

    assert second_turn["assistant_message"] == "Esse horario acabou de ser reservado. Vou buscar outro disponivel."


def test_parse_natural_date_supports_common_inputs():
    reference = pytz.timezone("America/Sao_Paulo").localize(datetime(2026, 3, 9, 10, 0))

    assert parse_natural_date("amanha", reference=reference) == "2026-03-10"
    assert parse_natural_date("sexta", reference=reference) == "2026-03-13"
    assert parse_natural_date("dia 13", reference=reference) == "2026-03-13"
    assert parse_natural_date("13/02/2026", reference=reference) == "2026-02-13"
    assert parse_natural_date("13 de fevereiro", reference=reference) == "2027-02-13"


def test_extract_appointment_id_ignores_dates():
    assert extract_appointment_id("remarcar 19/03/2026") is None
    assert extract_appointment_id("cancelar dia 13") is None
    assert extract_appointment_id("agendamento 42") == 42


def test_reschedule_intent_switch_asks_for_new_date(monkeypatch):
    thread_id = "5514999991005"

    appointment = {
        "id": 20,
        "service_name": "Consulta",
        "duration_minutes": 30,
        "appointment_time": "2030-01-01T09:00:00-03:00",
        "google_event_id": "evt_reschedule",
        "pet_name": "Luna",
    }

    monkeypatch.setattr("src.clinivet_brain.get_user_appointments", lambda _phone: [appointment])
    monkeypatch.setattr("src.clinivet_brain.get_appointment_by_id", lambda _appointment_id: appointment)
    monkeypatch.setattr("src.clinivet_brain.get_active_appointment_by_phone", lambda _phone: appointment)

    result = clinivet_agent.invoke(
        {
            "messages": [HumanMessage(content="quero mudar o dia")],
            "thread_id": thread_id,
        },
        config={"configurable": {"thread_id": thread_id}},
    )

    assert result["intent"] == "reschedule"
    assert result["assistant_message"] == "Claro! Vamos remarcar a consulta do seu pet. Qual novo dia voce prefere?"


def test_reschedule_date_message_does_not_detect_fake_time(monkeypatch):
    thread_id = "5514999991010"

    appointment = {
        "id": 20,
        "service_name": "Consulta",
        "duration_minutes": 30,
        "appointment_time": "2030-01-01T09:00:00-03:00",
        "google_event_id": "evt_reschedule",
        "pet_name": "Luna",
    }

    monkeypatch.setattr("src.clinivet_brain.get_user_appointments", lambda _phone: [appointment])
    monkeypatch.setattr("src.clinivet_brain.get_appointment_by_id", lambda _appointment_id: None)
    monkeypatch.setattr("src.clinivet_brain.get_active_appointment_by_phone", lambda _phone: appointment)

    result = clinivet_agent.invoke(
        {
            "messages": [HumanMessage(content="remarcar 19/03/2026")],
            "thread_id": thread_id,
        },
        config={"configurable": {"thread_id": thread_id}},
    )

    assert result["detected_date"] == "2026-03-19"
    assert result["detected_time"] is None
    assert "Voce prefere atendimento pela manha" in result["assistant_message"]


def test_is_valid_schedule_date_rejects_past_date():
    assert is_valid_schedule_date("2020-01-01") == (
        False,
        "Nao consigo agendar consultas em datas passadas.",
    )


def test_is_valid_schedule_date_rejects_sunday():
    assert is_valid_schedule_date("2026-03-15") == (
        False,
        "Nao realizamos agendamentos aos domingos.",
    )


def test_is_valid_schedule_date_rejects_date_beyond_window():
    is_valid, message = is_valid_schedule_date("2099-01-01")
    assert is_valid is False
    assert message == "Podemos agendar consultas ate 20 dias a frente."


def test_is_conversation_closing_detects_common_phrases():
    assert is_conversation_closing("obrigado")
    assert is_conversation_closing("valeu")
    assert is_conversation_closing("tchau")
    assert is_conversation_closing("ate logo")


def test_post_confirmation_closing_returns_polite_message(monkeypatch):
    thread_id = "5514999991006"
    fake_calendar = FakeCalendarService()

    monkeypatch.setattr(
        "src.clinivet_brain.structured_llm",
        DummyStructuredLLM(
            TriageOutput(
                tutor_name="Joao",
                pet_name="Rex",
                pet_species="Cao",
                urgency_level="routine",
                service_suggested="Consulta",
                phone="14999999999",
            )
        ),
    )
    monkeypatch.setattr("src.clinivet_brain.register_lead", lambda **_kwargs: 1)
    monkeypatch.setattr("src.clinivet_brain.upsert_pet_profile", lambda **_kwargs: {"id": 55})
    monkeypatch.setattr(
        "src.clinivet_brain.find_available_slots",
        lambda date, period, service_name="Consulta": ["09:00", "09:30", "10:30"],
    )
    monkeypatch.setattr("src.clinivet_brain.has_appointment_for_lead", lambda _lead_id: False)
    monkeypatch.setattr("src.clinivet_brain.get_service_id_by_name", lambda _name: 10)
    monkeypatch.setattr("src.clinivet_brain.confirm_appointment", lambda **_kwargs: {"id": 123})
    monkeypatch.setattr(
        "src.clinivet_brain.get_calendar_service", lambda *_args, **_kwargs: fake_calendar
    )
    monkeypatch.setattr("src.clinivet_brain.set_appointment_google_event_id", lambda *_args, **_kwargs: True)
    monkeypatch.setattr("src.clinivet_brain.update_lead_status", lambda _lead_id, _status: True)
    monkeypatch.setattr("src.clinivet_brain.build_slot_datetime", _slot_datetime)
    monkeypatch.setattr("src.clinivet_brain.build_next_business_day", lambda reference=None: "2030-01-01")

    config = {"configurable": {"thread_id": thread_id}}

    clinivet_agent.invoke(
        {
            "messages": [HumanMessage(content="Quero consulta de manha para o Rex. Meu nome e Joao.")],
            "thread_id": thread_id,
        },
        config=config,
    )
    clinivet_agent.invoke(
        {"messages": [HumanMessage(content="09:30")], "thread_id": thread_id},
        config=config,
    )

    closing_turn = clinivet_agent.invoke(
        {"messages": [HumanMessage(content="obrigado, ate logo")], "thread_id": thread_id},
        config=config,
    )

    assert "Ate logo" in closing_turn["assistant_message"]


def test_post_confirmation_does_not_restart_triage(monkeypatch):
    thread_id = "5514999991007"
    fake_calendar = FakeCalendarService()

    monkeypatch.setattr(
        "src.clinivet_brain.structured_llm",
        DummyStructuredLLM(
            TriageOutput(
                tutor_name="Joao",
                pet_name="Rex",
                pet_species="Cao",
                urgency_level="routine",
                service_suggested="Consulta",
                phone="14999999999",
            )
        ),
    )
    monkeypatch.setattr("src.clinivet_brain.register_lead", lambda **_kwargs: 1)
    monkeypatch.setattr("src.clinivet_brain.upsert_pet_profile", lambda **_kwargs: {"id": 55})
    monkeypatch.setattr(
        "src.clinivet_brain.find_available_slots",
        lambda date, period, service_name="Consulta": ["09:00", "09:30", "10:30"],
    )
    monkeypatch.setattr("src.clinivet_brain.has_appointment_for_lead", lambda _lead_id: False)
    monkeypatch.setattr("src.clinivet_brain.get_service_id_by_name", lambda _name: 10)
    monkeypatch.setattr("src.clinivet_brain.confirm_appointment", lambda **_kwargs: {"id": 123})
    monkeypatch.setattr(
        "src.clinivet_brain.get_calendar_service", lambda *_args, **_kwargs: fake_calendar
    )
    monkeypatch.setattr("src.clinivet_brain.set_appointment_google_event_id", lambda *_args, **_kwargs: True)
    monkeypatch.setattr("src.clinivet_brain.update_lead_status", lambda _lead_id, _status: True)
    monkeypatch.setattr("src.clinivet_brain.build_slot_datetime", _slot_datetime)
    monkeypatch.setattr("src.clinivet_brain.build_next_business_day", lambda reference=None: "2030-01-01")

    config = {"configurable": {"thread_id": thread_id}}

    clinivet_agent.invoke(
        {
            "messages": [HumanMessage(content="Quero consulta de manha para o Rex. Meu nome e Joao.")],
            "thread_id": thread_id,
        },
        config=config,
    )
    clinivet_agent.invoke(
        {"messages": [HumanMessage(content="09:30")], "thread_id": thread_id},
        config=config,
    )

    follow_up = clinivet_agent.invoke(
        {"messages": [HumanMessage(content="quero marcar outra consulta")], "thread_id": thread_id},
        config=config,
    )

    assert "ja esta confirmado" in follow_up["assistant_message"]


def test_cancel_uses_latest_active_appointment_by_phone_when_id_is_missing(monkeypatch):
    thread_id = "5514999991008"
    fake_calendar = FakeCalendarService()
    calls = {"cancelled_id": None}

    appointment = {
        "id": 42,
        "service_name": "Consulta",
        "appointment_time": "2030-01-01T09:00:00-03:00",
        "google_event_id": "evt_cancel_latest",
        "pet_name": "Rex",
    }

    monkeypatch.setattr("src.clinivet_brain.get_user_appointments", lambda _phone: [appointment])
    monkeypatch.setattr("src.clinivet_brain.get_appointment_by_id", lambda _appointment_id: None)
    monkeypatch.setattr("src.clinivet_brain.get_active_appointment_by_phone", lambda _phone: appointment)
    monkeypatch.setattr(
        "src.clinivet_brain.cancel_appointment_record",
        lambda appointment_id: calls.__setitem__("cancelled_id", appointment_id) or appointment,
    )
    monkeypatch.setattr(
        "src.clinivet_brain.get_calendar_service", lambda *_args, **_kwargs: fake_calendar
    )

    result = clinivet_agent.invoke(
        {
            "messages": [HumanMessage(content="Quero cancelar meu agendamento")],
            "thread_id": thread_id,
        },
        config={"configurable": {"thread_id": thread_id}},
    )

    assert result["assistant_message"] == "Agendamento 42 cancelado com sucesso."
    assert calls["cancelled_id"] == 42
