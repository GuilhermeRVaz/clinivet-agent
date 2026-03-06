from datetime import datetime

import pytz
from langchain_core.messages import HumanMessage

from src.clinivet_brain import TriageOutput, clinivet_agent


class DummyStructuredLLM:
    def __init__(self, result: TriageOutput):
        self.result = result

    def invoke(self, _conversation):
        return self.result


class FakeCalendarService:
    def get_free_slots(self, _day: str):
        return ["09:00", "09:30"]

    def create_event(self, **_kwargs):
        return "evt_123"


def test_agent_flow_schedules_after_complete_data(monkeypatch):
    thread_id = "5514999990001"
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
    monkeypatch.setattr("src.clinivet_brain.get_calendar_service", lambda: FakeCalendarService())
    monkeypatch.setattr("src.clinivet_brain.get_service_id_by_name", lambda _name: 10)
    monkeypatch.setattr("src.clinivet_brain.has_appointment_for_lead", lambda _lead_id: False)
    monkeypatch.setattr("src.clinivet_brain.confirm_appointment", lambda **_kwargs: {"id": 999})
    monkeypatch.setattr("src.clinivet_brain.update_lead_status", lambda _lead_id, _status: True)
    monkeypatch.setattr(
        "src.clinivet_brain.build_slot_datetime",
        lambda day, slot: pytz.timezone("America/Sao_Paulo").localize(
            datetime.strptime(f"{day} {slot}", "%Y-%m-%d %H:%M")
        ),
    )

    result = clinivet_agent.invoke(
        {
            "messages": [HumanMessage(content="Meu cachorro esta vomitando, quero consulta.")],
            "thread_id": thread_id,
        },
        config={"configurable": {"thread_id": thread_id}},
    )

    assert result["lead_id"] == 1
    assert "Agendamento confirmado" in result["assistant_message"]


def test_agent_flow_prevents_duplicate_appointment(monkeypatch):
    thread_id = "5514999990002"
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
    monkeypatch.setattr("src.clinivet_brain.register_lead", lambda **_kwargs: 2)
    monkeypatch.setattr("src.clinivet_brain.get_calendar_service", lambda: FakeCalendarService())
    monkeypatch.setattr("src.clinivet_brain.get_service_id_by_name", lambda _name: 10)
    monkeypatch.setattr("src.clinivet_brain.has_appointment_for_lead", lambda _lead_id: True)

    confirm_called = {"value": False}

    def _confirm_appointment(**_kwargs):
        confirm_called["value"] = True
        return {"id": 1000}

    monkeypatch.setattr("src.clinivet_brain.confirm_appointment", _confirm_appointment)
    monkeypatch.setattr("src.clinivet_brain.update_lead_status", lambda _lead_id, _status: True)

    result = clinivet_agent.invoke(
        {
            "messages": [HumanMessage(content="Quero agendar consulta para meu cachorro Rex.")],
            "thread_id": thread_id,
        },
        config={"configurable": {"thread_id": thread_id}},
    )

    assert result["assistant_message"] == "Voce ja possui um agendamento registrado."
    assert confirm_called["value"] is False
