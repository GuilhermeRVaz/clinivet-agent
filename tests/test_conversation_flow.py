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
    thread_id = "5514999990003"
    llm_sequence = SequenceStructuredLLM(
        [
            TriageOutput(
                tutor_name=None,
                pet_name=None,
                pet_species="Cao",
                urgency_level="routine",
                service_suggested="Consulta",
                phone=None,
            ),
            TriageOutput(
                tutor_name=None,
                pet_name="Rex",
                pet_species="Cao",
                urgency_level="routine",
                service_suggested="Consulta",
                phone=None,
            ),
            TriageOutput(
                tutor_name="Joao",
                pet_name=None,
                pet_species="Cao",
                urgency_level="routine",
                service_suggested="Consulta",
                phone=None,
            ),
            TriageOutput(
                tutor_name=None,
                pet_name=None,
                pet_species="Cao",
                urgency_level="routine",
                service_suggested="Consulta",
                phone="14999999999",
            ),
        ]
    )

    monkeypatch.setattr("src.clinivet_brain.structured_llm", llm_sequence)
    monkeypatch.setattr("src.clinivet_brain.register_lead", lambda **_kwargs: 77)
    monkeypatch.setattr("src.clinivet_brain.get_calendar_service", lambda: FakeCalendarService())
    monkeypatch.setattr("src.clinivet_brain.get_service_id_by_name", lambda _name: 5)
    monkeypatch.setattr("src.clinivet_brain.has_appointment_for_lead", lambda _lead_id: False)
    monkeypatch.setattr("src.clinivet_brain.confirm_appointment", lambda **_kwargs: {"id": 500})
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
    assert turn_1["assistant_message"] == "Entendi. Qual e o nome do seu pet?"

    turn_2 = clinivet_agent.invoke(
        {"messages": [HumanMessage(content="Rex")], "thread_id": thread_id},
        config=config,
    )
    assert turn_2["assistant_message"] == "Perfeito. Qual e o seu nome?"

    turn_3 = clinivet_agent.invoke(
        {"messages": [HumanMessage(content="Joao")], "thread_id": thread_id},
        config=config,
    )
    assert turn_3["assistant_message"] == "Voce poderia me informar um telefone para contato?"

    turn_4 = clinivet_agent.invoke(
        {"messages": [HumanMessage(content="14999999999")], "thread_id": thread_id},
        config=config,
    )
    assert "Agendamento confirmado" in turn_4["assistant_message"]
    assert turn_4["lead_id"] == 77
