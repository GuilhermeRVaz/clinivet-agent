from langchain_core.messages import HumanMessage

from src.clinivet_brain import TriageOutput, triage_node


class DummyStructuredLLM:
    def __init__(self, result: TriageOutput):
        self.result = result

    def invoke(self, _conversation):
        return self.result


def _base_state(message: str, thread_id: str = "5514999999999"):
    return {"messages": [HumanMessage(content=message)], "thread_id": thread_id}


def test_normal_triage(monkeypatch):
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
    monkeypatch.setattr("src.clinivet_brain.register_lead", lambda **_kwargs: 101)

    result = triage_node(_base_state("Meu cachorro esta vomitando."))

    assert result["next_step"] == "scheduling"
    assert result["lead_id"] == 101
    assert result["missing_fields"] == []


def test_missing_pet_name(monkeypatch):
    monkeypatch.setattr(
        "src.clinivet_brain.structured_llm",
        DummyStructuredLLM(
            TriageOutput(
                tutor_name="Joao",
                pet_name=None,
                pet_species="Cao",
                urgency_level="routine",
                phone="14999999999",
            )
        ),
    )

    result = triage_node(_base_state("Meu cachorro esta vomitando."))

    assert result["next_step"] == "ask_missing_data"
    assert "pet_name" in result["missing_fields"]


def test_emergency_case(monkeypatch):
    monkeypatch.setattr(
        "src.clinivet_brain.structured_llm",
        DummyStructuredLLM(
            TriageOutput(
                tutor_name="Maria",
                pet_name="Simba",
                pet_species="Gato",
                urgency_level="emergency",
                service_suggested="Emergencia",
                phone="14999999999",
            )
        ),
    )
    monkeypatch.setattr("src.clinivet_brain.register_lead", lambda **_kwargs: 202)

    result = triage_node(_base_state("Meu gato esta tendo convulsao agora."))

    assert result["next_step"] == "end"
    assert "urgencia" in result["assistant_message"].lower()


def test_banho_e_tosa_detection(monkeypatch):
    monkeypatch.setattr(
        "src.clinivet_brain.structured_llm",
        DummyStructuredLLM(
            TriageOutput(
                tutor_name="Ana",
                pet_name="Luna",
                pet_species="Cao",
                urgency_level="routine",
                service_suggested="Banho e Tosa",
                phone="14999999999",
            )
        ),
    )
    monkeypatch.setattr("src.clinivet_brain.register_lead", lambda **_kwargs: 303)

    result = triage_node(_base_state("Quero banho e tosa para minha cachorra Luna."))

    assert result["triage_data"].service_suggested == "Banho e Tosa"
    assert result["next_step"] == "scheduling"


def test_phone_extraction_from_message(monkeypatch):
    monkeypatch.setattr(
        "src.clinivet_brain.structured_llm",
        DummyStructuredLLM(
            TriageOutput(
                tutor_name="Pedro",
                pet_name="Max",
                pet_species="Cao",
                urgency_level="routine",
                service_suggested="Consulta",
                phone="(14) 99999-1111",
            )
        ),
    )
    monkeypatch.setattr("src.clinivet_brain.register_lead", lambda **_kwargs: 404)

    result = triage_node(_base_state("Meu telefone e (14) 99999-1111."))

    assert result["triage_data"].phone == "14999991111"
