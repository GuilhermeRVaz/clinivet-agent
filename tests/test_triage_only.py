from langchain_core.messages import HumanMessage

from src.clinivet_brain import TriageOutput, triage_node
from src.clinivet_calendar import get_calendar_service, get_calendar_service_type
from src.services.triage_service import classify_pet_size


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
                tutor_cpf="11122233344",
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
                tutor_cpf="11122233344",
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
                tutor_cpf="11122233344",
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


def test_pet_size_is_derived_from_weight(monkeypatch):
    monkeypatch.setattr(
        "src.clinivet_brain.structured_llm",
        DummyStructuredLLM(
            TriageOutput(
                tutor_name="Carlos",
                pet_name="Thor",
                pet_species="Cao",
                urgency_level="routine",
                service_suggested="Consulta",
                phone="14999999999",
                pet_weight=12.5,
            )
        ),
    )
    monkeypatch.setattr("src.clinivet_brain.register_lead", lambda **_kwargs: 505)

    result = triage_node(_base_state("Meu cachorro pesa 12.5kg e precisa consulta."))

    assert result["triage_data"].pet_size == "medio"


def test_classify_pet_size_rules():
    assert classify_pet_size(None) == "unknown"
    assert classify_pet_size(5) == "pequeno"
    assert classify_pet_size(10) == "pequeno"
    assert classify_pet_size(11) == "medio"
    assert classify_pet_size(25) == "medio"
    assert classify_pet_size(26) == "grande"


def test_service_calendar_mapping():
    assert get_calendar_service_type("Consulta") == "calendar_consultas"
    assert get_calendar_service_type("Banho e Tosa") == "calendar_banho_tosa"
    assert get_calendar_service_type("Cirurgia") == "calendar_cirurgia"
    assert get_calendar_service_type("Vacinacao") == "calendar_vacinas"


def test_calendar_service_falls_back_to_mock_without_credentials(monkeypatch):
    monkeypatch.delenv("GOOGLE_SERVICE_ACCOUNT_JSON", raising=False)
    monkeypatch.setattr("src.clinivet_calendar._mock_calendar_service", None)
    monkeypatch.setattr("src.clinivet_calendar._calendar_service", None)
    monkeypatch.setattr("src.clinivet_calendar._calendar_services_by_type", {})

    service = get_calendar_service("Consulta")

    assert service.get_free_slots("2030-01-01") == ["09:00", "09:30", "10:00", "14:00", "14:30"]
