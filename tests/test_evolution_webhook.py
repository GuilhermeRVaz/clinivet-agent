from fastapi.testclient import TestClient

from src.main import app
from src.services.whatsapp_service import extract_phone, parse_evolution_payload

client = TestClient(app)


def test_extract_phone_removes_whatsapp_suffix():
    assert extract_phone("5514999999999@s.whatsapp.net") == "5514999999999"


def test_parse_evolution_payload():
    payload = {
        "event": "messages.upsert",
        "data": {
            "key": {"remoteJid": "5514999999999@s.whatsapp.net"},
            "message": {"conversation": "Ola"},
        },
    }

    parsed = parse_evolution_payload(payload)

    assert parsed["phone"] == "5514999999999"
    assert parsed["thread_id"] == "5514999999999"
    assert parsed["message"] == "Ola"


def test_webhook_evolution_calls_agent_and_sender(monkeypatch):
    calls = {"message": None, "thread_id": None, "number": None, "text": None}

    def fake_call_agent(message: str, thread_id: str) -> str:
        calls["message"] = message
        calls["thread_id"] = thread_id
        return "Resposta do agente"

    def fake_send_whatsapp_message(phone: str, text: str):
        calls["number"] = phone
        calls["text"] = text
        return {"status": "sent", "status_code": 200}

    monkeypatch.setattr("src.main.call_agent", fake_call_agent)
    monkeypatch.setattr("src.main.send_whatsapp_message", fake_send_whatsapp_message)

    payload = {
        "event": "messages.upsert",
        "data": {
            "key": {"remoteJid": "5514999999999@s.whatsapp.net"},
            "message": {"conversation": "Ola"},
        },
    }
    response = client.post("/webhook/evolution", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["assistant_message"] == "Resposta do agente"
    assert calls["message"] == "Ola"
    assert calls["thread_id"] == "5514999999999"
    assert calls["number"] == "5514999999999"
    assert calls["text"] == "Resposta do agente"


def test_webhook_evolution_returns_400_for_invalid_payload():
    response = client.post(
        "/webhook/evolution",
        json={"event": "messages.upsert", "data": {"key": {}, "message": {}}},
    )
    assert response.status_code == 400
