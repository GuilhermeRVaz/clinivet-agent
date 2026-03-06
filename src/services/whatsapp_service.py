import json
import logging
import os
from typing import Any, Dict, Optional
from urllib import error, request

from langchain_core.messages import AIMessage, HumanMessage

from src.clinivet_brain import clinivet_agent

logger = logging.getLogger("WhatsAppService")


def extract_phone(remote_jid: str) -> str:
    if not remote_jid:
        raise ValueError("Missing remoteJid in payload.")
    return remote_jid.replace("@s.whatsapp.net", "")


def parse_evolution_payload(payload: Dict[str, Any]) -> Optional[Dict[str, str]]:
    data = payload.get("data") or {}
    key = data.get("key") or {}
    message_data = data.get("message") or {}

    if key.get("fromMe"):
        logger.info("Ignoring message sent by the bot itself.")
        return None

    remote_jid = key.get("remoteJid")
    message = message_data.get("conversation") or message_data.get("extendedTextMessage", {}).get("text")

    if not remote_jid:
        raise ValueError("Missing data.key.remoteJid in payload.")
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Missing data.message.conversation or data.message.extendedTextMessage.text in payload.")

    phone = extract_phone(remote_jid)

    return {
        "phone": phone,
        "thread_id": phone,
        "message": message,
    }


def call_agent(message: str, thread_id: str) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = clinivet_agent.invoke(
        {
            "messages": [HumanMessage(content=message)],
            "thread_id": thread_id,
        },
        config=config,
    )

    assistant_message = result.get("assistant_message")
    if not assistant_message:
        for item in reversed(result.get("messages", [])):
            if isinstance(item, AIMessage):
                assistant_message = item.content
                break

    if not assistant_message:
        raise RuntimeError("Agent response is empty.")

    return str(assistant_message)


def send_whatsapp_message(phone: str, text: str) -> Dict[str, Any]:
    base_url = os.getenv("EVOLUTION_API_URL")
    api_key = os.getenv("EVOLUTION_API_KEY")

    if not base_url:
        raise RuntimeError("EVOLUTION_API_URL is not configured.")
    if not api_key:
        raise RuntimeError("EVOLUTION_API_KEY is not configured.")

    url = f"{base_url.rstrip('/')}/message/sendText"
    payload = json.dumps({"number": phone, "text": text}).encode("utf-8")

    req = request.Request(
        url=url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "apikey": api_key,
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=15) as response:
            response_body = response.read().decode("utf-8")
            logger.info("Evolution send success: phone=%s status=%s", phone, response.status)
            return {
                "status": "sent",
                "status_code": response.status,
                "response": response_body,
            }
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        logger.error("Evolution send failed: phone=%s status=%s body=%s", phone, exc.code, body)
        return {
            "status": "error",
            "status_code": exc.code,
            "response": body,
        }
    except Exception as exc:
        logger.exception("Evolution send exception: phone=%s error=%s", phone, exc)
        return {
            "status": "error",
            "status_code": None,
            "response": str(exc),
        }
