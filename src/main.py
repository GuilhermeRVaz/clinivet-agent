import logging
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import AIMessage, HumanMessage
from src.clinivet_brain import clinivet_agent
from src.services.whatsapp_service import (
    call_agent,
    parse_evolution_payload,
    send_whatsapp_message,
)

app = FastAPI(title="Clinivet Agent API")
logger = logging.getLogger("ClinivetAPI")


class ChatRequest(BaseModel):
    message: str
    thread_id: str


@app.post("/agent/chat")
async def chat(request: ChatRequest):
    try:
        config = {"configurable": {"thread_id": request.thread_id}}
        result = clinivet_agent.invoke(
            {
                "messages": [HumanMessage(content=request.message)],
                "thread_id": request.thread_id,
            },
            config=config,
        )

        assistant_message = result.get("assistant_message")
        if not assistant_message:
            for message in reversed(result.get("messages", [])):
                if isinstance(message, AIMessage):
                    assistant_message = message.content
                    break

        return {
            "status": "ok",
            "assistant_message": assistant_message,
            "lead_id": result.get("lead_id"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/webhook/evolution")
async def evolution_webhook(payload: Dict[str, Any]):
    try:
        parsed = parse_evolution_payload(payload)
        if parsed is None:
            return {"status": "ignored"}

        phone = parsed["phone"]
        thread_id = parsed["thread_id"]
        message = parsed["message"]

        logger.info("Incoming Evolution message: phone=%s message=%s", phone, message)

        assistant_message = call_agent(message=message, thread_id=thread_id)
        logger.info("Agent response: phone=%s response=%s", phone, assistant_message)

        send_status = send_whatsapp_message(phone=phone, text=assistant_message)
        logger.info("Send status: phone=%s status=%s", phone, send_status.get("status"))

        return {
            "status": "ok",
            "thread_id": thread_id,
            "assistant_message": assistant_message,
            "send_status": send_status,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Evolution webhook processing error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
