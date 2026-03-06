from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import AIMessage, HumanMessage
from src.clinivet_brain import clinivet_agent

app = FastAPI(title="Clinivet Agent API")


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
