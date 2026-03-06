from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
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
            {"messages": [HumanMessage(content=request.message)]},
            config=config,
        )

        assistant_message = result.get("assistant_message")

        return {
            "status": "ok",
            "message": assistant_message,
            "result": str(result),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
