import argparse
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage

from src.clinivet_brain import clinivet_agent


def _extract_bot_message(result: dict) -> str:
    assistant_message = result.get("assistant_message")
    if assistant_message:
        return str(assistant_message)

    for item in reversed(result.get("messages", [])):
        if isinstance(item, AIMessage):
            return str(item.content)
    return "Sem resposta do agente."


def run_simulation(thread_id: str) -> None:
    messages = [
        "Oi, meu nome e Joao.",
        "Meu cachorro Rex esta vomitando.",
        "Quero agendar consulta.",
        "Meu telefone e 14999999999.",
    ]

    config = {"configurable": {"thread_id": thread_id}}

    for text in messages:
        print(f"USER: {text}")
        result = clinivet_agent.invoke(
            {"messages": [HumanMessage(content=text)], "thread_id": thread_id},
            config=config,
        )
        print(f"BOT: {_extract_bot_message(result)}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulador local de conversa WhatsApp para Clinivet Agent.")
    parser.add_argument(
        "--thread-id",
        default=f"sim-whatsapp-{uuid4().hex[:8]}",
        help="Thread ID da conversa (padrao: gerado automaticamente).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_simulation(thread_id=args.thread_id)
