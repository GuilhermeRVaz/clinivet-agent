import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.clinivet_brain import clinivet_agent  # noqa: E402


def load_environment() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=True)


def extract_assistant_message(result: dict) -> str:
    assistant_message = result.get("assistant_message")
    if assistant_message:
        return str(assistant_message)

    for message in reversed(result.get("messages", [])):
        if isinstance(message, AIMessage):
            return str(message.content)

    return "Sem resposta do agente."


def print_debug_info(result: dict) -> None:
    intent = result.get("intent")
    available_slots = result.get("available_slots")
    next_step = result.get("next_step")
    selected_slot = result.get("selected_slot")
    detected_date = result.get("detected_date")
    detected_time = result.get("detected_time") or selected_slot

    print("[debug] detected_intent:", intent)
    print("[debug] detected_date:", detected_date)
    print("[debug] detected_time:", detected_time)
    print("[debug] next_step:", next_step)
    if available_slots:
        print("[debug] slots sugeridos:", ", ".join(available_slots))
    if selected_slot:
        print("[debug] horario selecionado:", selected_slot)
    if result.get("lead_id"):
        print("[debug] lead_id:", result["lead_id"])


def run_simulator(thread_id: str, debug: bool) -> None:
    print("Simulador Clinivet iniciado. Digite 'exit' para encerrar.")

    while True:
        try:
            user_input = input("Cliente: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando simulador.")
            break

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("Encerrando simulador.")
            break

        config = {"configurable": {"thread_id": thread_id}}
        result = clinivet_agent.invoke(
            {
                "messages": [HumanMessage(content=user_input)],
                "thread_id": thread_id,
            },
            config=config,
        )

        print(f"Bot: {extract_assistant_message(result)}")
        if debug:
            print_debug_info(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simula uma conversa com o agente Clinivet no terminal."
    )
    parser.add_argument(
        "--thread-id",
        default=os.getenv("SIMULATOR_THREAD_ID", "5514999999999"),
        help="Thread ID usado para manter o contexto da conversa.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Exibe informacoes de depuracao do estado retornado pelo agente.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    load_environment()
    run_simulator(thread_id=args.thread_id, debug=args.debug)
