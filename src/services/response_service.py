import os
import sys
from typing import Optional


def _should_use_llm_response_layer() -> bool:
    if "pytest" in sys.modules:
        return False
    return bool(os.getenv("OPENAI_API_KEY"))


def generate_conversational_response(
    action: str,
    base_message: str,
    context: Optional[dict] = None,
) -> str:
    if not _should_use_llm_response_layer():
        return base_message

    try:
        from langchain_openai import ChatOpenAI
    except ModuleNotFoundError:
        return base_message

    prompt = (
        "Reescreva a mensagem do assistente veterinario em portugues do Brasil. "
        "Mantenha o mesmo significado, seja cordial, natural e conciso. "
        "Nao invente horarios, datas, IDs ou informacoes novas. "
        "Se houver lista de horarios, preserve todos exatamente como vieram.\n"
        f"ACTION: {action}\n"
        f"CONTEXT: {context or {}}\n"
        f"MESSAGE: {base_message}"
    )

    try:
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.4)
        response = llm.invoke(prompt)
        content = getattr(response, "content", None)
        return str(content).strip() if content else base_message
    except Exception:
        return base_message
