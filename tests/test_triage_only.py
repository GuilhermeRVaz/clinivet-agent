from langchain_core.messages import HumanMessage
from src.clinivet_brain import triage_node

state = {
    "messages": [
        HumanMessage(
            content="Oi, meu nome é João. Meu cachorro Rex está vomitando desde ontem."
        )
    ],
    "thread_id": "test1"
}

result = triage_node(state)

print("\nRESULTADO TRIAGEM")
print(result)