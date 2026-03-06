from langchain_core.messages import HumanMessage
from src.clinivet_brain import clinivet_agent

def run_test(message, thread_id="5511999999999"):

    result = clinivet_agent.invoke(
        {
            "messages": [HumanMessage(content=message)],
            "thread_id": thread_id
        }
    )

    print("\nRESULTADO FINAL:")
    print(result)


print("\nTESTE 1 — TRIAGEM NORMAL")
run_test(
    "Oi, meu nome é João. Meu cachorro Rex está vomitando desde ontem."
)


print("\nTESTE 2 — EMERGÊNCIA")
run_test(
    "Meu gato está tendo convulsão agora!"
)


print("\nTESTE 3 — DADOS INCOMPLETOS")
run_test(
    "Meu cachorro está doente"
)


print("\nTESTE 4 — COM TELEFONE")
run_test(
    "Sou Maria, meu gato Simba está mancando. Meu telefone é 14999123456"
)
