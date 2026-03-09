ativar venv
.\venv\Scripts\Activate

rodar todos testes
pytest -v -s

rodar teste de fluxo
pytest tests/test_agent_flow.py -v -s

rodar simulador
python scripts/simulate_chat.py --debug