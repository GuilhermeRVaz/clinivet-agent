# Clinivet Agent Workflow

## Objetivo
Documentar o fluxo funcional do agente da Clinivet, do recebimento da mensagem ate a resposta final ao tutor.

## Entradas do sistema
- `POST /agent/chat`
  - Entrada direta para testes e integracoes.
  - Corpo esperado: `message`, `thread_id`.
- `POST /webhook/evolution`
  - Entrada de mensagens vindas do WhatsApp (Evolution API).
  - O payload e validado e convertido para: `phone`, `thread_id`, `message`.

## Fluxo principal (LangGraph)
O grafo compilado em `src/clinivet_brain.py` agora suporta dois trilhos:

1. Fluxo clinico e agendamento
   - `triage`
   - `ask_missing_data`
   - `ask_time_preference`
   - `suggest_slots`
   - `confirm_slot`
   - `scheduling`
   - `conversion`
   - `END`
2. Fluxo de gerenciamento
   - `check_appointment`
   - `cancel_appointment`
   - `reschedule_appointment`
   - `load_pet_history`
   - `END`

## Regras por etapa

### 1) Triage (`triage_node`)
- Usa LLM estruturado (`TriageOutput`) para extrair dados clinicos e cadastrais.
- Normaliza e faz merge com dados previos da conversa.
- Se telefone nao vier da conversa, tenta inferir de `thread_id`.
- Se faltarem campos obrigatorios:
  - Define `next_step = ask_missing_data`.
- Se dados minimos estiverem completos:
  - Registra lead no banco (quando ainda nao existe `lead_id`).
- Se urgencia = `emergency`:
  - Responde orientando atendimento imediato.
  - Encerra fluxo.
- Se urgencia = `routine`:
  - Encaminha para agendamento.

### 2) Solicitar dados faltantes (`ask_missing_data_node`)
- Monta mensagem objetiva com os campos obrigatorios ausentes.
- Retorna mensagem ao tutor e encerra fluxo.

### 3) Agendamento (`scheduling_node`)
- Resolve contexto de agenda por tipo de servico.
- Busca dia alvo e horarios disponiveis.
- Se nao houver horario:
  - Lanca erro de indisponibilidade.
- Se houver horario:
  - Encaminha para `conversion`.

### 4) Preferencia e sugestao de horarios
- `ask_time_preference_node`
  - Pergunta se o tutor prefere `manha`, `tarde` ou `qualquer horario`.
- `suggest_slots_node`
  - Filtra horarios disponiveis por periodo.
  - Retorna ate 3 sugestoes.
- `confirm_slot_node`
  - Confirma o horario escolhido pelo tutor.
  - Mantem protecao contra agendamento duplicado.

### 5) Conversao/confirmacao automatica (`conversion_node`)
- Valida pre-condicoes: `lead_id`, `triage_data`, slots e data.
- Evita duplicidade com `has_appointment_for_lead`.
- Tenta confirmar horario no banco com controle de conflito:
  - Se houver conflito, tenta proximo horario.
- Com horario confirmado:
  - Cria evento no Google Calendar.
  - Persiste `google_event_id` no agendamento (quando disponivel).
  - Atualiza status do lead para `Agendado`.
  - Retorna mensagem final de confirmacao com data/hora.

### 6) Gerenciamento de agendamentos
- `check_appointment_node`
  - Busca agendamentos ativos pelo telefone.
- `cancel_appointment_node`
  - Cancela agendamento no banco.
  - Remove evento correspondente do Google Calendar.
- `reschedule_appointment_node`
  - Sugere novos horarios.
  - Atualiza horario no banco.
  - Atualiza evento correspondente no Google Calendar.
- `load_pet_history_node`
  - Recupera pets, consultas e vacinas por telefone do tutor.

## Roteamento entre nos
- `router()` decide proximo passo com base em:
  - `urgency_level`
  - `next_step`
- Mapeamento:
  - `emergency` -> `END`
  - `ask_missing_data` -> no correspondente
  - `ask_time_preference` -> no correspondente
  - `suggest_slots` -> no correspondente
  - `confirm_slot` -> no correspondente
  - `scheduling` -> no correspondente
  - `conversion` -> no correspondente
  - `check_appointment` -> no correspondente
  - `cancel_appointment` -> no correspondente
  - `reschedule_appointment` -> no correspondente
  - `load_pet_history` -> no correspondente

## Persistencia e integracoes
- Banco local (`src/clinivet_db.py`)
  - Leads, servicos, agendamentos e status.
- Google Calendar (`src/clinivet_calendar.py`)
  - Criacao de evento apos confirmar agendamento.
- Evolution API (`src/services/whatsapp_service.py`)
  - Recebe mensagem WhatsApp.
  - Envia resposta final ao tutor.

## Estados de saida esperados
- `missing_data`: agente pede informacoes faltantes.
- `emergency`: agente orienta atendimento imediato.
- `scheduled`: agendamento confirmado com data/hora.
- `error`: falha tecnica no processamento ou indisponibilidade.

## Checklist rapido de operacao
1. Configurar `.env` (OpenAI, Evolution e credenciais de agenda).
2. Subir API FastAPI (`src/main.py`).
3. Apontar webhook da Evolution para `/webhook/evolution`.
4. Validar fluxo com testes em `tests/`.
