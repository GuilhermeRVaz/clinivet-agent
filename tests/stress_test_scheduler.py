import argparse
import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from langchain_core.messages import HumanMessage

from src.clinivet_brain import clinivet_agent
from src.clinivet_db import get_supabase_client


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        for key in ("user_id", "requested_slot", "final_slot", "status"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        return json.dumps(payload, ensure_ascii=True)


@dataclass
class BookingResult:
    user_id: int
    requested_slot: str
    final_slot: Optional[str]
    success: bool
    error: Optional[str] = None


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("StressTestScheduler")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    return logger


def _extract_final_slot_from_db(phone: str) -> Optional[str]:
    client = get_supabase_client()
    lead_resp = (
        client.table("leads")
        .select("id")
        .eq("phone", phone)
        .order("id", desc=True)
        .limit(1)
        .execute()
    )
    if not lead_resp.data:
        return None

    lead_id = lead_resp.data[0]["id"]
    appt_resp = (
        client.table("appointments")
        .select("appointment_time")
        .eq("lead_id", lead_id)
        .order("appointment_time", desc=False)
        .limit(1)
        .execute()
    )
    if not appt_resp.data:
        return None

    appointment_time = appt_resp.data[0]["appointment_time"]
    if not appointment_time:
        return None

    try:
        dt = datetime.fromisoformat(str(appointment_time).replace("Z", "+00:00"))
        return dt.strftime("%H:%M")
    except Exception:
        return str(appointment_time)


def _run_single_user_sync(user_id: int, requested_slot: str) -> BookingResult:
    phone = f"551499990{user_id:04d}"
    thread_id = phone
    message = (
        f"Oi, meu nome e Cliente {user_id}. "
        f"Meu cachorro Rex (cao) precisa de consulta e quero horario {requested_slot}. "
        f"Meu telefone e {phone}."
    )

    result = clinivet_agent.invoke(
        {"messages": [HumanMessage(content=message)], "thread_id": thread_id},
        config={"configurable": {"thread_id": thread_id}},
    )
    assistant_message = str(result.get("assistant_message", ""))
    final_slot = _extract_final_slot_from_db(phone)
    success = bool(final_slot) and "Agendamento confirmado" in assistant_message
    if success:
        return BookingResult(
            user_id=user_id,
            requested_slot=requested_slot,
            final_slot=final_slot,
            success=True,
        )
    return BookingResult(
        user_id=user_id,
        requested_slot=requested_slot,
        final_slot=final_slot,
        success=False,
        error=assistant_message or "Agendamento nao confirmado.",
    )


async def _run_single_user(user_id: int, requested_slot: str) -> BookingResult:
    try:
        return await asyncio.to_thread(_run_single_user_sync, user_id, requested_slot)
    except Exception as exc:
        return BookingResult(
            user_id=user_id,
            requested_slot=requested_slot,
            final_slot=None,
            success=False,
            error=str(exc),
        )


async def run_stress_test(users: int, requested_slot: str = "09:00") -> None:
    logger = _setup_logger()
    tasks = [_run_single_user(user_id=i + 1, requested_slot=requested_slot) for i in range(users)]
    results = await asyncio.gather(*tasks)

    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    for item in results:
        logger.info(
            "booking_result",
            extra={
                "user_id": item.user_id,
                "requested_slot": item.requested_slot,
                "final_slot": item.final_slot,
                "status": "success" if item.success else "failed",
            },
        )

    assigned_slots = [r.final_slot for r in successful if r.final_slot]
    unique_slots = set(assigned_slots)
    duplicate_exists = len(unique_slots) != len(assigned_slots)
    moved_to_next_slot = any(slot != requested_slot for slot in assigned_slots)

    print("\n--- RESULTADO POR USUARIO ---")
    for item in results:
        final_slot = item.final_slot or f"FALHA ({item.error})"
        print(f"User {item.user_id} -> {final_slot}")

    print("\n--- RESUMO ---")
    print(f"TOTAL USERS: {users}")
    print(f"SUCCESSFUL BOOKINGS: {len(successful)}")
    print(f"FAILED BOOKINGS: {len(failed)}")
    print(f"DUPLICATE APPOINTMENTS: {'YES' if duplicate_exists else 'NO'}")
    print(
        "NEXT SLOT SELECTED WHEN REQUESTED SLOT WAS TAKEN: "
        f"{'YES' if moved_to_next_slot else 'NO'}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stress test de agendamento concorrente no Clinivet Agent.")
    parser.add_argument(
        "--users",
        type=int,
        default=50,
        help="Numero de usuarios concorrentes (padrao: 50).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.users < 1:
        raise ValueError("--users deve ser maior que zero.")
    asyncio.run(run_stress_test(users=args.users))
