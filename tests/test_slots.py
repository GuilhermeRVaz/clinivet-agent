from src.services.scheduling_service import find_available_slots


def test_find_available_slots():
    slots = find_available_slots(
        date="2026-03-10",
        period="morning",
        service_name="Consulta"
    )

    print("Slots disponíveis:", slots)