from typing import Dict

from src.clinivet_db import get_pet_history


def load_pet_history(phone: str) -> Dict[str, list]:
    return get_pet_history(phone)
