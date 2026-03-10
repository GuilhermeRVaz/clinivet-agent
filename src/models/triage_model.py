from typing import Literal, Optional

from pydantic import BaseModel


class TriageOutput(BaseModel):
    tutor_name: Optional[str] = None
    tutor_cpf: Optional[str] = None
    pet_name: Optional[str] = None
    pet_species: Literal["Cão", "Cao", "Gato", "Desconhecido"] = "Desconhecido"
    urgency_level: Literal["emergency", "routine"] = "routine"
    service_suggested: str = "Consulta"
    symptoms_summary: Optional[str] = None
    phone: Optional[str] = None
    pet_weight: Optional[float] = None
    pet_breed: Optional[str] = None
    pet_age: Optional[str] = None
    pet_size: Optional[str] = None
