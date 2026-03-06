from datetime import datetime, timedelta
from typing import Optional

from src.clinivet_calendar import TIMEZONE


def build_next_business_day(reference: Optional[datetime] = None) -> str:
    base = reference or datetime.now(TIMEZONE)
    return (base + timedelta(days=1)).strftime("%Y-%m-%d")
