-- Migration: pet metadata + multi-calendar + appointment lock
-- Date: 2026-03-07

BEGIN;

ALTER TABLE IF EXISTS leads
    ADD COLUMN IF NOT EXISTS pet_weight DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS pet_breed TEXT,
    ADD COLUMN IF NOT EXISTS pet_age TEXT,
    ADD COLUMN IF NOT EXISTS pet_size TEXT DEFAULT 'unknown';

ALTER TABLE IF EXISTS services
    ADD COLUMN IF NOT EXISTS calendar_type TEXT DEFAULT 'calendar_consultas';

UPDATE services
SET calendar_type = CASE
    WHEN lower(name) = 'banho e tosa' THEN 'calendar_banho_tosa'
    WHEN lower(name) IN ('cirurgia', 'cirurgia tecidos moles', 'anestesia inalatoria') THEN 'calendar_cirurgia'
    WHEN lower(name) IN ('vacina', 'vacinacao') THEN 'calendar_vacinas'
    ELSE 'calendar_consultas'
END
WHERE calendar_type IS NULL
   OR calendar_type = ''
   OR calendar_type = 'calendar_consultas';

ALTER TABLE IF EXISTS appointments
    ADD COLUMN IF NOT EXISTS pet_weight DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS pet_breed TEXT,
    ADD COLUMN IF NOT EXISTS pet_age TEXT,
    ADD COLUMN IF NOT EXISTS pet_size TEXT DEFAULT 'unknown';

COMMIT;

CREATE UNIQUE INDEX IF NOT EXISTS ux_appointments_service_time
ON appointments(service_id, appointment_time);
