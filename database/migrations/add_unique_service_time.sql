CREATE UNIQUE INDEX IF NOT EXISTS unique_service_time
ON appointments (service_id, appointment_time)
WHERE status <> 'Cancelado';
