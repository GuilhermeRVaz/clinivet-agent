BEGIN;

ALTER TABLE IF EXISTS leads
    ADD COLUMN IF NOT EXISTS tutor_cpf TEXT;

CREATE INDEX IF NOT EXISTS idx_leads_tutor_cpf
ON leads(tutor_cpf);

COMMIT;
