-- Принцип наименьших привилегий.
--   tourflow_etl    — пайплайн: читает и пишет raw/stg/mart/ops, но не меняет схему;
--   tourflow_reader — дашборд: только SELECT в mart и ops, raw/stg недоступны.
-- Роли создаёт docker/initdb/00_roles.sh (пароли из .env). Если ролей нет — блок пропускается.

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'tourflow_etl') THEN
        GRANT USAGE ON SCHEMA raw, stg, mart, ops TO tourflow_etl;
        GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA raw, stg, mart, ops TO tourflow_etl;
        GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA raw, stg, mart, ops TO tourflow_etl;
    END IF;

    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'tourflow_reader') THEN
        GRANT USAGE ON SCHEMA mart, ops TO tourflow_reader;
        GRANT SELECT ON ALL TABLES IN SCHEMA mart, ops TO tourflow_reader;
        REVOKE ALL ON SCHEMA raw, stg FROM tourflow_reader;
    END IF;
END
$$;

REVOKE CREATE ON SCHEMA public FROM PUBLIC;
