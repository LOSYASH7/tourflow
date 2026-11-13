#!/usr/bin/env bash
# Выполняется контейнером postgres один раз при первом создании тома.
# Создаёт роли приложения с паролями из переменных окружения (.env), без захардкоженных секретов.
set -euo pipefail

: "${TOURFLOW_ETL_PASSWORD:?нужно задать TOURFLOW_ETL_PASSWORD в .env}"
: "${TOURFLOW_READER_PASSWORD:?нужно задать TOURFLOW_READER_PASSWORD в .env}"

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
     -v etl_pw="$TOURFLOW_ETL_PASSWORD" -v reader_pw="$TOURFLOW_READER_PASSWORD" <<'SQL'
SELECT format('CREATE ROLE tourflow_etl LOGIN PASSWORD %L', :'etl_pw')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'tourflow_etl') \gexec
SELECT format('CREATE ROLE tourflow_reader LOGIN PASSWORD %L', :'reader_pw')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'tourflow_reader') \gexec
REVOKE ALL ON DATABASE :"DBNAME" FROM PUBLIC;
SQL

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
     -c "GRANT CONNECT ON DATABASE \"$POSTGRES_DB\" TO tourflow_etl, tourflow_reader;"

# Схема и права — теми же идемпотентными скриптами, что и `tourflow db-init`.
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" -f /sql/001_schema.sql
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" -f /sql/002_grants.sql
