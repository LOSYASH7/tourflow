-- Схема хранилища. Идемпотентна: можно применять повторно.
-- Слои: raw (как получили) -> stg (единый вид) -> mart (витрина, прогнозы) + ops (служебное).

CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS stg;
CREATE SCHEMA IF NOT EXISTS mart;
CREATE SCHEMA IF NOT EXISTS ops;

-- ---------- raw ----------
CREATE TABLE IF NOT EXISTS raw.payload (
    id            BIGSERIAL PRIMARY KEY,
    source        TEXT        NOT NULL,
    request_key   TEXT        NOT NULL,            -- что запрашивали (статья, точка, валюта, файл)
    period_start  DATE,
    period_end    DATE,
    content_type  TEXT        NOT NULL,
    content       BYTEA       NOT NULL,
    size_bytes    INTEGER     NOT NULL,
    sha256        CHAR(64)    NOT NULL,
    fetched_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    run_id        UUID,
    parsed_at     TIMESTAMPTZ,
    UNIQUE (source, request_key, sha256)            -- тот же ответ повторно не пишется
);
CREATE INDEX IF NOT EXISTS payload_unparsed_idx ON raw.payload (source) WHERE parsed_at IS NULL;

-- ---------- stg ----------
CREATE TABLE IF NOT EXISTS stg.accommodation (
    region_code  CHAR(8)  NOT NULL,
    acc_type     TEXT     NOT NULL,
    month        DATE     NOT NULL,
    guests       BIGINT   NOT NULL CHECK (guests >= 0),
    payload_id   BIGINT   REFERENCES raw.payload(id),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (region_code, acc_type, month)
);

CREATE TABLE IF NOT EXISTS stg.weather_daily (
    point_id     TEXT    NOT NULL,
    region_code  CHAR(8) NOT NULL,
    day          DATE    NOT NULL,
    t_mean       DOUBLE PRECISION,
    t_max        DOUBLE PRECISION,
    precip_mm    DOUBLE PRECISION,
    sunshine_h   DOUBLE PRECISION,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (point_id, day)
);

CREATE TABLE IF NOT EXISTS stg.pageviews_daily (
    article      TEXT    NOT NULL,
    region_code  CHAR(8) NOT NULL,
    day          DATE    NOT NULL,
    views        INTEGER NOT NULL CHECK (views >= 0),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (article, day)
);

CREATE TABLE IF NOT EXISTS stg.fx_daily (
    currency     CHAR(3) NOT NULL,
    day          DATE    NOT NULL,
    rate         NUMERIC(12, 4) NOT NULL CHECK (rate > 0),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (currency, day)
);

CREATE TABLE IF NOT EXISTS stg.calendar_daily (
    day          DATE PRIMARY KEY,
    day_type     SMALLINT NOT NULL,                -- 0 рабочий, 1 нерабочий, 2 сокращённый
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------- mart ----------
CREATE TABLE IF NOT EXISTS mart.features (
    region_code   CHAR(8) NOT NULL,
    acc_type      TEXT    NOT NULL,
    origin_month  DATE    NOT NULL,                -- последний месяц, известный на дату прогноза
    horizon       SMALLINT NOT NULL,
    target_month  DATE    NOT NULL,
    target        DOUBLE PRECISION,                -- NULL для будущих месяцев
    features      JSONB   NOT NULL,
    data_hash     CHAR(16) NOT NULL,
    PRIMARY KEY (region_code, acc_type, origin_month, horizon)
);

CREATE TABLE IF NOT EXISTS mart.model_registry (
    model_version   TEXT PRIMARY KEY,
    model_name      TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    data_hash       CHAR(16) NOT NULL,
    seed            INTEGER NOT NULL,
    features        JSONB NOT NULL,
    params          JSONB NOT NULL,
    metrics         JSONB NOT NULL,
    artifact_path   TEXT,
    artifact_sha256 CHAR(64),
    is_active       BOOLEAN NOT NULL DEFAULT false
);
CREATE UNIQUE INDEX IF NOT EXISTS one_active_model ON mart.model_registry (is_active) WHERE is_active;

CREATE TABLE IF NOT EXISTS mart.forecasts (
    model_version  TEXT    NOT NULL,
    model_name     TEXT    NOT NULL,
    kind           TEXT    NOT NULL CHECK (kind IN ('backtest', 'holdout', 'future')),
    region_code    CHAR(8) NOT NULL,
    acc_type       TEXT    NOT NULL,
    origin_month   DATE    NOT NULL,
    horizon        SMALLINT NOT NULL,
    target_month   DATE    NOT NULL,
    y_true         DOUBLE PRECISION,
    y_pred         DOUBLE PRECISION NOT NULL,
    y_lo           DOUBLE PRECISION,
    y_hi           DOUBLE PRECISION,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (model_version, kind, region_code, acc_type, origin_month, horizon)
);

CREATE TABLE IF NOT EXISTS mart.metrics (
    model_version  TEXT NOT NULL,
    model_name     TEXT NOT NULL,
    split          TEXT NOT NULL,                  -- backtest | holdout
    horizon        SMALLINT NOT NULL,              -- 0 = все горизонты
    mae            DOUBLE PRECISION,
    rmse           DOUBLE PRECISION,
    wape           DOUBLE PRECISION,
    mae_naive      DOUBLE PRECISION,
    n              INTEGER,
    PRIMARY KEY (model_version, split, horizon)
);

CREATE TABLE IF NOT EXISTS mart.importance (
    model_version  TEXT NOT NULL,
    method         TEXT NOT NULL,                  -- shap | permutation | coef
    period         TEXT NOT NULL,                  -- all | год | сезон
    feature        TEXT NOT NULL,
    importance     DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (model_version, method, period, feature)
);

CREATE TABLE IF NOT EXISTS mart.explanations (
    model_version  TEXT    NOT NULL,
    region_code    CHAR(8) NOT NULL,
    acc_type       TEXT    NOT NULL,
    origin_month   DATE    NOT NULL,
    horizon        SMALLINT NOT NULL,
    base_value     DOUBLE PRECISION NOT NULL,
    contributions  JSONB   NOT NULL,               -- {признак: [значение, вклад]}
    PRIMARY KEY (model_version, region_code, acc_type, origin_month, horizon)
);

CREATE TABLE IF NOT EXISTS mart.pdp (
    model_version  TEXT NOT NULL,
    feature        TEXT NOT NULL,
    grid_value     DOUBLE PRECISION NOT NULL,
    pd_value       DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (model_version, feature, grid_value)
);

-- ---------- ops ----------
CREATE TABLE IF NOT EXISTS ops.pipeline_runs (
    run_id       UUID PRIMARY KEY,
    started_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at  TIMESTAMPTZ,
    status       TEXT NOT NULL DEFAULT 'running',
    trigger      TEXT NOT NULL DEFAULT 'manual',
    message      TEXT
);

CREATE TABLE IF NOT EXISTS ops.load_log (
    id               BIGSERIAL PRIMARY KEY,
    run_id           UUID,
    source           TEXT NOT NULL,
    step             TEXT NOT NULL,                 -- fetch | parse | transform | train ...
    started_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at      TIMESTAMPTZ,
    status           TEXT NOT NULL DEFAULT 'running', -- running | success | skipped | failed
    requests         INTEGER DEFAULT 0,
    rows_fetched     INTEGER DEFAULT 0,
    rows_inserted    INTEGER DEFAULT 0,
    rows_updated     INTEGER DEFAULT 0,
    watermark_before DATE,
    watermark_after  DATE,
    message          TEXT
);
CREATE INDEX IF NOT EXISTS load_log_run_idx ON ops.load_log (run_id);

CREATE TABLE IF NOT EXISTS ops.source_state (
    source      TEXT NOT NULL,
    key         TEXT NOT NULL,                      -- статья / точка / валюта / '*'
    watermark   DATE NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (source, key)
);

CREATE TABLE IF NOT EXISTS ops.dq_checks (
    id          BIGSERIAL PRIMARY KEY,
    run_id      UUID,
    table_name  TEXT NOT NULL,
    check_name  TEXT NOT NULL,
    passed      BOOLEAN NOT NULL,
    details     TEXT,
    checked_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
