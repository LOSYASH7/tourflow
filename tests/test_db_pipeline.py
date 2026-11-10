"""Интеграционный тест на PostgreSQL: инкрементальность и идемпотентность загрузки.

Запускается, если задан TOURFLOW_TEST_DATABASE_URL (DSN владельца отдельной тестовой БД),
например: postgresql+psycopg://postgres:pw@localhost:5432/tourflow_test
"""
import os
from datetime import date

import pytest
from sqlalchemy import text

from tourflow import db, pipeline
from tourflow.config import Settings

URL = os.getenv("TOURFLOW_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TOURFLOW_TEST_DATABASE_URL не задан")
TABLES = ["stg.accommodation", "stg.weather_daily", "stg.pageviews_daily", "stg.fx_daily", "stg.calendar_daily"]


def _counts(engine):
    with engine.connect() as c:
        return {t: c.execute(text(f"SELECT count(*) FROM {t}")).scalar_one() for t in TABLES}


@pytest.fixture(scope="module")
def engine():
    eng = db.get_engine(URL)
    with eng.begin() as c:
        c.execute(text("DROP SCHEMA IF EXISTS raw, stg, mart, ops CASCADE"))
    db.apply_schema(eng)
    return eng


def _settings(as_of):
    return Settings(database_url=URL, source_mode="synthetic", synthetic_as_of=as_of)


def test_rerun_is_idempotent_and_shift_is_incremental(engine, cfg):
    s1 = _settings(date(2026, 6, 15))
    pipeline.ingest(engine, cfg, s1, None)
    first = _counts(engine)
    assert all(v > 0 for v in first.values())

    pipeline.ingest(engine, cfg, s1, None)                      # тот же день ещё раз
    assert _counts(engine) == first
    with engine.connect() as c:
        last = c.execute(text("SELECT sum(rows_inserted), sum(rows_updated) FROM ops.load_log "
                              "WHERE id > (SELECT max(id) - 5 FROM ops.load_log)")).one()
    assert tuple(last) == (0, 0)

    pipeline.ingest(engine, cfg, _settings(date(2026, 7, 15)), None)   # прошёл месяц
    after = _counts(engine)
    assert after["stg.accommodation"] == first["stg.accommodation"] + len(cfg.regions) * len(cfg.accommodation_types)
    assert after["stg.weather_daily"] > first["stg.weather_daily"]


def test_quality_and_features(engine, cfg):
    pipeline.check_quality(engine, cfg, _settings(date(2026, 7, 15)), None)
    df, h = pipeline.make_features(engine, cfg, None)
    df2, h2 = pipeline.make_features(engine, cfg, None)
    assert h == h2 and len(df) == len(df2)


def test_train_end_to_end_is_reproducible(engine, cfg, tmp_path):
    s = _settings(date(2026, 7, 15)).model_copy(update={"artifacts_dir": tmp_path})
    r1 = pipeline.train(engine, cfg, s, None)
    r2 = pipeline.train(engine, cfg, s, None)
    assert r1["version"] == r2["version"] and r1["wape_holdout"] == r2["wape_holdout"]
    with engine.connect() as c:
        assert c.execute(text("SELECT count(*) FROM mart.model_registry WHERE is_active")).scalar_one() == 1
        n_future = c.execute(text("SELECT count(*) FROM mart.forecasts WHERE kind = 'future'")).scalar_one()
        assert n_future == len(cfg.regions) * len(cfg.accommodation_types) * len(cfg.forecast.horizons)
        assert c.execute(text("SELECT count(*) FROM mart.explanations")).scalar_one() > 0
