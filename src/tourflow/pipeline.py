"""Шаги системы как обычные функции. Их вызывают CLI и Prefect — логика в одном месте.

Порядок слоёв и зависимостей односторонний:
  sources (внешний мир) → stg → features → models/explain → mart → dashboard (только чтение).
"""
from __future__ import annotations

import logging
import uuid

from sqlalchemy import Engine

from tourflow import db, runlog
from tourflow.config import AppConfig, Settings
from tourflow.etl import quality
from tourflow.features import build_features, load_inputs, save_features
from tourflow.sources import build_sources, today_for

log = logging.getLogger(__name__)


def ingest(engine: Engine, cfg: AppConfig, settings: Settings, run_id: uuid.UUID | None,
           only: list[str] | None = None) -> dict[str, str]:
    """Загрузка источников. Сбой одного источника не останавливает остальные."""
    status = {}
    for src in build_sources(cfg, settings, only):
        status[src.name] = src.run(engine, run_id).status
    return status


def check_quality(engine: Engine, cfg: AppConfig, settings: Settings, run_id: uuid.UUID | None) -> None:
    with runlog.step(engine, run_id, "stg", "quality", reraise=True) as st:
        res = quality.run_checks(engine, cfg, today_for(settings), run_id)
        bad = quality.blocking_failures(res)
        st.rows_fetched = len(res)
        st.note(f"проверок {len(res)}, не пройдено {sum(not r.passed for r in res)}, блокирующих {len(bad)}")
        if bad:
            raise RuntimeError("блокирующие проверки качества не пройдены: "
                               + "; ".join(f"{r.table}.{r.name}" for r in bad))


def make_features(engine: Engine, cfg: AppConfig, run_id: uuid.UUID | None):
    with runlog.step(engine, run_id, "features", "build", reraise=True) as st:
        inp = load_inputs(engine)
        h = inp.data_hash()
        df = build_features(inp, cfg)
        st.rows_inserted = save_features(engine, df, h)
        st.note(f"data_hash={h}")
    return df, h


def train(engine: Engine, cfg: AppConfig, settings: Settings, run_id: uuid.UUID | None,
          df=None, data_hash: str | None = None) -> dict:
    from tourflow.models.train import train_and_forecast

    if df is None:
        df, data_hash = make_features(engine, cfg, run_id)
    return train_and_forecast(engine, df, cfg, settings, data_hash, run_id)


def run_all(cfg: AppConfig, settings: Settings, trigger: str = "manual") -> dict:
    engine = db.get_engine()
    run_id = runlog.start_run(engine, trigger)
    try:
        status = ingest(engine, cfg, settings, run_id)
        if status.get("rosstat") == "failed":
            log.warning("целевая не обновилась — прогноз строится по последним загруженным данным")
        check_quality(engine, cfg, settings, run_id)
        df, h = make_features(engine, cfg, run_id)
        result = train(engine, cfg, settings, run_id, df, h)
        failed = [k for k, v in status.items() if v == "failed"]
        runlog.finish_run(engine, run_id, "partial" if failed else "success",
                          f"источники с ошибкой: {failed}" if failed else None)
        return {"run_id": str(run_id), "sources": status, **result}
    except Exception as exc:
        runlog.finish_run(engine, run_id, "failed", str(exc)[:1000])
        raise
