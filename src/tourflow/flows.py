"""Оркестрация в Prefect: те же шаги, что в pipeline.py, с повторами и расписанием.

Запуск по расписанию (сервис scheduler в docker-compose):
    python -m tourflow.flows            # ежедневно в 06:30 по Москве (см. CRON)
Разовый запуск без Prefect — `tourflow run`.
"""
from __future__ import annotations

import os

from prefect import flow, get_run_logger, task

from tourflow import db, pipeline, runlog
from tourflow.config import get_config, get_settings

CRON = os.getenv("TOURFLOW_CRON", "30 6 * * *")


@task(retries=2, retry_delay_seconds=300)
def ingest_source(name: str, run_id: str) -> str:
    st = pipeline.ingest(db.get_engine(), get_config(), get_settings(), run_id, only=[name])
    return st[name]


@task
def quality(run_id: str) -> None:
    pipeline.check_quality(db.get_engine(), get_config(), get_settings(), run_id)


@task
def features_and_train(run_id: str) -> dict:
    eng = db.get_engine()
    df, h = pipeline.make_features(eng, get_config(), run_id)
    return pipeline.train(eng, get_config(), get_settings(), run_id, df, h)


@flow(name="tourflow-daily")
def daily() -> dict:
    log = get_run_logger()
    eng = db.get_engine()
    run_id = runlog.start_run(eng, "prefect")
    try:
        # источники независимы: сбой одного не останавливает остальные
        futures = {n: ingest_source.submit(n, run_id) for n in ("calendar", "cbr", "weather", "pageviews", "rosstat")}
        status = {n: f.result(raise_on_failure=False) for n, f in futures.items()}
        log.info("источники: %s", status)
        quality(run_id)
        result = features_and_train(run_id)
        failed = [n for n, s in status.items() if s != "success" and s != "skipped"]
        runlog.finish_run(eng, run_id, "partial" if failed else "success",
                          f"источники с ошибкой: {failed}" if failed else None)
        return result
    except Exception as exc:
        runlog.finish_run(eng, run_id, "failed", str(exc)[:1000])
        raise


if __name__ == "__main__":
    daily.serve(name="tourflow-daily", cron=CRON, timezone="Europe/Moscow")
