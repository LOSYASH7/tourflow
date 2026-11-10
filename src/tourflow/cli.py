"""Командная строка: `tourflow <команда>` или `python -m tourflow.cli <команда>`."""
from __future__ import annotations

import argparse
import getpass
import json
import logging
import sys

from tourflow import db, pipeline, runlog
from tourflow.config import get_config, get_settings


def _setup_logging(level: str) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _check_secrets(settings) -> None:
    dsn = settings.database_url.get_secret_value()
    if settings.source_mode == "live" and ("change-me" in dsn.lower() or "changeme" in dsn.lower()):
        sys.exit("В .env остались пароли по умолчанию. Задайте свои (см. .env.example).")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="tourflow", description="Прогноз турпотока по видам размещения")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("db-init", help="создать/обновить схему (нужны права владельца БД)").add_argument(
        "--admin-url", required=True, help="DSN администратора; не сохраняется")
    ing = sub.add_parser("ingest", help="загрузить новые данные из источников")
    ing.add_argument("--only", nargs="*", help="calendar cbr weather pageviews rosstat")
    sub.add_parser("quality", help="проверки качества stg")
    sub.add_parser("features", help="пересчитать витрину признаков")
    sub.add_parser("train", help="бэктест, выбор модели, прогноз, объяснения")
    sub.add_parser("run", help="весь конвейер: ingest → quality → features → train")
    sub.add_parser("status", help="последние запуски и загрузки")
    sub.add_parser("hash-password", help="хеш пароля для дашборда")
    a = p.parse_args(argv)

    settings, cfg = get_settings(), get_config()
    _setup_logging(settings.log_level)

    if a.cmd == "hash-password":
        pw = getpass.getpass("Пароль дашборда: ")
        if len(pw) < 10:
            sys.exit("Пароль должен быть не короче 10 символов")
        from tourflow.security import hash_password

        print("TOURFLOW_DASHBOARD_PASSWORD_HASH=" + hash_password(pw))
        return
    if a.cmd == "db-init":
        db.apply_schema(db.get_engine(a.admin_url))
        return

    _check_secrets(settings)
    engine = db.get_engine()
    if a.cmd == "run":
        res = pipeline.run_all(cfg, settings)
        print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
        return
    if a.cmd == "status":
        print(db.read_df(engine, "SELECT run_id, started_at, finished_at, status, message FROM ops.pipeline_runs "
                                 "ORDER BY started_at DESC LIMIT 5").to_string())
        print(db.read_df(engine, "SELECT source, step, status, rows_fetched, rows_inserted, rows_updated, "
                                 "watermark_after, left(message, 80) AS message FROM ops.load_log "
                                 "ORDER BY id DESC LIMIT 15").to_string())
        return
    run_id = runlog.start_run(engine, f"cli:{a.cmd}")
    try:
        if a.cmd == "ingest":
            print(pipeline.ingest(engine, cfg, settings, run_id, a.only))
        elif a.cmd == "quality":
            pipeline.check_quality(engine, cfg, settings, run_id)
        elif a.cmd == "features":
            pipeline.make_features(engine, cfg, run_id)
        elif a.cmd == "train":
            print(json.dumps(pipeline.train(engine, cfg, settings, run_id), ensure_ascii=False, indent=2,
                             default=str))
        runlog.finish_run(engine, run_id, "success")
    except Exception as exc:
        runlog.finish_run(engine, run_id, "failed", str(exc)[:1000])
        raise


if __name__ == "__main__":
    main()
