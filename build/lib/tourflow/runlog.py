"""Журнал запусков и загрузок (ops.pipeline_runs, ops.load_log) и водяные знаки источников."""
from __future__ import annotations

import logging
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date

from sqlalchemy import Engine, text

log = logging.getLogger(__name__)


@dataclass
class StepStats:
    requests: int = 0
    rows_fetched: int = 0
    rows_inserted: int = 0
    rows_updated: int = 0
    watermark_before: date | None = None
    watermark_after: date | None = None
    status: str = "success"
    messages: list[str] = field(default_factory=list)

    def note(self, msg: str) -> None:
        self.messages.append(msg)


def start_run(engine: Engine, trigger: str = "manual") -> uuid.UUID:
    run_id = uuid.uuid4()
    with engine.begin() as c:
        c.execute(text("INSERT INTO ops.pipeline_runs (run_id, trigger) VALUES (:r, :t)"),
                  {"r": run_id, "t": trigger})
    return run_id


def finish_run(engine: Engine, run_id: uuid.UUID, status: str, message: str | None = None) -> None:
    with engine.begin() as c:
        c.execute(text("UPDATE ops.pipeline_runs SET finished_at = now(), status = :s, message = :m "
                       "WHERE run_id = :r"), {"r": run_id, "s": status, "m": message})


@contextmanager
def step(engine: Engine, run_id: uuid.UUID | None, source: str, step_name: str, reraise: bool = False):
    """Записывает шаг в ops.load_log. Ошибка шага фиксируется и по умолчанию не роняет весь прогон."""
    with engine.begin() as c:
        log_id = c.execute(
            text("INSERT INTO ops.load_log (run_id, source, step) VALUES (:r, :s, :st) RETURNING id"),
            {"r": run_id, "s": source, "st": step_name},
        ).scalar_one()
    stats = StepStats()
    try:
        yield stats
    except Exception as exc:  # noqa: BLE001 — фиксируем любую ошибку источника в журнале
        stats.status = "failed"
        stats.note(f"{type(exc).__name__}: {exc}")
        log.error("step %s/%s failed: %s", source, step_name, exc)
        log.debug("%s", traceback.format_exc())
        if reraise:
            raise
    finally:
        with engine.begin() as c:
            c.execute(
                text("""UPDATE ops.load_log SET finished_at = now(), status = :st, requests = :rq,
                        rows_fetched = :rf, rows_inserted = :ri, rows_updated = :ru,
                        watermark_before = :wb, watermark_after = :wa, message = :m WHERE id = :id"""),
                {"id": log_id, "st": stats.status, "rq": stats.requests, "rf": stats.rows_fetched,
                 "ri": stats.rows_inserted, "ru": stats.rows_updated, "wb": stats.watermark_before,
                 "wa": stats.watermark_after, "m": "; ".join(stats.messages)[:4000] or None},
            )
        log.info("%s/%s: %s fetched=%d ins=%d upd=%d", source, step_name, stats.status,
                 stats.rows_fetched, stats.rows_inserted, stats.rows_updated)


def get_watermark(engine: Engine, source: str, key: str = "*") -> date | None:
    with engine.connect() as c:
        return c.execute(text("SELECT watermark FROM ops.source_state WHERE source = :s AND key = :k"),
                         {"s": source, "k": key}).scalar_one_or_none()


def set_watermark(engine: Engine, source: str, key: str, wm: date) -> None:
    """Водяной знак только растёт — повторный запуск со старыми данными его не откатит."""
    with engine.begin() as c:
        c.execute(text("""INSERT INTO ops.source_state (source, key, watermark) VALUES (:s, :k, :w)
                          ON CONFLICT (source, key) DO UPDATE
                          SET watermark = GREATEST(ops.source_state.watermark, EXCLUDED.watermark),
                              updated_at = now()"""), {"s": source, "k": key, "w": wm})
