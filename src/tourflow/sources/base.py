"""Общий цикл загрузки источника: план → запрос → raw → разбор → stg → водяной знак.

Каждый источник описывает только три вещи: что запросить (plan), как запросить (request)
и как разобрать ответ (parse). Инкрементальность, идемпотентность и журнал — здесь, один раз.
"""
from __future__ import annotations

import hashlib
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
from sqlalchemy import Engine, text

from tourflow import db, runlog
from tourflow.config import AppConfig
from tourflow.sources.http import NotFound, Transport

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Request:
    key: str                 # ключ водяного знака (статья, точка, валюта, год, файл)
    url: str
    params: dict | None
    period_start: date | None
    period_end: date | None


class Source(ABC):
    name: str
    stg_table: str
    stg_key: tuple[str, ...]
    date_col: str = "day"

    def __init__(self, cfg: AppConfig, transport: Transport, today: date):
        self.cfg = cfg
        self.transport = transport
        self.today = today
        self.scfg = getattr(cfg.sources, self.name)

    @abstractmethod
    def plan(self, engine: Engine) -> list[Request]: ...

    @abstractmethod
    def parse(self, req_key: str, content: bytes, content_type: str) -> pd.DataFrame: ...

    def watermark_key(self, row: pd.Series) -> str:
        return "*"

    # ---------- общий цикл ----------
    def start_after(self, engine: Engine, key: str, reload_days: int) -> date:
        """Первая дата, которую нужно запросить: история с нуля либо хвост после водяного знака."""
        wm = runlog.get_watermark(engine, self.name, key)
        if wm is None:
            return self.cfg.project.history_start
        return max(self.cfg.project.history_start, wm - timedelta(days=reload_days) + timedelta(days=1))

    def run(self, engine: Engine, run_id: uuid.UUID | None) -> runlog.StepStats:
        with runlog.step(engine, run_id, self.name, "ingest") as st:
            st.watermark_before = runlog.get_watermark(engine, self.name, "*")
            requests = self.plan(engine)
            if not requests:
                st.status = "skipped"
                st.note("нет новых периодов для загрузки")
            for req in requests:
                st.requests += 1
                try:
                    content, ctype = self.transport.get(req.url, req.params)
                except NotFound:
                    st.note(f"{req.key}: нет данных за {req.period_start}..{req.period_end}")
                    continue
                self._store_raw(engine, req, content, ctype, run_id)
            self._parse_pending(engine, st)
            if st.status == "skipped" and st.rows_fetched:
                st.status = "success"            # данные пришли не по сети, а из inbox
            st.watermark_after = runlog.get_watermark(engine, self.name, "*")
        return st

    def _store_raw(self, engine: Engine, req: Request, content: bytes, ctype: str,
                   run_id: uuid.UUID | None) -> None:
        sha = hashlib.sha256(content).hexdigest()
        with engine.begin() as c:
            c.execute(text("""
                INSERT INTO raw.payload (source, request_key, period_start, period_end, content_type,
                                         content, size_bytes, sha256, run_id)
                VALUES (:s, :k, :ps, :pe, :ct, :c, :n, :h, :r)
                ON CONFLICT (source, request_key, sha256) DO NOTHING"""),
                {"s": self.name, "k": req.key, "ps": req.period_start, "pe": req.period_end,
                 "ct": ctype[:200], "c": content, "n": len(content), "h": sha, "r": run_id})

    def _parse_pending(self, engine: Engine, st: runlog.StepStats) -> None:
        with engine.connect() as c:
            pending = c.execute(text("""SELECT id, request_key, content, content_type FROM raw.payload
                                        WHERE source = :s AND parsed_at IS NULL ORDER BY id"""),
                                {"s": self.name}).all()
        for pid, key, content, ctype in pending:
            try:
                df = self.parse(key, bytes(content), ctype)
            except Exception as exc:  # noqa: BLE001 — битый ответ не должен ронять остальные
                st.status = "partial"
                st.note(f"payload {pid} ({key}) не разобран: {exc}")
                continue  # останется parsed_at IS NULL и будет разобран после исправления парсера
            st.rows_fetched += len(df)
            if not df.empty:
                if "payload_id" in db.table(engine, self.stg_table).c:
                    df = df.assign(payload_id=pid)
                ins, upd = db.upsert(engine, self.stg_table, df, self.stg_key)
                st.rows_inserted += ins
                st.rows_updated += upd
                for wkey, part in df.groupby(df.apply(self.watermark_key, axis=1)):
                    runlog.set_watermark(engine, self.name, wkey, pd.Timestamp(part[self.date_col].max()).date())
                runlog.set_watermark(engine, self.name, "*", pd.Timestamp(df[self.date_col].max()).date())
            with engine.begin() as c:
                c.execute(text("UPDATE raw.payload SET parsed_at = now() WHERE id = :i"), {"i": pid})


def month_start(d: date) -> date:
    return d.replace(day=1)


def add_months(d: date, n: int) -> date:
    m = d.month - 1 + n
    return date(d.year + m // 12, m % 12 + 1, 1)
