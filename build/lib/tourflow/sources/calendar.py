"""Производственный календарь РФ (isdayoff.ru): тип каждого дня года.

Ответ — строка по одному символу на день: 0 рабочий, 1 нерабочий, 2 сокращённый, 4 рабочий (особый).
Загружаем прошлые годы один раз, текущий и следующий — при каждом запуске (календарь на
следующий год публикуется осенью, и он нужен для признаков будущих месяцев).
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
from sqlalchemy import Engine

from tourflow import runlog
from tourflow.sources.base import Request, Source


class CalendarSource(Source):
    name = "calendar"
    stg_table = "stg.calendar_daily"
    stg_key = ("day",)

    def plan(self, engine: Engine) -> list[Request]:
        reqs = []
        for year in range(self.cfg.project.history_start.year, self.today.year + 2):
            wm = runlog.get_watermark(engine, self.name, str(year))
            if wm is not None and wm >= date(year, 12, 31) and year < self.today.year:
                continue
            reqs.append(Request(str(year), self.scfg["base_url"], {"year": year, "cc": "ru"},
                                date(year, 1, 1), date(year, 12, 31)))
        return reqs

    def parse(self, req_key: str, content: bytes, content_type: str) -> pd.DataFrame:
        body = content.decode("ascii", errors="replace").strip()
        year = int(req_key)
        n_days = (date(year + 1, 1, 1) - date(year, 1, 1)).days
        if len(body) != n_days or not body.isdigit():
            # коды ошибок сервиса (100, 101, 199) — календаря на этот год ещё нет
            return pd.DataFrame(columns=["day", "day_type"])
        days = [date(year, 1, 1) + timedelta(days=i) for i in range(n_days)]
        types = [{"4": 0}.get(ch, int(ch)) for ch in body]
        return pd.DataFrame({"day": days, "day_type": types})

    def watermark_key(self, row: pd.Series) -> str:
        return str(row["day"].year)
