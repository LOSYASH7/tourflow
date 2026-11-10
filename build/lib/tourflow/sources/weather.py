"""Open-Meteo Historical Weather API — суточные показания по курортным точкам региона.

Слабоструктурированный источник (ряды датчиков): в stg лежат сутки по точкам,
в признаки идут месячные агрегаты по региону (см. features.py).
"""
from __future__ import annotations

import json
from datetime import date, timedelta

import pandas as pd
from sqlalchemy import Engine

from tourflow.sources.base import Request, Source

DAILY_VARS = "temperature_2m_mean,temperature_2m_max,precipitation_sum,sunshine_duration"


class WeatherSource(Source):
    name = "weather"
    stg_table = "stg.weather_daily"
    stg_key = ("point_id", "day")

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.point_region = {p.id: r.code for r in self.cfg.regions for p in r.points}
        self.points = {p.id: p for r in self.cfg.regions for p in r.points}

    def plan(self, engine: Engine) -> list[Request]:
        end = self.today - timedelta(days=1)
        reqs = []
        for pid, p in self.points.items():
            start = self.start_after(engine, pid, self.scfg.get("reload_days", 7))
            # режем по годам: меньше ответ, а при сбое сохраняется прогресс по уже загруженным годам
            while start <= end:
                chunk_end = min(end, date(start.year, 12, 31))
                reqs.append(Request(pid, self.scfg["base_url"], {
                    "latitude": p.lat, "longitude": p.lon, "start_date": start.isoformat(),
                    "end_date": chunk_end.isoformat(), "daily": DAILY_VARS, "timezone": "Europe/Moscow",
                }, start, chunk_end))
                start = chunk_end + timedelta(days=1)
        return reqs

    def parse(self, req_key: str, content: bytes, content_type: str) -> pd.DataFrame:
        d = json.loads(content)["daily"]
        df = pd.DataFrame({
            "day": pd.to_datetime(d["time"]).date,
            "t_mean": d.get("temperature_2m_mean"),
            "t_max": d.get("temperature_2m_max"),
            "precip_mm": d.get("precipitation_sum"),
            "sunshine_h": pd.Series(d.get("sunshine_duration"), dtype="float") / 3600.0,
        })
        df = df.dropna(subset=["t_mean"])  # последние дни архива бывают ещё пустыми
        df.insert(0, "point_id", req_key)
        df.insert(1, "region_code", self.point_region[req_key])
        return df

    def watermark_key(self, row: pd.Series) -> str:
        return row["point_id"]
