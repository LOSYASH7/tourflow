"""Wikimedia Pageviews API — суточные просмотры статей о курортах (журнал обращений).

Интерес к статье о городе растёт до поездки, поэтому это опережающий индикатор турпотока.
"""
from __future__ import annotations

import json
from datetime import timedelta
from urllib.parse import quote

import pandas as pd
from sqlalchemy import Engine

from tourflow.sources.base import Request, Source


class PageviewsSource(Source):
    name = "pageviews"
    stg_table = "stg.pageviews_daily"
    stg_key = ("article", "day")

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.article_region: dict[str, str] = {}
        for r in self.cfg.regions:
            for a in r.articles:
                if a in self.article_region:
                    raise ValueError(f"статья {a} указана у двух регионов")
                self.article_region[a] = r.code

    def plan(self, engine: Engine) -> list[Request]:
        end = self.today - timedelta(days=1)
        reqs = []
        for art in self.article_region:
            start = self.start_after(engine, art, self.scfg.get("reload_days", 3))
            if start > end:
                continue
            url = f"{self.scfg['base_url']}/{quote(art, safe='')}/daily/{start:%Y%m%d}00/{end:%Y%m%d}00"
            reqs.append(Request(art, url, None, start, end))
        return reqs

    def parse(self, req_key: str, content: bytes, content_type: str) -> pd.DataFrame:
        items = json.loads(content).get("items", [])
        if not items:
            return pd.DataFrame(columns=["article", "region_code", "day", "views"])
        df = pd.DataFrame(items)
        return pd.DataFrame({
            "article": req_key,
            "region_code": self.article_region[req_key],
            "day": pd.to_datetime(df["timestamp"].str[:8], format="%Y%m%d").dt.date,
            "views": df["views"].astype(int),
        })

    def watermark_key(self, row: pd.Series) -> str:
        return row["article"]
