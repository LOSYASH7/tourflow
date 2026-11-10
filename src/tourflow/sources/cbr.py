"""ЦБ РФ — динамика официальных курсов валют (XML).

XML разбирается через defusedxml: защита от XXE и «XML-бомб».
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
from defusedxml import ElementTree
from sqlalchemy import Engine

from tourflow.sources.base import Request, Source


class CbrSource(Source):
    name = "cbr"
    stg_table = "stg.fx_daily"
    stg_key = ("currency", "day")

    def plan(self, engine: Engine) -> list[Request]:
        reqs = []
        for cur, code in self.scfg["currencies"].items():
            start = self.start_after(engine, cur, self.scfg.get("reload_days", 7))
            if start > self.today:
                continue
            reqs.append(Request(cur, self.scfg["base_url"], {
                "date_req1": start.strftime("%d/%m/%Y"), "date_req2": self.today.strftime("%d/%m/%Y"),
                "VAL_NM_RQ": code}, start, self.today))
        return reqs

    def parse(self, req_key: str, content: bytes, content_type: str) -> pd.DataFrame:
        root = ElementTree.fromstring(content)
        rows = []
        for rec in root.iter("Record"):
            value = float(rec.findtext("Value").replace(",", "."))
            nominal = float(rec.findtext("Nominal") or 1)
            rows.append({"currency": req_key,
                         "day": datetime.strptime(rec.get("Date"), "%d.%m.%Y").date(),
                         "rate": round(value / nominal, 4)})
        return pd.DataFrame(rows, columns=["currency", "day", "rate"])

    def watermark_key(self, row: pd.Series) -> str:
        return row["currency"]

