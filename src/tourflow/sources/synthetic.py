"""Синтетический мир для офлайн-режима (TOURFLOW_SOURCE_MODE=synthetic).

Нужен для демонстрации, тестов и CI без доступа к сети. Отдаёт ответы в тех же форматах,
что и настоящие API (JSON Open-Meteo и Wikimedia, XML ЦБ, строка isdayoff, XLSX Росстата),
поэтому весь остальной код системы работает без изменений.

Данные согласованы между источниками: турпоток зависит от сезона, погоды, курса,
праздников; интерес в Википедии опережает поездки примерно на месяц. Всё детерминировано
(seed из config.yaml), «сегодня» задаётся TOURFLOW_SYNTHETIC_AS_OF — так можно показать
инкрементальную догрузку, сдвинув дату.
"""
from __future__ import annotations

import io
import json
from datetime import date, timedelta
from functools import cached_property
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd
from openpyxl import Workbook

from tourflow.config import AppConfig
from tourflow.sources.http import NotFound

FIXED_HOLIDAYS = {(1, d) for d in range(1, 9)} | {(2, 23), (3, 8), (5, 1), (5, 9), (6, 12), (11, 4)}
MONTH_RU = ["январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август", "сентябрь",
            "октябрь", "ноябрь", "декабрь"]
# профиль региона: масштаб (тыс. чел./мес. в гостиницах), морской курорт, тренд
PROFILE = {
    "03000000": (900, True, 0.05), "45000000": (1600, False, 0.04), "40000000": (900, False, 0.05),
    "46000000": (450, False, 0.03), "35000000": (500, True, 0.07), "92000000": (300, False, 0.06),
    "27000000": (220, True, 0.06), "07000000": (260, False, 0.04),
}
ACC_SCALE = {"hotels": 1.0, "sanatoriums": 0.18, "recreation": 0.25}


class SyntheticWorld:
    def __init__(self, cfg: AppConfig, as_of: date):
        self.cfg = cfg
        self.as_of = as_of
        self.start = cfg.project.history_start
        # Мир строится на фиксированном отрезке и не зависит от as_of и порядка вызовов:
        # повторный запуск отдаёт ровно те же числа (иначе нельзя проверить идемпотентность).
        self.days = pd.date_range(self.start, "2030-12-31", freq="D")

    def _rng(self, *name: str) -> np.random.Generator:
        key = [self.cfg.project.seed] + [int.from_bytes(n.encode()[:8].ljust(8, b"\0"), "little") for n in name]
        key += [len(n) for n in name]
        return np.random.default_rng(key)

    # ---------- латентные ряды ----------
    @cached_property
    def calendar(self) -> pd.Series:
        t = [1 if (d.weekday() >= 5 or (d.month, d.day) in FIXED_HOLIDAYS) else 0 for d in self.days]
        return pd.Series(t, index=self.days)

    @cached_property
    def weather(self) -> dict[str, pd.DataFrame]:
        out = {}
        doy = self.days.dayofyear.values
        for r in self.cfg.regions:
            for p in r.points:
                amp = 7 + 0.45 * (p.lat - 43)       # годовая амплитуда растёт к северу
                mean = 24 - 0.75 * (p.lat - 40)
                clim = mean - amp * np.cos(2 * np.pi * (doy - 20) / 365.25) - 6
                rng = self._rng("weather", p.id)
                noise = np.zeros(len(doy))
                eps = rng.normal(0, 2.2, len(doy))
                for i in range(1, len(doy)):
                    noise[i] = 0.8 * noise[i - 1] + eps[i]
                t = clim + noise
                precip = np.where(rng.random(len(doy)) < 0.3, rng.gamma(1.5, 4, len(doy)), 0.0)
                sun = np.clip(4 + 6 * np.sin(np.pi * (doy - 80) / 365.25) - precip * 0.3
                              + rng.normal(0, 1, len(doy)), 0, 15)
                out[p.id] = pd.DataFrame({"t_mean": t, "t_max": t + 5 + rng.normal(0, 1, len(doy)),
                                          "precip": precip, "sun_h": sun}, index=self.days)
        return out

    @cached_property
    def fx(self) -> pd.Series:
        steps = self._rng("fx").normal(0.0002, 0.006, len(self.days))
        lvl = 65 * np.exp(np.cumsum(steps))
        return pd.Series(lvl, index=self.days)

    @cached_property
    def guests(self) -> pd.DataFrame:
        months = pd.date_range(self.start, self.days[-1], freq="MS")
        rows = []
        cal_m = self.calendar.resample("MS").sum()
        fx_m = self.fx.resample("MS").mean()
        for r in self.cfg.regions:
            scale, seaside, trend = PROFILE.get(r.code, (300, False, 0.04))
            w = self.weather[r.points[0].id].resample("MS").agg({"t_mean": "mean"})["t_mean"]
            w_anom = w - w.groupby(w.index.month).transform("mean")
            for acc, a_scale in ACC_SCALE.items():
                rng = self._rng("guests", r.code, acc)
                for m in months:
                    k = m.month
                    if seaside:
                        season = 0.35 + 1.9 * np.exp(-((k - 7.5) ** 2) / 2.2) + (0.15 if k in (1, 5) else 0)
                    else:
                        season = 0.8 + 0.35 * np.exp(-((k - 7) ** 2) / 4) + (0.2 if k in (1, 5, 12) else 0)
                    if acc == "sanatoriums":
                        season = 0.6 * season + 0.4
                    yrs = (m - pd.Timestamp(self.start)).days / 365.25
                    lvl = scale * a_scale * 1000 * season * (1 + trend) ** yrs
                    lvl *= 1 + 0.05 * (cal_m.get(m, 9) - 9)              # праздники и длинные выходные
                    lvl *= (fx_m.get(m, 70) / 70) ** 0.35                  # дорогой доллар -> больше внутреннего туризма
                    if seaside and k in (5, 6, 7, 8, 9):
                        lvl *= 1 + 0.04 * w_anom.get(m, 0)                 # тёплое лето -> больше гостей
                    if m.year == 2020 and k in (4, 5, 6):
                        lvl *= 0.12
                    elif m.year == 2020 and k in (7, 8, 9, 10, 11, 12):
                        lvl *= 0.65
                    lvl *= np.exp(rng.normal(0, 0.05))
                    rows.append({"region_code": r.code, "acc_type": acc, "month": m, "guests": round(lvl)})
        return pd.DataFrame(rows)

    @cached_property
    def pageviews(self) -> dict[str, pd.Series]:
        g = self.guests.groupby(["region_code", "month"])["guests"].sum()
        out = {}
        for r in self.cfg.regions:
            gm = g.loc[r.code]
            lead = (gm.shift(-1).bfill() / gm.mean())            # интерес опережает поездки на месяц
            daily = lead.reindex(self.days, method="ffill").fillna(1.0)
            for j, a in enumerate(r.articles):
                base = 3000 / (j + 1)
                weekly = np.where(self.days.weekday >= 5, 1.15, 1.0)
                v = base * daily.values ** 0.9 * weekly * np.exp(self._rng("pv", a).normal(0, 0.12, len(self.days)))
                out[a] = pd.Series(np.round(v).astype(int), index=self.days)
        return out

    # ---------- ответы «как у настоящих API» ----------
    def respond(self, url: str, params: dict) -> tuple[bytes, str]:
        host = urlparse(url).hostname
        if host == "archive-api.open-meteo.com":
            return self._weather(params)
        if host == "wikimedia.org":
            return self._pageviews(url)
        if host == "www.cbr.ru":
            return self._cbr(params)
        if host == "isdayoff.ru":
            return self._calendar(params)
        if host == "rosstat.gov.ru":
            return self._rosstat()
        raise NotFound(url)

    def _visible(self, s: pd.DataFrame | pd.Series, start: date, end: date, lag_days: int = 1):
        end = min(end, self.as_of - timedelta(days=lag_days))
        return s.loc[pd.Timestamp(start):pd.Timestamp(end)]

    def _weather(self, p: dict) -> tuple[bytes, str]:
        pid = next(pt.id for r in self.cfg.regions for pt in r.points
                   if abs(pt.lat - float(p["latitude"])) < 1e-6 and abs(pt.lon - float(p["longitude"])) < 1e-6)
        df = self._visible(self.weather[pid], date.fromisoformat(p["start_date"]),
                           date.fromisoformat(p["end_date"]), lag_days=5)
        body = {"daily": {"time": [d.strftime("%Y-%m-%d") for d in df.index],
                          "temperature_2m_mean": df["t_mean"].round(1).tolist(),
                          "temperature_2m_max": df["t_max"].round(1).tolist(),
                          "precipitation_sum": df["precip"].round(1).tolist(),
                          "sunshine_duration": (df["sun_h"] * 3600).round(0).tolist()}}
        return json.dumps(body).encode(), "application/json"

    def _pageviews(self, url: str) -> tuple[bytes, str]:
        parts = urlparse(url).path.split("/")
        article, start, end = unquote(parts[-4]), parts[-2], parts[-1]
        s = self._visible(self.pageviews[article], pd.to_datetime(start[:8]).date(),
                          pd.to_datetime(end[:8]).date())
        if s.empty:
            raise NotFound(url)
        items = [{"article": article, "timestamp": d.strftime("%Y%m%d00"), "views": int(v)} for d, v in s.items()]
        return json.dumps({"items": items}).encode(), "application/json"

    def _cbr(self, p: dict) -> tuple[bytes, str]:
        to_d = lambda s: pd.to_datetime(s, format="%d/%m/%Y").date()  # noqa: E731
        s = self._visible(self.fx, to_d(p["date_req1"]), to_d(p["date_req2"]), lag_days=0)
        mult = 1.0 if p["VAL_NM_RQ"] == "R01235" else 1.08
        recs = "".join(
            f'<Record Date="{d:%d.%m.%Y}" Id="{p["VAL_NM_RQ"]}"><Nominal>1</Nominal>'
            f'<Value>{f"{v * mult:.4f}".replace(".", ",")}</Value></Record>'
            for d, v in s.items() if d.weekday() < 5)
        xml = f'<?xml version="1.0" encoding="windows-1251"?><ValCurs ID="{p["VAL_NM_RQ"]}">{recs}</ValCurs>'
        return xml.encode("windows-1251"), "application/xml"

    def _calendar(self, p: dict) -> tuple[bytes, str]:
        year = int(p["year"])
        if year > self.as_of.year + (1 if self.as_of.month >= 10 else 0):
            return b"101", "text/plain"  # как настоящий сервис: календарь ещё не опубликован
        s = self.calendar.loc[str(year)]
        return "".join(str(v) for v in s.values).encode(), "text/plain"

    def _rosstat(self) -> tuple[bytes, str]:
        """Широкая таблица: лист на вид КСР, регионы по строкам, месяцы по столбцам."""
        lag = self.cfg.forecast.publication_lag_months
        last = (pd.Timestamp(self.as_of).to_period("M") - lag).to_timestamp()
        g = self.guests[self.guests["month"] <= last]
        wb = Workbook()
        wb.remove(wb.active)
        for acc, title in self.cfg.accommodation_types.items():
            ws = wb.create_sheet(title=acc)
            ws.append([f"Численность лиц, размещённых в КСР: {title}, человек (синтетические данные)"])
            ws.append([])
            sub = g[g["acc_type"] == acc].pivot(index="region_code", columns="month", values="guests")
            ws.append(["Субъект РФ"] + [f"{MONTH_RU[m.month - 1]} {m.year}" for m in sub.columns])
            for code, row in sub.iterrows():
                ws.append([self.cfg.region_by_code(code).name] + [int(v) for v in row.values])
        buf = io.BytesIO()
        wb.save(buf)
        return buf.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
