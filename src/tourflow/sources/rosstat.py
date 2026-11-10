"""Росстат / ЕМИСС — численность лиц, размещённых в КСР, по региону, виду КСР и месяцу (целевая).

Файлы приходят двумя путями: по прямым ссылкам из config.yaml или из папки data/inbox/rosstat
(резервный ручной режим, если сайт недоступен или меняет ссылки).

Парсер терпим к раскладке, потому что выгрузки Росстата и ЕМИСС отличаются:
* «широкая» таблица: строки — регионы, столбцы — месяцы; вид КСР — в названии листа
  или в заголовке над таблицей;
* «длинная» таблица (выгрузка ЕМИСС): столбцы регион / вид КСР / период / значение.
Значения «…», «-», «х» считаются пропуском. Опция cumulative=true переводит
нарастающий итог с начала года в помесячные значения.
"""
from __future__ import annotations

import io
import logging
import re
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from sqlalchemy import Engine

from tourflow.config import PROJECT_ROOT
from tourflow.sources.base import Request, Source

log = logging.getLogger(__name__)

MONTHS = {
    "январ": 1, "янв": 1, "феврал": 2, "фев": 2, "март": 3, "мар": 3, "апрел": 4, "апр": 4,
    "май": 5, "мая": 5, "июн": 6, "июл": 7, "август": 8, "авг": 8, "сентябр": 9, "сен": 9,
    "октябр": 10, "окт": 10, "ноябр": 11, "ноя": 11, "декабр": 12, "дек": 12,
}
ACC_PATTERNS = {
    "hotels": ("гостиниц",),
    "sanatoriums": ("санатор",),
    "recreation": ("организации отдыха", "организациях отдыха", "базы отдыха", "отдыха"),
}
SYNTHETIC_URL = "https://rosstat.gov.ru/synthetic/ksr_monthly.xlsx"


def norm_region(name: str) -> str:
    """«Город Москва столица Российской Федерации город федерального значения» → «москва»."""
    s = str(name).lower().replace("ё", "е")
    s = re.sub(r"столица российской федерации|федерального значения|\(.*?\)", " ", s)
    s = re.sub(r"\b(г|город|гор)\b\.?", " ", s)
    return re.sub(r"[^a-zа-я]", "", s)


def parse_period_label(v, year: int | None = None) -> tuple[date | None, bool]:
    """Период из подписи столбца. Возвращает (месяц, нарастающий_итог).

    Понимает «январь 2024», «2024-01», а также «январь-август» с годом из строки выше —
    так ЕМИСС подписывает данные нарастающим итогом с начала года.
    """
    d = parse_period(v)
    if d is not None:
        return d, False
    if year is None or v is None:
        return None, False
    words = re.findall(r"[а-я]+", str(v).lower())
    months = [m for m in (next((num for stem, num in MONTHS.items() if w.startswith(stem)), None)
                          for w in words) if m is not None][:2]
    if not months or len(words) > 3:
        return None, False
    cumulative = len(months) == 2 and months[0] == 1
    return date(year, months[-1], 1), cumulative


def read_rows(content: bytes) -> list[tuple[str, list[list]]]:
    """Листы книги как списки строк. XLSX — openpyxl (read-only), старый XLS из ЕМИСС — xlrd."""
    if content[:4] == b"\xd0\xcf\x11\xe0":                      # OLE2 = .xls
        import xlrd

        wb = xlrd.open_workbook(file_contents=content, on_demand=True)
        out = [(sh.name, [sh.row_values(r) for r in range(sh.nrows)]) for sh in wb.sheets()]
        wb.release_resources()
        return out
    wb = load_workbook(io.BytesIO(content), read_only=True, data_only=True)
    out = [(ws.title, [list(r) for r in ws.iter_rows(values_only=True)]) for ws in wb.worksheets]
    wb.close()
    return out


def parse_period(v) -> date | None:
    if isinstance(v, (datetime, date)):
        return date(v.year, v.month, 1)
    if v is None:
        return None
    s = str(v).strip().lower()
    m = re.fullmatch(r"(\d{4})[-./](\d{1,2})(?:[-./]\d{1,2})?", s)
    if m:
        return date(int(m.group(1)), int(m.group(2)), 1)
    m = re.fullmatch(r"(\d{1,2})[./](\d{4})", s)
    if m:
        return date(int(m.group(2)), int(m.group(1)), 1)
    m = re.search(r"([а-я]+)\.?\s*(\d{4}|\d{2})\b", s)
    if m:
        word, yr = m.group(1), int(m.group(2))
        yr = yr + 2000 if yr < 100 else yr
        for stem, num in MONTHS.items():
            if word.startswith(stem):
                return date(yr, num, 1)
    return None


def to_number(v) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace("\xa0", "").replace(" ", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None  # «…», «-», «х» и прочие отметки Росстата


def detect_acc_type(text: str) -> str | None:
    t = str(text).lower()
    for code, pats in ACC_PATTERNS.items():
        if any(p in t for p in pats):
            return code
    return None


class RosstatSource(Source):
    name = "rosstat"
    stg_table = "stg.accommodation"
    stg_key = ("region_code", "acc_type", "month")
    date_col = "month"

    def __init__(self, *a, synthetic: bool = False, **kw):
        super().__init__(*a, **kw)
        self.synthetic = synthetic
        self.region_map = {norm_region(r.name): r.code for r in self.cfg.regions}
        self.cumulative = bool(self.scfg.get("cumulative", False))
        # если в файле нет разреза по видам КСР (как в оперативных данных ЕМИСС) — этот вид
        types = list(self.cfg.accommodation_types)
        self.default_acc = self.scfg.get("default_acc_type") or (types[0] if len(types) == 1 else None)

    def plan(self, engine: Engine) -> list[Request]:
        if self.synthetic:
            return [Request("ksr_monthly.xlsx", SYNTHETIC_URL, None, None, None)]
        # Файлы перекачиваются каждый раз: Росстат пересматривает прошлые месяцы.
        # Идемпотентность обеспечивает sha256 в raw.payload — тот же файл не сохранится повторно.
        return [Request(Path(u).name or f"file{i}", u, None, None, None)
                for i, u in enumerate(self.scfg.get("urls", []))]

    def run(self, engine, run_id):
        if not self.synthetic:          # в демо-режиме реальные файлы не смешиваем с синтетикой
            self._ingest_inbox(engine, run_id)
        return super().run(engine, run_id)

    def _ingest_inbox(self, engine: Engine, run_id) -> None:
        inbox = PROJECT_ROOT / self.scfg.get("inbox_dir", "data/inbox/rosstat")
        if not inbox.exists():
            return
        for f in sorted(inbox.glob("*.xls*")):
            if f.stat().st_size > self.cfg.sources.http.max_response_mb * 1024 * 1024:
                log.warning("файл %s слишком большой, пропущен", f.name)
                continue
            self._store_raw(engine, Request(f.name, f"file://{f.name}", None, None, None),
                            f.read_bytes(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            run_id)

    # ---------- разбор ----------
    def parse(self, req_key: str, content: bytes, content_type: str) -> pd.DataFrame:
        frames = []
        self._cum_detected = False
        for title, rows in read_rows(content):
            rows = [r for r in rows if any(c is not None and str(c).strip() for c in r)]
            if not rows or title.lower().startswith("паспорт"):
                continue
            df = self._parse_long(rows)
            if df is None:
                df = self._parse_wide(rows, title)
            if df is not None:
                frames.append(df)
        if not frames:
            raise ValueError("не найдено ни одной таблицы с регионами и месяцами")
        out = pd.concat(frames, ignore_index=True)
        if self.default_acc:
            out["acc_type"] = out["acc_type"].fillna(self.default_acc)
        out = out[out["region_code"].notna() & out["acc_type"].notna() & out["guests"].notna()]
        if self.cumulative or self._cum_detected:
            out = out.sort_values("month")
            out["year"] = [m.year for m in out["month"]]
            prev = out.groupby(["region_code", "acc_type", "year"])["guests"].shift()
            first = [m.month == 1 for m in out["month"]]
            out["guests"] = out["guests"] - prev.where(~pd.Series(first, index=out.index), 0).fillna(0)
            out = out.drop(columns="year")
        out["guests"] = out["guests"].round().astype("int64")
        return out.drop_duplicates(["region_code", "acc_type", "month"], keep="last")[
            ["region_code", "acc_type", "month", "guests"]]

    def _region(self, v) -> str | None:
        return self.region_map.get(norm_region(v)) if v is not None else None

    def _parse_long(self, rows: list[list]) -> pd.DataFrame | None:
        head = [str(c or "").lower() for c in rows[0]]

        def col(*keys):
            return next((i for i, h in enumerate(head) if any(k in h for k in keys)), None)

        ri, ti, pi, vi = col("регион", "субъект", "территор"), col("вид", "тип"), col("период", "месяц", "дата"), \
            col("значение", "численность", "value")
        if None in (ri, ti, pi, vi):
            return None
        recs = [{"region_code": self._region(r[ri]), "acc_type": detect_acc_type(r[ti]),
                 "month": parse_period(r[pi]), "guests": to_number(r[vi])} for r in rows[1:]]
        df = pd.DataFrame(recs).dropna(subset=["month"])
        return df if not df.empty else None

    def _parse_wide(self, rows: list[list], sheet_title: str) -> pd.DataFrame | None:
        # строка-шапка: первая, где не меньше трёх ячеек — периоды; год может стоять строкой выше
        # (объединённые ячейки ЕМИСС: «2022» над «январь», «январь-февраль», …)
        hdr_idx, periods = None, {}
        for i, r in enumerate(rows):
            years = self._years_row(rows[i - 1]) if i > 0 else {}
            cand = {}
            for j, c in enumerate(r):
                y = max(((yj, yv) for yj, yv in years.items() if yj <= j), default=(None, None))[1]
                d, cum = parse_period_label(c, y)
                if d is not None:
                    cand[j] = d
                    self._cum_detected |= cum
            if len(cand) >= 3:
                hdr_idx, periods = i, cand
                break
        if hdr_idx is None:
            return None
        acc = detect_acc_type(sheet_title) or next(
            (detect_acc_type(c) for r in rows[:hdr_idx] for c in r if c and detect_acc_type(c)), None)
        recs = []
        for r in rows[hdr_idx + 1:]:
            label = r[0]
            block = detect_acc_type(label) if label and not self._region(label) else None
            if block:
                acc = block
                continue
            code = self._region(label)
            if not code:
                continue
            for j, p in periods.items():
                if j < len(r):
                    recs.append({"region_code": code, "acc_type": acc, "month": p, "guests": to_number(r[j])})
        return pd.DataFrame(recs) if recs else None

    @staticmethod
    def _years_row(r: list) -> dict[int, int]:
        """{номер столбца: год} для строки с годами (объединённые ячейки → год действует до следующего)."""
        out = {}
        for j, c in enumerate(r):
            s = str(c).strip().split(".")[0] if c not in (None, "") else ""
            if re.fullmatch(r"(19|20)\d\d", s):
                out[j] = int(s)
        return out
