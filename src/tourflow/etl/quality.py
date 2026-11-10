"""Контроль качества слоя stg: схемы pandera + свежесть и полнота. Итоги — в ops.dq_checks.

Проверки делятся на блокирующие (данные нельзя использовать: отрицательные значения,
невозможная температура) и предупреждения (скачок ряда, отставание источника).
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import date

import pandas as pd
import pandera.pandas as pa
from sqlalchemy import Engine, text

from tourflow import db
from tourflow.config import AppConfig

log = logging.getLogger(__name__)


@dataclass
class CheckResult:
    table: str
    name: str
    passed: bool
    details: str = ""
    blocking: bool = True


SCHEMAS = {
    "stg.accommodation": pa.DataFrameSchema({
        "region_code": pa.Column(str, pa.Check.str_matches(r"^\d{8}$")),
        "acc_type": pa.Column(str, pa.Check.isin(["total", "hotels", "sanatoriums", "recreation"])),
        "guests": pa.Column(int, pa.Check.ge(0)),
    }, unique=["region_code", "acc_type", "month"]),
    "stg.weather_daily": pa.DataFrameSchema({
        "t_mean": pa.Column(float, pa.Check.in_range(-60, 50), nullable=True),
        "precip_mm": pa.Column(float, pa.Check.ge(0), nullable=True),
        "sunshine_h": pa.Column(float, pa.Check.in_range(0, 24), nullable=True),
    }),
    "stg.pageviews_daily": pa.DataFrameSchema({"views": pa.Column(int, pa.Check.ge(0))}),
    "stg.fx_daily": pa.DataFrameSchema({"rate": pa.Column(float, pa.Check.in_range(1, 1000))}),
    "stg.calendar_daily": pa.DataFrameSchema({"day_type": pa.Column(int, pa.Check.isin([0, 1, 2]))}),
}
FRESHNESS_DAYS = {"stg.weather_daily": 14, "stg.pageviews_daily": 7, "stg.fx_daily": 10}
DATE_COL = {"stg.accommodation": "month"}


def _load(engine: Engine, table: str) -> pd.DataFrame:
    df = db.read_df(engine, f"SELECT * FROM {db.safe_ident(table)}")  # noqa: S608  # nosec B608
    if "rate" in df:
        df["rate"] = df["rate"].astype(float)
    return df


def run_checks(engine: Engine, cfg: AppConfig, today: date, run_id: uuid.UUID | None = None) -> list[CheckResult]:
    results: list[CheckResult] = []
    for table, schema in SCHEMAS.items():
        df = _load(engine, table)
        if df.empty:
            results.append(CheckResult(table, "not_empty", False, "таблица пуста"))
            continue
        try:
            schema.validate(df, lazy=True)
            results.append(CheckResult(table, "schema", True, f"{len(df)} строк"))
        except pa.errors.SchemaErrors as e:
            fc = e.failure_cases
            results.append(CheckResult(table, "schema", False,
                                       f"{len(fc)} нарушений: " + fc.head(5).to_string(index=False)))
        dcol = DATE_COL.get(table, "day")
        last = pd.Timestamp(df[dcol].max()).date()
        if table in FRESHNESS_DAYS:
            lag = (today - last).days
            results.append(CheckResult(table, "freshness", lag <= FRESHNESS_DAYS[table],
                                       f"последняя дата {last}, отставание {lag} дн.", blocking=False))
    results += _accommodation_checks(engine, cfg)
    _save(engine, results, run_id)
    return results


def _accommodation_checks(engine: Engine, cfg: AppConfig) -> list[CheckResult]:
    df = _load(engine, "stg.accommodation")
    out = []
    if df.empty:
        return out
    df["month"] = pd.to_datetime(df["month"])
    # полнота: у всех регионов и видов КСР есть последний месяц
    last = df["month"].max()
    expected = {(r, a) for r in cfg.region_codes for a in cfg.accommodation_types}
    have = set(map(tuple, df.loc[df["month"] == last, ["region_code", "acc_type"]].values))
    missing = sorted(expected - have)
    out.append(CheckResult("stg.accommodation", "completeness_last_month", not missing,
                           f"{last:%Y-%m}: нет {missing[:5]}" if missing else f"{last:%Y-%m}: все ряды есть",
                           blocking=False))
    # пропуски внутри рядов
    gaps = []
    for (r, a), g in df.groupby(["region_code", "acc_type"]):
        full = pd.date_range(g["month"].min(), g["month"].max(), freq="MS")
        if len(full) != len(g):
            gaps.append(f"{r}/{a}: {len(full) - len(g)}")
    out.append(CheckResult("stg.accommodation", "no_gaps", not gaps, "; ".join(gaps[:10]) or "пропусков нет",
                           blocking=False))
    # резкие скачки год к году (вне исключённых периодов) — сигнал проверить источник
    df = df.sort_values("month")
    df["yoy"] = df.groupby(["region_code", "acc_type"])["guests"].pct_change(12, fill_method=None)
    excl = pd.Series(False, index=df.index)
    for a, b in cfg.forecast.exclude_periods:
        excl |= df["month"].between(pd.Timestamp(a), pd.Timestamp(b) + pd.DateOffset(months=12))
    jumps = df[(~excl) & (df["yoy"].abs() > 1.5)]
    out.append(CheckResult("stg.accommodation", "yoy_jumps", jumps.empty,
                           f"{len(jumps)} скачков >150% г/г" if len(jumps) else "резких скачков нет",
                           blocking=False))
    return out


def _save(engine: Engine, results: list[CheckResult], run_id) -> None:
    with engine.begin() as c:
        for r in results:
            c.execute(text("""INSERT INTO ops.dq_checks (run_id, table_name, check_name, passed, details)
                              VALUES (:r, :t, :n, :p, :d)"""),
                      {"r": run_id, "t": r.table, "n": r.name, "p": r.passed, "d": r.details[:2000]})
    for r in results:
        (log.info if r.passed else log.warning)("DQ %s.%s: %s %s", r.table, r.name,
                                                "OK" if r.passed else "FAIL", r.details[:200])


def blocking_failures(results: list[CheckResult]) -> list[CheckResult]:
    return [r for r in results if r.blocking and not r.passed]


