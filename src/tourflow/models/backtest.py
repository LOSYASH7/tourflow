"""Схема валидации с учётом времени: rolling origin + отложенная выборка.

Для каждой точки прогноза o модель обучается только на строках, чья целевая известна к o
(target_month <= o), и прогнозирует все горизонты от o. Последние holdout_months месяцев
целевой — отложенная выборка: она не участвует ни в обучении при выборе, ни в выборе модели.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class Split:
    backtest_origins: list[pd.Timestamp]
    holdout_origins: list[pd.Timestamp]
    holdout_start: pd.Timestamp       # первый целевой месяц отложенной выборки
    last_month: pd.Timestamp


def make_split(df: pd.DataFrame, holdout_months: int, n_origins: int) -> Split:
    last = df.loc[df["target"].notna(), "target_month"].max()
    holdout_start = last - pd.DateOffset(months=holdout_months - 1)
    origins = sorted(df["origin_month"].unique())
    hold = [o for o in origins if holdout_start - pd.DateOffset(months=1) <= o < last]
    back = [o for o in origins if o < holdout_start - pd.DateOffset(months=1)][-n_origins:]
    return Split(back, hold, holdout_start, last)


def train_rows(df: pd.DataFrame, origin: pd.Timestamp, before: pd.Timestamp | None = None) -> pd.DataFrame:
    m = (df["target"].notna() & (df["target_month"] <= origin) & ~df["in_excluded_period"]
         & ~df["anomaly_window"])
    if before is not None:
        m &= df["target_month"] < before
    return df[m]


def to_guests(df: pd.DataFrame, y_rel_pred: np.ndarray) -> np.ndarray:
    return df["level"].to_numpy() * np.exp(y_rel_pred)


def rolling_forecast(df: pd.DataFrame, factory, origins: list[pd.Timestamp],
                     limit_target_before: pd.Timestamp | None, kind: str) -> pd.DataFrame:
    out = []
    for o in origins:
        tr = train_rows(df, o)
        te = df[(df["origin_month"] == o) & df["target"].notna()]
        if limit_target_before is not None:
            te = te[te["target_month"] < limit_target_before]
        if te.empty or len(tr) < 50:
            continue
        model = factory().fit(tr, tr["y_rel"])
        p = model.predict(te)
        out.append(te.assign(y_rel_pred=p, y_pred=to_guests(te, p), kind=kind))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def metrics(pred: pd.DataFrame, naive: pd.DataFrame | None = None) -> pd.DataFrame:
    """MAE, RMSE, WAPE по горизонтам и в целом (horizon = 0). Исключённые периоды не оцениваются."""
    p = pred[~pred["in_excluded_period"]]
    rows = []
    for h, g in [(0, p)] + list(p.groupby("horizon")):
        err = g["y_pred"] - g["target"]
        r = {"horizon": int(h), "mae": err.abs().mean(), "rmse": float(np.sqrt((err ** 2).mean())),
             "wape": err.abs().sum() / g["target"].abs().sum(), "n": len(g)}
        if naive is not None and not naive.empty:
            nk = naive[~naive["in_excluded_period"]]
            nk = nk if h == 0 else nk[nk["horizon"] == h]
            nk = nk.merge(g[["region_code", "acc_type", "origin_month", "horizon"]])
            r["mae_naive"] = (nk["y_pred"] - nk["target"]).abs().mean()
        rows.append(r)
    return pd.DataFrame(rows)


def residual_quantiles(pred: pd.DataFrame, level: float) -> pd.DataFrame:
    """Квантили ошибки в лог-пространстве по горизонту — для интервала прогноза."""
    p = pred[~pred["in_excluded_period"]].assign(res=lambda x: x["y_rel"] - x["y_rel_pred"])
    lo, hi = (1 - level) / 2, (1 + level) / 2
    return p.groupby("horizon")["res"].quantile([lo, hi]).unstack().rename(columns={lo: "q_lo", hi: "q_hi"})
