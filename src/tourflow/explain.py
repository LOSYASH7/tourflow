"""Интерпретируемость модели (компонент по выбору).

1. Важность признаков двумя независимыми способами: средний |SHAP| и permutation importance
   на отложенной выборке (рост WAPE при перемешивании признака).
2. Объяснение отдельного прогноза: вклад каждого признака (SHAP) для любой строки витрины.
3. Зависимость прогноза от значения признака: partial dependence (PDP).
4. Структура важности по периодам: |SHAP| по годам и по сезонам.
5. Абляция групп признаков: насколько ухудшится модель без погоды, просмотров, курса, календаря.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from tourflow.features import ALL_FEATURES, CATEGORICAL, FEATURE_GROUPS

log = logging.getLogger(__name__)


def shap_values(model, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """SHAP в пространстве исходных признаков (категории не раскладываются на one-hot)."""
    if hasattr(model, "contributions"):  # LightGBM: точный TreeSHAP
        c, base = model.contributions(X)
        return pd.DataFrame(c, columns=ALL_FEATURES, index=X.index), base
    import shap

    pipe = model.pipeline
    Xt = pipe.named_steps["pre"].transform(X[pipe.feature_names_in_])
    names = pipe.named_steps["pre"].get_feature_names_out()
    est = pipe.named_steps["model"]
    if hasattr(est, "coef_"):
        expl = shap.LinearExplainer(est, shap.maskers.Independent(Xt, max_samples=2000))
    else:
        expl = shap.TreeExplainer(est, feature_perturbation="tree_path_dependent")
    sv = expl.shap_values(Xt)
    sv = pd.DataFrame(sv, columns=names, index=X.index)
    # сворачиваем one-hot и индикаторы пропусков обратно в исходные признаки
    out = pd.DataFrame(0.0, index=X.index, columns=[c for c in ALL_FEATURES])
    for col in names:
        src = next((c for c in CATEGORICAL if col.startswith(c + "_")), None) or col.replace("missingindicator_", "")
        if src in out:
            out[src] += sv[col]
    return out, np.full(len(X), float(np.atleast_1d(expl.expected_value)[0]))


def importance_shap(sv: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    out = [pd.DataFrame({"period": "all", "feature": sv.columns, "importance": sv.abs().mean().values})]
    years = rows["target_month"].dt.year
    for y in sorted(years.unique()):
        m = (years == y).to_numpy()
        if m.sum() >= 50:
            out.append(pd.DataFrame({"period": str(y), "feature": sv.columns,
                                     "importance": sv[m].abs().mean().values}))
    summer = rows["target_month"].dt.month.isin([6, 7, 8]).to_numpy()
    for label, m in (("лето", summer), ("не лето", ~summer)):
        out.append(pd.DataFrame({"period": label, "feature": sv.columns, "importance": sv[m].abs().mean().values}))
    return pd.concat(out, ignore_index=True).assign(method="shap")


def wape(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.abs(p - y).sum() / np.abs(y).sum())


def importance_permutation(model, X: pd.DataFrame, seed: int, repeats: int = 5) -> pd.DataFrame:
    """Рост WAPE (п.п.) при случайном перемешивании признака на отложенной выборке."""
    rng = np.random.default_rng(seed)
    y = X["target"].to_numpy()
    lvl = X["level"].to_numpy()
    base = wape(y, lvl * np.exp(model.predict(X)))
    rows = []
    for f in ALL_FEATURES:
        deltas = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[f] = rng.permutation(Xp[f].to_numpy())
            deltas.append(wape(y, lvl * np.exp(model.predict(Xp))) - base)
        rows.append({"feature": f, "importance": float(np.mean(deltas)) * 100})
    return pd.DataFrame(rows).assign(method="permutation", period="all")


def partial_dependence(model, X: pd.DataFrame, features: list[str], grid: int = 20) -> pd.DataFrame:
    """PDP: средний прогноз (в % к уровню ряда) при фиксированном значении признака."""
    rows = []
    sample = X.sample(min(len(X), 1500), random_state=0)
    for f in features:
        vals = sample[f].dropna()
        if vals.nunique() < 2:
            continue
        qs = np.unique(np.quantile(vals, np.linspace(0.02, 0.98, grid)))
        for v in qs:
            Xp = sample.copy()
            Xp[f] = v
            rows.append({"feature": f, "grid_value": float(v),
                         "pd_value": float(np.mean(np.exp(model.predict(Xp))) * 100 - 100)})
    return pd.DataFrame(rows)


def ablation(factory_for, df: pd.DataFrame, origins, holdout_start, base_wape: float) -> pd.DataFrame:
    """Вклад группы признаков: WAPE бэктеста без группы минус WAPE полной модели (п.п.)."""
    from tourflow.models.backtest import metrics, rolling_forecast

    rows = []
    for group in ("weather", "pageviews", "fx", "calendar"):
        cols = [c for c in FEATURE_GROUPS[group] if c != "horizon"]
        d = df.copy()
        d[cols] = np.nan          # признак «неизвестен»: модели умеют работать с пропусками
        pred = rolling_forecast(d, factory_for, origins, holdout_start, "ablation")
        w = metrics(pred).query("horizon == 0")["wape"].iloc[0]
        rows.append({"feature": f"group:{group}", "importance": (w - base_wape) * 100})
        log.info("ablation без %s: WAPE %.4f (полная %.4f)", group, w, base_wape)
    return pd.DataFrame(rows).assign(method="ablation", period="all")


def contributions_json(sv: pd.DataFrame, X: pd.DataFrame, ly_base: np.ndarray) -> list[dict]:
    """{признак: [значение, вклад в log]}; '__base_ly' — сезонная база «как год назад»."""
    out = []
    for i in range(len(sv)):
        rec = {"__base_ly": [None, round(float(ly_base[i]), 6)]}
        for f in sv.columns:
            v = X.iloc[i][f]
            v = None if (isinstance(v, float) and np.isnan(v)) else (v.item() if hasattr(v, "item") else v)
            rec[f] = [v, round(float(sv.iloc[i][f]), 6)]
        out.append(rec)
    return out
