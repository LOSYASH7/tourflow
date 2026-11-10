"""Модели. Все предсказывают y_rel = log(y_t / level_o); перевод в гостей — в backtest.py.

Классы моделей разные, как требует методичка:
  naive        — сезонный наивный прогноз «как год назад» (базовая модель);
  elasticnet   — линейная регрессия с L1+L2-регуляризацией;
  random_forest— бэггинг деревьев;
  lightgbm     — градиентный бустинг деревьев.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tourflow.features import CATEGORICAL, NUMERIC


def seasonal_base(X: pd.DataFrame) -> np.ndarray:
    """Сезонный наивный прогноз в относительных единицах (см. f_base в features.py)."""
    return X["f_base"].to_numpy(dtype=float)


class NaiveSeasonal:
    """Базовая модель: y_t = y_{t-12} (если тот год аномален — y_{t-24} с поправкой на уровень)."""

    name = "naive"
    params: dict = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> NaiveSeasonal:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return seasonal_base(X)


class _OnTopOfNaive:
    """ML-модели учат ПОПРАВКУ к сезонному наивному прогнозу: y_rel − base.

    Прогноз = «как год назад» × поправки (тренд, календарь, курс, интерес, погода).
    Деревьям не нужно аппроксимировать тождество y ≈ f_ly ступеньками, а объяснение
    прогноза читается естественно: от прошлого года к прогнозу.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self._fit(X, np.asarray(y, dtype=float) - seasonal_base(X))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return seasonal_base(X) + self._predict(X)

    def base_of(self, X: pd.DataFrame) -> np.ndarray:
        return seasonal_base(X)


def _onehot_pipeline(est, scale: bool) -> Pipeline:
    # Категорий мало (8 регионов, 3 вида КСР) — one-hot без потери информации и без риска утечки,
    # который есть у target encoding.
    num = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), *( [StandardScaler()] if scale else []))
    pre = ColumnTransformer([
        ("num", num, NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ], verbose_feature_names_out=False)
    return Pipeline([("pre", pre), ("model", est)])


class SkModel(_OnTopOfNaive):
    def __init__(self, name: str, pipeline: Pipeline, params: dict):
        self.name, self.pipeline, self.params = name, pipeline, params

    def _fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self.pipeline.fit(X[NUMERIC + CATEGORICAL], y)

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X[NUMERIC + CATEGORICAL])


class LgbmModel(_OnTopOfNaive):
    """LightGBM со встроенной обработкой пропусков и категорий (без one-hot)."""

    name = "lightgbm"

    def __init__(self, params: dict, seed: int):
        self.params = params
        self.model = LGBMRegressor(random_state=seed, n_jobs=1, deterministic=True, force_row_wise=True, **params)
        self.categories: dict[str, list] = {}

    def _prep(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[NUMERIC + CATEGORICAL].copy()
        for c in CATEGORICAL:
            X[c] = pd.Categorical(X[c], categories=self.categories[c])
        return X

    def _fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self.categories = {c: sorted(X[c].dropna().unique().tolist()) for c in CATEGORICAL}
        self.model.fit(self._prep(X), y)

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._prep(X))

    def contributions(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Точные SHAP-значения TreeSHAP для поправки к наивному прогнозу. Возвращает (вклады, базу)."""
        c = self.model.predict(self._prep(X), pred_contrib=True)
        return c[:, :-1], c[:, -1]


def make_models(cfg_models: dict, seed: int) -> dict:
    en = cfg_models["elasticnet"]
    rf = cfg_models["random_forest"]
    return {
        "naive": lambda: NaiveSeasonal(),
        "elasticnet": lambda: SkModel("elasticnet", _onehot_pipeline(ElasticNet(random_state=seed, **en), True), en),
        "random_forest": lambda: SkModel("random_forest", _onehot_pipeline(
            RandomForestRegressor(random_state=seed, n_jobs=-1, **rf), False), rf),
        "lightgbm": lambda: LgbmModel(cfg_models["lightgbm"], seed),
    }
