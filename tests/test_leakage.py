"""Доказательство отсутствия утечки целевой переменной.

1. Обрезание данных: признаки строки с origin o, посчитанные на ВСЕХ данных, совпадают
   с признаками, посчитанными на данных, какими они были на дату прогноза
   (целевая до o, внешние источники до конца месяца o + lag − 1).
2. Перемешанная целевая: модель, обученная на случайной целевой, не лучше наивного прогноза.
"""
import numpy as np
import pandas as pd
import pytest

from tourflow.features import ALL_FEATURES, KEYS, build_features
from tourflow.models.backtest import make_split, metrics, rolling_forecast
from tourflow.models.estimators import make_models

NUM = [c for c in ALL_FEATURES if c not in KEYS and c not in ("region_code", "acc_type")]


@pytest.mark.parametrize("origin", ["2022-11-01", "2024-06-01", "2025-12-01"])
def test_features_do_not_see_future(cfg, inputs, features, origin):
    o = pd.Timestamp(origin)
    cutoff = o + pd.DateOffset(months=cfg.forecast.publication_lag_months - 1) + pd.offsets.MonthEnd(0)
    trunc = build_features(inputs.truncated(o.date(), cutoff.date()), cfg)
    a = features[features.origin_month == o].set_index(KEYS)[NUM + ["level"]].sort_index().astype(float)
    b = trunc[trunc.origin_month == o].set_index(KEYS)[NUM + ["level"]].sort_index().astype(float)
    assert len(a) == len(b) > 0
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9, atol=1e-9)


def test_target_is_never_a_feature(features):
    for f in ALL_FEATURES:
        assert not f.startswith(("target", "y_t", "y_rel")), f


def test_shuffled_target_does_not_beat_naive(cfg, features):
    df = features.copy()
    rng = np.random.default_rng(cfg.project.seed)
    known = df["y_rel"].notna()
    df.loc[known, "y_rel"] = rng.permutation(df.loc[known, "y_rel"].to_numpy())
    split = make_split(features, cfg.forecast.holdout_months, 4)
    models = make_models(cfg.models, cfg.project.seed)
    shuffled = rolling_forecast(df, models["elasticnet"], split.backtest_origins, split.holdout_start, "t")
    naive = rolling_forecast(features, models["naive"], split.backtest_origins, split.holdout_start, "t")
    assert metrics(shuffled).iloc[0]["wape"] > metrics(naive).iloc[0]["wape"]
