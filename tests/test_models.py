"""Модели и интерпретация."""
import numpy as np
import pytest

from tourflow import explain
from tourflow.models.backtest import make_split, train_rows
from tourflow.models.estimators import make_models
from tourflow.security import hash_password, verify_password


@pytest.mark.parametrize("name", ["elasticnet", "lightgbm"])
def test_shap_contributions_add_up_to_prediction(cfg, features, name):
    split = make_split(features, cfg.forecast.holdout_months, 3)
    tr = train_rows(features, split.last_month)
    model = make_models(cfg.models, cfg.project.seed)[name]().fit(tr, tr["y_rel"])
    rows = features[features.origin_month == split.last_month].head(40)
    sv, base = explain.shap_values(model, rows)
    total = model.base_of(rows) + base + sv.sum(axis=1).to_numpy()
    np.testing.assert_allclose(total, model.predict(rows), atol=1e-6)


def test_split_keeps_holdout_out_of_backtest(cfg, features):
    s = make_split(features, cfg.forecast.holdout_months, cfg.forecast.backtest_origins)
    assert max(s.backtest_origins) < s.holdout_start
    assert all(o >= s.holdout_start - np.timedelta64(31, "D") for o in s.holdout_origins)


def test_training_is_reproducible(cfg, features):
    tr = train_rows(features, features["origin_month"].max())
    f = make_models(cfg.models, cfg.project.seed)["lightgbm"]
    p1 = f().fit(tr, tr["y_rel"]).predict(tr.head(50))
    p2 = f().fit(tr, tr["y_rel"]).predict(tr.head(50))
    np.testing.assert_array_equal(p1, p2)


def test_password_hashing():
    h = hash_password("correct horse battery")
    assert verify_password("correct horse battery", h)
    assert not verify_password("wrong", h)
    assert not verify_password("x", "garbage")
