"""Обучение, сравнение, выбор и прогноз.

Порядок: бэктест всех моделей → выбор по WAPE бэктеста → проверка на отложенной выборке →
обучение выбранной модели на всех данных → прогноз будущих месяцев с интервалом →
объяснения → запись в mart и реестр моделей (+ MLflow, если настроен).
"""
from __future__ import annotations

import hashlib
import json
import logging
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

from tourflow import db, explain, runlog
from tourflow.config import AppConfig, Settings
from tourflow.features import ALL_FEATURES, FEATURE_GROUPS
from tourflow.models.backtest import (
    make_split,
    metrics,
    residual_quantiles,
    rolling_forecast,
    to_guests,
    train_rows,
)
from tourflow.models.estimators import make_models

log = logging.getLogger(__name__)


def model_version(name: str, params: dict, seed: int, data_hash: str) -> str:
    """Версия детерминирована: те же данные + параметры + seed = та же версия (идемпотентность)."""
    blob = json.dumps({"name": name, "params": params, "seed": seed, "data": data_hash,
                       "features": ALL_FEATURES}, sort_keys=True, default=str)
    return f"{name}-{hashlib.sha256(blob.encode()).hexdigest()[:10]}"


def _forecast_rows(pred: pd.DataFrame, version: str, name: str) -> pd.DataFrame:
    out = pred[["region_code", "acc_type", "origin_month", "horizon", "target_month", "kind"]].copy()
    out["y_true"] = pred["target"]
    out["y_pred"] = pred["y_pred"]
    out["y_lo"] = pred.get("y_lo")
    out["y_hi"] = pred.get("y_hi")
    out["model_version"], out["model_name"] = version, name
    for c in ("origin_month", "target_month"):
        out[c] = pd.to_datetime(out[c]).dt.date
    return out


def train_and_forecast(engine: Engine, df: pd.DataFrame, cfg: AppConfig, settings: Settings,
                       data_hash: str, run_id: uuid.UUID | None = None) -> dict:
    fc = cfg.forecast
    seed = cfg.project.seed
    np.random.seed(seed)
    factories = make_models(cfg.models, seed)
    split = make_split(df, fc.holdout_months, fc.backtest_origins)
    log.info("бэктест: %d точек (%s…%s), отложенная выборка с %s",
             len(split.backtest_origins), split.backtest_origins[0].date(), split.backtest_origins[-1].date(),
             split.holdout_start.date())

    results: dict[str, dict] = {}
    with runlog.step(engine, run_id, "models", "backtest", reraise=True) as st:
        for name, factory in factories.items():
            bt = rolling_forecast(df, factory, split.backtest_origins, split.holdout_start, "backtest")
            ho = rolling_forecast(df, factory, split.holdout_origins, None, "holdout")
            results[name] = {"backtest": bt, "holdout": ho}
            st.rows_fetched += len(bt) + len(ho)
        naive_bt, naive_ho = results["naive"]["backtest"], results["naive"]["holdout"]
        for name, r in results.items():
            r["m_bt"] = metrics(r["backtest"], naive_bt)
            r["m_ho"] = metrics(r["holdout"], naive_ho)
            st.note(f"{name}: WAPE бэктест {r['m_bt'].iloc[0]['wape']:.3f}, отложенная {r['m_ho'].iloc[0]['wape']:.3f}")

    # выбор только по бэктесту; отложенная выборка — независимая проверка выбранной модели
    metric = cfg.models.get("selection_metric", "wape")
    best = min((n for n in results if n != "naive"), key=lambda n: results[n]["m_bt"].iloc[0][metric])
    log.info("выбрана модель %s", best)

    # финальное обучение на всех известных данных и прогноз будущего
    last = split.last_month
    full_train = train_rows(df, last)
    final = factories[best]().fit(full_train, full_train["y_rel"])
    fut = df[df["origin_month"] == last].copy()
    fut["y_rel_pred"] = final.predict(fut)
    fut["y_pred"] = to_guests(fut, fut["y_rel_pred"])
    q = residual_quantiles(pd.concat([results[best]["backtest"], results[best]["holdout"]]), fc.interval_level)
    fut = fut.join(q, on="horizon")
    fut["y_lo"] = fut["level"] * np.exp(fut["y_rel_pred"] + fut["q_lo"])
    fut["y_hi"] = fut["level"] * np.exp(fut["y_rel_pred"] + fut["q_hi"])
    fut["kind"] = "future"

    versions = {n: model_version(n, getattr(factories[n](), "params", {}), seed, data_hash) for n in results}
    _save_results(engine, results, versions, fut, best)
    art_path, art_sha = _save_artifact(settings, versions[best], final)
    _register(engine, versions, results, best, data_hash, seed, cfg, art_path, art_sha)

    expl_info = _explain(engine, final, df, results, best, versions[best], split, factories[best], seed, run_id)
    _mlflow(settings, versions, results, best, data_hash, seed, cfg, art_path)
    m = results[best]
    return {"best": best, "version": versions[best],
            "wape_backtest": float(m["m_bt"].iloc[0]["wape"]), "wape_holdout": float(m["m_ho"].iloc[0]["wape"]),
            "wape_naive_holdout": float(results["naive"]["m_ho"].iloc[0]["wape"]), **expl_info}


def _save_results(engine, results, versions, fut, best) -> None:
    for name, r in results.items():
        v = versions[name]
        rows = pd.concat([_forecast_rows(r["backtest"], v, name), _forecast_rows(r["holdout"], v, name)])
        if name == best:
            rows = pd.concat([rows, _forecast_rows(fut, v, name)])
        db.replace_where(engine, "mart.forecasts", "model_version = :v", {"v": v}, rows)
        mets = pd.concat([r["m_bt"].assign(split="backtest"), r["m_ho"].assign(split="holdout")])
        mets = mets.assign(model_version=v, model_name=name)
        mets = mets[["model_version", "model_name", "split", "horizon", "mae", "rmse", "wape", "mae_naive", "n"]]
        db.replace_where(engine, "mart.metrics", "model_version = :v", {"v": v}, mets)


def _save_artifact(settings: Settings, version: str, model) -> tuple[str, str]:
    d = Path(settings.artifacts_dir) / "models"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{version}.joblib"
    joblib.dump(model, path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return str(path), sha


def load_artifact(path: str, expected_sha: str):
    """Загрузка pickle только после сверки sha256 с реестром — чужой/подменённый файл не исполнится."""
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError(f"контрольная сумма артефакта {path} не совпадает с реестром")
    return joblib.load(path)


def _register(engine, versions, results, best, data_hash, seed, cfg, art_path, art_sha) -> None:
    with engine.begin() as c:
        c.execute(text("UPDATE mart.model_registry SET is_active = false WHERE is_active"))
        for name, v in versions.items():
            m = results[name]
            mets = {"backtest": m["m_bt"].iloc[0][["mae", "rmse", "wape"]].to_dict(),
                    "holdout": m["m_ho"].iloc[0][["mae", "rmse", "wape"]].to_dict()}
            c.execute(text("""
                INSERT INTO mart.model_registry (model_version, model_name, data_hash, seed, features, params,
                                                 metrics, artifact_path, artifact_sha256, is_active)
                VALUES (:v, :n, :h, :s, CAST(:f AS jsonb), CAST(:p AS jsonb), CAST(:m AS jsonb), :ap, :as_, :act)
                ON CONFLICT (model_version) DO UPDATE SET metrics = EXCLUDED.metrics, is_active = EXCLUDED.is_active,
                    artifact_path = COALESCE(EXCLUDED.artifact_path, mart.model_registry.artifact_path),
                    artifact_sha256 = COALESCE(EXCLUDED.artifact_sha256, mart.model_registry.artifact_sha256)"""),
                {"v": v, "n": name, "h": data_hash, "s": seed, "f": json.dumps(ALL_FEATURES),
                 "p": json.dumps(cfg.models.get(name, {})), "m": json.dumps(mets, default=float),
                 "ap": art_path if name == best else None, "as_": art_sha if name == best else None,
                 "act": name == best})


def _explain(engine, final, df, results, best, version, split, factory, seed, run_id) -> dict:
    abl: dict = {}
    with runlog.step(engine, run_id, "models", "explain") as st:  # сбой объяснений не отменяет прогноз
        rows = df[df["level"].notna()].copy()
        if best == "random_forest":   # TreeSHAP по лесу дорогой — будущее, отложенная и случайная выборка
            keep = rows["target"].isna() | (rows["target_month"] >= split.holdout_start)
            rows = pd.concat([rows[keep], rows[~keep].sample(min(1500, (~keep).sum()), random_state=seed)])
        sv, base = explain.shap_values(final, rows)
        expl = pd.DataFrame({
            "model_version": version, "region_code": rows["region_code"].values, "acc_type": rows["acc_type"].values,
            "origin_month": rows["origin_month"].dt.date.values, "horizon": rows["horizon"].values,
            "base_value": base,
            "contributions": explain.contributions_json(sv, rows[ALL_FEATURES], final.base_of(rows),
                                                       rows["level"].to_numpy())})
        db.replace_where(engine, "mart.explanations", "model_version = :v", {"v": version}, expl)

        trained = rows["target"].notna().to_numpy()
        imp_shap = explain.importance_shap(sv[trained], rows[trained])
        imp = [imp_shap]
        ho = results[best]["holdout"]
        imp.append(explain.importance_permutation(final, ho, seed))
        base_wape = float(results[best]["m_bt"].iloc[0]["wape"])
        imp.append(explain.ablation(factory, df, split.backtest_origins, split.holdout_start, base_wape))
        coef = _coefficients(final)
        if coef is not None:
            imp.append(coef)
        imp = pd.concat(imp, ignore_index=True).assign(model_version=version)
        db.replace_where(engine, "mart.importance", "model_version = :v", {"v": version},
                         imp[["model_version", "method", "period", "feature", "importance"]])
        # PDP — для самых важных числовых признаков по SHAP (кроме самой сезонной базы)
        top = (imp_shap.query("period == 'all'").set_index("feature")["importance"]
               .drop(["region_code", "acc_type", "f_base"], errors="ignore").nlargest(8).index.tolist())
        pdp = explain.partial_dependence(final, rows[rows["target"].notna()], top).assign(model_version=version)
        db.replace_where(engine, "mart.pdp", "model_version = :v", {"v": version}, pdp)
        st.rows_inserted = len(expl)
        abl = imp[imp["method"] == "ablation"].set_index("feature")["importance"].round(2).to_dict()
        st.note(f"ablation ΔWAPE п.п.: {abl}")
    return {"ablation": abl}


def _coefficients(model) -> pd.DataFrame | None:
    est = getattr(getattr(model, "pipeline", None), "named_steps", {}).get("model")
    if est is None or not hasattr(est, "coef_"):
        return None
    names = model.pipeline.named_steps["pre"].get_feature_names_out()
    return pd.DataFrame({"feature": names, "importance": np.abs(est.coef_), "method": "coef", "period": "all"})


def _mlflow(settings, versions, results, best, data_hash, seed, cfg, art_path) -> None:
    """Журнал экспериментов в MLflow (если задан TOURFLOW_MLFLOW_TRACKING_URI)."""
    if not settings.mlflow_tracking_uri:
        return
    try:
        import mlflow

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment("tourflow")
        for name, v in versions.items():
            with mlflow.start_run(run_name=v):
                mlflow.log_params({"model": name, "seed": seed, "data_hash": data_hash,
                                   "n_features": len(ALL_FEATURES), **{f"p_{k}": val for k, val in
                                                                       cfg.models.get(name, {}).items()}})
                for split in ("bt", "ho"):
                    for _, r in results[name][f"m_{split}"].iterrows():
                        for k in ("mae", "rmse", "wape"):
                            mlflow.log_metric(f"{split}_{k}_h{int(r['horizon'])}", float(r[k]))
                mlflow.set_tag("selected", str(name == best))
                if name == best:
                    mlflow.log_artifact(art_path)
                    mlflow.log_dict({g: f for g, f in FEATURE_GROUPS.items()}, "features.json")
    except Exception as exc:  # noqa: BLE001 — недоступный MLflow не должен ломать прогноз
        log.warning("MLflow недоступен: %s", exc)
