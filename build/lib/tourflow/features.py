"""Единый код признаков — для обучения, бэктеста и прогноза.

Строка витрины = (ряд «регион × вид КСР», origin, горизонт h):
  origin o      — последний опубликованный месяц целевой (известен на дату прогноза);
  target t      — o + h;
  cutoff e      — последний полный месяц внешних данных на дату прогноза: o + lag − 1
                  (Росстат публикует месяц o примерно в o + lag, к этому моменту
                  погода, курсы и просмотры известны до конца месяца o + lag − 1).

Правило против утечки: признак строки зависит только от целевой до o и внешних данных до e.
Исключение — данные, известные заранее по своей природе: производственный календарь,
школьные каникулы и климатическая норма (считается только по прошлым годам).
Тест tests/test_leakage.py проверяет это обрезанием данных: признаки не должны меняться.

Масштабы рядов отличаются в сотни раз, поэтому целевая и лаги — в логарифме
относительно уровня ряда (среднее за 12 месяцев до o). Так одна глобальная модель
учится на всей панели регионов и видов размещения.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import Engine

from tourflow import db
from tourflow.config import AppConfig

KEYS = ["region_code", "acc_type", "origin_month", "horizon"]
CATEGORICAL = ["region_code", "acc_type"]
FEATURE_GROUPS = {
    "target_lags": ["f_base", "f_lag0", "f_lag1", "f_lag2", "f_ly", "f_ly2", "f_ly_excluded", "f_yoy_level",
                    "f_yoy_recent", "f_std12", "f_covid_share"],
    "calendar": ["horizon", "month_sin", "month_cos", "cal_nonworking", "cal_holidays", "cal_long_weekend",
                 "cal_school", "cal_d_nonworking"],
    "weather": ["w_clim_t", "w_clim_beach", "w_anom_e", "w_anom_ly"],
    "pageviews": ["pv_rel", "pv_yoy", "pv_ly_t"],
    "fx": ["fx_usd_log", "fx_usd_chg3", "fx_usd_yoy"],
    "categorical": CATEGORICAL,
}
NUMERIC = [f for g, fs in FEATURE_GROUPS.items() if g != "categorical" for f in fs]
ALL_FEATURES = NUMERIC + CATEGORICAL
FIXED_HOLIDAYS = {(1, d) for d in range(1, 9)} | {(2, 23), (3, 8), (5, 1), (5, 9), (6, 12), (11, 4)}


@dataclass
class Inputs:
    target: pd.DataFrame          # region_code, acc_type, month, guests
    weather: pd.DataFrame         # point_id, region_code, day, t_mean, t_max, precip_mm, sunshine_h
    pageviews: pd.DataFrame       # article, region_code, day, views
    fx: pd.DataFrame              # currency, day, rate
    calendar: pd.DataFrame        # day, day_type
    meta: dict = field(default_factory=dict)

    def truncated(self, last_target_month: date, last_exog_day: date) -> Inputs:
        """Данные такими, какими они были на дату прогноза (для тестов на утечку)."""
        lt, le = pd.Timestamp(last_target_month), pd.Timestamp(last_exog_day)
        return Inputs(
            target=self.target[pd.to_datetime(self.target["month"]) <= lt],
            weather=self.weather[pd.to_datetime(self.weather["day"]) <= le],
            pageviews=self.pageviews[pd.to_datetime(self.pageviews["day"]) <= le],
            fx=self.fx[pd.to_datetime(self.fx["day"]) <= le],
            calendar=self.calendar,  # известен заранее
        )

    def data_hash(self) -> str:
        h = hashlib.sha256()
        for df in (self.target, self.weather, self.pageviews, self.fx, self.calendar):
            h.update(pd.util.hash_pandas_object(df.sort_values(list(df.columns[:2])).reset_index(drop=True),
                                                index=False).values.tobytes())
        return h.hexdigest()[:16]


def load_inputs(engine: Engine) -> Inputs:
    fx = db.read_df(engine, "SELECT currency, day, rate FROM stg.fx_daily")
    fx["rate"] = fx["rate"].astype(float)
    return Inputs(
        target=db.read_df(engine, "SELECT region_code, acc_type, month, guests FROM stg.accommodation"),
        weather=db.read_df(engine, "SELECT point_id, region_code, day, t_mean, t_max, precip_mm, sunshine_h "
                                   "FROM stg.weather_daily"),
        pageviews=db.read_df(engine, "SELECT article, region_code, day, views FROM stg.pageviews_daily"),
        fx=fx,
        calendar=db.read_df(engine, "SELECT day, day_type FROM stg.calendar_daily"),
    )


# ---------- месячные агрегаты внешних данных ----------
def _full_months(daily: pd.DataFrame, value_cols: list[str], group: str | None, how: dict) -> pd.DataFrame:
    """Агрегирует сутки в месяцы; неполные месяцы (меньше 90% дней) отбрасываются."""
    d = daily.copy()
    d["day"] = pd.to_datetime(d["day"])
    d["month"] = d["day"].dt.to_period("M").dt.to_timestamp()
    gcols = ([group] if group else []) + ["month"]
    agg = d.groupby(gcols).agg(**{c: (c, how[c]) for c in value_cols}, n_days=("day", "nunique")).reset_index()
    agg = agg[agg["n_days"] >= 0.9 * agg["month"].dt.days_in_month]
    return agg.drop(columns="n_days")


def weather_monthly(w: pd.DataFrame) -> pd.DataFrame:
    if w.empty:
        return pd.DataFrame(columns=["region_code", "month", "t_mean", "beach_days"])
    w = w.sort_values(["point_id", "day"]).copy()
    # короткие пропуски датчика (до 3 дней) — интерполяция внутри точки
    w["day"] = pd.to_datetime(w["day"])
    parts = []
    for pid, g in w.groupby("point_id"):
        g = g.set_index("day").asfreq("D")
        g[["t_mean", "t_max", "precip_mm"]] = g[["t_mean", "t_max", "precip_mm"]].interpolate(limit=3)
        g["point_id"], g["region_code"] = pid, g["region_code"].ffill().bfill()
        parts.append(g.reset_index())
    w = pd.concat(parts)
    w["beach"] = ((w["t_max"] >= 24) & (w["precip_mm"].fillna(0) < 1)).astype(float)
    # аномальные показания (выброс датчика) — в пропуск
    w.loc[(w["t_mean"] < -60) | (w["t_mean"] > 50), "t_mean"] = np.nan
    reg = w.groupby(["region_code", "day"]).agg(t_mean=("t_mean", "mean"), beach=("beach", "mean")).reset_index()
    reg = reg.dropna(subset=["t_mean"])
    m = _full_months(reg, ["t_mean", "beach"], "region_code", {"t_mean": "mean", "beach": "sum"})
    return m.rename(columns={"beach": "beach_days"})


def calendar_monthly(cal: pd.DataFrame, months: pd.DatetimeIndex, school: list[tuple[str, str]]) -> pd.DataFrame:
    """Календарь на любые месяцы: официальный, а где его ещё нет — правило «выходные + праздники»."""
    # запас по краям: серия выходных на стыке месяцев должна считаться одинаково
    # независимо от того, где заканчивается запрошенный диапазон
    days = pd.date_range(months.min() - pd.Timedelta(days=31), months.max() + pd.offsets.MonthEnd(0)
                         + pd.Timedelta(days=31), freq="D")
    official = cal.assign(day=pd.to_datetime(cal["day"])).set_index("day")["day_type"] if len(cal) else pd.Series(dtype=int)
    rule = pd.Series([1 if (d.weekday() >= 5 or (d.month, d.day) in FIXED_HOLIDAYS) else 0 for d in days], index=days)
    dt = official.reindex(days).fillna(rule).astype(int)
    nonwork = (dt == 1).astype(int)
    holiday = ((dt == 1) & (days.weekday < 5)).astype(int)
    # длинные выходные: серии нерабочих дней длиной от 3
    run_id = (nonwork != nonwork.shift()).cumsum()
    run_len = nonwork.groupby(run_id).transform("sum")
    long_we = ((nonwork == 1) & (run_len >= 3)).astype(int)
    md = days.strftime("%m-%d")
    sch = pd.Series(0, index=days)
    for a, b in school:
        sch |= ((md >= a) & (md <= b)).astype(int)
    df = pd.DataFrame({"nonworking": nonwork.values, "holidays": holiday.values, "long_weekend": long_we.values,
                       "school": sch.values}, index=days)
    out = df.resample("MS").sum()
    out.index.name = "month"
    return out.loc[months.min():months.max()]


# ---------- сборка витрины ----------
def _months_between(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a.dt.year - b.dt.year) * 12 + (a.dt.month - b.dt.month)


def build_features(inp: Inputs, cfg: AppConfig, future_only: bool = False) -> pd.DataFrame:
    lag = cfg.forecast.publication_lag_months
    horizons = cfg.forecast.horizons
    tgt = inp.target.copy()
    tgt["month"] = pd.to_datetime(tgt["month"])
    tgt["guests"] = tgt["guests"].astype(float)
    tgt.loc[tgt["guests"] <= 0, "guests"] = np.nan            # ноль в КСР — пропуск/ошибка, не реальный ноль
    last_month = tgt["month"].max()
    excl = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in cfg.forecast.exclude_periods]

    # полная сетка месяцев по каждому ряду, чтобы сдвиги были по календарю, а не по строкам
    rows = []
    for (r, a), g in tgt.groupby(["region_code", "acc_type"]):
        idx = pd.date_range(g["month"].min(), last_month, freq="MS")
        s = g.set_index("month")["guests"].reindex(idx)
        ly = np.log(s)
        level = s.rolling(12, min_periods=12).mean()
        d = pd.DataFrame({"region_code": r, "acc_type": a, "origin_month": idx, "y_o": s.values,
                          "level": level.values})
        d["f_lag0"] = (ly - np.log(level)).values
        d["f_lag1"] = (ly.shift(1) - np.log(level)).values
        d["f_lag2"] = (ly.shift(2) - np.log(level)).values
        d["f_yoy_level"] = (np.log(level) - np.log(level.shift(12))).values
        d["f_yoy_recent"] = (np.log(s.rolling(3, min_periods=2).mean())
                             - np.log(s.shift(12).rolling(3, min_periods=2).mean())).values
        d["f_std12"] = ly.diff().rolling(12, min_periods=9).std().values
        in_excl = pd.Series([any(x <= m <= y for x, y in excl) for m in idx], index=idx).astype(float)
        d["f_covid_share"] = in_excl.rolling(12, min_periods=1).mean().values
        for h in horizons:
            dh = d.copy()
            dh["horizon"] = h
            dh["target_month"] = dh["origin_month"] + pd.DateOffset(months=h)
            dh["y_t"] = s.shift(-h).reindex(idx).values if h > 0 else s.values
            dh["f_ly"] = (ly.shift(12 - h) - np.log(level)).values if h <= 12 else np.nan
            dh["f_ly2"] = (ly.shift(24 - h) - np.log(level.shift(12))).values  # два года назад к уровню того года
            rows.append(dh)
    df = pd.concat(rows, ignore_index=True)
    df = df[df["level"].notna() & df["f_lag0"].notna()]
    if future_only:
        df = df[df["origin_month"] == last_month]
    df["cutoff_month"] = df["origin_month"] + pd.DateOffset(months=lag - 1)

    # календарь целевого месяца и того же месяца год назад
    all_months = pd.date_range(df["origin_month"].min() - pd.DateOffset(months=12),
                               df["target_month"].max(), freq="MS")
    cal = calendar_monthly(inp.calendar, all_months, cfg.school_holidays)
    cal_t = cal.reindex(df["target_month"]).reset_index(drop=True)
    cal_ly = cal.reindex(df["target_month"] - pd.DateOffset(months=12)).reset_index(drop=True)
    df = df.reset_index(drop=True)
    df["cal_nonworking"] = cal_t["nonworking"].values
    df["cal_holidays"] = cal_t["holidays"].values
    df["cal_long_weekend"] = cal_t["long_weekend"].values
    df["cal_school"] = cal_t["school"].values
    df["cal_d_nonworking"] = (cal_t["nonworking"] - cal_ly["nonworking"]).values
    tm = df["target_month"].dt.month
    df["month_sin"], df["month_cos"] = np.sin(2 * np.pi * tm / 12), np.cos(2 * np.pi * tm / 12)

    df = _add_weather(df, weather_monthly(inp.weather))
    df = _add_pageviews(df, inp.pageviews)
    df = _add_fx(df, inp.fx)

    # винзоризация логарифмических отношений: единичный выброс источника не раскачивает модель
    rel = [c for c in NUMERIC if c.startswith(("f_lag", "f_yoy", "pv_", "fx_usd_chg", "fx_usd_yoy"))
           or c in ("f_ly", "f_ly2")]
    df[rel] = df[rel].clip(-3, 3)
    df["in_excluded_period"] = [any(x <= m <= y for x, y in excl) for m in df["target_month"]]
    ly_month = df["target_month"] - pd.DateOffset(months=12)
    df["f_ly_excluded"] = [float(any(x <= m <= y for x, y in excl)) for m in ly_month]
    # База сезонного прогноза: тот же месяц год назад; если тот год аномален — два года назад
    # (с поправкой на изменение уровня ряда), если истории нет — последний известный месяц.
    alt = df["f_ly2"] + df["f_yoy_level"].fillna(0)
    df["f_base"] = np.where(df["f_ly_excluded"] == 1, alt, df["f_ly"])
    df["f_base"] = df["f_base"].fillna(df["f_ly"]).fillna(df["f_lag0"])
    # Строки, где окно истории задевает аномальный период: уровень и база искажены.
    # Для обучения они не используются (прогноз для них всё равно строится).
    df["anomaly_window"] = (df["f_covid_share"] > 0) | (df["f_ly_excluded"] == 1)
    df["target"] = df["y_t"]
    df["y_rel"] = np.log(df["y_t"]) - np.log(df["level"])
    return df[KEYS + ["target_month", "cutoff_month", "level", "target", "y_rel", "in_excluded_period",
                      "anomaly_window"]
              + [c for c in ALL_FEATURES if c not in KEYS]]


def _lookup(df: pd.DataFrame, table: pd.DataFrame, key_col: str | None, month_col: str, value: str) -> np.ndarray:
    """Значение `value` из месячной таблицы по (регион, месяц) для каждой строки df."""
    if table.empty:
        return np.full(len(df), np.nan)
    t = table.set_index(([key_col] if key_col else []) + ["month"])[value]
    idx = pd.MultiIndex.from_arrays([df[key_col], df[month_col]]) if key_col else pd.Index(df[month_col])
    return t.reindex(idx).values


def _add_weather(df: pd.DataFrame, wm: pd.DataFrame) -> pd.DataFrame:
    if wm.empty:
        for c in FEATURE_GROUPS["weather"]:
            df[c] = np.nan
        return df
    wm = wm.copy()
    wm["cal_month"] = wm["month"].dt.month
    # накопленная климатическая норма каждого календарного месяца по мере поступления лет
    clim = wm.sort_values("month").assign(
        clim_t=lambda x: x.groupby(["region_code", "cal_month"])["t_mean"].transform(lambda s: s.expanding().mean()),
        clim_beach=lambda x: x.groupby(["region_code", "cal_month"])["beach_days"].transform(
            lambda s: s.expanding().mean()),
    ).rename(columns={"month": "limit"})[["region_code", "cal_month", "limit", "clim_t", "clim_beach"]]
    clim["cal_month"] = clim["cal_month"].astype("int64")
    clim["limit"] = clim["limit"].astype("datetime64[ns]")

    def clim_for(month: pd.Series) -> pd.DataFrame:
        """Норма для месяца `month`: только прошлые годы и только наблюдения не позже cutoff."""
        prev_dec = pd.to_datetime(month.dt.year.sub(1).astype("Int64").astype(str) + "-12-01", errors="coerce")
        limit = pd.concat([df["cutoff_month"], prev_dec], axis=1).min(axis=1)
        q = pd.DataFrame({"region_code": df["region_code"], "cal_month": month.dt.month, "limit": limit,
                          "_i": np.arange(len(df))}).dropna(subset=["limit", "cal_month"])
        q["cal_month"] = q["cal_month"].astype("int64")
        q["limit"] = q["limit"].astype("datetime64[ns]")
        m = pd.merge_asof(q.sort_values("limit"), clim.sort_values("limit"), on="limit",
                          by=["region_code", "cal_month"], direction="backward")
        return m.set_index("_i").reindex(np.arange(len(df)))

    ct = clim_for(df["target_month"])
    df["w_clim_t"], df["w_clim_beach"] = ct["clim_t"].values, ct["clim_beach"].values
    # аномалия температуры: последний известный месяц и тот же месяц год назад (база сезонного прогноза)
    ly = (df["target_month"] - pd.DateOffset(months=12)).where(
        lambda x: x <= df["cutoff_month"])
    for col, month in (("w_anom_e", df["cutoff_month"]), ("w_anom_ly", ly)):
        actual = _lookup(df.assign(_m=month), wm, "region_code", "_m", "t_mean")
        df[col] = actual - clim_for(month)["clim_t"].values
    return df


def _add_pageviews(df: pd.DataFrame, pv: pd.DataFrame) -> pd.DataFrame:
    if pv.empty:
        for c in FEATURE_GROUPS["pageviews"]:
            df[c] = np.nan
        return df
    reg = pv.groupby(["region_code", "day"], as_index=False)["views"].sum()
    m = _full_months(reg, ["views"], "region_code", {"views": "sum"}).sort_values(["region_code", "month"])
    m["mean12"] = m.groupby("region_code")["views"].transform(lambda s: s.rolling(12, min_periods=9).mean())
    v_e = _lookup(df, m, "region_code", "cutoff_month", "views")
    mean_e = _lookup(df, m, "region_code", "cutoff_month", "mean12")
    v_e12 = _lookup(df.assign(_m=df["cutoff_month"] - pd.DateOffset(months=12)), m, "region_code", "_m", "views")
    # интерес год назад за месяц до целевого: показывает сезонный пик интереса
    ly_t = df["target_month"] - pd.DateOffset(months=13)
    ly_t = ly_t.where(ly_t <= df["cutoff_month"])
    v_ly = _lookup(df.assign(_m=ly_t), m, "region_code", "_m", "views")
    df["pv_rel"] = np.log(v_e / mean_e)
    df["pv_yoy"] = np.log(v_e / v_e12)
    df["pv_ly_t"] = np.log(v_ly / mean_e)
    return df


def _add_fx(df: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    usd = fx[fx["currency"] == "USD"][["day", "rate"]]
    if usd.empty:
        for c in FEATURE_GROUPS["fx"]:
            df[c] = np.nan
        return df
    m = usd.assign(day=pd.to_datetime(usd["day"]))
    m = m.groupby(m["day"].dt.to_period("M").dt.to_timestamp())["rate"].mean().rename_axis("month").reset_index()
    e = _lookup(df, m, None, "cutoff_month", "rate")
    e3 = _lookup(df.assign(_m=df["cutoff_month"] - pd.DateOffset(months=3)), m, None, "_m", "rate")
    e12 = _lookup(df.assign(_m=df["cutoff_month"] - pd.DateOffset(months=12)), m, None, "_m", "rate")
    df["fx_usd_log"] = np.log(e)
    df["fx_usd_chg3"] = np.log(e / e3)
    df["fx_usd_yoy"] = np.log(e / e12)
    return df


def save_features(engine: Engine, df: pd.DataFrame, data_hash: str) -> int:
    """Полная замена витрины одним атомарным действием — повторный запуск даёт то же состояние."""
    feats = df[[c for c in ALL_FEATURES if c not in KEYS]]
    out = df[KEYS + ["target_month", "target"]].copy()
    out["features"] = [
        {k: (None if (isinstance(v, float) and np.isnan(v)) else (v.item() if hasattr(v, "item") else v))
         for k, v in rec.items()} for rec in feats.to_dict("records")]
    out["data_hash"] = data_hash
    out["origin_month"] = out["origin_month"].dt.date
    out["target_month"] = out["target_month"].dt.date
    return db.replace_where(engine, "mart.features", "TRUE", {}, out)
