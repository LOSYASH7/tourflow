"""Дашборд: турпоток в регион по видам размещения.

Каждый блок отвечает на один вопрос пользователя (органа по туризму / гостиничной ассоциации).
Дашборд подключается к БД ролью только для чтения и видит лишь витрину mart и журнал ops.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, text

from tourflow.config import get_config, get_settings
from tourflow.security import verify_password
from tourflow.sources import today_for

st.set_page_config(page_title="Турпоток: прогноз", layout="wide")
cfg, settings = get_config(), get_settings()

FEATURE_RU = {
    "__base_ly": "База: тот же месяц год назад", "f_base": "Уровень базы к среднему за год", "f_lag0": "Последний месяц к уровню",
    "f_lag1": "Месяц −1 к уровню", "f_lag2": "Месяц −2 к уровню", "f_ly": "Тот же месяц год назад",
    "f_ly2": "Тот же месяц два года назад", "f_ly_excluded": "Год назад — аномальный период",
    "f_yoy_level": "Рост уровня за год", "f_yoy_recent": "Рост последних 3 мес. к прошлому году",
    "f_std12": "Волатильность ряда", "f_covid_share": "Доля аномальных месяцев в окне", "horizon": "Горизонт",
    "month_sin": "Сезон (sin)", "month_cos": "Сезон (cos)", "cal_nonworking": "Нерабочие дни",
    "cal_holidays": "Праздники в будни", "cal_long_weekend": "Дни длинных выходных", "cal_school": "Дни каникул",
    "cal_d_nonworking": "Нерабочих дней больше, чем год назад", "w_clim_t": "Климатическая норма t°",
    "w_clim_beach": "Норма пляжных дней", "w_anom_e": "Аномалия t° последнего месяца",
    "w_anom_ly": "Аномалия t° год назад", "pv_rel": "Интерес в Википедии к уровню",
    "pv_yoy": "Интерес в Википедии, г/г", "pv_ly_t": "Интерес год назад перед этим месяцем",
    "fx_usd_log": "Курс доллара", "fx_usd_chg3": "Изменение курса за 3 мес.", "fx_usd_yoy": "Курс доллара, г/г",
    "region_code": "Регион", "acc_type": "Вид размещения",
}
ACC_RU = {"total": "Все КСР", "hotels": "Гостиницы", "sanatoriums": "Санатории", "recreation": "Организации отдыха"}
ACC_RU = {k: v for k, v in ACC_RU.items() if k in cfg.accommodation_types}
REGION_RU = {r.code: r.name for r in cfg.regions}
MONTHS_RU = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]


def mlabel(d) -> str:
    d = pd.Timestamp(d)
    return f"{MONTHS_RU[d.month - 1]} {d.year}"


def fmt(n: float) -> str:
    if abs(n) >= 1e6:
        return f"{n / 1e6:.2f} млн".replace(".", ",")
    if abs(n) >= 1e3:
        return f"{n / 1e3:.0f} тыс."
    return f"{n:.0f}"


# ---------- доступ ----------
def require_login() -> None:
    stored = settings.dashboard_password_hash
    if stored is None or not stored.get_secret_value():
        st.error("Пароль дашборда не задан. Выполните `tourflow hash-password` и добавьте хеш в .env.")
        st.stop()
    if st.session_state.get("auth_ok"):
        return
    st.title("Вход")
    fails = st.session_state.get("fails", 0)
    if fails:
        st.error(f"Неверный пароль (попыток: {fails})")
    with st.form("login"):                      # Enter в поле тоже отправляет форму
        pw = st.text_input("Пароль", type="password")
        sent = st.form_submit_button("Войти")
    if sent and pw:
        if verify_password(pw, stored.get_secret_value()):
            st.session_state["auth_ok"] = True
            st.session_state["fails"] = 0
            st.rerun()
        st.session_state["fails"] = fails + 1
        time.sleep(min(fails + 1, 5))           # замедление перебора, но без долгого «зависания»
        st.rerun()
    st.stop()


require_login()


@st.cache_resource
def engine():
    dsn = (settings.reader_database_url or settings.database_url).get_secret_value()
    return create_engine(dsn, pool_pre_ping=True)


@st.cache_data(ttl=300)
def q(sql: str, **params) -> pd.DataFrame:
    with engine().connect() as c:
        return pd.read_sql(text(sql), c, params=params)


active = q("SELECT model_version, model_name, created_at, data_hash FROM mart.model_registry WHERE is_active")
if active.empty:
    st.warning("Модель ещё не обучена — запустите `tourflow run`.")
    st.stop()
models = q("SELECT DISTINCT model_version, model_name FROM mart.metrics ORDER BY model_name")
actual = q("SELECT region_code, acc_type, target_month AS month, target AS guests FROM mart.features "
           "WHERE horizon = 1 AND target IS NOT NULL")
actual["month"] = pd.to_datetime(actual["month"])

# ---------- фильтры ----------
with st.sidebar:
    st.header("Фильтры")
    region = st.selectbox("Регион", cfg.region_codes, format_func=REGION_RU.get)
    acc = st.selectbox("Вид размещения", list(ACC_RU), format_func=ACC_RU.get)
    names = models["model_name"].tolist()
    best = active.iloc[0]["model_name"]
    model_name = st.selectbox("Модель", names, index=names.index(best),
                              format_func=lambda n: f"{n} (рабочая)" if n == best else n)
    version = models.set_index("model_name").loc[model_name, "model_version"]
    hist_h = st.select_slider("Горизонт для истории прогнозов", cfg.forecast.horizons, value=3)

runs = q("SELECT started_at, finished_at, status, message FROM ops.pipeline_runs ORDER BY started_at DESC LIMIT 1")
last_pub = actual["month"].max()
st.title(f"Турпоток: {REGION_RU[region]} · {ACC_RU[acc]}")
r0 = runs.iloc[0] if len(runs) else None
st.caption(f"Последний опубликованный месяц Росстата: **{mlabel(last_pub)}** · последний запуск: "
           f"{pd.Timestamp(r0['started_at']):%d.%m.%Y %H:%M} ({r0['status']})" if r0 is not None else "")
if r0 is not None and r0["status"] != "success":
    st.warning(f"Последний запуск завершился со статусом «{r0['status']}»: {r0['message'] or ''}")

fc = q("SELECT * FROM mart.forecasts WHERE model_version = :v AND region_code = :r AND acc_type = :a",
       v=version, r=region, a=acc)
for c in ("origin_month", "target_month"):
    fc[c] = pd.to_datetime(fc[c])
fut = q("SELECT * FROM mart.forecasts WHERE kind = 'future' AND model_version = :v", v=active.iloc[0]["model_version"])
fut["target_month"] = pd.to_datetime(fut["target_month"])
series_act = actual[(actual.region_code == region) & (actual.acc_type == acc)].set_index("month")["guests"]

# ---------- 1. Сколько гостей ждать ----------
st.subheader("Сколько гостей ждать")
f_sel = fut[(fut.region_code == region) & (fut.acc_type == acc)].sort_values("target_month")
today_m = pd.Timestamp(today_for(settings)).to_period("M").to_timestamp()
cols = st.columns(len(f_sel) or 1)
for col, (_, r) in zip(cols, f_sel.iterrows(), strict=False):
    ly = series_act.get(r["target_month"] - pd.DateOffset(months=12))
    delta = f"{(r['y_pred'] / ly - 1) * 100:+.1f}% к прошлому году" if ly else None
    tag = "оценка" if r["target_month"] < today_m else "прогноз"
    col.metric(f"{mlabel(r['target_month'])} · {tag}", fmt(r["y_pred"]), delta,
               help="Оценка — месяц уже прошёл, но Росстат его ещё не опубликовал")
    col.caption(f"интервал {int(cfg.forecast.interval_level * 100)}%: {fmt(r['y_lo'])} – {fmt(r['y_hi'])}")
if model_name != best:
    st.caption(f"Прогноз на будущее строит рабочая модель ({best}); выбранная модель влияет на блоки ниже.")

# ---------- 2. Факт и прогноз ----------
st.subheader("Как прогноз согласуется с фактом")
hist = fc[(fc.kind != "future") & (fc.horizon == hist_h)].sort_values("target_month")
start = last_pub - pd.DateOffset(months=36)
fig = go.Figure()
sa = series_act[series_act.index >= start]
fig.add_scatter(x=sa.index, y=sa.values, name="Факт", line=dict(color="#94a3b8", width=2.5))  # виден в светлой и тёмной теме
for kind, color, label in (("backtest", "#60a5fa", "Прогноз (бэктест)"), ("holdout", "#2563eb", "Прогноз (отложенная выборка)")):
    h = hist[hist.kind == kind]
    fig.add_scatter(x=h.target_month, y=h.y_pred, name=f"{label}, h={hist_h}", mode="lines+markers",
                    line=dict(color=color, dash="dot"))
if len(f_sel):
    x = pd.concat([pd.Series([last_pub]), f_sel.target_month])
    y = np.r_[series_act.get(last_pub, np.nan), f_sel.y_pred]
    fig.add_scatter(x=list(f_sel.target_month) + list(f_sel.target_month[::-1]),
                    y=list(f_sel.y_hi) + list(f_sel.y_lo[::-1]), fill="toself", fillcolor="rgba(234,88,12,0.15)",
                    mode="lines", line=dict(width=0), name="Интервал прогноза", hoverinfo="skip")
    fig.add_scatter(x=x, y=y, name="Прогноз", line=dict(color="#ea580c", width=2))
fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="гостей в месяц",
                  legend=dict(orientation="h", y=-0.15))
st.plotly_chart(fig, width="stretch")

# ---------- 3. Насколько можно доверять ----------
st.subheader("Насколько можно доверять прогнозу")
met = q("SELECT * FROM mart.metrics")
m_sel = met[(met.model_version == version) & (met.split == "holdout") & (met.horizon == 0)].iloc[0]
gain = (1 - m_sel["mae"] / m_sel["mae_naive"]) * 100 if m_sel["mae_naive"] else np.nan
c1, c2, c3 = st.columns(3)
c1.metric("Средняя ошибка на ряд в месяц", f"±{fmt(m_sel['mae'])} гостей")
c2.metric("Ошибка в % от потока (WAPE)", f"{m_sel['wape'] * 100:.1f}%")
c3.metric("Точнее прогноза «как год назад»", f"на {gain:.0f}%" if gain == gain else "—")
st.caption("Оценка на отложенной выборке: последние "
           f"{cfg.forecast.holdout_months} месяцев, которые не участвовали в выборе модели.")
cmp = met[(met.horizon == 0)].pivot_table(index="model_name", columns="split", values="wape").mul(100).round(2)
cmp = cmp.rename(columns={"backtest": "WAPE бэктест, %", "holdout": "WAPE отложенная, %"}).sort_values("WAPE бэктест, %")
st.dataframe(cmp, width="stretch")

# ---------- 4. Где рост и где падение ----------
st.subheader("Где ожидается рост и где падение")
months = sorted(fut.target_month.unique())
tm = st.select_slider("Месяц", months, value=months[min(2, len(months) - 1)], format_func=mlabel)
g = fut[fut.target_month == tm][["region_code", "acc_type", "y_pred"]].copy()
ly = actual.assign(month=actual["month"] + pd.DateOffset(months=12)).rename(columns={"guests": "ly"})
g = g.merge(ly[ly.month == tm][["region_code", "acc_type", "ly"]], how="left")
g["Изменение к прошлому году, %"] = ((g.y_pred / g.ly - 1) * 100).round(1)
g["Регион"], g["Вид"] = g.region_code.map(REGION_RU), g.acc_type.map(ACC_RU)
g["Прогноз, гостей"], g["Год назад"] = g.y_pred.round(-2).astype(int), g.ly.round(-2)
st.dataframe(g.sort_values("Изменение к прошлому году, %", ascending=False)[
    ["Регион", "Вид", "Прогноз, гостей", "Год назад", "Изменение к прошлому году, %"]],
    width="stretch", hide_index=True)

# ---------- 5. Почему прогноз такой ----------
st.subheader("Почему прогноз именно такой")
av = active.iloc[0]["model_version"]
tab1, tab2, tab3, tab4 = st.tabs(["Объяснение прогноза", "Важность признаков", "Важность по периодам",
                                  "Зависимость от признака"])
with tab1:
    ex_keys = q("SELECT origin_month, horizon FROM mart.explanations WHERE model_version = :v AND region_code = :r "
                "AND acc_type = :a ORDER BY origin_month DESC, horizon", v=av, r=region, a=acc)
    ex_keys["target_month"] = pd.to_datetime(ex_keys.origin_month) + ex_keys.horizon.apply(lambda h: pd.DateOffset(months=int(h)))
    c1, c2 = st.columns(2)
    tsel = c1.selectbox("Целевой месяц", sorted(ex_keys.target_month.unique(), reverse=True), format_func=mlabel)
    hs = sorted(ex_keys[ex_keys.target_month == tsel].horizon.tolist())
    hsel = c2.selectbox("Горизонт (месяцев от последних опубликованных данных)", hs)
    origin = (pd.Timestamp(tsel) - pd.DateOffset(months=int(hsel))).date()
    ex = q("SELECT base_value, contributions FROM mart.explanations WHERE model_version = :v AND region_code = :r "
           "AND acc_type = :a AND origin_month = :o AND horizon = :h", v=av, r=region, a=acc, o=origin, h=int(hsel))
    lvl = q("SELECT target FROM mart.features WHERE region_code = :r AND acc_type = :a AND origin_month = :o "
            "AND horizon = :h", r=region, a=acc, o=origin, h=int(hsel))
    if len(ex):
        contrib = dict(ex.iloc[0]["contributions"])
        base_ly = contrib.pop("__base_ly")[1]
        level = contrib.pop("__level")[0]
        items = sorted(((k, v[0], v[1]) for k, v in contrib.items()), key=lambda t: -abs(t[2]))
        top, rest = items[:10], items[10:]

        def lbl(k, v):
            if isinstance(v, str):
                return f"{FEATURE_RU.get(k, k)}: {REGION_RU.get(v, ACC_RU.get(v, v))}"
            return f"{FEATURE_RU.get(k, k)} = {v:.2f}" if v is not None else FEATURE_RU.get(k, k)

        labels = ["Общая поправка модели (тренд)"] + [lbl(k, v) for k, v, _ in top] + ["Остальные признаки"]
        logs = [ex.iloc[0]["base_value"]] + [c for *_, c in top] + [sum(c for *_, c in rest)]
        pct = [(np.exp(v) - 1) * 100 for v in logs]
        total = (np.exp(sum(logs)) - 1) * 100
        wf = go.Figure(go.Waterfall(
            orientation="h", y=(labels + ["Итого к базе"])[::-1], x=(pct + [total])[::-1],
            measure=(["relative"] * len(pct) + ["total"])[::-1],
            text=[f"{p:+.1f}%" for p in (pct + [total])[::-1]]))
        wf.update_layout(height=460, margin=dict(l=10, r=10, t=30, b=10),
                         title="Поправки модели к базе «как год назад», % (в логарифмах вклады складываются)")
        st.plotly_chart(wf, width="stretch")
        base_guests = level * np.exp(base_ly)
        pred = level * np.exp(base_ly + sum(logs))
        fact = lvl.iloc[0]["target"] if len(lvl) else None
        fact_txt = f"факт **{fmt(fact)}**" if fact is not None and fact == fact else "факт ещё не опубликован"
        st.markdown(f"База «как год назад»: **{fmt(base_guests)}** → прогноз модели: **{fmt(pred)}** · {fact_txt}. "
                    "Для прошлых месяцев показано объяснение рабочей модели, обученной на всех данных.")
with tab2:
    imp = q("SELECT method, feature, importance FROM mart.importance WHERE model_version = :v AND period = 'all'", v=av)
    c1, c2 = st.columns(2)
    for col, method, title in ((c1, "shap", "SHAP: средний |вклад|"),
                               (c2, "permutation", "Permutation: рост WAPE, п.п.")):
        d = imp[imp.method == method].nlargest(12, "importance").sort_values("importance")
        fig = go.Figure(go.Bar(x=d.importance, y=d.feature.map(lambda f: FEATURE_RU.get(f, f)), orientation="h"))
        fig.update_layout(height=400, title=title, margin=dict(l=10, r=10, t=40, b=10))
        col.plotly_chart(fig, width="stretch")
    s = imp[imp.method == "shap"].set_index("feature")["importance"].rank(ascending=False)
    p = imp[imp.method == "permutation"].set_index("feature")["importance"].rank(ascending=False)
    st.caption(f"Согласие двух способов (ранговая корреляция Спирмена): {s.corr(p.reindex(s.index), method='pearson'):.2f}. "
               "Permutation для сезонной базы велик, потому что база напрямую входит в прогноз.")
    ab = imp[imp.method == "ablation"].copy()
    if len(ab):
        ab["Группа"] = ab.feature.str.replace("group:", "").map(
            {"weather": "Погода", "pageviews": "Просмотры Википедии", "fx": "Курс валют", "calendar": "Календарь"})
        ab["Рост WAPE без группы, п.п."] = ab.importance.round(2)
        st.write("Абляция: насколько ухудшится модель без группы признаков (бэктест)")
        st.dataframe(ab[["Группа", "Рост WAPE без группы, п.п."]], hide_index=True, width="stretch")
with tab3:
    per = q("SELECT period, feature, importance FROM mart.importance WHERE model_version = :v AND method = 'shap' "
            "AND period <> 'all'", v=av)
    tbl = per.pivot(index="feature", columns="period", values="importance")
    tbl = tbl.loc[tbl.mean(axis=1).nlargest(12).index]
    tbl.index = tbl.index.map(lambda f: FEATURE_RU.get(f, f))
    # тепловая карта на plotly — без зависимости от matplotlib
    hm = go.Figure(go.Heatmap(z=tbl.values, x=list(tbl.columns), y=list(tbl.index), colorscale="Blues",
                              text=np.round(tbl.values, 3), texttemplate="%{text}", hoverongaps=False,
                              colorbar=dict(title="|SHAP|")))
    hm.update_layout(height=40 * len(tbl) + 100, margin=dict(l=10, r=10, t=40, b=10), yaxis=dict(autorange="reversed"),
                     xaxis=dict(type="category", side="top"))  # годы — подписи, а не числа
    st.plotly_chart(hm, width="stretch")
    years = [c for c in tbl.columns if c.isdigit()]
    if len(years) >= 2:
        rc = tbl[years].rank(ascending=False).corr(method="spearman")
        st.caption(f"Устойчивость структуры важности: ранговая корреляция {years[-2]} и {years[-1]} — "
                   f"{rc.loc[years[-2], years[-1]]:.2f}")
with tab4:
    pdp = q("SELECT feature, grid_value, pd_value FROM mart.pdp WHERE model_version = :v ORDER BY grid_value", v=av)
    rng = pdp.groupby("feature")["pd_value"].agg(lambda s: s.max() - s.min()).sort_values(ascending=False)
    f = st.selectbox("Признак (по силе влияния)", rng.index.tolist(), format_func=lambda x: FEATURE_RU.get(x, x))
    d = pdp[pdp.feature == f]
    fig = go.Figure(go.Scatter(x=d.grid_value, y=d.pd_value, mode="lines+markers"))
    fig.update_layout(height=350, xaxis_title=FEATURE_RU.get(f, f), yaxis_title="средний прогноз к уровню ряда, %",
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
