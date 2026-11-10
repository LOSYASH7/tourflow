"""Доступ к PostgreSQL: подключение, применение схемы, идемпотентный upsert.

Все запросы параметризованы (SQLAlchemy Core) — значения никогда не подставляются в текст SQL.
Имена таблиц приходят только из кода, не из пользовательского ввода.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, MetaData, Table, create_engine, literal_column, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from tourflow.config import PROJECT_ROOT, get_settings

log = logging.getLogger(__name__)

_engines: dict[str, Engine] = {}
_tables: dict[str, Table] = {}
NON_DATA_COLS = {"payload_id", "updated_at"}


ALLOWED_SCHEMAS = {"raw", "stg", "mart", "ops"}


def safe_ident(qualified: str) -> str:
    """Имя таблицы в SQL-тексте допускается только вида схема.таблица из известных схем."""
    import re

    if not re.fullmatch(r"[a-z_]+\.[a-z_]+", qualified) or qualified.split(".")[0] not in ALLOWED_SCHEMAS:
        raise ValueError(f"недопустимое имя таблицы: {qualified!r}")
    return qualified


def get_engine(url: str | None = None) -> Engine:
    dsn = url or get_settings().database_url.get_secret_value()
    if dsn not in _engines:
        _engines[dsn] = create_engine(dsn, pool_pre_ping=True, future=True)
    return _engines[dsn]


def masked(dsn: str) -> str:
    """DSN без пароля — для логов."""
    from sqlalchemy.engine import make_url

    return make_url(dsn).render_as_string(hide_password=True)


def apply_schema(engine: Engine) -> None:
    for name in ("001_schema.sql", "002_grants.sql"):
        sql = (PROJECT_ROOT / "sql" / name).read_text(encoding="utf-8")
        with engine.begin() as conn:
            conn.exec_driver_sql(sql)
    log.info("schema applied to %s", engine.url.render_as_string(hide_password=True))


def table(engine: Engine, qualified: str) -> Table:
    if qualified not in _tables:
        schema, name = qualified.split(".")
        _tables[qualified] = Table(name, MetaData(schema=schema), autoload_with=engine)
    return _tables[qualified]


def upsert(
    engine: Engine,
    qualified: str,
    rows: Sequence[dict] | pd.DataFrame,
    key: Iterable[str],
    chunk: int = 2000,
) -> tuple[int, int]:
    """INSERT ... ON CONFLICT DO UPDATE. Возвращает (вставлено, обновлено).

    Повторный вызов с теми же данными не создаёт дублей — это и есть идемпотентность записи.
    Строки, значения которых не изменились, не переписываются (WHERE ... IS DISTINCT FROM).
    """
    if isinstance(rows, pd.DataFrame):
        rows = rows.astype(object).where(rows.notna(), None).to_dict("records")
    if not rows:
        return 0, 0
    t = table(engine, qualified)
    key = list(key)
    value_cols = [c for c in rows[0] if c not in key]
    inserted = updated = 0
    with engine.begin() as conn:
        for i in range(0, len(rows), chunk):
            stmt = pg_insert(t).values(rows[i : i + chunk])
            if value_cols:
                set_ = {c: stmt.excluded[c] for c in value_cols}
                if "updated_at" in t.c:
                    set_["updated_at"] = text("now()")
                changed = None
                # служебные столбцы (ссылка на raw) не считаются изменением данных
                for c in [c for c in value_cols if c not in NON_DATA_COLS] or value_cols:
                    cond = t.c[c].is_distinct_from(stmt.excluded[c])
                    changed = cond if changed is None else (changed | cond)
                stmt = stmt.on_conflict_do_update(index_elements=key, set_=set_, where=changed)
            else:
                stmt = stmt.on_conflict_do_nothing(index_elements=key)
            # xmax = 0 у только что вставленной строки, иначе это обновление
            res = conn.execute(stmt.returning(literal_column("(xmax = 0)").label("ins")))
            flags = [r.ins for r in res]
            inserted += sum(flags)
            updated += len(flags) - sum(flags)
    return inserted, updated


def read_df(engine: Engine, sql: str, **params) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


def replace_where(engine: Engine, qualified: str, where_sql: str, params: dict, df: pd.DataFrame) -> int:
    """Атомарно заменить срез таблицы (DELETE + INSERT в одной транзакции) — идемпотентно."""
    t = table(engine, qualified)
    rows = df.astype(object).where(df.notna(), None).to_dict("records")
    with engine.begin() as conn:
        # имя проверено safe_ident, условие — константа из кода, значения — только параметры
        conn.execute(text(f"DELETE FROM {safe_ident(qualified)} WHERE {where_sql}"), params)  # noqa: S608  # nosec B608
        if rows:
            conn.execute(t.insert(), rows)
    return len(rows)


def schema_files() -> list[Path]:
    return sorted((PROJECT_ROOT / "sql").glob("*.sql"))
