"""Конфигурация: доменные параметры из config.yaml + секреты и окружение из .env.

Секреты (пароли, DSN) читаются только из переменных окружения и хранятся как SecretStr,
чтобы случайно не попасть в логи.
"""
from __future__ import annotations

import os
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

def _project_root() -> Path:
    """Корень проекта (там config.yaml, sql/, data/).

    TOURFLOW_HOME → исходники рядом с пакетом (режим разработки) → текущая папка (Docker: /app).
    """
    if os.getenv("TOURFLOW_HOME"):
        return Path(os.environ["TOURFLOW_HOME"]).resolve()
    src_root = Path(__file__).resolve().parents[2]
    if (src_root / "config.yaml").exists():
        return src_root
    return Path.cwd()


PROJECT_ROOT = _project_root()


class Settings(BaseSettings):
    """Окружение. Значения по умолчанию безопасны для локальной разработки."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="TOURFLOW_", extra="ignore")

    config_path: Path = PROJECT_ROOT / "config.yaml"
    # DSN роли ETL (запись в raw/stg/mart/ops) и роли только для чтения (дашборд)
    database_url: SecretStr = SecretStr("postgresql+psycopg://tourflow_etl:change-me@localhost:5432/tourflow")
    reader_database_url: SecretStr | None = None
    # synthetic — офлайн-режим с синтетическими данными той же структуры (демо, тесты, CI)
    source_mode: Literal["live", "synthetic"] = "live"
    synthetic_as_of: date | None = None  # «сегодня» для синтетического мира
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    mlflow_tracking_uri: str | None = None
    dashboard_password_hash: SecretStr | None = None  # scrypt-хеш пароля дашборда
    log_level: str = "INFO"


class Point(BaseModel):
    id: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class Region(BaseModel):
    code: str = Field(pattern=r"^\d{8}$")
    name: str
    articles: list[str]
    points: list[Point]


class HttpCfg(BaseModel):
    timeout_s: float = 30
    max_retries: int = 5
    max_response_mb: int = 50
    user_agent: str


class SourcesCfg(BaseModel):
    allowed_hosts: list[str]
    http: HttpCfg
    rosstat: dict
    weather: dict
    pageviews: dict
    cbr: dict
    calendar: dict


class ForecastCfg(BaseModel):
    horizons: list[int]
    publication_lag_months: int = Field(ge=0, le=12)
    holdout_months: int = Field(ge=1)
    backtest_origins: int = Field(ge=1)
    interval_level: float = Field(gt=0, lt=1)
    exclude_periods: list[tuple[date, date]] = []


class ProjectCfg(BaseModel):
    seed: int
    history_start: date


class AppConfig(BaseModel):
    project: ProjectCfg
    forecast: ForecastCfg
    accommodation_types: dict[str, str]
    regions: list[Region]
    school_holidays: list[tuple[str, str]]
    sources: SourcesCfg
    models: dict

    @field_validator("regions")
    @classmethod
    def _unique_codes(cls, v: list[Region]) -> list[Region]:
        codes = [r.code for r in v]
        if len(codes) != len(set(codes)):
            raise ValueError("коды регионов в config.yaml повторяются")
        return v

    @property
    def region_codes(self) -> list[str]:
        return [r.code for r in self.regions]

    def region_by_code(self, code: str) -> Region:
        return next(r for r in self.regions if r.code == code)


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_config(path: str | None = None) -> AppConfig:
    p = Path(path) if path else get_settings().config_path
    with open(p, encoding="utf-8") as f:
        return AppConfig.model_validate(yaml.safe_load(f))
