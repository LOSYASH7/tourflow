"""Сборка источников. Остальная система не знает, живые данные или синтетические."""
from __future__ import annotations

from datetime import date

from tourflow.config import AppConfig, Settings
from tourflow.sources.base import Source
from tourflow.sources.calendar import CalendarSource
from tourflow.sources.cbr import CbrSource
from tourflow.sources.http import HttpTransport, SyntheticTransport
from tourflow.sources.pageviews import PageviewsSource
from tourflow.sources.rosstat import RosstatSource
from tourflow.sources.weather import WeatherSource

SOURCE_NAMES = ("calendar", "cbr", "weather", "pageviews", "rosstat")


def today_for(settings: Settings) -> date:
    if settings.source_mode == "synthetic" and settings.synthetic_as_of:
        return settings.synthetic_as_of
    return date.today()


def build_sources(cfg: AppConfig, settings: Settings, only: list[str] | None = None) -> list[Source]:
    today = today_for(settings)
    synthetic = settings.source_mode == "synthetic"
    if synthetic:
        from tourflow.sources.synthetic import SyntheticWorld

        transport = SyntheticTransport(cfg, SyntheticWorld(cfg, today))
    else:
        transport = HttpTransport(cfg)
    classes = {"calendar": CalendarSource, "cbr": CbrSource, "weather": WeatherSource,
               "pageviews": PageviewsSource}
    out: list[Source] = []
    for name in only or SOURCE_NAMES:
        if name == "rosstat":
            out.append(RosstatSource(cfg, transport, today, synthetic=synthetic))
        else:
            out.append(classes[name](cfg, transport, today))
    return out
