"""Парсеры источников и защитные свойства транспорта (без сети)."""
import io
from datetime import date

import pandas as pd
import pytest
from openpyxl import Workbook

from tourflow.sources.calendar import CalendarSource
from tourflow.sources.cbr import CbrSource
from tourflow.sources.http import HttpTransport
from tourflow.sources.rosstat import RosstatSource, norm_region, parse_period


def _xlsx(sheets: dict[str, list[list]]) -> bytes:
    wb = Workbook()
    wb.remove(wb.active)
    for title, rows in sheets.items():
        ws = wb.create_sheet(title)
        for r in rows:
            ws.append(r)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


@pytest.mark.parametrize("raw,expected", [
    ("январь 2024", date(2024, 1, 1)), ("Февраль 2023 г.", date(2023, 2, 1)), ("2025-07", date(2025, 7, 1)),
    ("03.2022", date(2022, 3, 1)), ("дек.24", date(2024, 12, 1)), ("итого", None),
])
def test_parse_period(raw, expected):
    assert parse_period(raw) == expected


def test_region_names_are_normalized():
    assert norm_region("г. Москва") == norm_region("Москва") == norm_region("город Москва")
    assert norm_region("Республика Крым") != norm_region("Крым")


def test_rosstat_wide_layout_with_blocks(cfg):
    content = _xlsx({"Лист1": [
        ["Численность размещённых лиц, человек"],
        ["", "январь 2024", "февраль 2024", "март 2024"],
        ["Гостиницы и аналогичные средства размещения"],
        ["Краснодарский край", 100, "1 200", "…"],
        ["г. Москва", 300, 310, 320],
        ["Санаторно-курортные организации"],
        ["Краснодарский край", 10, 11, 12],
        ["Неизвестный регион", 1, 1, 1],
    ]})
    src = RosstatSource(cfg, transport=None, today=date(2026, 1, 1))
    df = src.parse("f.xlsx", content, "")
    assert len(df) == 8                        # «…» — пропуск, неизвестный регион отброшен
    row = df[(df.region_code == "03000000") & (df.acc_type == "hotels") & (df.month == date(2024, 2, 1))]
    assert row.guests.item() == 1200


def test_rosstat_long_layout_and_cumulative(cfg):
    content = _xlsx({"data": [
        ["Субъект", "Вид КСР", "Период", "Значение"],
        ["Республика Татарстан", "Организации отдыха", "январь 2024", 50],
        ["Республика Татарстан", "Организации отдыха", "февраль 2024", 120],
    ]})
    src = RosstatSource(cfg, transport=None, today=date(2026, 1, 1))
    src.cumulative = True
    df = src.parse("f.xlsx", content, "").sort_values("month")
    assert df.guests.tolist() == [50, 70]      # нарастающий итог → помесячно


def test_cbr_xml_parsing(cfg):
    xml = ('<?xml version="1.0" encoding="windows-1251"?><ValCurs><Record Date="09.01.2024" Id="R01235">'
           '<Nominal>1</Nominal><Value>90,2675</Value></Record></ValCurs>').encode("windows-1251")
    df = CbrSource(cfg, transport=None, today=date(2024, 1, 10)).parse("USD", xml, "")
    assert df.rate.item() == pytest.approx(90.2675)


def test_cbr_rejects_xml_entities(cfg):
    bomb = b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa"><!ENTITY b "&a;&a;">]><ValCurs>&b;</ValCurs>'
    from defusedxml import EntitiesForbidden

    with pytest.raises(EntitiesForbidden):
        CbrSource(cfg, transport=None, today=date(2024, 1, 1)).parse("USD", bomb, "")


def test_calendar_error_code_is_not_data(cfg):
    src = CalendarSource(cfg, transport=None, today=date(2024, 1, 1))
    assert src.parse("2027", b"101", "").empty
    df = src.parse("2024", b"0" * 366, "")
    assert len(df) == 366 and set(df.day_type) == {0}


@pytest.mark.parametrize("url", ["http://www.cbr.ru/x", "https://evil.example.com/x", "file:///etc/passwd",
                                 "https://169.254.169.254/latest/meta-data"])
def test_transport_blocks_non_allowlisted_urls(cfg, url):
    with pytest.raises(ValueError):
        HttpTransport(cfg).get(url)


def test_synthetic_world_is_deterministic(cfg):
    from tourflow.sources.synthetic import SyntheticWorld
    a = SyntheticWorld(cfg, date(2026, 1, 1)).guests
    b = SyntheticWorld(cfg, date(2026, 6, 1)).guests
    pd.testing.assert_frame_equal(a, b)


def test_emiss_operative_xls_with_cumulative_totals(cfg):
    """Реальная выгрузка ЕМИСС: .xls, год над месяцами, «январь-…» нарастающим итогом."""
    from pathlib import Path

    content = (Path(__file__).parent / "fixtures" / "emiss_ksr_operative.xls").read_bytes()
    src = RosstatSource(cfg, transport=None, today=date(2026, 11, 1))
    src.default_acc = "total"
    df = src.parse("data.xls", content, "")
    assert df.region_code.nunique() == len(cfg.regions)
    kras = df[df.region_code == "03000000"].set_index("month").guests
    assert kras[date(2022, 1, 1)] == 436894
    assert kras[date(2022, 2, 1)] == 813369 - 436894        # из нарастающего итога
    assert (df.guests > 0).all()


def test_emiss_like_xlsx_header(cfg):
    content = _xlsx({"Данные": [
        ["Численность размещенных лиц"],
        ["", "2024", "", "", "2025"],
        ["", "январь", "январь-февраль", "январь-март", "январь"],
        ["        Город Москва столица Российской Федерации город федерального значения", 10, 25, 45, 12],
    ]})
    src = RosstatSource(cfg, transport=None, today=date(2026, 1, 1))
    src.default_acc = "total"
    df = src.parse("f.xlsx", content, "").sort_values("month")
    assert df.guests.tolist() == [10, 15, 20, 12] and set(df.region_code) == {"45000000"}
