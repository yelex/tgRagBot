"""
Быстрые детерминированные тесты для фикса P1/P2/P3 (см. docs/TESTING.md).

В отличие от tests/test_scenarios.py эти тесты не дёргают GigaChat —
проверяют чистую логику: state_checklist.py и _find_bouquet_fuzzy.

Запуск: python -m pytest tests/test_state_checklist.py -v
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from state_checklist import MIN_ORDER_AMOUNT, is_below_minimum, is_node_complete, missing_fields
from flower_logic import FlowerLogic, _find_bouquet_fuzzy, INTERRUPT_NODES


CATALOG = [
    {"Название": "Букет Модена", "Цена": 5900, "Ссылка": "https://example.com/modena"},
    {"Название": "Букет из 19 белых эустом", "Цена": 7490, "Ссылка": "https://example.com/eustoma"},
    {"Название": "Букет Прояно", "Цена": 7900, "Ссылка": "https://example.com/proyano"},
]


# ── missing_fields / P1 ────────────────────────────────────────────────────

def test_missing_fields_n10_delivery_all_present():
    collected = {"address": "Арбат 5", "delivery_date": "завтра", "delivery_time": "15:00"}
    assert missing_fields("N10_delivery", collected) == []


def test_missing_fields_n10_delivery_time_missing():
    collected = {"address": "Арбат 5", "delivery_date": "завтра"}
    assert missing_fields("N10_delivery", collected) == ["delivery_time"]


def test_missing_fields_n10_delivery_all_missing():
    assert missing_fields("N10_delivery", {}) == ["address", "delivery_date", "delivery_time"]


def test_missing_fields_n9_offer():
    assert missing_fields("N9_offer", {}) == ["budget_max"]
    assert missing_fields("N9_offer", {"budget_max": 7000}) == []


def test_missing_fields_unknown_node_has_no_requirements():
    # Узлы вне REQUIRED_FIELDS (N11_complaint и т.п.) ничего не требуют —
    # роутинг для них не гейтится.
    assert missing_fields("N11_complaint", {}) == []


def test_interrupt_nodes_bypass_gate():
    # Прерывающие фазы должны быть в allow-list независимо от collected_data —
    # иначе жалоба/эскалация посреди брифа перестанет перехватывать фазу.
    for node in ("N11_complaint", "N12_escalation", "N5_urgent", "N7_pickup", "N8_faq", "N4_reference"):
        assert node in INTERRUPT_NODES


# ── is_node_complete / P1 (второй уровень — опциональный телефон) ─────────

def test_n10_delivery_not_complete_without_phone_resolution():
    # Адрес+дата+время есть, но телефон не дан и не пропущен явно —
    # узел ещё не закончил (иначе роутер прыгнет мимо шага "не знаю телефон")
    collected = {"address": "Арбат 5", "delivery_date": "завтра", "delivery_time": "15:00"}
    assert is_node_complete("N10_delivery", collected) is False


def test_n10_delivery_complete_with_phone():
    collected = {
        "address": "Арбат 5", "delivery_date": "завтра", "delivery_time": "15:00",
        "recipient_phone": "+79161234567",
    }
    assert is_node_complete("N10_delivery", collected) is True


def test_n10_delivery_complete_with_phone_skipped():
    collected = {
        "address": "Арбат 5", "delivery_date": "завтра", "delivery_time": "15:00",
        "phone_skipped": True,
    }
    assert is_node_complete("N10_delivery", collected) is True


def test_n9_offer_not_complete_without_bouquet_choice():
    # Бюджет один не должен пускать роутер мимо N9_offer_node — иначе
    # теряется сообщение "Отлично, {букет} — хороший выбор!" (P2)
    assert is_node_complete("N9_offer", {"budget_max": 7000}) is False


def test_n9_offer_complete_once_bouquet_chosen():
    assert is_node_complete("N9_offer", {"budget_max": 7000, "bouquet_name": "Букет Модена"}) is True


def test_n9_offer_incomplete_without_budget():
    assert is_node_complete("N9_offer", {}) is False


# ── is_below_minimum / P3 ──────────────────────────────────────────────────

def test_is_below_minimum_true():
    assert is_below_minimum({"budget_max": 2000}) is True


def test_is_below_minimum_false_at_threshold():
    assert is_below_minimum({"budget_max": MIN_ORDER_AMOUNT}) is False


def test_is_below_minimum_false_above():
    assert is_below_minimum({"budget_max": 8000}) is False


def test_is_below_minimum_no_budget():
    assert is_below_minimum({}) is False


# ── _find_bouquet_fuzzy / P2 ───────────────────────────────────────────────

def test_find_bouquet_exact_substring():
    match = _find_bouquet_fuzzy("модена", CATALOG)
    assert match is not None
    assert match["Название"] == "Букет Модена"


def test_find_bouquet_from_raw_sentence_fallback():
    # Имитирует случай P2: LLM не извлёк name_query, но текст содержит имя букета
    match = _find_bouquet_fuzzy("модена", CATALOG, preferred=[])
    assert match["Название"] == "Букет Модена"


def test_find_bouquet_prefers_preferred_pool():
    preferred = [CATALOG[1]]  # эустомы
    match = _find_bouquet_fuzzy("эустом", CATALOG, preferred=preferred)
    assert match["Название"] == "Букет из 19 белых эустом"


def test_find_bouquet_empty_query_returns_none():
    assert _find_bouquet_fuzzy("", CATALOG) is None
    assert _find_bouquet_fuzzy(None, CATALOG) is None


def test_find_bouquet_long_sentence_without_name_returns_none():
    # Полное сообщение без упоминания букета не должно давать ложных срабатываний
    text = "Хочу букет для мамы на юбилей, бюджет 8000"
    assert _find_bouquet_fuzzy(text, CATALOG) is None


def test_find_bouquet_typo_fuzzy_match():
    # Небольшая опечатка всё ещё должна находиться через difflib
    match = _find_bouquet_fuzzy("модина", CATALOG)
    assert match is not None
    assert match["Название"] == "Букет Модена"


# ── _is_real_value / GigaChat "null"-string quirk ──────────────────────────

def test_is_real_value_rejects_null_like_strings():
    for bad in ("null", "None", "NULL", "нет данных", "", "  ", "-"):
        assert FlowerLogic._is_real_value(bad) is False, repr(bad)


def test_is_real_value_rejects_none():
    assert FlowerLogic._is_real_value(None) is False


def test_is_real_value_accepts_real_data():
    assert FlowerLogic._is_real_value("+7 916 123-45-67") is True
    assert FlowerLogic._is_real_value(7000.0) is True
    assert FlowerLogic._is_real_value("Модена") is True


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
