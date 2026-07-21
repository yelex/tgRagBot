"""Единый источник истины о том, какие поля обязательны для каждой фазы FSM.

Используется роутером (guard в _nlu_node) и самими узлами графа, чтобы решение
"данные для фазы собраны" принималось по факту содержимого collected_data,
а не по пересказу истории чата LLM-роутером.
"""
from typing import Any, Dict, List

MIN_ORDER_AMOUNT = 5000

REQUIRED_FIELDS: Dict[str, List[str]] = {
    "N9_offer": ["budget_max"],
    "N10_delivery": ["address", "delivery_date", "delivery_time"],
}


def missing_fields(phase_node: str, collected: Dict[str, Any]) -> List[str]:
    """Список обязательных полей фазы, которых ещё нет в collected_data."""
    required = REQUIRED_FIELDS.get(phase_node, [])
    return [field for field in required if not collected.get(field)]


def is_node_complete(phase_node: str, collected: Dict[str, Any]) -> bool:
    """Действительно ли фаза выполнила свою работу и вправе передать
    управление дальше.

    В отличие от missing_fields() это не просто "все обязательные поля
    есть", а точное условие выхода из узла — для N10_delivery телефон
    опционален (клиент может нажать "Не знаю телефон"), поэтому одного
    списка обязательных полей недостаточно: без этой проверки роутер мог
    посчитать фазу завершённой сразу как только адрес/дата/время собраны,
    и перепрыгнуть мимо _N10_delivery_node в момент явного пропуска
    телефона — узел не успевал зафиксировать phone_skipped и создать заказ.
    """
    if missing_fields(phase_node, collected):
        return False
    if phase_node == "N10_delivery":
        return bool(collected.get("recipient_phone")) or bool(collected.get("phone_skipped"))
    if phase_node == "N9_offer":
        # Бюджет — не единственное условие: клиент ещё должен выбрать
        # букет. Это может сделать только сам N9_offer node (позиционный
        # или fuzzy-выбор по имени) — иначе роутер прыгает в N10_delivery
        # мимо сообщения "Отлично, {букет} — хороший выбор!" (P2).
        return bool(collected.get("bouquet_name"))
    return True


def is_below_minimum(collected: Dict[str, Any]) -> bool:
    """Бюджет клиента ниже минимального заказа (5000 ₽)."""
    budget = collected.get("budget_max")
    return bool(budget) and budget < MIN_ORDER_AMOUNT
