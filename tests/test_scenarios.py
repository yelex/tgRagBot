"""
Автоматический тест FSM Floribot v2.

Запуск:
    PATH_BOUQUETS=/path/to/data/bouquets.json python tests/test_scenarios.py

Каждый сценарий — последовательность сообщений с ожидаемыми результатами.
Оценивается:
  - фаза после каждого шага (phase_prefix / phase_exact)
  - наличие сущностей в collected_data
  - строки в ответе (response_contains / response_not_contains)
  - бюджетный фильтр (prices_within_budget)
  - эскалация (escalated)
"""

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("PATH_BOUQUETS", os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "bouquets.json"
))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

from flower_logic import FlowerLogic


# ─────────────────────────────────────────────────────────────────────────────
# Структуры данных
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Step:
    input: str
    phase_prefix: Optional[str] = None
    phase_exact: Optional[str] = None
    collected_keys: List[str] = field(default_factory=list)
    collected_values: Dict[str, Any] = field(default_factory=dict)
    response_contains: List[str] = field(default_factory=list)
    response_not_contains: List[str] = field(default_factory=list)
    prices_within_budget: Optional[float] = None
    escalated: Optional[bool] = None
    note: str = ""


@dataclass
class Scenario:
    name: str
    steps: List[Step]


# ─────────────────────────────────────────────────────────────────────────────
# Сценарии
# ─────────────────────────────────────────────────────────────────────────────

SCENARIOS: List[Scenario] = [

    # ── 1. Happy path ─────────────────────────────────────────────────────────
    # Роутер видит бюджет + повод → прыгает сразу в N9_offer. Это корректно.
    Scenario("1. Happy path: полная информация → N9_offer → доставка → оплата", steps=[
        Step("Привет",
             phase_exact="N0_greet",
             response_contains=["Добрый день", "Flori Pacco"]),
        Step("Хочу букет для мамы на юбилей, бюджет 8000",
             phase_prefix="N9_offer",
             collected_keys=["budget_max"],
             prices_within_budget=8800,
             response_contains=["Вариант"],
             note="Бюджет+повод в одном → N9_offer, бюджет персистирован"),
        Step("Второй",
             phase_prefix="N10_delivery",
             note="Позиционный выбор 'второй'"),
        Step("Москва, Арбат 5",
             phase_prefix="N10_delivery",
             collected_keys=["address"]),
        Step("Завтра",
             phase_prefix="N10_delivery",
             collected_keys=["delivery_date"]),
        Step("В 15:00",
             phase_prefix="N10_delivery",
             collected_keys=["delivery_time"]),
        Step("+7 916 123-45-67",
             phase_prefix="payment",
             collected_keys=["recipient_phone"],
             response_contains=["#"],
             note="Сводка с номером заказа перед оплатой"),
    ]),

    # ── 2. Нет бюджета — бриф обязателен ─────────────────────────────────────
    Scenario("2. Только 'хочу букет' без бюджета → N2_brief", steps=[
        Step("Хочу букет",
             phase_prefix="N2_brief",
             note="Без бюджета — должен спросить"),
        Step("7000",
             phase_prefix="N9_offer",
             collected_keys=["budget_max"],
             prices_within_budget=7700,
             note="После бюджета — варианты"),
    ]),

    # ── 3. Бюджет '10000 рублей' = max_price ─────────────────────────────────
    Scenario("3. '10000 рублей' — max_price, цены в пределах бюджета", steps=[
        Step("Букет до 10000, коллеге",
             phase_prefix="N9_offer",
             collected_keys=["budget_max"],
             prices_within_budget=11000,
             response_not_contains=["99 000", "50 000"],
             note="Цены строго <= 11000 (10000 + 10%)"),
    ]),

    # ── 4. Жалоба прерывает любую фазу ───────────────────────────────────────
    Scenario("4. Жалоба из N9_offer → N11_complaint", steps=[
        Step("Хочу букет до 7000",
             phase_prefix="N9_offer"),
        Step("Букет из прошлого заказа завял через день",
             phase_prefix="N11_complaint",
             response_contains=["извин"],
             note="Жалоба должна переключить фазу независимо от контекста"),
    ]),

    # ── 5. Самовывоз ──────────────────────────────────────────────────────────
    Scenario("5. Самовывоз → N7_pickup", steps=[
        Step("Хочу забрать заказ сам",
             phase_prefix="N7_pickup",
             response_contains=["самовывоз", "предоплат"]),
    ]),

    # ── 6. Срочная доставка ───────────────────────────────────────────────────
    Scenario("6. Срочная доставка → N5_urgent", steps=[
        Step("Срочно нужен букет через 1.5 часа",
             phase_prefix="N5_urgent",
             response_contains=["срочн"]),
    ]),

    # ── 7. Корпоратив → эскалация ─────────────────────────────────────────────
    Scenario("7. Свадьба/корпоратив → N12_escalation + оператор", steps=[
        Step("Нам нужно оформление зала на свадьбу для 50 гостей",
             phase_prefix="N12_escalat",
             escalated=True,
             note="Крупное мероприятие → оператор с маркером ||escalate:||"),
    ]),

    # ── 8. Юрлицо в платёжном контексте ─────────────────────────────────────
    Scenario("8. Юрлицо во время оплаты → реквизиты + оператор", steps=[
        Step("Букет до 6000 коллеге",
             phase_prefix="N9_offer",
             prices_within_budget=6600),
        Step("Первый",
             phase_prefix="N10_delivery"),
        Step("Ленинградский проспект 37",
             phase_prefix="N10_delivery",
             collected_keys=["address"]),
        Step("Завтра в 12:00",
             note="Дата+время одним сообщением — должен пройти к следующему шагу"),
        Step("Не знаю телефон",
             phase_prefix="payment",
             note="Пропуск телефона → payment"),
        Step("У нас ООО, нам нужен счёт и договор",
             phase_prefix="N12_escalat",
             escalated=True,
             response_contains=["ИНН", "организац"],
             note="Любой вординг юрлица → реквизиты + оператор"),
    ]),

    # ── 9. Пропуск телефона ───────────────────────────────────────────────────
    Scenario("9. 'Не знаю телефон' → пропуск, переход к оплате", steps=[
        Step("Букет до 9000 для девушки",
             phase_prefix="N9_offer"),
        Step("Последний",
             phase_prefix="N10_delivery"),
        Step("Тверская 1",
             phase_prefix="N10_delivery",
             collected_keys=["address"]),
        Step("Завтра в 19:00",
             phase_prefix="N10_delivery",
             note="Дата+время: должен перейти к телефону"),
        Step("Не знаю телефон",
             phase_prefix="payment",
             note="Пропуск телефона — должен идти к оплате"),
    ]),

    # ── 10. Бюджет ниже минимума ──────────────────────────────────────────────
    Scenario("10. Бюджет < 5000 → предупреждение о минимуме", steps=[
        Step("Нужен букет за 2000 рублей",
             phase_prefix="N2_brief",
             response_contains=["5 000"],
             note="Бот должен объяснить минимальный заказ 5000₽ и вернуть в бриф"),
    ]),

    # ── 11. Размытые ответы — не зависать ────────────────────────────────────
    Scenario("11. Размытые ответы ('не знаю') — бот двигается дальше", steps=[
        Step("Нужен букет, не знаю какой",
             phase_prefix="N2_brief",
             note="Без бюджета — спрашивает"),
        Step("Около 10000",
             phase_prefix="N9_offer",
             collected_keys=["budget_max"],
             prices_within_budget=11000,
             note="После бюджета показывает варианты, не зависает"),
    ]),

    # ── 12. Каталог с диапазоном цен ─────────────────────────────────────────
    Scenario("12. Каталог 'покажи' + диапазон '5000-10000'", steps=[
        Step("Покажи что есть в каталоге",
             phase_prefix="N3_catalog",
             response_contains=["категори", "₽"]),
        Step("5000-10000",
             phase_prefix="N3_catalog",
             note="Диапазон должен парситься, цены <= 10000"),
    ]),

    # ── 13. FAQ во время диалога ──────────────────────────────────────────────
    Scenario("13. Вопрос о доставке → N8_faq", steps=[
        Step("Сколько стоит доставка за МКАД?",
             phase_prefix="N8_faq",
             response_contains=["МКАД", "₽"]),
    ]),

    # ── 14. Смена варианта ────────────────────────────────────────────────────
    Scenario("14. Выбор по имени букета в N9_offer", steps=[
        Step("Хочу букет до 8000 маме",
             phase_prefix="N9_offer",
             prices_within_budget=8800),
        Step("Букет Модена",
             phase_prefix="N10_delivery",
             collected_keys=["bouquet_name"],
             response_contains=["Модена"],
             note="Выбор по имени → N10, bouquet_name сохранён"),
    ]),

    # ── 15. Все данные доставки в одном сообщении ─────────────────────────────
    Scenario("15. Адрес + дата + время + телефон одним сообщением", steps=[
        Step("Хочу букет 7000 девушке",
             phase_prefix="N9_offer"),
        Step("Третий",
             phase_prefix="N10_delivery"),
        Step("Доставьте завтра в 14:00 на ул. Ленина 10, телефон +7 916 000-00-00",
             phase_prefix="payment",
             collected_keys=["address", "delivery_date", "delivery_time", "recipient_phone"],
             note="Всё сразу — должен пропустить уточняющие вопросы"),
    ]),

    # ── 16. Greeting только с приветствием ───────────────────────────────────
    Scenario("16. Только приветствие → greeting, затем бриф", steps=[
        Step("Привет",
             phase_exact="N0_greet",
             response_contains=["Добрый день"]),
        Step("Мне нужны розы",
             phase_prefix="N2_brief",
             note="После приветствия — бриф без бюджета"),
        Step("8000",
             phase_prefix="N9_offer",
             prices_within_budget=8800,
             note="После бюджета — варианты с ценами в пределах"),
    ]),

]


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ─────────────────────────────────────────────────────────────────────────────

def extract_phase(response: str) -> str:
    m = re.search(r'\|\|phase:([^\|]+)\|\|', response)
    return m.group(1) if m else ""


def extract_collected(response: str) -> Dict[str, Any]:
    # brace-counting чтобы правильно парсить вложенный JSON
    start = response.find("||data:{")
    if start == -1:
        return {}
    json_start = start + len("||data:")
    depth = 0
    for i, ch in enumerate(response[json_start:], json_start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(response[json_start:i + 1])
                except Exception:
                    return {}
    return {}


def extract_prices_from_response(text: str) -> List[float]:
    # Только конкретные цены букетов (формат "5 900 ₽"), не диапазоны категорий
    clean = re.sub(r'\|\|.*?\|\|', '', text)
    # Ищем паттерны типа "💰 5 900 ₽" или "— 5900 ₽"
    prices = []
    for m in re.finditer(r'(?:💰|—)\s*(\d[\d\s]{1,6})\s*₽', clean):
        try:
            prices.append(float(m.group(1).replace(" ", "").replace(" ", "")))
        except ValueError:
            pass
    return prices


def clean_response(text: str) -> str:
    return re.sub(r'\|\|[^|]+\|\|', '', text).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Оценщик
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StepResult:
    step_input: str
    response_clean: str
    phase: str
    collected: Dict[str, Any]
    checks: List[str]
    passed: bool


def evaluate_step(step: Step, response: str) -> StepResult:
    phase = extract_phase(response)
    collected = extract_collected(response)
    clean = clean_response(response)
    checks = []
    all_pass = True

    def ok(msg):  checks.append(f"  ✅ {msg}")
    def fail(msg):
        nonlocal all_pass
        all_pass = False
        checks.append(f"  ❌ {msg}")

    if step.phase_exact:
        if phase == step.phase_exact:
            ok(f"phase == '{phase}'")
        else:
            fail(f"phase_exact: ожидалось '{step.phase_exact}', получено '{phase}'")

    if step.phase_prefix:
        if phase.startswith(step.phase_prefix):
            ok(f"phase='{phase}' ∈ '{step.phase_prefix}*'")
        else:
            fail(f"phase_prefix: ожидалось '{step.phase_prefix}*', получено '{phase}'")

    for key in step.collected_keys:
        if collected.get(key) is not None:
            ok(f"collected['{key}'] = {str(collected[key])[:40]!r}")
        else:
            fail(f"collected['{key}'] отсутствует. collected={list(collected.keys())}")

    for key, expected in step.collected_values.items():
        actual = collected.get(key)
        if actual == expected:
            ok(f"collected['{key}'] == {expected}")
        else:
            fail(f"collected['{key}']: ожидалось {expected}, получено {actual}")

    for substr in step.response_contains:
        if substr.lower() in clean.lower():
            ok(f"response ∋ '{substr}'")
        else:
            fail(f"response НЕ содержит '{substr}'")

    for substr in step.response_not_contains:
        if substr.lower() not in clean.lower():
            ok(f"response ∌ '{substr}'")
        else:
            fail(f"response содержит запрещённое '{substr}'!")

    if step.prices_within_budget is not None:
        prices = extract_prices_from_response(response)
        violations = [p for p in prices if p > step.prices_within_budget]
        if not violations:
            ok(f"все цены {prices} <= {step.prices_within_budget}")
        else:
            fail(f"цены превышают бюджет {step.prices_within_budget}: {violations}")

    if step.escalated is not None:
        has = "||escalate:" in response
        if has == step.escalated:
            ok(f"escalated={step.escalated}")
        else:
            fail(f"escalated: ожидалось {step.escalated}, получено {has}")

    return StepResult(
        step_input=step.input,
        response_clean=clean[:200],
        phase=phase,
        collected=collected,
        checks=checks,
        passed=all_pass,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Раннер
# ─────────────────────────────────────────────────────────────────────────────

def run_scenario(fl: FlowerLogic, scenario: Scenario, verbose: bool = True) -> Dict:
    history = []
    user_id = abs(hash(scenario.name)) % 90000 + 10000
    step_results = []
    scenario_passed = True

    for i, step in enumerate(scenario.steps):
        history.append({"role": "user", "content": step.input})
        try:
            response = fl.get_bouquet_recommendation(
                user_input=step.input,
                user_id=user_id,
                user_name="test_user",
                conversation_history=history[:-1],
            )
        except Exception as e:
            response = f"||phase:ERROR|| ERROR: {e}"

        history.append({"role": "assistant", "content": response})
        result = evaluate_step(step, response)
        step_results.append(result)
        if not result.passed:
            scenario_passed = False

        if verbose:
            icon = "✅" if result.passed else "❌"
            note = f" [{step.note}]" if step.note else ""
            print(f"    Шаг {i+1}{note}: {icon} '{step.input[:55]}'")
            print(f"           phase={result.phase}  collected={list(result.collected.keys())}")
            if not result.passed or verbose:
                for c in result.checks:
                    if "❌" in c or "✅" in c:
                        print(f"         {c}")

    return {
        "scenario": scenario.name,
        "passed": scenario_passed,
        "steps": step_results,
        "total": len(step_results),
        "failed": sum(1 for r in step_results if not r.passed),
    }


def run_all(verbose: bool = True) -> None:
    print("🌸 Floribot FSM — автотест\n" + "=" * 70)
    fl = FlowerLogic()
    print("FlowerLogic готов.\n")

    results = []
    t0 = time.time()

    for i, sc in enumerate(SCENARIOS):
        if i > 0:
            time.sleep(3)  # пауза между сценариями — GigaChat rate limit ~30 req/min
        print(f"\n{'─'*70}")
        print(f"📋 {sc.name}")
        r = run_scenario(fl, sc, verbose=verbose)
        results.append(r)
        status = "✅ PASS" if r["passed"] else f"❌ FAIL ({r['failed']}/{r['total']} шагов)"
        print(f"   → {status}")

    elapsed = time.time() - t0
    passed = sum(1 for r in results if r["passed"])
    total_steps = sum(r["total"] for r in results)
    failed_steps = sum(r["failed"] for r in results)

    print(f"\n{'=' * 70}")
    print(f"📊 {passed}/{len(results)} сценариев  |  "
          f"{total_steps - failed_steps}/{total_steps} шагов  |  ⏱ {elapsed:.0f}с")

    if passed < len(results):
        print("\n❌ Провальные сценарии:")
        for r in results:
            if not r["passed"]:
                print(f"   • {r['scenario']}")
                for s in r["steps"]:
                    if not s.passed:
                        for c in s.checks:
                            if "❌" in c:
                                print(f"       {s.step_input[:40]!r}: {c.strip()}")
    else:
        print("✅ Все сценарии прошли!")

    report = {
        "summary": {
            "scenarios_passed": passed, "scenarios_total": len(results),
            "steps_passed": total_steps - failed_steps, "steps_total": total_steps,
            "elapsed_seconds": round(elapsed, 1),
        },
        "scenarios": [
            {"name": r["scenario"], "passed": r["passed"],
             "steps": [{"input": s.step_input, "phase": s.phase,
                        "passed": s.passed, "checks": s.checks,
                        "response": s.response_clean}
                       for s in r["steps"]]}
            for r in results
        ],
    }
    path = os.path.join(os.path.dirname(__file__), "last_run.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n📄 Отчёт: {path}")


if __name__ == "__main__":
    run_all(verbose="--quiet" not in sys.argv)
