# Floribot v2 — план дальнейших шагов

## Приоритет 1 — Критично для продакшена

### 1.1 APScheduler persistence
**Проблема:** при рестарте бота все запланированные фидбэки теряются.

```python
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore

jobstores = {
    'default': SQLAlchemyJobStore(url='mysql+mysqlconnector://user:pass@host/db')
}
scheduler = AsyncIOScheduler(jobstores=jobstores)
```

Также добавить `job_id` в таблицу `orders` чтобы уметь отменять задачу при эскалации.

### 1.2 Нормализация дат
**Проблема:** `delivery_date = "завтра"` хранится как строка — нельзя сортировать и сравнивать.

Нужен парсер: `"завтра"` → `datetime.date.today() + timedelta(1)`, `"15 июня"` → `datetime(2026, 6, 15)`.

Библиотека: `natasha` или `dateparser` (поддерживает русский).

### 1.3 Проверка что DB-операции не роняют бота
Текущие вызовы `create_order` / `update_order` синхронные внутри async-обработчика.
Нужно обернуть в `asyncio.run_in_executor` или перейти на `aiomysql`.

---

## Приоритет 2 — Важные бизнес-функции

### 2.1 Upsell-узел
После выбора букета в N9_offer, до перехода в N10_delivery:

```
"Отличный выбор! Если хотите — можем сделать его чуть пышнее.
 Текущий: Букет Модена — 5 900 ₽
 Версия M: +800 ₽ (больше объём)
 Версия L: +1 800 ₽ (добавим пионы)"
```

### 2.2 Relay-режим оператора
Сейчас при эскалации оператор получает уведомление но отвечает клиенту напрямую.
Нужен relay: сообщения оператора из рабочего чата пересылаются клиенту через бота.

Схема:
- Оператор отвечает на форвард в группе → бот парсит `reply_to_message` → пересылает клиенту
- Кнопка «Завершить» возвращает управление боту

### 2.3 Статус заказа по запросу
Клиент пишет «где мой заказ» → бот показывает статус из `orders` по `user_id`.

```python
# intent routing: "status" → N12_escalation сейчас
# нужно: "status" → новый N_status узел
order = get_active_order(user_id)
"Ваш заказ #42 сейчас в статусе: Доставка 🚚"
```

### 2.4 Повторный заказ
`repeat_order` intent → загрузить последний `order` клиента → предзаполнить collected_data → перейти сразу в N10_delivery.

---

## Приоритет 3 — Качество и надёжность

### 3.1 Тесты FSM переходов
```python
# pytest + VCR для GigaChat
def test_greeting_to_brief():
    fl = FlowerLogic()
    decision = fl._predict_routing_decision(state_with("Привет"))
    assert decision.next_phase == "N0_greet"

def test_budget_is_max_price():
    decision = fl._predict_routing_decision(state_with("10000 рублей"))
    assert decision.max_price == 10000
    assert decision.min_price is None
```

### 3.2 NLU кэш с TTL
Текущий кэш `nlu_cache` растёт без ограничений. Нужен TTL или LRU.

### 3.3 Rate limiting GigaChat
Сейчас `time.sleep(0.5)` после каждого вызова. При высокой нагрузке — bottleneck.
Нужен `asyncio.Semaphore` и retry с exponential backoff.

### 3.4 Логирование в формате JSON
Для Loki/Grafana:
```python
import structlog
log = structlog.get_logger()
log.info("routing_decision", phase=decision.next_phase, uid=user_id)
```

---

## Приоритет 4 — Будущие функции

### 4.1 Каталог в базе данных
Сейчас `bouquets.json` — статичный файл. Нужна таблица `bouquets` с возможностью обновления через админку или импорт из CMS (Tilda, Ecwid).

### 4.2 Vision через внешний API
GigaChat не поддерживает vision. Варианты:
- OpenRouter (Claude Haiku / GPT-4o Vision) — для анализа фото букетов от клиентов
- Только для сценария «клиент прислал референс»

### 4.3 Интеграция с платёжной системой
Сейчас бот говорит «оплатите переводом» — нет проверки оплаты.
- ЮKassa или Robokassa → webhook → `update_order(order_id, payment_status="paid")`
- Telegram Payments API

### 4.4 Админ-панель
Простая веб-страница для оператора:
- Список активных заказов с фильтром по статусу
- Обновление статуса заказа
- История переписки с клиентом

---

## Техдолг

| Файл | Проблема |
|------|----------|
| `flower_logic.py` | `_N9_confirm_node` — мёртвый код, не подключён к графу |
| `flower_logic.py` | `ALLOWED_INTENTS`, `_normalize_intent` — устарели после перехода на structured routing |
| `tg_bot.py` | `_sanitize_for_markdown` — не используется (parse_mode убран) |
| `flower_logic.py` | `time.sleep(0.5)` в sync коде внутри async — блокирует event loop |
