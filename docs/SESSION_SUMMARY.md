# Floribot v2 — итоги разработки (сессия 2026-06-03)

## Ветка

`feature/floribot-v2` (база: `optimize-glm-speed`)

---

## Что было сделано

### 1. Персистентность заказов (commit `942aae7`)

Добавлена таблица `orders` в MySQL со статусами жизненного цикла:

```
new → in_progress → delivery → delivered → feedback → closed / cancelled / escalated
```

- `interfaces/mysql_interface.py`: `create_order`, `update_order`, `get_active_order`, `get_order`
- `flower_logic.py`: хелперы `_upsert_order`, `_set_order_status`
- Хуки в FSM: N10_delivery создаёт заказ, N13_photo → `in_progress`, N14 → `delivery`, N15 → `delivered`
- `order_id` персистируется через `||data:{...}||` маркер в истории

### 2. Обработка входящих фото (commit `d5a55a9`)

- `tg_bot.py`: `filters.PHOTO` handler
- Если есть caption — обрабатывается как текст через FSM
- Без caption — просит описать словами (GigaChat не поддерживает vision)

### 3. Смена LLM: GLM → GigaChat (commit `29d4253`)

- GLM (`api.z.ai`) перестал работать: баланс исчерпан, старые модели не поддерживаются
- Интеграция через `langchain-gigachat`
- Модель: `GigaChat-2-Max` (задаётся `GIGACHAT_MODEL` в `.env`)
- Vision удалён полностью

### 4. Human handoff — уведомление оператора (commit `aba4836`)

- N12_escalation вшивает `||escalate:{json}||` маркер
- `tg_bot.py`: `_notify_operator` отправляет в `OPERATOR_CHAT_ID` форматированное сообщение:
  - причина эскалации, данные клиента, последние 5 сообщений, ссылка `tg://user?id=...`
- Триггеры: жалоба, конфликт, корпоратив, «позвать менеджера»

### 5. Фоновые задачи — фидбэк после доставки (commit `6e953aa`)

- `APScheduler` (`AsyncIOScheduler`) стартует вместе с ботом
- После фазы `N15_close`: задача на отправку фидбэка через `FEEDBACK_DELAY_HOURS` (дефолт 2ч)
- Инлайн-кнопки: «Да, всё отлично!» / «Есть вопрос»
- Негативный фидбэк → уведомление оператору

### 6. Сбор даты, времени, телефона в N10 (commit `83c555e`)

N10_delivery теперь пошагово собирает 4 поля:
1. Адрес (+ расчёт МКАД/за МКАД)
2. Дата доставки (кнопки: Сегодня / Завтра / Другая дата)
3. Время доставки
4. Телефон получателя (кнопка «Не знаю телефон»)

Перед оплатой показывается полная сводка с номером заказа.

NLU обновлён: извлекает `delivery_date`, `delivery_time`, `recipient_phone`.

### 7. LLM-рутинг вместо keyword matching (commit `3f9bc65`)

**Главное архитектурное изменение сессии.**

Заменён трёхслойный keyword-рутер на один вызов `with_structured_output(NLUDecision)`:

```python
class NLUDecision(BaseModel):
    next_phase: Literal["N0_greet", "N2_brief", "N3_catalog", ...]
    max_price: Optional[float]
    min_price: Optional[float]
    occasion: Optional[str]
    # ... все сущности
```

- Модель видит: текущую фазу + краткое резюме собранных данных + последние 6 сообщений
- Принимает решение о следующей фазе И извлекает сущности одним вызовом
- Удалено ~300 строк keyword-логики
- Аварийный fallback: keyword-роутер из 15 строк

**Тест 4 сценариев:**
- "Привет, хочу букет" → `N2_brief` ✅
- "7000 рублей" в N2_brief_budget → `N9_offer` + `max_price=7000` ✅
- "Завял букет" из N9_offer → `N11_complaint` ✅
- "Хочу Амели" в N9_offer → `N10_delivery` + `name_query=Амели` ✅

### 8. Исправления по результатам тестирования

| Коммит | Фикс |
|--------|------|
| `e589b48` | JSON-парсинг: brace-counting вместо жадного regex; позиционный выбор («последний», «вариант 2») |
| `e2631f6` | N0_greet: приветствие не пропускается; бюджет «X рублей» без «от» → max_price |

---

## Текущее состояние

Бот запускается локально:
```bash
PATH_BOUQUETS=/path/to/data/bouquets.json .venv/bin/python3 tg_bot.py
```

MySQL-таблицы: `messages` + `orders` — оба существуют на `95.142.42.28`.

`OPERATOR_CHAT_ID=7180426531` настроен в `.env`.

---

## Известные ограничения

| Пункт | Статус |
|-------|--------|
| APScheduler in-memory | Jobs теряются при рестарте. Для прода нужен `SQLAlchemyJobStore` |
| Vision | Не реализован (GigaChat не поддерживает) |
| "Сегодня/Завтра" → реальная дата | Хранится как строка, не нормализуется в дату |
| Upsell-узел | Не добавлен |
