import json
import logging
import os
import re
from typing import Any, Dict, List, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_gigachat.chat_models import GigaChat
from langgraph.graph import END, START, StateGraph

from interfaces import mysql_interface

load_dotenv()
logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    user_input: str
    user_id: Optional[int]
    user_name: Optional[str]
    conversation_history: List[Dict[str, str]]
    full_conversation_text: str
    bouquets_data: List[Dict[str, Any]]
    intent: Literal[
        "greet", "greet_and_offer", "catalog", "recommend", "chosen_by_name",
        "delivery_info", "payment_info", "complaint", "urgent", "corporate",
        "occasion", "photo_request", "change_order", "cancel_order", "repeat_order",
        "contact_owner", "faq", "unknown"
    ]
    entities: Dict[str, Any]
    response: str


class FlowerLogic:
    def __init__(self):
        self.PATH_BOUQUETS = os.getenv("PATH_BOUQUETS")
        self.AUTHORIZATION_KEY = os.getenv("AUTHORIZATION_KEY")
        self.bouquets_info: str = ""
        self.bouquets_data: List[Dict[str, Any]] = []
        self.intent_llm: Optional[GigaChat] = None
        self.graph = None
        self._last_bouquets: Dict[int, List[Dict[str, Any]]] = {}
        self.initialize_components()

    def load_bouquets_data(self):
        """Загружает данные о букетах из JSON."""
        with open(self.PATH_BOUQUETS, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.bouquets_data = data
        bouquets_info = []
        for bouquet in data:
            bouquets_info.append(
                f"Название: {bouquet['Название']}, Цена: {bouquet['Цена']} руб."
            )
        self.bouquets_info = "\n".join(bouquets_info)

    def initialize_components(self):
        """Инициализация минимального LangGraph-агента без RAG."""
        print("🌸 Инициализация LangGraph-агента...")
        self.load_bouquets_data()
        self._initialize_intent_llm()
        self.graph = self._build_graph()
        print("✅ LangGraph-агент готов!")

    def _initialize_intent_llm(self):
        if not self.AUTHORIZATION_KEY:
            logger.warning("AUTHORIZATION_KEY не задан, intent LLM отключен.")
            self.intent_llm = None
            return
        try:
            self.intent_llm = GigaChat(
                verify_ssl_certs=False,
                credentials=self.AUTHORIZATION_KEY,
                model="GigaChat-2-Max",
            )
            logger.info("Intent LLM инициализирован.")
        except Exception as exc:
            logger.exception("Не удалось инициализировать intent LLM: %s", exc)
            self.intent_llm = None

    def _build_graph(self):
        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("nlu", self._nlu_node)
        graph_builder.add_node("greet", self._greet_node)
        graph_builder.add_node("offer", self._offer_node)
        graph_builder.add_node("delivery", self._delivery_node)
        graph_builder.add_node("payment", self._payment_node)
        graph_builder.add_node("complaint", self._complaint_node)
        graph_builder.add_node("urgent", self._urgent_node)
        graph_builder.add_node("corporate", self._corporate_node)
        graph_builder.add_node("occasion", self._occasion_node)
        graph_builder.add_node("faq", self._faq_node)
        graph_builder.add_node("contact_owner", self._contact_owner_node)
        graph_builder.add_node("photo_request", self._photo_request_node)
        graph_builder.add_node("change_order", self._change_order_node)
        graph_builder.add_node("cancel_order", self._cancel_order_node)
        graph_builder.add_node("repeat_order", self._repeat_order_node)
        graph_builder.add_node("unknown", self._unknown_node)

        graph_builder.add_edge(START, "nlu")
        graph_builder.add_conditional_edges(
            "nlu",
            self._route_from_nlu,
            {
                "greet": "greet",
                "offer": "offer",
                "greet_and_offer": "greet",
                "catalog": "offer",
                "recommend": "offer",
                "chosen_by_name": "offer",
                "delivery_info": "delivery",
                "payment_info": "payment",
                "complaint": "complaint",
                "urgent": "urgent",
                "corporate": "corporate",
                "occasion": "occasion",
                "photo_request": "photo_request",
                "change_order": "change_order",
                "cancel_order": "cancel_order",
                "repeat_order": "repeat_order",
                "faq": "faq",
                "contact_owner": "contact_owner",
                "unknown": "unknown",
            },
        )
        graph_builder.add_edge("greet", END)
        graph_builder.add_edge("offer", END)
        graph_builder.add_edge("delivery", END)
        graph_builder.add_edge("payment", END)
        graph_builder.add_edge("complaint", END)
        graph_builder.add_edge("urgent", END)
        graph_builder.add_edge("corporate", END)
        graph_builder.add_edge("occasion", END)
        graph_builder.add_edge("faq", END)
        graph_builder.add_edge("contact_owner", END)
        graph_builder.add_edge("photo_request", END)
        graph_builder.add_edge("change_order", END)
        graph_builder.add_edge("cancel_order", END)
        graph_builder.add_edge("repeat_order", END)
        graph_builder.add_edge("unknown", END)

        return graph_builder.compile()

    def _user_context(self, state: AgentState) -> str:
        return f"user_id={state.get('user_id')} user_name={state.get('user_name')}"

    def _route_from_nlu(self, state: AgentState) -> str:
        intent = state["intent"]
        logger.info(
            "Agent route decision: intent=%s %s",
            intent,
            self._user_context(state),
        )
        allowed_intents = {
            "greet", "greet_and_offer", "catalog", "recommend", "chosen_by_name",
            "delivery_info", "payment_info", "complaint", "urgent", "corporate",
            "occasion", "photo_request", "change_order", "cancel_order", "repeat_order",
            "contact_owner", "faq", "unknown"
        }
        if intent in allowed_intents:
            return intent
        logger.warning(
            "Unexpected intent=%s, fallback to unknown %s",
            intent,
            self._user_context(state),
        )
        return "unknown"

    def _normalize_intent(self, raw_intent: str) -> AgentState["intent"]:
        allowed = {
            "greet", "greet_and_offer", "catalog", "recommend", "chosen_by_name",
            "delivery_info", "payment_info", "complaint", "urgent", "corporate",
            "occasion", "photo_request", "change_order", "cancel_order", "repeat_order",
            "contact_owner", "faq", "unknown"
        }
        intent = (raw_intent or "").strip().lower()
        if intent in allowed:
            return intent  # type: ignore[return-value]
        return "unknown"

    def _predict_nlu_with_llm(self, state: AgentState) -> Dict[str, Any]:
        if self.intent_llm is None:
            return {"intent": "unknown", "entities": {}}

        prompt = f"""
Ты NLU-модуль для цветочного Telegram-бота (darkstore, Москва, только доставка).
Определи intent и извлеки сущности из последнего сообщения с учетом истории.

Доступные intent:
- greet — только приветствие, без конкретного запроса
- greet_and_offer — приветствие + запрос букета
- catalog — просьба показать каталог/все варианты
- recommend — просьба подобрать/порекомендовать по бюджету/описанию
- chosen_by_name — запрос конкретного букета по названию
- delivery_info — вопросы о доставке (стоимость, время, зоны, условия)
- payment_info — вопросы об оплате (способы, предоплата, юрлица)
- complaint — претензия/жалоба (завял, не понравилось, проблема с качеством)
- urgent — срочная доставка ("срочно", "побыстрее", "нужно через час")
- corporate — корпоративный заказ (на мероприятие, в офис, много букетов)
- occasion — подбор по поводу (свадьба, день рождения, извинения, юбилей)
- photo_request — запрос фото букета перед отправкой
- change_order — изменить заказ (адрес, время, состав)
- cancel_order — отменить заказ
- repeat_order — повторный заказ ("как в прошлый раз", "повторите мой заказ")
- contact_owner — просьба связать с владельцем/мастером, сложный вопрос
- faq — вопросы о политике (возврат, гарантия свежести, минимальный заказ, анонимность)
- unknown — если невозможно уверенно классифицировать

Правила:
- Если пользователь спрашивает "сколько стоит доставка", "доставляете ли в X" — delivery_info
- Если говорит "срочно", "нужно быстро", "доставьте за час" — urgent
- Если спрашивает "как оплатить", "принимаете ли карты", "можно наличными" — payment_info
- Если жалуется, что цветы завяли, не понравились — complaint
- Если хочет букет на свадьбу, день рождения, для мамы, для любимой — occasion
- Если просит повторить прошлый заказ или "как в прошлый раз" — repeat_order

Извлечение цен (критично):
- "до X тыс" — max_price = X*1000
- "от X тыс" — min_price = X*1000
- "от X до Y тыс" — min_price = X*1000, max_price = Y*1000
- "не дороже X" — max_price = X
- Числа вроде "хочу 101 розу" — это количество, НЕ бюджет

Ответ верни строго JSON без пояснений:
{{
  "intent": "one_of_allowed_values",
  "entities": {{
    "max_price": number|null,
    "min_price": number|null,
    "name_query": "string|null",
    "query_text": "normalized user request",
    "occasion": "string|null",
    "urgent": true|false|null,
    "complaint_type": "wilted|bad_looking|wrong_flowers|delivery|other|null",
    "corporate_size": number|null,
    "change_request": "address|time|composition|other|null",
    "photo_request": true|false|null,
    "repeat_reference": "string|null"
  }}
}}

История:
{state["full_conversation_text"]}

Последнее сообщение:
{state["user_input"]}
""".strip()

        try:
            llm_result = self.intent_llm.invoke(prompt)
            content = getattr(llm_result, "content", "") if llm_result is not None else ""
            if isinstance(content, list):
                content = " ".join(str(item) for item in content)
            if not isinstance(content, str):
                content = str(content)

            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            payload = json.loads(match.group(0) if match else content)
            intent = self._normalize_intent(str(payload.get("intent", "")))
            entities_raw = payload.get("entities", {}) if isinstance(payload, dict) else {}
            if not isinstance(entities_raw, dict):
                entities_raw = {}

            entities: Dict[str, Any] = {}

            # Извлечение цен
            max_price = entities_raw.get("max_price")
            if isinstance(max_price, (int, float)):
                entities["max_price"] = float(max_price)
            elif isinstance(max_price, str):
                try:
                    entities["max_price"] = float(max_price.replace(",", ".").strip())
                except ValueError:
                    pass

            min_price = entities_raw.get("min_price")
            if isinstance(min_price, (int, float)):
                entities["min_price"] = float(min_price)
            elif isinstance(min_price, str):
                try:
                    entities["min_price"] = float(min_price.replace(",", ".").strip())
                except ValueError:
                    pass

            name_query = entities_raw.get("name_query")
            if isinstance(name_query, str) and name_query.strip():
                entities["name_query"] = name_query.lower()

            query_text = entities_raw.get("query_text")
            if isinstance(query_text, str) and query_text.strip():
                entities["query_text"] = query_text
            else:
                entities["query_text"] = state["user_input"].lower()

            # Дополнительные сущности
            for key in ("occasion", "urgent", "complaint_type", "corporate_size",
                        "change_request", "photo_request", "repeat_reference"):
                val = entities_raw.get(key)
                if val is not None:
                    entities[key] = val

            logger.info(
                "NLU predicted by LLM: intent=%s entities=%s %s",
                intent,
                entities,
                self._user_context(state),
            )
            return {"intent": intent, "entities": entities}
        except Exception as exc:
            logger.exception("NLU LLM prediction failed, fallback to rules: %s", exc)
            return {"intent": "unknown", "entities": {}}

    def _fallback_intent_from_rules(self, state: AgentState) -> AgentState["intent"]:
        text = state["user_input"].lower()

        # Проверка на приветствие
        has_greeting = any(marker in text for marker in ("привет", "здравствуйте", "добрый", "hello", "hi"))

        # Проверка на запрос каталога
        has_catalog = any(marker in text for marker in ("каталог", "все букеты", "покажи все"))

        # Проверка на запрос букета/цветов
        has_flower_request = any(token in text for token in ("букет", "цвет", "роза", "роз", "пион", "тюльпан", "хризантем", "лили", "гортенз", "эустом", "подсолну"))

        # Проверка на вопросы о доставке
        has_delivery_question = any(token in text for token in (
            "достав", "привезти", "везете", "мкад", "москв", "подмосков",
        ))

        # Проверка на оплату
        has_payment_question = any(token in text for token in (
            "оплат", "деньги", "перевод", "наличн", "карт", "эквайринг",
            "счет", "юрлиц", "нал", "безнал",
        ))

        # Проверка на претензию
        has_complaint = any(token in text for token in (
            "завял", "увял", "не свеж", "плох", "не нравит", "недовол",
            "жалоб", "претенз", "возврат", "брак",
        ))

        # Проверка на срочность
        has_urgent = any(token in text for token in (
            "срочн", "быстр", "побыстре", "через час", "сейчас", "быстро",
        ))

        # Проверка на корпоративный запрос
        has_corporate = any(token in text for token in (
            "корпоратив", "офис", "мероприятие", "свадьб", "много букет",
            "оптом", "компани",
        ))

        # Проверка на подбор по поводу
        has_occasion = any(token in text for token in (
            "свадьб", "день рожд", "юбилей", "мам", "любим", "девушк",
            "жен", "коллег", "учител", "врач", "извин",
        ))

        # Проверка на запрос фото
        has_photo_request = any(token in text for token in (
            "фото", "покажи", "пришли",
        ))

        # Проверка на изменение заказа
        has_change = any(token in text for token in (
            "измен", "поменя", "передвин", "перенес", "другой адрес",
        ))

        # Проверка на отмену
        has_cancel = any(token in text for token in ("отмен", "аннулир"))

        # Проверка на повторный заказ
        has_repeat = any(token in text for token in (
            "повтор", "как в прошл", "еще раз", "снова", "опять",
        ))

        # Проверка на вопросы о политике (FAQ)
        has_faq = any(token in text for token in (
            "минимальн", "сколько стоит", "гаранти", "свежест", "анонимн",
            "открытк", "ваз", "сладк",
        ))

        # Проверка на просьбу соединить с владельцем
        has_contact_owner = any(token in text for token in (
            "позов", "владельц", "хозяин", "поговорить", "свяж", "менеджер",
            "соедин", "позвонить",
        ))

        # Приоритетная проверка
        if has_cancel:
            return "cancel_order"
        if has_complaint:
            return "complaint"
        if has_contact_owner:
            return "contact_owner"
        if has_urgent and has_flower_request:
            return "urgent"
        if has_corporate:
            return "corporate"
        if has_delivery_question:
            return "delivery_info"
        if has_payment_question:
            return "payment_info"
        if has_occasion:
            return "occasion"
        if has_repeat:
            return "repeat_order"
        if has_change:
            return "change_order"
        if has_photo_request and (has_flower_request or has_catalog):
            return "photo_request"
        if has_faq:
            return "faq"
        if has_catalog and has_greeting:
            return "greet_and_offer"
        if has_catalog:
            return "catalog"
        if has_greeting and has_flower_request:
            return "greet_and_offer"
        if has_greeting:
            return "greet"
        if has_flower_request:
            return "recommend"

        return "unknown"

    def _extract_price_from_text(self, text: str) -> Dict[str, Optional[float]]:
        """Fallback-извлечение min_price/max_price из текста через regex."""
        result: Dict[str, Optional[float]] = {}
        text_lower = text.lower()

        # Сначала ищем "от X до Y тыс" — комбинацию обоих цен
        m = re.search(
            r'(?:^|\s)от\s+(\d+(?:[.,]\d+)?)\s+до\s+(\d+(?:[.,]\d+)?)\s*тыс(?:яч)?',
            text_lower,
        )
        if m:
            result["min_price"] = float(m.group(1).replace(",", ".")) * 1000
            result["max_price"] = float(m.group(2).replace(",", ".")) * 1000
            return result

        # "от X до Y" (без тыс)
        m = re.search(r'(?:^|\s)от\s+(\d+(?:[.,]\d+)?)\s+до\s+(\d+(?:[.,]\d+)?)', text_lower)
        if m:
            result["min_price"] = float(m.group(1).replace(",", "."))
            result["max_price"] = float(m.group(2).replace(",", "."))
            return result

        # "от X тыс" (с умножением) — min_price
        m = re.search(r'(?:^|\s)от\s+(\d+(?:[.,]\d+)?)\s*тыс(?:яч)?', text_lower)
        if m:
            result["min_price"] = float(m.group(1).replace(",", ".")) * 1000

        # "до X тыс" (с умножением) — max_price (если ещё не нашли выше)
        if "max_price" not in result:
            m = re.search(r'(?:^|\s)до\s+(\d+(?:[.,]\d+)?)\s*тыс(?:яч)?', text_lower)
            if m:
                result["max_price"] = float(m.group(1).replace(",", ".")) * 1000

        # "от X" (без тыс) — min_price
        if "min_price" not in result:
            m = re.search(r'(?:^|\s)от\s+(\d+(?:[.,]\d+)?)', text_lower)
            if m:
                result["min_price"] = float(m.group(1).replace(",", "."))

        # "до X" (без тыс) — max_price
        if "max_price" not in result:
            m = re.search(r'(?:^|\s)до\s+(\d+(?:[.,]\d+)?)', text_lower)
            if m:
                result["max_price"] = float(m.group(1).replace(",", "."))

        # "не дороже X тыс", "в пределах X тыс" — max_price
        if "max_price" not in result:
            m = re.search(r'(?:не дороже|в пределах|не больше)\s+(\d+(?:[.,]\d+)?)\s*тыс(?:яч)?', text_lower)
            if m:
                result["max_price"] = float(m.group(1).replace(",", ".")) * 1000

        # "не дороже X", "в пределах X" — max_price
        if "max_price" not in result:
            m = re.search(r'(?:не дороже|в пределах|не больше)\s+(\d+(?:[.,]\d+)?)', text_lower)
            if m:
                result["max_price"] = float(m.group(1).replace(",", "."))

        # "не меньше X тыс", "начиная от X тыс" — min_price
        if "min_price" not in result:
            m = re.search(r'(?:не меньше|начиная от)\s+(\d+(?:[.,]\d+)?)\s*тыс(?:яч)?', text_lower)
            if m:
                result["min_price"] = float(m.group(1).replace(",", ".")) * 1000

        # "не меньше X", "начиная от X" — min_price
        if "min_price" not in result:
            m = re.search(r'(?:не меньше|начиная от)\s+(\d+(?:[.,]\d+)?)', text_lower)
            if m:
                result["min_price"] = float(m.group(1).replace(",", "."))

        return result

    def _nlu_node(self, state: AgentState) -> AgentState:
        logger.info("Agent node enter: nlu %s", self._user_context(state))
        nlu_result = self._predict_nlu_with_llm(state)
        intent = self._normalize_intent(str(nlu_result.get("intent", "unknown")))
        entities: Dict[str, Any] = nlu_result.get("entities", {})
        if not isinstance(entities, dict):
            entities = {}

        if intent == "unknown":
            intent = self._fallback_intent_from_rules(state)
            logger.info("Intent fallback used: %s %s", intent, self._user_context(state))

        if "query_text" not in entities:
            entities["query_text"] = state["user_input"].lower()

        # Fallback-извлечение min_price/max_price из текста, если LLM не извлёк
        if entities.get("min_price") is None and entities.get("max_price") is None:
            price_fallback = self._extract_price_from_text(state["user_input"])
            for key, val in price_fallback.items():
                if val is not None and key not in entities:
                    entities[key] = val
                    logger.info(
                        "Price fallback extracted: %s=%s %s",
                        key, val, self._user_context(state),
                    )

        if intent == "chosen_by_name" and "name_query" not in entities:
            intent = "recommend"

        state["intent"] = intent
        state["entities"] = entities
        logger.info(
            "Agent node exit: nlu, intent=%s, entities=%s history_items=%s %s",
            intent,
            entities,
            len(state["conversation_history"]),
            self._user_context(state),
        )
        return state

    def _greet_node(self, state: AgentState) -> AgentState:
        logger.info("Agent node enter: greet %s", self._user_context(state))

        # Если это greet_and_offer, сразу переводим в offer с приветствием
        if state["intent"] == "greet_and_offer":
            # Формируем response через offer-логику
            offer_state = self._offer_node(state)
            if offer_state["response"]:
                state["response"] = f"Здравствуйте!\n\n{offer_state['response']}"
                logger.info("Agent node exit: greet (with offer) %s", self._user_context(state))
                return state

        state["response"] = (
            "🌸 *Добрый день!* Рады приветствовать в нашей цветочной мастерской.\n\n"
            "Я — ИИ-помощник. Могу помочь:\n"
            "• Подобрать букет по бюджету и пожеланиям\n"
            "• Рассказать о доставке (Москва и область, 24/7)\n"
            "• Ответить на вопросы об оплате\n"
            "• Оформить срочный заказ\n\n"
            "Напишите, что вас интересует, например:\n"
            "— *«Нужен букет до 12 000 рублей»*\n"
            "— *«Сколько стоит доставка в Подмосковье?»*\n"
            "— *«Хочу срочно букет роз через 2 часа»*"
        )
        logger.info("Agent node exit: greet %s", self._user_context(state))
        return state

    @staticmethod
    def _pluralize_variant(n: int) -> str:
        if 11 <= n % 100 <= 14:
            return "вариантов"
        if n % 10 == 1:
            return "вариант"
        if n % 10 in (2, 3, 4):
            return "варианта"
        return "вариантов"

    def _format_short_offer(self, bouquets: List[Dict[str, Any]], max_items: int = 10) -> str:
        lines = ["Подобрал варианты:"]
        total = len(bouquets)
        show_count = min(total, max_items)
        for bouquet in bouquets[:show_count]:
            lines.append(
                f"- {bouquet['Название']} — {int(bouquet['Цена'])} руб.\n"
                f"  {bouquet['Ссылка']}"
            )
        if total > max_items:
            remaining = total - max_items
            variant = self._pluralize_variant(remaining)
            lines.append(f"\n||more:{remaining}:{max_items}||\nХотите посмотреть ещё {remaining} {variant}?")
        return "\n".join(lines)

    def _format_bouquets_page(self, bouquets: List[Dict[str, Any]], start: int, count: int) -> str:
        """Показывает страницу букетов начиная с индекса start."""
        lines = ["Ещё варианты:"]
        for bouquet in bouquets[start:start + count]:
            lines.append(
                f"- {bouquet['Название']} — {int(bouquet['Цена'])} руб.\n"
                f"  {bouquet['Ссылка']}"
            )
        return "\n".join(lines)

    def _search_bouquets_by_query(self, bouquets: List[Dict[str, Any]], query_text: str) -> List[Dict[str, Any]]:
        words = re.findall(r"[a-zA-Zа-яА-Я0-9]+", query_text.lower())
        stop_words = {
            "привет", "здравствуйте", "добрый", "день", "вечер", "утро", "хочу", "нужен",
            "нужна", "нужно", "подберите", "покажи", "покажи", "варианты", "букет", "букеты",
            "до", "от", "руб", "рублей", "тыс", "тысяч", "пожалуйста",
        }
        tokens = [w for w in words if w not in stop_words and len(w) > 1]
        if not tokens:
            return []

        scored: List[tuple[int, Dict[str, Any]]] = []
        for bouquet in bouquets:
            name = bouquet["Название"].lower()
            score = sum(1 for token in tokens if token in name)
            if score > 0:
                scored.append((score, bouquet))

        scored.sort(key=lambda item: (-item[0], item[1]["Цена"]))
        return [item[1] for item in scored]

    def _offer_node(self, state: AgentState) -> AgentState:
        logger.info(
            "Agent node enter: offer, intent=%s %s",
            state["intent"],
            self._user_context(state),
        )
        intent = state["intent"]
        entities = state["entities"]
        bouquets = state["bouquets_data"]
        user_id = state.get("user_id")

        if intent == "catalog":
            sorted_bouquets = sorted(bouquets, key=lambda b: b["Цена"])
            if user_id is not None:
                self._last_bouquets[user_id] = sorted_bouquets
            state["response"] = self._format_short_offer(sorted_bouquets)
            logger.info("Agent node exit: offer, branch=catalog %s", self._user_context(state))
            return state

        if intent == "chosen_by_name":
            query = entities.get("name_query", "")
            # Токен-матчинг: разбиваем запрос на слова, проверяем что каждое
            # слово присутствует хотя бы в одном из названий букета
            query_tokens = set(re.findall(r"[a-zA-Zа-яА-Я0-9]+", query.lower()))
            scored = []
            for b in bouquets:
                name_tokens = set(re.findall(r"[a-zA-Zа-яА-Я0-9]+", b["Название"].lower()))
                matches = len(query_tokens & name_tokens)
                if matches > 0:
                    scored.append((matches, b))
            scored.sort(key=lambda x: (-x[0], x[1]["Цена"]))
            found = [b for _, b in scored]
            if user_id is not None:
                self._last_bouquets[user_id] = found
            if found:
                state["response"] = self._format_short_offer(found)
            else:
                state["response"] = (
                    "Не нашёл букет по точному названию. "
                    "Могу подобрать 2-3 варианта по бюджету."
                )
            logger.info(
                "Agent node exit: offer, branch=chosen_by_name, found=%s %s",
                bool(found),
                self._user_context(state),
            )
            return state

        if intent in ("recommend", "greet_and_offer"):
            max_price = entities.get("max_price")
            min_price = entities.get("min_price")

            # Если есть хотя бы один ценовой фильтр
            if max_price is not None or min_price is not None:
                filtered = bouquets
                if max_price is not None:
                    filtered = [b for b in filtered if b["Цена"] <= max_price]
                if min_price is not None:
                    filtered = [b for b in filtered if b["Цена"] >= min_price]

                # Дополнительно фильтруем по текстовому запросу, если он есть
                query_text = entities.get("query_text", state["user_input"])
                text_filtered = self._search_bouquets_by_query(filtered, query_text)
                if text_filtered:
                    filtered = text_filtered

                if not filtered:
                    parts = []
                    if min_price is not None:
                        parts.append(f"от {int(min_price)}")
                    if max_price is not None:
                        parts.append(f"до {int(max_price)}")
                    price_desc = " ".join(parts)
                    cheapest = sorted(bouquets, key=lambda b: b["Цена"])[:10]
                    if user_id is not None:
                        self._last_bouquets[user_id] = cheapest
                    state["response"] = (
                        f"В диапазоне {price_desc} руб. вариантов не нашёл. "
                        "Могу показать ближайшие по цене."
                    )
                    state["response"] += "\n\n" + self._format_short_offer(cheapest)
                    logger.info(
                        "Agent node exit: offer, branch=%s, within_budget=0 %s",
                        intent,
                        self._user_context(state),
                    )
                    return state

                filtered = sorted(filtered, key=lambda b: b["Цена"], reverse=True)
                if user_id is not None:
                    self._last_bouquets[user_id] = filtered
                state["response"] = self._format_short_offer(filtered)
                logger.info(
                    "Agent node exit: offer, branch=%s, within_budget=%s %s",
                    intent,
                    len(filtered),
                    self._user_context(state),
                )
                return state

            query_text = entities.get("query_text", state["user_input"])
            relevant = self._search_bouquets_by_query(bouquets, query_text)
            if relevant:
                if user_id is not None:
                    self._last_bouquets[user_id] = relevant
                state["response"] = self._format_short_offer(relevant[:3])
                logger.info(
                    "Agent node exit: offer, branch=%s, matched_by_query=%s %s",
                    intent,
                    len(relevant),
                    self._user_context(state),
                )
                return state

            top_items = sorted(bouquets, key=lambda b: b["Цена"])[:3]
            state["response"] = self._format_short_offer(top_items)
            logger.info(
                "Agent node exit: offer, branch=%s, fallback=cheapest %s",
                intent,
                self._user_context(state),
            )
            return state

        state["response"] = (
            "Уточните, какой бюджет или какие цветы хотите. "
            "Например: 'Пионы до 15000'."
        )
        logger.info("Agent node exit: offer %s", self._user_context(state))
        return state

    def _delivery_node(self, state: AgentState) -> AgentState:
        logger.info("Agent node enter: delivery %s", self._user_context(state))
        state["response"] = (
            "🚚 *Условия доставки:*\n\n"
            "📍 *Зона:* Москва и ближайшее Подмосковье. В теории — куда угодно, вопрос цены.\n\n"
            "💰 *Стоимость:*\n"
            "• В пределах МКАД — 600 ₽\n"
            "• За МКАД — 600 ₽ + 50 ₽/км\n\n"
            "⏰ *Время:*\n"
            "• Доставка 24/7\n"
            "• Приём заказов с 9:00 до 23:00\n"
            "• Стандартно: от заказа до доставки 1,5–2 часа\n"
            "• Возможна доставка ко времени\n\n"
            "⚡ *Срочная доставка:*\n"
            "• Букет 10–15 тыс. ₽ — 1,5–2 часа с момента заказа\n\n"
            "📦 *Минимальный заказ:* от 5 000 ₽\n\n"
            "Дополнительные вопросы по доставке? Напишите адрес — уточню стоимость."
        )
        logger.info("Agent node exit: delivery %s", self._user_context(state))
        return state

    def _payment_node(self, state: AgentState) -> AgentState:
        logger.info("Agent node enter: payment %s", self._user_context(state))
        state["response"] = (
            "💳 *Способы оплаты:*\n\n"
            "• 💸 Перевод на карту (предпочтительно)\n"
            "• 🔗 Ссылка на оплату\n"
            "• 💵 Наличные при получении\n"
            "• 🏢 Работаем с юрлицами (счёт, акт)\n\n"
            "*Условия:*\n"
            "• Для новых клиентов — предоплата\n"
            "• Постоянным/по рекомендации — оплата при получении\n"
            "• Можно поставить заказ в работу до оплаты\n"
            "• Можем созвониться с получателем для согласования доставки\n\n"
            "Как вам удобнее оплатить?"
        )
        logger.info("Agent node exit: payment %s", self._user_context(state))
        return state

    def _complaint_node(self, state: AgentState) -> AgentState:
        logger.info("Agent node enter: complaint %s", self._user_context(state))
        complaint_type = state["entities"].get("complaint_type", "other")

        if complaint_type == "wilted":
            state["response"] = (
                "😔 Очень жаль, что цветы подвели. Мы дорожим своей репутацией.\n\n"
                "*Наша политика по качеству:*\n"
                "• Если цветы завяли в течение 1–2 дней — предоставляем скидку 50% на следующие два заказа\n"
                "• Если просто не понравился букет — скидка 50% на следующий заказ\n\n"
                "Пришлите, пожалуйста, фото букета — разберёмся и предложим лучшее решение."
            )
        elif complaint_type == "bad_looking":
            state["response"] = (
                "😔 Нам жаль, что букет не оправдал ожиданий.\n\n"
                "*Что можем предложить:*\n"
                "• Скидка 50% на следующий заказ\n"
                "• При повторном заказе учтём все пожелания\n\n"
                "Напишите, что именно не понравилось — мы хотим стать лучше."
            )
        elif complaint_type == "wrong_flowers":
            state["response"] = (
                "😔 Извините за несоответствие. Бывает, что оптовые поставки вносят коррективы.\n\n"
                "• Все замены мы стараемся делать в той же цветовой гамме и ценовом диапазоне\n"
                "• Если замена критична — предлагаем скидку 50% на следующий заказ\n\n"
                "Если хотите обсудить детали — могу соединить с владельцем."
            )
        elif complaint_type == "delivery":
            state["response"] = (
                "😔 Извините за проблемы с доставкой. Давайте разберёмся.\n\n"
                "• Если курьер опоздал — примите наши извинения\n"
                "• Если не смогли вручить — организуем повторную доставку\n"
                "• Если хотите перенести время — напишите, согласуем\n\n"
                "Могу соединить с владельцем для детального разговора."
            )
        else:
            state["response"] = (
                "😔 Примите наши извинения. Нам важно ваше мнение.\n\n"
                "Расскажите подробнее, что случилось, и мы обязательно найдём решение.\n"
                "• Предоставим скидку на следующий заказ\n"
                "• Если нужно — соединю с владельцем напрямую"
            )

        state["response"] += (
            "\n\nЕсли хотите обсудить с владельцем лично — просто скажите «позовите владельца»."
        )
        logger.info("Agent node exit: complaint %s", self._user_context(state))
        return state

    def _urgent_node(self, state: AgentState) -> AgentState:
        logger.info("Agent node enter: urgent %s", self._user_context(state))
        state["response"] = (
            "⚡ *Срочная доставка!*\n\n"
            "Букет 10–15 тыс. ₽ можем доставить за 1,5–2 часа с момента заказа.\n\n"
            "Для оформления срочного заказа нужно:\n"
            "1️⃣ Бюджет (от 5 000 ₽)\n"
            "2️⃣ Предпочтения по цветам/составу\n"
            "3️⃣ Адрес доставки\n"
            "4️⃣ Контакт получателя\n\n"
            "Напишите бюджет и пожелания — запустим сборку!"
        )
        logger.info("Agent node exit: urgent %s", self._user_context(state))
        return state

    def _corporate_node(self, state: AgentState) -> AgentState:
        logger.info("Agent node enter: corporate %s", self._user_context(state))
        state["response"] = (
            "🏢 *Корпоративные заказы*\n\n"
            "Работаем с юрлицами — предоставляем счёт и закрывающие документы.\n\n"
            "*Что можем предложить:*\n"
            "• Оформление столов на мероприятия\n"
            "• Большие корзины из роз\n"
            "• Авторские букеты для сотрудников и партнёров\n"
            "• Свадебные букеты\n"
            "• Кашпо и коробки\n\n"
            "*Форматы:* от небольших заказов до полного оформления.\n\n"
            "Напишите:\n"
            "• Количество букетов\n"
            "• Бюджет на единицу\n"
            "• Повод/мероприятие\n"
            "• Сроки\n\n"
            "Просчитаю варианты. Или могу соединить с владельцем для обсуждения."
        )
        logger.info("Agent node exit: corporate %s", self._user_context(state))
        return state

    def _occasion_node(self, state: AgentState) -> AgentState:
        logger.info("Agent node enter: occasion %s", self._user_context(state))
        occasion = state["entities"].get("occasion", "")
        user_input = state["user_input"].lower()

        if any(w in user_input for w in ("свадьб", "свадебн")):
            state["response"] = (
                "💍 *Свадебная флористика*\n\n"
                "Делаем:\n"
                "• Свадебные букеты невесты\n"
                "• Бутоньерки\n"
                "• Оформление столов и зала\n"
                "• Кортежную флористику\n\n"
                "Напишите бюджет и пожелания по стилю — подберём варианты. "
                "Для сложных проектов могу соединить с владельцем."
            )
        elif any(w in user_input for w in ("день рожд", "юбилей", "др")):
            state["response"] = (
                "🎂 *На день рождения — отличный выбор!*\n\n"
                "Что можем предложить:\n"
                "• Авторские букеты от 5 000 ₽\n"
                "• Корзины из роз\n"
                "• Композиции в коробках и кашпо\n\n"
                "Скажите бюджет — соберём красивый вариант. "
                "Можно добавить открытку (бесплатно) и вазу (от 1 500 ₽)."
            )
        elif any(w in user_input for w in ("любим", "девушк", "жен", "романтик")):
            state["response"] = (
                "💕 *Романтический сюрприз*\n\n"
                "Классика:\n"
                "• Красные розы — символ страсти\n"
                "• Нежные пионы — для признания\n"
                "• Авторские сборные букеты\n\n"
                "Можно сделать анонимную доставку и добавить рукописную открытку (бесплатно).\n\n"
                "Какой бюджет рассматриваете? От 5 000 ₽."
            )
        elif any(w in user_input for w in ("мам", "матер")):
            state["response"] = (
                "🌷 *Для мамы — с любовью!*\n\n"
                "Популярные варианты:\n"
                "• Нежные букеты в пастельных тонах\n"
                "• Корзины с розами\n"
                "• Композиции в кашпо (долго стоят)\n\n"
                "Напишите бюджет — подберём идеальный вариант."
            )
        elif any(w in user_input for w in ("извин", "прости")):
            state["response"] = (
                "🙏 *Букет для извинений*\n\n"
                "Крупные композиции и корзины роз — беспроигрышный вариант.\n"
                "Можем приложить открытку с текстом.\n\n"
                "Какой бюджет рассматриваете? Обычно от 7 000 ₽."
            )
        elif any(w in user_input for w in ("коллег", "учител", "врач", "начальн")):
            state["response"] = (
                "🎁 *Отличная идея для подарка!*\n\n"
                "Рекомендуем:\n"
                "• Элегантные букеты 5–10 тыс. ₽\n"
                "• Композиции в коробках\n"
                "• Добавить открытку с тёплыми словами\n\n"
                "Напишите бюджет — подберём достойный вариант."
            )
        else:
            state["response"] = (
                "🎉 *Отличный повод для букета!*\n\n"
                "Подберём идеальный вариант под ваш случай.\n"
                "Напишите:\n"
                "• Повод\n"
                "• Бюджет (от 5 000 ₽)\n"
                "• Предпочтения по цветам"
            )
        logger.info("Agent node exit: occasion %s", self._user_context(state))
        return state

    def _faq_node(self, state: AgentState) -> AgentState:
        logger.info("Agent node enter: faq %s", self._user_context(state))
        user_input = state["user_input"].lower()

        if any(w in user_input for w in ("минимальн", "от скольк")):
            state["response"] = (
                "📌 *Минимальный заказ — 5 000 ₽.* "
                "Средний чек 10–12 тыс. ₽."
            )
        elif any(w in user_input for w in ("анонимн", "тайно", "секрет")):
            state["response"] = (
                "🤫 *Анонимная доставка — возможна.*\n"
                "Можем не указывать отправителя. "
                "Открытка подписывается анонимно или с любым текстом."
            )
        elif any(w in user_input for w in ("открытк", "записк")):
            state["response"] = (
                "✉️ *Открытка — бесплатно!*\n"
                "• Рукописная — можем написать любой текст\n"
                "• Печатная — если нужно официально\n"
                "• Можно анонимно\n\n"
                "Текст открытки согласуем с вами."
            )
        elif any(w in user_input for w in ("ваз", "вазу")):
            state["response"] = (
                "🏺 *Вазы в наличии:* стеклянные, от 1 500 ₽.\n"
                "Спросите при заказе — добавим к букету."
            )
        elif any(w in user_input for w in ("свежест", "гаранти", "долго сто")):
            state["response"] = (
                "🌷 *Гарантия свежести:*\n"
                "• Цветы закупаем ежедневно на оптовых базах\n"
                "• Если что-то не устроило — свяжитесь с нами\n"
                "• При увядании за 1–2 дня — скидка 50% на следующие два заказа"
            )
        elif any(w in user_input for w in ("замена", "замен")):
            state["response"] = (
                "🔄 *Политика замен:*\n"
                "• Иногда оптовые поставки вносят коррективы\n"
                "• Замены стараемся делать в той же цветовой гамме и ценовом диапазоне\n"
                "• По референсу — собираем максимально похожий букет\n"
                "• Если замена критична — согласуем с вами"
            )
        elif any(w in user_input for w in ("работа", "график", "часы")):
            state["response"] = (
                "🕐 *График работы:*\n"
                "• Приём заказов: 9:00–23:00\n"
                "• Доставка: 24/7\n"
                "• Сборка букетов — после подтверждения заказа"
            )
        else:
            state["response"] = (
                "❓ Чем могу помочь?\n\n"
                "Я могу рассказать:\n"
                "• О доставке (стоимость, зоны, время)\n"
                "• Об оплате (способы, условия)\n"
                "• О минимальном заказе\n"
                "• Об открытках и вазах\n"
                "• О гарантии свежести\n"
                "• О политике замен\n\n"
                "Что вас интересует?"
            )
        logger.info("Agent node exit: faq %s", self._user_context(state))
        return state

    def _contact_owner_node(self, state: AgentState) -> AgentState:
        logger.info("Agent node enter: contact_owner %s", self._user_context(state))
        state["response"] = (
            "📞 *Соединяю с владельцем...*\n\n"
            "Передаю ваш запрос Дмитрию. Он свяжется с вами в ближайшее время.\n\n"
            "Если хотите ускорить — напишите кратко суть вопроса, я передам."
        )
        logger.info("Agent node exit: contact_owner %s", self._user_context(state))
        return state

    def _photo_request_node(self, state: AgentState) -> AgentState:
        logger.info("Agent node enter: photo_request %s", self._user_context(state))
        state["response"] = (
            "📸 *Фото перед отправкой*\n\n"
            "Постоянным клиентам отправляем фото после доставки.\n"
            "По запросу — можем отправить фото букета перед отправкой.\n\n"
            "Если заказ уже оформлен — напишите номер заказа.\n"
            "Если только выбираете — могу показать фото из каталога."
        )
        logger.info("Agent node exit: photo_request %s", self._user_context(state))
        return state

    def _change_order_node(self, state: AgentState) -> AgentState:
        logger.info("Agent node enter: change_order %s", self._user_context(state))
        change_request = state["entities"].get("change_request", "other")
        state["response"] = (
            "✏️ *Изменение заказа*\n\n"
            "До отправки заказа:\n"
            "• Перенос времени — бесплатно\n"
            "• Изменение адреса — возможно\n"
            "• Изменение состава — по согласованию\n\n"
            "Если заказ уже в работе — напишите, что именно хотите изменить.\n"
            "Для сложных изменений могу соединить с владельцем."
        )
        logger.info("Agent node exit: change_order %s", self._user_context(state))
        return state

    def _cancel_order_node(self, state: AgentState) -> AgentState:
        logger.info("Agent node enter: cancel_order %s", self._user_context(state))
        state["response"] = (
            "❌ *Отмена заказа*\n\n"
            "• Если заказ оплачен, но ещё не собран — отмена бесплатно\n"
            "• Если оплачен и уже собран — ничего страшного, отменяем\n"
            "• Если получателя нет на месте — курьер подождёт (за отдельную плату)\n\n"
            "Напишите номер заказа — отменю."
        )
        logger.info("Agent node exit: cancel_order %s", self._user_context(state))
        return state

    def _repeat_order_node(self, state: AgentState) -> AgentState:
        logger.info("Agent node enter: repeat_order %s", self._user_context(state))
        state["response"] = (
            "🔄 *Повторный заказ*\n\n"
            "Рады, что вам понравилось! Можем:\n"
            "• Повторить тот же букет\n"
            "• Собрать похожий, но с нюансами\n"
            "• Сделать сюрприз — другой, но не менее красивый\n\n"
            "Напишите:\n"
            "• Какой заказ повторить (или данные получателя)\n"
            "• Те же адрес и время или что-то изменить?"
        )
        logger.info("Agent node exit: repeat_order %s", self._user_context(state))
        return state

    def _unknown_node(self, state: AgentState) -> AgentState:
        logger.info("Agent node enter: unknown %s", self._user_context(state))
        state["response"] = (
            "🤔 Не совсем понял ваш запрос.\n\n"
            "Я могу:\n"
            "• Подобрать букет по бюджету (например: *«букет до 10 000»*)\n"
            "• Рассказать о доставке\n"
            "• Ответить про оплату\n"
            "• Оформить срочный заказ\n\n"
            "Напишите подробнее, что вас интересует.\n"
            "Или свяжу с владельцем — просто скажите *«позовите Дмитрия»*."
        )
        logger.info("Agent node exit: unknown %s", self._user_context(state))
        return state

    def get_bouquet_recommendation(
        self,
        user_input: str,
        user_id: Optional[int] = None,
        user_name: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Обрабатывает запрос пользователя через LangGraph."""
        if user_id is not None:
            mysql_interface.save_message(user_id, user_input)
        history = conversation_history or []

        history_lines: List[str] = []
        for item in history:
            role = item.get("role", "unknown")
            content = item.get("content", "")
            if isinstance(content, str) and content.strip():
                history_lines.append(f"{role}: {content}")

        # Если последнее сообщение в истории еще не содержит текущий user_input,
        # добавляем его явно, чтобы граф всегда видел актуальный контекст.
        if not history_lines or not history_lines[-1].endswith(user_input):
            history_lines.append(f"user: {user_input}")

        full_conversation_text = "\n".join(history_lines)
        logger.info(
            "Agent input received: user_chars=%s history_items=%s full_chars=%s %s",
            len(user_input),
            len(history),
            len(full_conversation_text),
            f"user_id={user_id} user_name={user_name}",
        )
        logger.info("Agent input last message: %s", user_input[:1000])
        if len(full_conversation_text) > 2000:
            logger.warning(
                "Agent input conversation is large: %s chars, %s history items",
                len(full_conversation_text),
                len(history),
            )
        logger.debug("Agent input full conversation: %s", full_conversation_text)

        graph_input: AgentState = {
            "user_input": user_input,
            "user_id": user_id,
            "user_name": user_name,
            "conversation_history": history,
            "full_conversation_text": full_conversation_text,
            "bouquets_data": self.bouquets_data,
            "intent": "unknown",
            "entities": {},
            "response": "",
        }

        result = self.graph.invoke(graph_input)
        return result["response"]

    def get_more_bouquets(self, user_id: int, start: int, count: int) -> Optional[str]:
        """Возвращает следующую страницу букетов для пользователя."""
        bouquets = self._last_bouquets.get(user_id)
        if not bouquets or start >= len(bouquets):
            return None
        return self._format_bouquets_page(bouquets, start, count)

    def filter_bouquets_by_price(self, max_price: float):
        return [b for b in self.bouquets_data if b['Цена'] <= max_price]

    def add_new_messages_to_index(self):
        """Совместимость со старым API: RAG отключен, индексация не требуется."""
        print("ℹ️ RAG отключен. add_new_messages_to_index пропущен.")


def format_bouquet_message(bouquet):
    parts = [f"💐 {bouquet['Название']}"]
    if bouquet.get("Цена") and bouquet["Цена"] > 0:
        parts.append(f"💰 Цена: {int(bouquet['Цена'])} руб.")
    if bouquet.get("Описание"):
        parts.append(f"📝 {bouquet['Описание']}")
    if bouquet.get("Состав"):
        состав = ", ".join(bouquet["Состав"])
        parts.append(f"🌷 Состав: {состав}")
    parts.append(f"🔗 [Ссылка на букет]({bouquet['Ссылка']})")
    return "\n\n".join(parts)


def get_bouquet_image(bouquet):
    """Возвращает URL изображения букета, если есть."""
    return bouquet.get("Изображение")


def create_price_ranges():
    return [
        ("До 5 000 руб.", "5000"),
        ("5 000-10 000 руб.", "10000"),
        ("10 000-15 000 руб.", "15000"),
        ("15 000-20 000 руб.", "20000"),
        ("Свыше 20 000 руб.", "20000+")
    ]