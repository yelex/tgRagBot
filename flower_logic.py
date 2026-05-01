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
    intent: Literal["greet", "greet_and_offer", "catalog", "recommend", "chosen_by_name", "unknown"]
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

        graph_builder.add_edge(START, "nlu")
        graph_builder.add_conditional_edges(
            "nlu",
            self._route_from_nlu,
            {
                "greet": "greet",
                "offer": "offer",
            },
        )
        graph_builder.add_edge("greet", END)
        graph_builder.add_edge("offer", END)

        return graph_builder.compile()

    def _user_context(self, state: AgentState) -> str:
        return f"user_id={state.get('user_id')} user_name={state.get('user_name')}"

    def _route_from_nlu(self, state: AgentState) -> str:
        logger.info(
            "Agent route decision: intent=%s %s",
            state["intent"],
            self._user_context(state),
        )
        if state["intent"] == "greet":
            logger.info("Agent transition: nlu -> greet %s", self._user_context(state))
            return "greet"
        logger.info("Agent transition: nlu -> offer %s", self._user_context(state))
        return "offer"

    def _normalize_intent(self, raw_intent: str) -> AgentState["intent"]:
        allowed = {"greet", "greet_and_offer", "catalog", "recommend", "chosen_by_name", "unknown"}
        intent = (raw_intent or "").strip().lower()
        if intent in allowed:
            return intent  # type: ignore[return-value]
        return "unknown"

    def _predict_nlu_with_llm(self, state: AgentState) -> Dict[str, Any]:
        if self.intent_llm is None:
            return {"intent": "unknown", "entities": {}}

        prompt = f"""
Ты NLU-модуль для цветочного Telegram-бота.
Определи intent и извлеки сущности из последнего сообщения с учетом истории.

Доступные intent:
- greet
- greet_and_offer
- catalog
- recommend
- chosen_by_name
- unknown

Правила:
- greet: пользователь только здоровается/начинает диалог без конкретного запроса.
- greet_and_offer: есть приветствие и одновременно конкретный запрос по букетам.
- catalog: просит показать весь каталог/все варианты.
- recommend: просит подобрать/порекомендовать/найти по бюджету или описанию.
- chosen_by_name: запрашивает конкретную позицию/модель букета по названию.
- unknown: если невозможно уверенно классифицировать.

Важно:
- Числа в сообщении НЕ всегда бюджет (пример: "хочу 101 розу" -> это количество/характеристика, НЕ бюджет).
- Извлекай `max_price` когда в тексте явно речь о верхней границе бюджета ("до X", "не дороже X", "в пределах X").
- Извлекай `min_price` когда в тексте явно речь о нижней границе бюджета ("от X", "не меньше X", "начиная от X", "X+", "X тыс" в контексте "от").
- Если сказано "от 10 тыс" — это min_price=10000.
- Если сказано "до 10 тыс" — это max_price=10000.
- Если сказано "от 10 до 15 тыс" — это min_price=10000, max_price=15000.

Ответ верни строго JSON без пояснений:
{{
  "intent": "one_of_allowed_values",
  "entities": {{
    "max_price": number|null,
    "min_price": number|null,
    "name_query": "string|null",
    "query_text": "normalized user request"
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

            max_price = entities_raw.get("max_price")
            if isinstance(max_price, (int, float)):
                parsed_max_price: Optional[float] = float(max_price)
            elif isinstance(max_price, str):
                try:
                    parsed_max_price = float(max_price.replace(",", ".").strip())
                except ValueError:
                    parsed_max_price = None
            else:
                parsed_max_price = None

            min_price = entities_raw.get("min_price")
            if isinstance(min_price, (int, float)):
                parsed_min_price: Optional[float] = float(min_price)
            elif isinstance(min_price, str):
                try:
                    parsed_min_price = float(min_price.replace(",", ".").strip())
                except ValueError:
                    parsed_min_price = None
            else:
                parsed_min_price = None

            name_query = entities_raw.get("name_query")
            if not isinstance(name_query, str) or not name_query.strip():
                name_query = None

            query_text = entities_raw.get("query_text")
            if not isinstance(query_text, str) or not query_text.strip():
                query_text = state["user_input"].lower()

            entities = {"query_text": query_text}
            if parsed_max_price is not None:
                entities["max_price"] = parsed_max_price
            if parsed_min_price is not None:
                entities["min_price"] = parsed_min_price
            if name_query is not None:
                entities["name_query"] = name_query.lower()

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
        has_greeting = any(marker in text for marker in ("привет", "здравствуйте", "добрый", "hello", "hi"))
        has_catalog = any(marker in text for marker in ("каталог", "все букеты", "покажи все"))
        has_request = any(token in text for token in ("букет", "цвет", "роза", "роз", "пион"))

        if has_catalog and has_greeting:
            return "greet_and_offer"
        if has_catalog:
            return "catalog"
        if has_greeting and has_request:
            return "greet_and_offer"
        if has_greeting:
            return "greet"
        if has_request:
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
        state["response"] = (
            "Здравствуйте! Помогу подобрать букет. "
            "Напишите бюджет или пожелания, например: "
            "'Нужен букет до 10000'."
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
                    if intent == "greet_and_offer":
                        state["response"] = f"Здравствуйте!\n\n{state['response']}"
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
                if intent == "greet_and_offer":
                    state["response"] = f"Здравствуйте!\n\n{state['response']}"
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
                if intent == "greet_and_offer":
                    state["response"] = f"Здравствуйте!\n\n{state['response']}"
                return state

            top_items = sorted(bouquets, key=lambda b: b["Цена"])[:3]
            state["response"] = self._format_short_offer(top_items)
            logger.info(
                "Agent node exit: offer, branch=%s, fallback=cheapest %s",
                intent,
                self._user_context(state),
            )
            if intent == "greet_and_offer":
                state["response"] = f"Здравствуйте!\n\n{state['response']}"
            return state

        state["response"] = (
            "Уточните, какой бюджет или какие цветы хотите. "
            "Например: 'Пионы до 15000'."
        )
        logger.info("Agent node exit: offer %s", self._user_context(state))
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
    return (
        f"💐 {bouquet['Название']}\n"
        f"💰 Цена: {bouquet['Цена']} руб.\n"
        f"🔗 [Ссылка на букет]({bouquet['Ссылка']})"
    )


def create_price_ranges():
    return [
        ("До 5 000 руб.", "5000"),
        ("5 000-10 000 руб.", "10000"),
        ("10 000-15 000 руб.", "15000"),
        ("15 000-20 000 руб.", "20000"),
        ("Свыше 20 000 руб.", "20000+")
    ]