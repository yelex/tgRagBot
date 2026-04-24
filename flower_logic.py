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
- Извлекай `max_price` только когда в тексте явно речь о бюджете/ценовом лимите.

Ответ верни строго JSON без пояснений:
{{
  "intent": "one_of_allowed_values",
  "entities": {{
    "max_price": number|null,
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

            name_query = entities_raw.get("name_query")
            if not isinstance(name_query, str) or not name_query.strip():
                name_query = None

            query_text = entities_raw.get("query_text")
            if not isinstance(query_text, str) or not query_text.strip():
                query_text = state["user_input"].lower()

            entities = {"query_text": query_text}
            if parsed_max_price is not None:
                entities["max_price"] = parsed_max_price
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

    def _format_short_offer(self, bouquets: List[Dict[str, Any]]) -> str:
        lines = ["Подобрал варианты:"]
        for bouquet in bouquets[:3]:
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
            "до", "руб", "рублей", "пожалуйста",
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

        if intent == "catalog":
            sorted_bouquets = sorted(bouquets, key=lambda b: b["Цена"])
            state["response"] = self._format_short_offer(sorted_bouquets)
            logger.info("Agent node exit: offer, branch=catalog %s", self._user_context(state))
            return state

        if intent == "chosen_by_name":
            query = entities.get("name_query", "")
            found = [b for b in bouquets if query in b["Название"].lower()]
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
            if max_price is not None:
                filtered = [b for b in bouquets if b["Цена"] <= max_price]
                if not filtered:
                    state["response"] = (
                        f"В бюджете до {int(max_price)} руб. вариантов не нашёл. "
                        "Могу показать ближайшие по цене."
                    )
                    cheapest = sorted(bouquets, key=lambda b: b["Цена"])[:3]
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