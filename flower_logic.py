import json
import logging
import os
import re
from typing import Any, Dict, List, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from interfaces import mysql_interface

load_dotenv()
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    user_input: str
    user_id: Optional[int]
    user_name: Optional[str]
    conversation_history: List[Dict[str, str]]
    bouquets_data: List[Dict[str, Any]]
    intent: Literal["greet", "catalog", "recommend", "chosen_by_name", "unknown"]
    entities: Dict[str, Any]
    response: str


class FlowerLogic:
    def __init__(self):
        self.PATH_BOUQUETS = os.getenv("PATH_BOUQUETS")
        self.bouquets_info: str = ""
        self.bouquets_data: List[Dict[str, Any]] = []
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
        self.graph = self._build_graph()
        print("✅ LangGraph-агент готов!")

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

    def _extract_budget(self, user_input: str) -> Optional[float]:
        clean_text = user_input.lower().replace(" ", "")
        matches = re.findall(r"\d+(?:[.,]\d+)?", clean_text)
        if not matches:
            return None
        try:
            return float(matches[0].replace(",", "."))
        except ValueError:
            return None

    def _nlu_node(self, state: AgentState) -> AgentState:
        logger.info("Agent node enter: nlu %s", self._user_context(state))
        text = state["user_input"].lower()
        intent: AgentState["intent"] = "unknown"
        entities: Dict[str, Any] = {}

        greeting_markers = ("привет", "здравствуйте", "добрый", "hello", "hi")
        if any(marker in text for marker in greeting_markers):
            intent = "greet"
        elif "каталог" in text or "все букеты" in text or "покажи все" in text:
            intent = "catalog"
        else:
            normalized = text
            for bouquet in state["bouquets_data"]:
                name = bouquet["Название"].lower()
                if name in normalized:
                    intent = "chosen_by_name"
                    entities["name_query"] = name
                    break

            if intent == "unknown":
                budget = self._extract_budget(text)
                if budget is not None:
                    intent = "recommend"
                    entities["max_price"] = budget
                elif any(token in text for token in ("букет", "цвет", "роза", "пион")):
                    intent = "recommend"

        state["intent"] = intent
        state["entities"] = entities
        logger.info(
            "Agent node exit: nlu, intent=%s, entities=%s %s",
            intent,
            entities,
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

        if intent == "recommend":
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
                        "Agent node exit: offer, branch=recommend, within_budget=0 %s",
                        self._user_context(state),
                    )
                    return state

                filtered = sorted(filtered, key=lambda b: b["Цена"], reverse=True)
                state["response"] = self._format_short_offer(filtered)
                logger.info(
                    "Agent node exit: offer, branch=recommend, within_budget=%s %s",
                    len(filtered),
                    self._user_context(state),
                )
                return state

            top_items = sorted(bouquets, key=lambda b: b["Цена"])[:3]
            state["response"] = self._format_short_offer(top_items)
            logger.info(
                "Agent node exit: offer, branch=recommend, fallback=cheapest %s",
                self._user_context(state),
            )
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
        graph_input: AgentState = {
            "user_input": user_input,
            "user_id": user_id,
            "user_name": user_name,
            "conversation_history": conversation_history or [],
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