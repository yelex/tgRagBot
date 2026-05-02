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


DEFAULT_DELIVERY_ZONES: Dict[str, Dict[str, Any]] = {
    "mкад": {"min_km": 0, "max_km": 0, "price": 600, "label": "В пределах МКАД"},
    "5-10": {"min_km": 5, "max_km": 10, "price": 1500, "label": "5–10 км за МКАД"},
    "10-20": {"min_km": 10, "max_km": 20, "price": 2000, "label": "10–20 км за МКАД"},
    "20-30": {"min_km": 20, "max_km": 30, "price": 3000, "label": "20–30 км за МКАД"},
    "30-40": {"min_km": 30, "max_km": 40, "price": 4000, "label": "30–40 км за МКАД"},
    "40-50": {"min_km": 40, "max_km": 50, "price": 5000, "label": "40–50 км за МКАД"},
}

FREE_DELIVERY_THRESHOLD = 20000


class AgentState(TypedDict):
    user_input: str
    user_id: Optional[int]
    user_name: Optional[str]
    conversation_history: List[Dict[str, str]]
    full_conversation_text: str
    bouquets_data: List[Dict[str, Any]]
    intent: str
    entities: Dict[str, Any]
    response: str
    phase: str
    collected_data: Dict[str, Any]


ALLOWED_INTENTS = {
    "greet", "catalog", "recommend", "chosen_by_name", "delivery_info",
    "payment_info", "complaint", "urgent", "corporate", "occasion",
    "photo_request", "change_order", "cancel_order", "repeat_order",
    "contact_owner", "faq", "unknown", "greet_and_offer",
    "pickup", "scheduled", "reference", "status",
}

ALLOWED_PHASES = {
    "init", "N0_greet", "N1_router", "N2_brief_budget", "N2_brief_occasion",
    "N2_brief_format", "N2_brief_gamma", "N2_brief_restrictions",
    "N2_brief_reference", "N2_brief_card", "N3_catalog_choice",
    "N4_reference_critical", "N5_urgent_data", "N6_scheduled_data",
    "N7_pickup_data", "N8_faq", "N9_offer", "N10_delivery",
    "N11_complaint_facts", "N12_escalated", "N13_photo_approval",
    "N14_delivery_exec", "N15_close", "payment",
}


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

    # ──────────────────────────────────────────────
    # Инициализация
    # ──────────────────────────────────────────────

    def load_bouquets_data(self):
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
        print("🌸 Инициализация LangGraph-агента (v2 — многошаговый диалог)...")
        self.load_bouquets_data()
        self._initialize_intent_llm()
        self.graph = self._build_graph()
        print("✅ LangGraph-агент v2 готов!")

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

    # ──────────────────────────────────────────────
    # Построение графа
    # ──────────────────────────────────────────────

    def _build_graph(self):
        builder = StateGraph(AgentState)

        builder.add_node("nlu", self._nlu_node)
        builder.add_node("router", self._router_node)
        builder.add_node("N0_greet", self._N0_greet_node)
        builder.add_node("N2_brief", self._N2_brief_node)
        builder.add_node("N3_catalog", self._N3_catalog_node)
        builder.add_node("N4_reference", self._N4_reference_node)
        builder.add_node("N5_urgent", self._N5_urgent_node)
        builder.add_node("N6_scheduled", self._N6_scheduled_node)
        builder.add_node("N7_pickup", self._N7_pickup_node)
        builder.add_node("N8_faq", self._N8_faq_node)
        builder.add_node("N9_offer", self._N9_offer_node)
        builder.add_node("N10_delivery", self._N10_delivery_node)
        builder.add_node("N11_complaint", self._N11_complaint_node)
        builder.add_node("N12_escalation", self._N12_escalation_node)
        builder.add_node("N13_photo", self._N13_photo_node)
        builder.add_node("N14_delivery_exec", self._N14_delivery_exec_node)
        builder.add_node("N15_close", self._N15_close_node)
        builder.add_node("payment", self._payment_node)
        builder.add_node("unknown", self._unknown_node)

        builder.add_edge(START, "nlu")

        builder.add_conditional_edges(
            "nlu",
            self._route_after_nlu,
            {
                "router": "router",
                "N0_greet": "N0_greet",
                "N2_brief": "N2_brief",
                "N3_catalog": "N3_catalog",
                "N4_reference": "N4_reference",
                "N5_urgent": "N5_urgent",
                "N6_scheduled": "N6_scheduled",
                "N7_pickup": "N7_pickup",
                "N8_faq": "N8_faq",
                "N9_offer": "N9_offer",
                "N10_delivery": "N10_delivery",
                "N11_complaint": "N11_complaint",
                "N12_escalation": "N12_escalation",
                "N13_photo": "N13_photo",
                "N14_delivery_exec": "N14_delivery_exec",
                "N15_close": "N15_close",
                "payment": "payment",
                "unknown": "unknown",
            },
        )

        builder.add_conditional_edges(
            "router",
            self._route_from_router,
            {
                "N0_greet": "N0_greet",
                "N2_brief": "N2_brief",
                "N3_catalog": "N3_catalog",
                "N4_reference": "N4_reference",
                "N5_urgent": "N5_urgent",
                "N6_scheduled": "N6_scheduled",
                "N7_pickup": "N7_pickup",
                "N8_faq": "N8_faq",
                "N9_offer": "N9_offer",
                "N10_delivery": "N10_delivery",
                "N11_complaint": "N11_complaint",
                "N12_escalation": "N12_escalation",
                "N13_photo": "N13_photo",
                "N14_delivery_exec": "N14_delivery_exec",
                "N15_close": "N15_close",
                "payment": "payment",
                "unknown": "unknown",
            },
        )

        for node_name in (
            "N0_greet", "N2_brief", "N3_catalog", "N4_reference", "N5_urgent",
            "N6_scheduled", "N7_pickup", "N8_faq", "N9_offer", "N10_delivery",
            "N11_complaint", "N12_escalation", "N13_photo", "N14_delivery_exec",
            "N15_close", "payment", "unknown",
        ):
            builder.add_edge(node_name, END)

        return builder.compile()

    # ──────────────────────────────────────────────
    # Хелперы
    # ──────────────────────────────────────────────

    @staticmethod
    def _user_context(state: AgentState) -> str:
        uid = state.get("user_id")
        uname = state.get("user_name")
        phase = state.get("phase", "?")
        return f"uid={uid} name={uname} phase={phase}"

    @staticmethod
    def _normalize_phase(raw: Optional[str]) -> str:
        if raw and raw in ALLOWED_PHASES:
            return raw
        return "init"

    @staticmethod
    def _normalize_intent(raw: Optional[str]) -> str:
        intent = (raw or "").strip().lower()
        if intent in ALLOWED_INTENTS:
            return intent
        return "unknown"

    def _extract_phase_from_history(self, state: AgentState) -> Optional[str]:
        """Извлекает фазу из метаданных последнего ответа ассистента."""
        for item in reversed(state["conversation_history"]):
            content = item.get("content", "")
            if isinstance(content, str):
                m = re.search(r'\|\|phase:([a-zA-Z0-9_]+)\|\|', content)
                if m:
                    return m.group(1)
        return None

    def _extract_collected_data_from_history(self, state: AgentState) -> Dict[str, Any]:
        """Извлекает собранные данные из метаданных истории."""
        for item in reversed(state["conversation_history"]):
            content = item.get("content", "")
            if isinstance(content, str):
                m = re.search(r'\|\|data:(\{.*?\})\|\|', content)
                if m:
                    try:
                        return json.loads(m.group(1))
                    except (json.JSONDecodeError, ValueError):
                        pass
        return {}

    # ──────────────────────────────────────────────
    # NLU
    # ──────────────────────────────────────────────

    def _nlu_node(self, state: AgentState) -> AgentState:
        logger.info("Agent node enter: nlu %s", self._user_context(state))

        # Восстанавливаем фазу и собранные данные из истории
        phase = self._extract_phase_from_history(state)
        state["phase"] = self._normalize_phase(phase)
        state["collected_data"] = self._extract_collected_data_from_history(state)

        # Если есть фаза — значит диалог уже идёт, NLU только для извлечения сущностей
        if state["phase"] != "init":
            # Для уже идущего диалога определяем интент/сущности через LLM или fallback
            nlu_result = self._predict_nlu_with_llm(state)
            intent = self._normalize_intent(str(nlu_result.get("intent", "unknown")))
            entities: Dict[str, Any] = nlu_result.get("entities", {})
            if not isinstance(entities, dict):
                entities = {}
            if intent == "unknown":
                intent = self._fallback_intent_from_rules(state)
            state["intent"] = intent
            state["entities"] = entities
            logger.info(
                "NLU (phase=%s): intent=%s entities=%s %s",
                state["phase"], intent, entities, self._user_context(state),
            )
            return state

        # Первый вход: определяем интент и фазу
        nlu_result = self._predict_nlu_with_llm(state)
        intent = self._normalize_intent(str(nlu_result.get("intent", "unknown")))
        entities = nlu_result.get("entities", {})
        if not isinstance(entities, dict):
            entities = {}

        if intent == "unknown":
            intent = self._fallback_intent_from_rules(state)

        # Fallback-извлечение цен
        price_fallback = self._extract_price_from_text(state["user_input"])
        for key, val in price_fallback.items():
            if val is not None and key not in entities:
                entities[key] = val

        if "query_text" not in entities:
            entities["query_text"] = state["user_input"].lower()

        state["intent"] = intent
        state["entities"] = entities
        state["phase"] = "init"

        logger.info(
            "NLU (init): intent=%s entities=%s %s",
            intent, entities, self._user_context(state),
        )
        return state

    def _route_after_nlu(self, state: AgentState) -> str:
        """Маршрутизация после NLU: учитывает фазу и интент."""
        phase = state.get("phase", "init")

        # Если диалог уже идёт — идём в соответствующий обработчик
        if phase != "init":
            return phase

        # Первый вход: по интенту
        intent = state.get("intent", "unknown")
        phase_map: Dict[str, str] = {
            "greet": "N0_greet",
            "greet_and_offer": "N0_greet",
            "catalog": "N3_catalog",
            "recommend": "N0_greet",
            "chosen_by_name": "N9_offer",
            "delivery_info": "N10_delivery",
            "payment_info": "payment",
            "complaint": "N11_complaint",
            "urgent": "N5_urgent",
            "corporate": "N12_escalation",
            "occasion": "N2_brief",
            "photo_request": "N13_photo",
            "change_order": "N10_delivery",
            "cancel_order": "N12_escalation",
            "repeat_order": "N9_offer",
            "contact_owner": "N12_escalation",
            "faq": "N8_faq",
            "pickup": "N7_pickup",
            "scheduled": "N6_scheduled",
            "reference": "N4_reference",
            "status": "N12_escalation",
            "unknown": "router",
        }
        return phase_map.get(intent, "router")

    def _route_from_router(self, state: AgentState) -> str:
        """Роутер N1: определение сценария при неопределённом интенте."""
        text = state["user_input"].lower()
        collected = state.get("collected_data", {})

        # Проверка на самовывоз
        if any(w in text for w in ("самовывоз", "заберу", "подъеду", "приеду")):
            return "N7_pickup"

        # Проверка на срочность
        if any(w in text for w in ("срочн", "быстр", "побыстре", "через час", "asap")):
            return "N5_urgent"

        # Проверка на референс/фото
        if any(w in text for w in ("референс", "пример", "как на фото", "сделайте так же")):
            return "N4_reference"

        # Проверка на каталог
        if any(w in text for w in ("каталог", "что есть", "покажи", "посмотреть")):
            return "N3_catalog"

        # Проверка на жалобу
        if any(w in text for w in ("завял", "не нравит", "жалоб", "недовол")):
            return "N11_complaint"

        # Проверка на ивент/корпоратив
        if any(w in text for w in ("свадьб", "мероприяти", "корпоратив", "офис", "оптом")):
            return "N12_escalation"

        # Проверка на доставку
        if any(w in text for w in ("достав", "привез", "адрес", "мкад")):
            if collected.get("address"):
                return "N10_delivery"
            return "N10_delivery"

        # Проверка на FAQ
        if any(w in text for w in ("сколько", "цена", "минимальн", "оплата", "работа")):
            return "N8_faq"

        # По умолчанию — начало продажи
        if collected.get("budget") or any(w in text for w in ("букет", "цвет", "хочу", "нужен")):
            return "N2_brief"

        return "N0_greet"

    # ──────────────────────────────────────────────
    # NLU — LLM + Fallback
    # ──────────────────────────────────────────────

    def _predict_nlu_with_llm(self, state: AgentState) -> Dict[str, Any]:
        if self.intent_llm is None:
            return {"intent": "unknown", "entities": {}}

        phase = state.get("phase", "init")
        prompt = f"""
Ты NLU-модуль для цветочного Telegram-бота (darkstore, Москва, доставка 24/7).
Определи intent и извлеки сущности из сообщения с учетом истории.

Текущая фаза диалога: {phase}

Доступные intent:
- greet — приветствие без конкретного запроса
- greet_and_offer — приветствие + запрос букета
- catalog — просьба показать каталог
- recommend — просьба подобрать по бюджету/описанию
- chosen_by_name — запрос конкретного букета по названию
- delivery_info — вопросы о доставке (стоимость, зоны, время)
- payment_info — вопросы об оплате
- complaint — претензия/жалоба
- urgent — срочная доставка
- corporate — корпоративный/ивент-заказ
- occasion — подбор по поводу (свадьба, др, маме)
- photo_request — запрос фото
- change_order — изменить заказ
- cancel_order — отменить заказ
- repeat_order — повторный заказ
- contact_owner — позвать владельца
- faq — вопросы о политике
- pickup — самовывоз
- scheduled — доставка ко времени
- reference — референс/фото-пример
- status — статус заказа
- unknown — неопределено

Извлечение цен:
- "до X тыс" → max_price = X*1000
- "от X тыс" → min_price = X*1000
- "от X до Y тыс" → min_price = X*1000, max_price = Y*1000

Ответ строго JSON:
{{
  "intent": "...",
  "entities": {{
    "max_price": number|null,
    "min_price": number|null,
    "name_query": "string|null",
    "query_text": "string",
    "occasion": "string|null",
    "urgent": true|false|null,
    "complaint_type": "wilted|bad_looking|wrong_flowers|delivery|other|null",
    "corporate_size": number|null,
    "change_request": "address|time|composition|other|null",
    "photo_request": true|false|null,
    "repeat_reference": "string|null",
    "pickup": true|false|null,
    "product_type": "bouquet|composition|box|basket|null",
    "delivery_time": "asap|scheduled|night|null",
    "address": "string|null",
    "gamma": "pastel|bright|null",
    "recipient": "string|null",
    "card_text": "string|null",
    "anonymous": true|false|null,
    "critical_flower": "string|null",
    "critical_color": "string|null"
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
            for key in (
                "max_price", "min_price", "name_query", "query_text",
                "occasion", "urgent", "complaint_type", "corporate_size",
                "change_request", "photo_request", "repeat_reference",
                "pickup", "product_type", "delivery_time", "address",
                "gamma", "recipient", "card_text", "anonymous",
                "critical_flower", "critical_color",
            ):
                val = entities_raw.get(key)
                if val is not None:
                    entities[key] = val

            # Нормализация цен
            for price_key in ("max_price", "min_price"):
                val = entities_raw.get(price_key)
                if isinstance(val, (int, float)):
                    entities[price_key] = float(val)
                elif isinstance(val, str):
                    try:
                        entities[price_key] = float(val.replace(",", ".").strip())
                    except ValueError:
                        pass

            logger.info(
                "NLU LLM: intent=%s entities=%s %s",
                intent, entities, self._user_context(state),
            )
            return {"intent": intent, "entities": entities}

        except Exception as exc:
            logger.exception("NLU LLM failed, fallback: %s", exc)
            return {"intent": "unknown", "entities": {}}

    def _fallback_intent_from_rules(self, state: AgentState) -> str:
        text = state["user_input"].lower()

        has_greeting = any(m in text for m in ("привет", "здравствуйте", "добрый", "hello"))
        has_catalog = any(m in text for m in ("каталог", "все букеты", "покажи все", "ассортимент"))
        has_flower = any(m in text for m in ("букет", "цвет", "роз", "пион", "тюльпан"))
        has_delivery = any(m in text for m in ("достав", "привезти", "везете", "мкад"))
        has_payment = any(m in text for m in ("оплат", "деньги", "перевод", "наличн", "карт"))
        has_complaint = any(m in text for m in ("завял", "увял", "плох", "не нравит", "недовол", "жалоб"))
        has_urgent = any(m in text for m in ("срочн", "быстр", "побыстре", "через час"))
        has_corporate = any(m in text for m in ("корпоратив", "офис", "мероприятие", "свадьб", "оптом"))
        has_occasion = any(m in text for m in ("день рожд", "юбилей", "мам", "любим", "девушк", "извин"))
        has_photo = any(m in text for m in ("фото", "покажи", "пришли"))
        has_change = any(m in text for m in ("измен", "поменя", "перенес", "передвин"))
        has_cancel = any(m in text for m in ("отмен", "аннулир"))
        has_repeat = any(m in text for m in ("повтор", "как в прошл", "еще раз", "снова"))
        has_faq = any(m in text for m in ("минимальн", "сколько стоит", "гаранти", "анонимн", "открытк", "ваз"))
        has_owner = any(m in text for m in ("позов", "владельц", "хозяин", "соедин", "менеджер"))
        has_pickup = any(m in text for m in ("самовывоз", "заберу", "подъеду"))
        has_reference = any(m in text for m in ("референс", "как на фото", "пример"))

        if has_cancel:
            return "cancel_order"
        if has_complaint:
            return "complaint"
        if has_owner:
            return "contact_owner"
        if has_urgent and has_flower:
            return "urgent"
        if has_corporate:
            return "corporate"
        if has_pickup:
            return "pickup"
        if has_reference:
            return "reference"
        if has_delivery:
            return "delivery_info"
        if has_payment:
            return "payment_info"
        if has_occasion:
            return "occasion"
        if has_repeat:
            return "repeat_order"
        if has_change:
            return "change_order"
        if has_photo and (has_flower or has_catalog):
            return "photo_request"
        if has_faq:
            return "faq"
        if has_catalog and has_greeting:
            return "greet_and_offer"
        if has_catalog:
            return "catalog"
        if has_greeting and has_flower:
            return "greet_and_offer"
        if has_greeting:
            return "greet"
        if has_flower:
            return "recommend"
        return "unknown"

    def _extract_price_from_text(self, text: str) -> Dict[str, Optional[float]]:
        result: Dict[str, Optional[float]] = {}
        t = text.lower()

        m = re.search(r'(?:\s|^)от\s+(\d+(?:[.,]\d+)?)\s+до\s+(\d+(?:[.,]\d+)?)\s*тыс(?:яч)?', t)
        if m:
            result["min_price"] = float(m.group(1).replace(",",".")) * 1000
            result["max_price"] = float(m.group(2).replace(",",".")) * 1000
            return result

        m = re.search(r'(?:\s|^)от\s+(\d+(?:[.,]\d+)?)\s*тыс(?:яч)?', t)
        if m:
            result["min_price"] = float(m.group(1).replace(",",".")) * 1000

        if "max_price" not in result:
            m = re.search(r'(?:\s|^)до\s+(\d+(?:[.,]\d+)?)\s*тыс(?:яч)?', t)
            if m:
                result["max_price"] = float(m.group(1).replace(",",".")) * 1000

        if "max_price" not in result:
            m = re.search(r'(?:не дороже|в пределах|не больше)\s+(\d+(?:[.,]\d+)?)\s*тыс(?:яч)?', t)
            if m:
                result["max_price"] = float(m.group(1).replace(",",".")) * 1000

        return result

    # ──────────────────────────────────────────────
    # N0 — Приветствие и якоря
    # ──────────────────────────────────────────────

    def _N0_greet_node(self, state: AgentState) -> AgentState:
        logger.info("Agent node enter: N0_greet %s", self._user_context(state))

        # Если пользователь уже ответил на приветствие
        if state["phase"] != "init":
            collected = state.get("collected_data", {})
            intent = state.get("intent", "unknown")
            text = state["user_input"].lower()

            # Если пользователь сказал что хочет — переходим в N2_brief
            if intent in ("recommend", "catalog", "greet_and_offer") or \
               any(w in text for w in ("букет", "цветы", "хочу", "нужен", "подбери")):
                state["phase"] = "N2_brief"
                state["response"] = "Отлично! Давайте подберём букет."
                return self._N2_brief_node(state)

            # Если спросил про доставку
            if "достав" in text or "мкад" in text:
                state["phase"] = "N10_delivery"
                return self._N10_delivery_node(state)

            # Если спросил про оплату
            if any(w in text for w in ("оплат", "деньги", "перевод")):
                state["phase"] = "payment"
                return self._payment_node(state)

            # Если срочно
            if any(w in text for w in ("срочн", "быстр", "asap")):
                state["phase"] = "N5_urgent"
                return self._N5_urgent_node(state)

            # Если хочет каталог
            if any(w in text for w in ("каталог", "покажи", "что есть")):
                state["phase"] = "N3_catalog"
                return self._N3_catalog_node(state)

            # Общая реакция
            state["phase"] = "N2_brief"
            state["response"] = (
                "Подскажите, на какой бюджет рассчитываете? "
                "Минимальный заказ — 5 000 ₽."
            )
            state["response"] += " ||phase:N2_brief_budget||"
            logger.info("N0_greet → N2_brief (budget question) %s", self._user_context(state))
            return state

        # Первое приветствие
        state["phase"] = "N0_greet"
        state["collected_data"] = {}

        # Если есть бюджет от NLU — сразу в N2_brief
        entities = state.get("entities", {})
        if entities.get("max_price") or entities.get("min_price"):
            state["phase"] = "N2_brief"
            state["response"] = "Здравствуйте! Сразу к делу — давайте подберём букет."
            state["response"] += " ||phase:N2_brief_budget||"
            logger.info("N0_greet → N2_brief (has budget) %s", self._user_context(state))
            return state

        state["response"] = (
            "🌸 *Добрый день!* Рады приветствовать в нашей цветочной мастерской.\n\n"
            "Подскажите, пожалуйста, какой букет/композиция/коробка вас интересует?\n\n"
            "Или просто напишите бюджет — подберём варианты."
        )
        state["response"] += " ||phase:N0_greet|| ||data:{}||"
        logger.info("N0_greet — first greeting %s", self._user_context(state))
        return state

    # ──────────────────────────────────────────────
    # N1 — Router (только если NLU не смог определить)
    # ──────────────────────────────────────────────

    def _router_node(self, state: AgentState) -> AgentState:
        logger.info("Agent node enter: router (N1) %s", self._user_context(state))
        # Роутер логика уже в _route_from_router
        # Здесь просто ставим fallback-ответ, если ничего не подошло
        state["phase"] = "N0_greet"
        state["response"] = (
            "Не совсем понял ваш запрос. Давайте начнём сначала:\n\n"
            "• *Хочу заказать букет* — напишите бюджет и пожелания\n"
            "• *Узнать о доставке* — «доставка»\n"
            "• *Посмотреть каталог* — «каталог»"
        )
        state["response"] += " ||phase:N0_greet||"
        logger.info("Router fallback → N0_greet %s", self._user_context(state))
        return state

    # ──────────────────────────────────────────────
    # N2 — Мини-бриф (пошаговый сбор данных)
    # ──────────────────────────────────────────────

    def _N2_brief_node(self, state: AgentState) -> AgentState:
        logger.info("Agent node enter: N2_brief %s", self._user_context(state))

        collected = state.get("collected_data", {})
        entities = state.get("entities", {})
        text = state["user_input"].lower()
        phase = state.get("phase", "N2_brief_budget")

        # Сохраняем данные из сообщения
        if entities.get("max_price") or entities.get("min_price"):
            collected["budget_max"] = entities.get("max_price")
            collected["budget_min"] = entities.get("min_price")
            if collected["budget_min"] and collected["budget_min"] < 5000:
                collected["budget_min"] = 5000

        if entities.get("occasion"):
            collected["occasion"] = entities["occasion"]
        if entities.get("recipient"):
            collected["recipient"] = entities["recipient"]
        if entities.get("product_type"):
            collected["product_type"] = entities["product_type"]
        if entities.get("gamma"):
            collected["gamma"] = entities["gamma"]
        if entities.get("card_text"):
            collected["card_text"] = entities["card_text"]
        if entities.get("anonymous"):
            collected["anonymous"] = entities["anonymous"]
        if entities.get("address"):
            collected["address"] = entities["address"]

        # Извлекаем бюджет из текста, если не извлёк LLM
        if not collected.get("budget_max") and not collected.get("budget_min"):
            pf = self._extract_price_from_text(text)
            if pf.get("max_price") or pf.get("min_price"):
                collected["budget_max"] = pf.get("max_price")
                collected["budget_min"] = pf.get("min_price")
                if collected.get("budget_min") and collected["budget_min"] < 5000:
                    collected["budget_min"] = 5000

        # Проверка минимального бюджета
        budget = collected.get("budget_max") or collected.get("budget_min") or 0
        budget_ok = budget >= 5000 or (collected.get("budget_min") and collected["budget_min"] >= 5000)

        # Определяем, какой вопрос задать
        if not collected.get("budget_max") and not collected.get("budget_min"):
            state["phase"] = "N2_brief_budget"
            state["response"] = (
                "На какой бюджет ориентируемся? "
                "Минимальный заказ — 5 000 ₽.\n\n"
                "Комфортно будет собрать от 5 000–7 000 ₽."
            )
            state["response"] += " ||phase:N2_brief_budget||"
            logger.info("N2_brief: ask budget %s", self._user_context(state))
            return self._wrap_response(state)

        if not budget_ok:
            state["phase"] = "N2_brief_budget"
            state["response"] = (
                "Минимальный заказ — 5 000 ₽. "
                "Комфортно будет собрать от 5 000–7 000 ₽.\n"
                "Какой бюджет рассматриваете?"
            )
            state["response"] += " ||phase:N2_brief_budget||"
            logger.info("N2_brief: budget too low %s", self._user_context(state))
            return self._wrap_response(state)

        if not collected.get("occasion") and not collected.get("recipient"):
            state["phase"] = "N2_brief_occasion"
            state["response"] = (
                "К какому поводу и кому выбираем: "
                "девушка/мама/коллега/свадьба?"
            )
            state["response"] += " ||phase:N2_brief_occasion||"
            logger.info("N2_brief: ask occasion %s", self._user_context(state))
            return self._wrap_response(state)

        if not collected.get("product_type"):
            state["phase"] = "N2_brief_format"
            state["response"] = (
                "Вам ближе букет, композиция (коробка/корзина) или кашпо?"
            )
            state["response"] += " ||phase:N2_brief_format||"
            logger.info("N2_brief: ask format %s", self._user_context(state))
            return self._wrap_response(state)

        if not collected.get("gamma"):
            state["phase"] = "N2_brief_gamma"
            state["response"] = (
                "Какой стиль предпочитаете: нежно-пастельный или яркий/насыщенный?"
            )
            state["response"] += " ||phase:N2_brief_gamma||"
            logger.info("N2_brief: ask gamma %s", self._user_context(state))
            return self._wrap_response(state)

        if "restrictions_asked" not in collected:
            collected["restrictions_asked"] = True
            state["phase"] = "N2_brief_restrictions"
            state["response"] = (
                "Есть ли цветы, которые точно нельзя "
                "(аллергия на запах, лилии, сильный аромат)?"
            )
            state["response"] += " ||phase:N2_brief_restrictions||"
            logger.info("N2_brief: ask restrictions %s", self._user_context(state))
            return self._wrap_response(state)

        if "reference_asked" not in collected:
            collected["reference_asked"] = True
            state["phase"] = "N2_brief_reference"
            state["response"] = (
                "Если есть фото-пример — пришлите, пожалуйста. "
                "Сделаем максимально похоже."
            )
            state["response"] += " ||phase:N2_brief_reference||"
            logger.info("N2_brief: ask reference %s", self._user_context(state))
            return self._wrap_response(state)

        # Все данные собраны — переходим к офферу
        state["collected_data"] = collected
        state["phase"] = "N9_offer"
        return self._N9_offer_node(state)

    # ──────────────────────────────────────────────
    # N3 — Каталог
    # ──────────────────────────────────────────────

    def _N3_catalog_node(self, state: AgentState) -> AgentState:
        logger.info("Agent node enter: N3_catalog %s", self._user_context(state))

        collected = state.get("collected_data", {})
        entities = state.get("entities", {})

        # Сохраняем параметры
        if entities.get("max_price"):
            collected["budget_max"] = entities["max_price"]
        if entities.get("min_price"):
            collected["budget_min"] = entities["min_price"]
        if entities.get("product_type"):
            collected["product_type"] = entities["product_type"]
        if entities.get("gamma"):
            collected["gamma"] = entities["gamma"]

        bouquets = state["bouquets_data"]
        filtered = list(bouquets)

        # Фильтруем по бюджету
        max_price = collected.get("budget_max")
        min_price = collected.get("budget_min")
        if max_price:
            filtered = [b for b in filtered if b["Цена"] <= max_price]
        if min_price:
            filtered = [b for b in filtered if b["Цена"] >= min_price]

        # Если нет фильтров — показываем все
        if not max_price and not min_price:
            # Группируем по ценовым категориям
            categories = [
                ("До 5 000 ₽", [b for b in filtered if b["Цена"] <= 5000]),
                ("5 000–10 000 ₽", [b for b in filtered if 5000 < b["Цена"] <= 10000]),
                ("10 000–15 000 ₽", [b for b in filtered if 10000 < b["Цена"] <= 15000]),
                ("15 000–20 000 ₽", [b for b in filtered if 15000 < b["Цена"] <= 20000]),
                ("Свыше 20 000 ₽", [b for b in filtered if b["Цена"] > 20000]),
            ]
            lines = ["📋 *Категории букетов:*\n"]
            for name, items in categories:
                if items:
                    lines.append(f"• *{name}* — {len(items)} {self._pluralize_variant(len(items))}")
            lines.append("\nНапишите бюджет или название категории — покажу варианты.")
            state["response"] = "\n".join(lines)
            state["phase"] = "N3_catalog_choice"
            state["collected_data"] = collected
            state["response"] += f" ||phase:N3_catalog_choice|| ||data:{json.dumps(collected, ensure_ascii=False)}||"
            logger.info("N3_catalog: show categories %s", self._user_context(state))
            return state

        # Показываем конкретные варианты
        sorted_b = sorted(filtered, key=lambda b: b["Цена"])
        if sorted_b:
            user_id = state.get("user_id")
            if user_id is not None:
                self._last_bouquets[user_id] = sorted_b
            lines = ["Вот подходящие варианты:\n"]
            for b in sorted_b[:5]:
                lines.append(
                    f"• {b['Название']} — {int(b['Цена'])} ₽\n"
                    f"  {b['Ссылка']}"
                )
            if len(sorted_b) > 5:
                lines.append(f"\nПоказано 5 из {len(sorted_b)} вариантов.")
            lines.append("\nНапишите, какой понравился, и перейдём к оформлению доставки.")
            state["response"] = "\n".join(lines)
        else:
            state["response"] = "К сожалению, в этом диапазоне ничего не нашлось."

        state["phase"] = "N3_catalog_choice"
        state["collected_data"] = collected
        state["response"] += f" ||phase:N3_catalog_choice|| ||data:{json.dumps(collected, ensure_ascii=False)}||"
        logger.info("N3_catalog: show items %s", self._user_context(state))
        return state

    # ──────────────────────────────────────────────
    # N4 — Референс
    # ──────────────────────────────────────────────

    def _N4_reference_node(self, state: AgentState) -> AgentState:
        logger.info("Agent node enter: N4_reference %s", self._user_context(state))

        collected = state.get("collected_data", {})
        entities = state.get("entities", {})
        text = state["user_input"].lower()

        # Сохраняем данные
        if entities.get("critical_flower"):
            collected["critical_flower"] = entities["critical_flower"]
        if entities.get("critical_color"):
            collected["critical_color"] = entities["critical_color"]

        if "reference_critical_asked" not in collected:
            collected["reference_critical_asked"] = True
            state["phase"] = "N4_reference_critical"
            state["response"] = (
                "Что для вас принципиально важно в референсе:\n"
                "• Цветовая гамма\n"
                "• Конкретный цветок\n"
                "• Форма/объём букета\n\n"
                "Возможны небольшие замены в пределах 10–15% состава "
                "без согласования, при сохранении стиля и качества."
            )
            state["response"] += " ||phase:N4_reference_critical||"
            state["collected_data"] = collected
            logger.info("N4_reference: ask critical %s", self._user_context(state))
            return self._wrap_response(state)

        # Проверка на требование "1-в-1 без замен"
        if any(w in text for w in ("1-в-1", "один в один", "точно так же", "без замен",
                                    "строго", "только эти", "именно эти")):
            state["phase"] = "N12_escalated"
            collected["escalation_reason"] = "reference_1v1"
            state["collected_data"] = collected
            state["response"] = (
                "Понимаю. Сейчас уточню возможность точного повтора "
                "и вернусь к вам."
            )
            state["response"] += " ||phase:N12_escalated|| ||data:{}||"
            logger.info("N4_reference → N12 (1v1 required) %s", self._user_context(state))
            return state

        # Если не критично — переходим к офферу
        state["phase"] = "N9_offer"
        state["collected_data"] = collected
        return self._N9_offer_node(state)

    # ──────────────────────────────────────────────
    # N5 — Срочно / ASAP
    # ──────────────────────────────────────────────

    def _N5_urgent_node(self, state: AgentState) -> AgentState:
        logger.info("Agent node enter: N5_urgent %s", self._user_context(state))

        collected = state.get("collected_data", {})
        entities = state.get("entities", {})
        text = state["user_input"].lower()

        # Сохраняем данные
        if entities.get("address"):
            collected["address"] = entities["address"]
        if entities.get("max_price"):
            collected["budget_max"] = entities["max_price"]
        if entities.get("min_price"):
            collected["budget_min"] = entities["min_price"]

        # Извлекаем адрес
        if not collected.get("address"):
            state["phase"] = "N5_urgent_data"
            state["response"] = (
                "⚡ Для срочного заказа напишите, пожалуйста:\n"
                "1️⃣ Адрес доставки\n"
                "2️⃣ На какое время нужно\n"
                "3️⃣ Бюджет (от 5 000 ₽)\n\n"
                "Букет 10–15 тыс. ₽ доставляем за 1,5–2 часа с момента заказа."
            )
            state["response"] += " ||phase:N5_urgent_data||"
            state["collected_data"] = collected
            logger.info("N5_urgent: ask data %s", self._user_context(state))
            return self._wrap_response(state)

        # Проверка за МКАД
        if "мкад" in text and any(w in text for w in ("за", "замкад", "область", "мо")):
            collected["escalation_reason"] = "urgent_beyond_mkad"
            state["collected_data"] = collected
            state["phase"] = "N12_escalated"
            state["response"] = (
                "Сейчас уточню возможность срочной доставки по вашему адресу "
                "и вернусь к вам."
            )
            state["response"] += " ||phase:N12_escalated||"
            logger.info("N5_urgent → N12 (beyond MKAD) %s", self._user_context(state))
            return state

        # Есть адрес — переходим к быстрому офферу
        state["phase"] = "N9_offer"
        state["collected_data"] = collected
        state["response"] = (
            "Отлично! Давайте быстро подберём вариант."
        )
        logger.info("N5_urgent → N9_offer %s", self._user_context(state))
        return self._N9_offer_node(state)

    # ──────────────────────────────────────────────
    # N6 — Ко времени / ночь
    # ──────────────────────────────────────────────

    def _N6_scheduled_node(self, state: AgentState) -> AgentState:
        logger.info("Agent node enter: N6_scheduled %s", self._user_context(state))

        collected = state.get("collected_data", {})
        entities = state.get("entities", {})

        if entities.get("address"):
            collected["address"] = entities["address"]
        if entities.get("delivery_time"):
            collected["delivery_time"] = entities["delivery_time"]

        if not collected.get("address"):
            state["phase"] = "N6_scheduled_data"
            state["response"] = (
                "Доставка ко времени — отличный выбор!\n\n"
                "Напишите, пожалуйста:\n"
                "• Точное время доставки\n"
                "• Адрес\n"
                "• Контакт получателя\n"
                "• Комментарии курьеру"
            )
            state["response"] += " ||phase:N6_scheduled_data||"
            state["collected_data"] = collected
            logger.info("N6_scheduled: ask data %s", self._user_context(state))
            return self._wrap_response(state)

        # Ночной режим
        hour = 0  # не знаем час
        if 0 <= hour < 9:
            state["response"] = (
                "Сейчас мы на связи с 09:00 до 23:00.\n"
                "Я зафиксирую заявку и утром сразу подтвердим доставку."
            )
        else:
            state["response"] = "Принял! Давайте подберём букет."

        state["collected_data"] = collected
        state["phase"] = "N9_offer"
        state["response"] += " ||phase:N9_offer||"
        logger.info("N6_scheduled → N9_offer %s", self._user_context(state))
        return self._wrap_response(state)

    # ──────────────────────────────────────────────
    # N7 — Самовывоз
    # ──────────────────────────────────────────────

    def _N7_pickup_node(self, state: AgentState) -> AgentState:
        logger.info("Agent node enter: N7_pickup %s", self._user_context(state))

        collected = state.get("collected_data", {})

        if "pickup_terms_shown" not in collected:
            collected["pickup_terms_shown"] = True
            state["collected_data"] = collected
            state["phase"] = "N7_pickup_data"
            state["response"] = (
                "🏪 *Самовывоз*\n\n"
                "Правила:\n"
                "• Самовывоз только по 100% предоплате\n"
                "• После сборки отправим фото\n\n"
                "Напишите, что хотите забрать и когда планируете подъехать."
            )
            state["response"] += " ||phase:N7_pickup_data||"
            logger.info("N7_pickup: show terms %s", self._user_context(state))
            return self._wrap_response(state)

        # Переходим в оплату
        state["collected_data"] = collected
        state["phase"] = "payment"
        state["response"] = "Отлично! Давайте оформим."
        state["response"] += " ||phase:payment||"
        logger.info("N7_pickup → payment %s", self._user_context(state))
        return self._wrap_response(state)

    # ──────────────────────────────────────────────
    # N8 — FAQ → конверсия
    # ──────────────────────────────────────────────

    def _N8_faq_node(self, state: AgentState) -> AgentState:
        logger.info("Agent node enter: N8_faq %s", self._user_context(state))

        text = state["user_input"].lower()
        phase = state.get("phase", "N8_faq")

        # Если это продолжение после первого FAQ-ответа
        if phase != "init" and phase != "N8_faq":
            # Пробуем перевести в заказ
            if any(w in text for w in ("букет", "хочу", "нужен", "заказать")):
                state["phase"] = "N2_brief"
                return self._N2_brief_node(state)
            state["phase"] = "N2_brief"
            state["response"] = (
                "На какой бюджет и на какое время нужна доставка? "
                "Минимальный заказ — 5 000 ₽."
            )
            state["response"] += " ||phase:N2_brief_budget||"
            logger.info("N8_faq → N2_brief %s", self._user_context(state))
            return state

        # Формируем ответ под вопрос
        if any(w in text for w in ("минимальн", "от скольк")):
            response = "📌 *Минимальный заказ — 5 000 ₽.*"
        elif any(w in text for w in ("анонимн", "тайно", "секрет")):
            response = (
                "🤫 *Анонимная доставка — возможна.* "
                "Можем не указывать отправителя."
            )
        elif any(w in text for w in ("открытк", "записк")):
            response = (
                "✉️ *Открытка — бесплатно!*\n"
                "• Рукописная — любой текст\n"
                "• Можно анонимно"
            )
        elif any(w in text for w in ("ваз", "вазу")):
            response = "🏺 *Вазы:* стеклянные, от 1 500 ₽."
        elif any(w in text for w in ("свежест", "гаранти")):
            response = (
                "🌷 *Гарантия свежести:* цветы закупаем ежедневно.\n"
                "Если завяли за 1–2 дня — скидка 50% на 2 заказа."
            )
        elif any(w in text for w in ("замена", "замен")):
            response = (
                "🔄 *Замены:* в той же гамме и ценовом диапазоне.\n"
                "По референсу — максимально похоже."
            )
        elif any(w in text for w in ("работа", "график", "часы")):
            response = (
                "🕐 Приём заказов: 9:00–23:00.\n"
                "🚚 Доставка: 24/7."
            )
        elif any(w in text for w in ("мкад", "достав")):
            response = (
                "🚚 *Доставка:*\n"
                "• МКАД — 600 ₽ (бесплатно при заказе >20 000 ₽)\n"
                "• За МКАД — от 1 500 ₽ (зависит от расстояния)"
            )
        elif any(w in text for w in ("оплат", "карт", "нал")):
            response = (
                "💳 *Оплата:* перевод, ссылка, наличные при получении.\n"
                "Работаем с юрлицами."
            )
        else:
            response = (
                "Чем могу помочь?\n\n"
                "Я могу рассказать:\n"
                "• О доставке (стоимость, зоны)\n"
                "• Об оплате\n"
                "• О минимальном заказе (5 000 ₽)\n"
                "• Об открытках и вазах\n"
                "• О гарантии свежести"
            )

        # Конверсия в заказ
        response += (
            "\n\nХотите оформить заказ? Напишите бюджет и пожелания — "
            "подберу варианты."
        )

        state["response"] = response + " ||phase:N2_brief_budget||"
        state["phase"] = "N2_brief"
        logger.info("N8_faq: answered, phase→N2_brief %s", self._user_context(state))
        return state

    # ──────────────────────────────────────────────
    # N9 — Оффер 2-3 варианта
    # ──────────────────────────────────────────────

    def _N9_offer_node(self, state: AgentState) -> AgentState:
        logger.info("Agent node enter: N9_offer %s", self._user_context(state))

        collected = state.get("collected_data", {})
        entities = state.get("entities", {})
        bouquets = state["bouquets_data"]
        user_id = state.get("user_id")

        max_price = collected.get("budget_max") or entities.get("max_price")
        min_price = collected.get("budget_min") or entities.get("min_price")

        # Фильтруем по бюджету
        filtered = list(bouquets)
        if max_price:
            filtered = [b for b in filtered if b["Цена"] <= max_price]
        if min_price:
            filtered = [b for b in filtered if b["Цена"] >= min_price]

        # Если ничего не нашли — показываем ближайшие
        if not filtered:
            cheapest = sorted(bouquets, key=lambda b: b["Цена"])[:3]
            if user_id is not None:
                self._last_bouquets[user_id] = cheapest
            lines = ["К сожалению, в этом бюджете вариантов нет.\n"]
            lines.append("Вот самые доступные варианты:\n")
            for b in cheapest:
                lines.append(
                    f"• {b['Название']} — {int(b['Цена'])} ₽\n"
                    f"  {b['Ссылка']}"
                )
            lines.append("\nКакой вариант присмотреть?")
            state["response"] = "\n".join(lines)
            state["phase"] = "N9_offer"
            state["collected_data"] = collected
            state["response"] += f" ||phase:N9_offer|| ||data:{json.dumps(collected, ensure_ascii=False)}||"
            logger.info("N9_offer: no results, show cheapest %s", self._user_context(state))
            return state

        # Good/Better/Best
        filtered_sorted = sorted(filtered, key=lambda b: b["Цена"])
        good = None
        better = None
        best = None

        # Good — самый дешёвый подходящий
        if filtered_sorted:
            good = filtered_sorted[0]

        # Better — средний
        if len(filtered_sorted) >= 2:
            mid = len(filtered_sorted) // 2
            better = filtered_sorted[mid]
        elif filtered_sorted:
            better = filtered_sorted[-1]

        # Best — самый дорогой
        if len(filtered_sorted) >= 3:
            best = filtered_sorted[-1]
        elif filtered_sorted:
            best = filtered_sorted[-1]

        picks = [p for p in (good, better, best) if p]
        # Убираем дубликаты
        seen = set()
        unique_picks = []
        for p in picks:
            if p["Название"] not in seen:
                seen.add(p["Название"])
                unique_picks.append(p)

        if user_id is not None:
            self._last_bouquets[user_id] = filtered_sorted

        lines = ["Подобрал для вас варианты:\n"]
        labels = ["✅ *Вариант 1 (Good)*", "⭐ *Вариант 2 (Better)*", "🏆 *Вариант 3 (Best)*"]
        for i, b in enumerate(unique_picks[:3]):
            label = labels[i] if i < len(labels) else f"*Вариант {i+1}*"
            lines.append(
                f"{label}\n"
                f"💐 {b['Название']}\n"
                f"💰 {int(b['Цена'])} ₽\n"
                f"🔗 {b['Ссылка']}\n"
            )

        lines.append("_Можно добавить:_")
        lines.append("• ✉️ Открытка — бесплатно")
        lines.append("• 🏺 Ваза стеклянная — от 1 500 ₽")
        lines.append("• 🎈 Шарики, сладкое — по запросу")
        lines.append("\nКакой вариант больше нравится? Или хотите что-то поменять?")

        state["response"] = "\n".join(lines)
        state["phase"] = "N10_delivery"
        state["collected_data"] = collected
        state["response"] += f" ||phase:N10_delivery|| ||data:{json.dumps(collected, ensure_ascii=False)}||"
        logger.info("N9_offer: %d options presented %s", len(unique_picks), self._user_context(state))
        return state

    # ──────────────────────────────────────────────
    # N10 — Доставка (сбор данных + цена)
    # ──────────────────────────────────────────────

    def _N10_delivery_node(self, state: AgentState) -> AgentState:
        logger.info("Agent node enter: N10_delivery %s", self._user_context(state))

        collected = state.get("collected_data", {})
        entities = state.get("entities", {})
        text = state["user_input"].lower()

        # Сохраняем адрес
        if entities.get("address"):
            collected["address"] = entities["address"]
        if not collected.get("address"):
            # Пробуем извлечь адрес из текста
            addr_match = re.search(
                r'(?:адрес|по адресу|на? )([а-яА-ЯёЁ\s\d.,/-]{5,})',
                text
            )
            if addr_match:
                collected["address"] = addr_match.group(1).strip()

        # Проверка на признаки "за МКАД"
        beyond_mkad = any(w in text for w in (
            "за мкад", "замкад", "область", "московская область",
            "подмосковье", "за город",
        ))

        if beyond_mkad:
            state["collected_data"] = collected
            state["response"] = (
                "Сейчас уточню точную стоимость доставки по этому адресу "
                "и вернусь к вам."
            )
            state["phase"] = "N12_escalated"
            collected["escalation_reason"] = "beyond_mkad"
            state["response"] += " ||phase:N12_escalated||"
            logger.info("N10_delivery → N12 (beyond MKAD) %s", self._user_context(state))
            return state

        # МКАД: цена доставки
        total = collected.get("budget_max") or 0
        delivery_price = 0 if total >= FREE_DELIVERY_THRESHOLD else 600

        if not collected.get("address"):
            state["response"] = (
                "Для расчёта доставки напишите, пожалуйста, адрес полностью."
            )
            state["phase"] = "N10_delivery"
            state["collected_data"] = collected
            state["response"] += " ||phase:N10_delivery||"
            logger.info("N10_delivery: ask address %s", self._user_context(state))
            return self._wrap_response(state)

        # Адрес есть — показываем итог
        delivery_line = "🚚 *Доставка:* 0 ₽ (бесплатно при заказе от 20 000 ₽)" \
            if delivery_price == 0 else f"🚚 *Доставка:* {delivery_price} ₽ (МКАД)"

        state["response"] = (
            f"{delivery_line}\n\n"
            "📋 *Подтвердите детали заказа:*\n"
            f"📍 Адрес: {collected['address']}\n"
            "⏰ Время: уточним\n\n"
            "Всё верно? Переходим к оплате?"
        )
        state["phase"] = "payment"
        state["collected_data"] = collected
        state["response"] += " ||phase:payment||"
        logger.info("N10_delivery: address collected → payment %s", self._user_context(state))
        return state

    # ──────────────────────────────────────────────
    # N11 — Жалоба: сбор фактов → эскалация
    # ──────────────────────────────────────────────

    def _N11_complaint_node(self, state: AgentState) -> AgentState:
        logger.info("Agent node enter: N11_complaint %s", self._user_context(state))

        collected = state.get("collected_data", {})

        if "complaint_facts_collected" not in collected:
            collected["complaint_facts_collected"] = True
            state["phase"] = "N11_complaint_facts"
            state["response"] = (
                "Понимаю. Примите наши извинения.\n\n"
                "Чтобы разобраться, напишите, пожалуйста:\n"
                "• Когда получили букет\n"
                "• Что именно не так (фото приложите)\n"
                "• Условия хранения (в вазе/без, температура)"
            )
            state["response"] += " ||phase:N11_complaint_facts||"
            state["collected_data"] = collected
            logger.info("N11_complaint: collect facts %s", self._user_context(state))
            return self._wrap_response(state)

        # Факты собраны → эскалация
        state["phase"] = "N12_escalated"
        collected["escalation_reason"] = "complaint"
        state["collected_data"] = collected
        state["response"] = (
            "Сейчас всё уточню и вернусь к вам."
        )
        state["response"] += " ||phase:N12_escalated||"
        logger.info("N11_complaint → N12 (escalated) %s", self._user_context(state))
        return state

    # ──────────────────────────────────────────────
    # N12 — Эскалация владельцу
    # ──────────────────────────────────────────────

    def _N12_escalation_node(self, state: AgentState) -> AgentState:
        logger.info("Agent node enter: N12_escalation %s", self._user_context(state))

        collected = state.get("collected_data", {})
        reason = collected.get("escalation_reason", "other")

        state["phase"] = "N12_escalated"
        state["response"] = (
            "📞 Сейчас всё уточню и вернусь к вам."
        )
        state["response"] += " ||phase:N12_escalated||"
        logger.info("N12_escalation: reason=%s %s", reason, self._user_context(state))
        return state

    # ──────────────────────────────────────────────
    # N13 — Фото перед доставкой
    # ──────────────────────────────────────────────

    def _N13_photo_node(self, state: AgentState) -> AgentState:
        logger.info("Agent node enter: N13_photo %s", self._user_context(state))

        collected = state.get("collected_data", {})

        if "photo_requested" not in collected:
            collected["photo_requested"] = True
            state["phase"] = "N13_photo_approval"
            state["response"] = (
                "📸 *Фото перед отправкой*\n\n"
                "Постоянным клиентам отправляем фото после доставки.\n"
                "По запросу — можем отправить фото букета перед отправкой.\n\n"
                "Если заказ уже оформлен — напишите номер, отправлю фото."
            )
            state["response"] += " ||phase:N13_photo_approval||"
            state["collected_data"] = collected
            logger.info("N13_photo: first request %s", self._user_context(state))
            return self._wrap_response(state)

        # Если клиент недоволен фото → эскалация
        text = state["user_input"].lower()
        if any(w in text for w in ("не нравит", "не то", "не так", "плох", "переделай")):
            collected["escalation_reason"] = "photo_dispute"
            state["collected_data"] = collected
            state["phase"] = "N12_escalated"
            state["response"] = (
                "Сейчас уточню и вернусь к вам."
            )
            state["response"] += " ||phase:N12_escalated||"
            logger.info("N13_photo → N12 (dispute) %s", self._user_context(state))
            return state

        # Фото одобрено → доставка
        state["phase"] = "N14_delivery_exec"
        state["collected_data"] = collected
        state["response"] = "Отлично! Передаю курьеру."
        state["response"] += " ||phase:N14_delivery_exec||"
        logger.info("N13_photo → N14_delivery_exec %s", self._user_context(state))
        return state

    # ──────────────────────────────────────────────
    # N14 — Вручение и проблемы
    # ──────────────────────────────────────────────

    def _N14_delivery_exec_node(self, state: AgentState) -> AgentState:
        logger.info("Agent node enter: N14_delivery_exec %s", self._user_context(state))

        text = state["user_input"].lower()

        # Проверка на спор/конфликт
        if any(w in text for w in ("опоздал", "не приехал", "не тот", "проблем", "конфликт")):
            collected = state.get("collected_data", {})
            collected["escalation_reason"] = "delivery_conflict"
            state["collected_data"] = collected
            state["phase"] = "N12_escalated"
            state["response"] = "Сейчас уточню и вернусь к вам."
            state["response"] += " ||phase:N12_escalated||"
            logger.info("N14_delivery_exec → N12 (conflict) %s", self._user_context(state))
            return state

        state["phase"] = "N15_close"
        state["response"] = (
            "Спасибо за заказ! Надеемся, букет порадует получателя 🌸\n\n"
            "_Гарантия свежести:_ если что-то не так — "
            "напишите, поможем."
        )
        state["response"] += " ||phase:N15_close||"
        logger.info("N14_delivery_exec → N15_close %s", self._user_context(state))
        return state

    # ──────────────────────────────────────────────
    # N15 — Закрытие
    # ──────────────────────────────────────────────

    def _N15_close_node(self, state: AgentState) -> AgentState:
        logger.info("Agent node enter: N15_close %s", self._user_context(state))

        # Проверка фоллоу-ап
        text = state["user_input"].lower()
        if any(w in text for w in ("спасиб", "благодар", "отлично", "хорошо", "да")):
            state["response"] = (
                "🌸 Всегда рады помочь! Если захотите повторить или "
                "посоветовать нас друзьям — будем благодарны.\n\n"
                "Хорошего дня!"
            )
        elif any(w in text for w in ("ещё", "другой", "повтор")):
            state["phase"] = "N2_brief"
            logger.info("N15_close → N2_brief (repeat) %s", self._user_context(state))
            return self._N2_brief_node(state)
        else:
            state["response"] = (
                "Спасибо за обращение! Если появятся вопросы — "
                "мы на связи с 09:00 до 23:00."
            )

        state["phase"] = "N15_close"
        state["response"] += " ||phase:N15_close||"
        logger.info("N15_close: done %s", self._user_context(state))
        return state

    # ──────────────────────────────────────────────
    # Оплата
    # ──────────────────────────────────────────────

    def _payment_node(self, state: AgentState) -> AgentState:
        logger.info("Agent node enter: payment %s", self._user_context(state))

        collected = state.get("collected_data", {})

        state["response"] = (
            "💳 *Способы оплаты:*\n\n"
            "• 💸 Перевод на карту\n"
            "• 🔗 Ссылка на оплату\n"
            "• 💵 Наличные при получении\n"
            "• 🏢 Работаем с юрлицами\n\n"
            "*Условия:*\n"
            "• Для новых клиентов — предоплата\n"
            "• Постоянным/по рекомендации — оплата при получении\n"
            "• Можно поставить в работу до оплаты\n\n"
            "Как вам удобнее оплатить?"
        )

        state["phase"] = "N13_photo"
        state["collected_data"] = collected
        state["response"] += " ||phase:N13_photo||"
        logger.info("Payment → N13_photo %s", self._user_context(state))
        return state

    # ──────────────────────────────────────────────
    # Unknown
    # ──────────────────────────────────────────────

    def _unknown_node(self, state: AgentState) -> AgentState:
        logger.info("Agent node enter: unknown %s", self._user_context(state))
        state["phase"] = "N0_greet"
        state["response"] = (
            "🤔 Не совсем понял ваш запрос.\n\n"
            "Я могу:\n"
            "• Подобрать букет по бюджету\n"
            "• Рассказать о доставке\n"
            "• Ответить про оплату\n"
            "• Оформить срочный заказ\n\n"
            "Напишите подробнее, что вас интересует.\n"
            "Или свяжу с владельцем — скажите *«позовите Дмитрия»*."
        )
        state["response"] += " ||phase:N0_greet||"
        logger.info("Unknown → N0_greet %s", self._user_context(state))
        return state

    # ──────────────────────────────────────────────
    # Хелперы для форматирования
    # ──────────────────────────────────────────────

    def _wrap_response(self, state: AgentState) -> AgentState:
        """Добавляет метаданные фазы и данных в response."""
        phase = state.get("phase", "init")
        data = state.get("collected_data", {})
        if "||phase:" not in state.get("response", ""):
            state["response"] += f" ||phase:{phase}|| ||data:{json.dumps(data, ensure_ascii=False)}||"
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
        show = min(total, max_items)
        for b in bouquets[:show]:
            lines.append(
                f"- {b['Название']} — {int(b['Цена'])} руб.\n"
                f"  {b['Ссылка']}"
            )
        if total > max_items:
            rem = total - max_items
            v = self._pluralize_variant(rem)
            lines.append(f"\n||more:{rem}:{max_items}||\nХотите посмотреть ещё {rem} {v}?")
        return "\n".join(lines)

    def _format_bouquets_page(self, bouquets: List[Dict[str, Any]], start: int, count: int) -> str:
        lines = ["Ещё варианты:"]
        for b in bouquets[start:start + count]:
            lines.append(
                f"- {b['Название']} — {int(b['Цена'])} руб.\n"
                f"  {b['Ссылка']}"
            )
        return "\n".join(lines)

    def _search_bouquets_by_query(self, bouquets: List[Dict[str, Any]], query_text: str) -> List[Dict[str, Any]]:
        words = re.findall(r"[a-zA-Zа-яА-Я0-9]+", query_text.lower())
        stop_words = {
            "привет", "здравствуйте", "добрый", "хочу", "нужен",
            "нужна", "нужно", "пожалуйста",
        }
        tokens = [w for w in words if w not in stop_words and len(w) > 1]
        if not tokens:
            return []

        scored: List[tuple[int, Dict[str, Any]]] = []
        for b in bouquets:
            name = b["Название"].lower()
            score = sum(1 for t in tokens if t in name)
            if score > 0:
                scored.append((score, b))
        scored.sort(key=lambda it: (-it[0], it[1]["Цена"]))
        return [it[1] for it in scored]

    # ──────────────────────────────────────────────
    # Публичное API (для tg_bot.py)
    # ──────────────────────────────────────────────

    def get_bouquet_recommendation(
        self,
        user_input: str,
        user_id: Optional[int] = None,
        user_name: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Обрабатывает запрос пользователя через LangGraph (многошаговый диалог)."""
        if user_id is not None:
            mysql_interface.save_message(user_id, user_input)

        history = conversation_history or []
        history_lines: List[str] = []
        for item in history:
            role = item.get("role", "unknown")
            content = item.get("content", "")
            if isinstance(content, str) and content.strip():
                history_lines.append(f"{role}: {content}")

        if not history_lines or not history_lines[-1].endswith(user_input):
            history_lines.append(f"user: {user_input}")

        full_conversation_text = "\n".join(history_lines)
        logger.info(
            "Agent input: chars=%s history=%s uid=%s",
            len(user_input), len(history), user_id,
        )

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
            "phase": "init",
            "collected_data": {},
        }

        result = self.graph.invoke(graph_input)
        return result["response"]

    def get_more_bouquets(self, user_id: int, start: int, count: int) -> Optional[str]:
        bouquets = self._last_bouquets.get(user_id)
        if not bouquets or start >= len(bouquets):
            return None
        return self._format_bouquets_page(bouquets, start, count)

    def filter_bouquets_by_price(self, max_price: float):
        return [b for b in self.bouquets_data if b['Цена'] <= max_price]

    def add_new_messages_to_index(self):
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
    return bouquet.get("Изображение")


def create_price_ranges():
    return [
        ("До 5 000 руб.", "5000"),
        ("5 000-10 000 руб.", "10000"),
        ("10 000-15 000 руб.", "15000"),
        ("15 000-20 000 руб.", "20000"),
        ("Свыше 20 000 руб.", "20000+")
    ]