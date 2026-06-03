import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_gigachat.chat_models import GigaChat
from langgraph.graph import END, START, StateGraph

from interfaces import mysql_interface
from interfaces.mysql_interface import create_order, update_order, get_active_order

load_dotenv()
logger = logging.getLogger(__name__)


DELIVERY_MKAD_PRICE = 600          # В пределах МКАД
DELIVERY_BEYOND_BASE = 1000         # Базовая цена за МКАД
DELIVERY_BEYOND_PER_KM = 50         # + 50 ₽/км от МКАД
FREE_DELIVERY_THRESHOLD = 20000     # Бесплатная доставка от этой суммы (МКАД)
COURIER_WAIT_FREE_MIN = 15          # Бесплатное ожидание курьера (мин)
COURIER_WAIT_PRICE_PER_10MIN = 100  # Цена за каждые 10 мин сверх лимита


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
        self.GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS")
        self.GIGACHAT_SCOPE = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
        self.GIGACHAT_MODEL = os.getenv("GIGACHAT_MODEL", "GigaChat")
        self.GIGACHAT_TIMEOUT = int(os.getenv("GIGACHAT_TIMEOUT", "60"))
        self.bouquets_info: str = ""
        self.bouquets_data: List[Dict[str, Any]] = []
        self.intent_llm: Optional[GigaChat] = None
        self.vision_llm = None  # GigaChat vision через file API — не реализован
        self.graph = None
        self._last_bouquets: Dict[int, List[Dict[str, Any]]] = {}
        self.nlu_cache: Dict[str, Dict[str, Any]] = {}
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
        if not self.GIGACHAT_CREDENTIALS:
            logger.warning("GIGACHAT_CREDENTIALS не заданы, intent LLM отключен.")
            self.intent_llm = None
            return
        try:
            self.intent_llm = GigaChat(
                credentials=self.GIGACHAT_CREDENTIALS,
                scope=self.GIGACHAT_SCOPE,
                model=self.GIGACHAT_MODEL,
                verify_ssl_certs=False,
                timeout=self.GIGACHAT_TIMEOUT,
                max_tokens=512,
            )
            logger.info("GigaChat LLM инициализирован: %s", self.GIGACHAT_MODEL)
        except Exception as exc:
            logger.exception("Не удалось инициализировать GigaChat LLM: %s", exc)
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

    # sub_phases → node_name mapping
    _PHASE_TO_NODE: Dict[str, str] = {}

    @classmethod
    def _phase_to_node_name(cls, phase: str) -> str:
        """Мапит подфазу (N2_brief_occasion) на узел графа (N2_brief)."""
        if not cls._PHASE_TO_NODE:
            # Инициализация: префикс → узел
            cls._PHASE_TO_NODE = {
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
                "N0_greet": "N0_greet",
            }
        for prefix, node in cls._PHASE_TO_NODE.items():
            if phase.startswith(prefix):
                return node
        return "router"

    def _route_after_nlu(self, state: AgentState) -> str:
        """Маршрутизация после NLU: учитывает фазу и интент."""
        phase = state.get("phase", "init")
        intent = state.get("intent", "unknown")

        # Интенты, которые могут прервать текущую фазу
        # (НЕ срабатывают внутри брифа/каталога/приветствия — там диалог идёт своим чередом)
        non_override_phases = {
            "init", "N0_greet", "N2_brief_budget", "N2_brief_occasion",
            "N2_brief_format", "N2_brief_gamma", "N2_brief_restrictions",
            "N2_brief_reference", "N3_catalog_choice",
        }
        override_intents: Dict[str, str] = {
            "recommend": "N3_catalog",
            "catalog": "N3_catalog",
            "greet": "N0_greet",
            "greet_and_offer": "N0_greet",
            "chosen_by_name": "N9_offer",
            "complaint": "N11_complaint",
            "urgent": "N5_urgent",
            "corporate": "N12_escalation",
            "faq": "N8_faq",
            "pickup": "N7_pickup",
            "occasion": "N2_brief",
            "photo_request": "N13_photo",
            "contact_owner": "N12_escalation",
        }
        if intent in override_intents and phase not in non_override_phases:
            logger.info(
                "INTENT OVERRIDE: intent=%s перехватывает фазу %s → %s uid=%s",
                intent, phase, override_intents[intent],
                state.get("user_id", "unknown"),
            )
            # Сбрасываем фазу на init, чтобы узел не думал что мы в середине диалога доставки
            state["phase"] = "init"
            return override_intents[intent]

        # Если диалог уже идёт — идём в соответствующий обработчик
        if phase != "init":
            return self._phase_to_node_name(phase)

        # Первый вход: по интенту
        phase_map: Dict[str, str] = {
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

        cache_key = state["user_input"].strip().lower()
        if cache_key in self.nlu_cache:
            return self.nlu_cache[cache_key]

        phase = state.get("phase", "init")
        prompt = f"""
Ты NLU-модуль для цветочного бота. Определи intent и сущности.

Доступные intent: greet, catalog, recommend, chosen_by_name, delivery_info, payment_info, complaint, urgent, occasion, photo_request, change_order, cancel_order, repeat_order, contact_owner, faq, unknown.

Правила извлечения:
- Цены: "до X тыс" → max_price=X*1000, "от X тыс" → min_price=X*1000
- delivery_date: дата доставки ("завтра", "15 июня", "15.06", "через 2 дня" — сохраняй как есть)
- delivery_time: время доставки ("в 14:00", "в 14 часов", "после 18" — сохраняй как есть)
- recipient_phone: телефон получателя (любой формат: +7..., 8..., 9... — сохраняй как есть)
- address: адрес доставки

Ответ JSON: {{"intent": "...", "entities": {{"max_price": null, "min_price": null, "name_query": null, "query_text": "...", "occasion": null, "urgent": null, "complaint_type": null, "photo_request": null, "product_type": null, "delivery_date": null, "delivery_time": null, "address": null, "recipient_phone": null, "gamma": null, "recipient": null, "card_text": null}}}}

История: {state["full_conversation_text"]}

Сообщение: {state["user_input"]}
""".strip()

        try:
            start_time = time.time()
            llm_result = self.intent_llm.invoke(prompt)
            end_time = time.time()
            logger.info(f"LLM call took {end_time - start_time:.2f} seconds")
            time.sleep(1)  # Rate limiting to avoid 429 errors
            content = getattr(llm_result, "content", "") if llm_result is not None else ""
            if isinstance(content, list):
                content = " ".join(str(item) for item in content)
            if not isinstance(content, str):
                content = str(content)

            payload = self._extract_first_json(content)
            intent = self._normalize_intent(str(payload.get("intent", "")))
            entities_raw = payload.get("entities", {}) if isinstance(payload, dict) else {}
            if not isinstance(entities_raw, dict):
                entities_raw = {}

            entities: Dict[str, Any] = {}
            for key in (
                "max_price", "min_price", "name_query", "query_text",
                "occasion", "urgent", "complaint_type", "corporate_size",
                "change_request", "photo_request", "repeat_reference",
                "pickup", "product_type", "delivery_date", "delivery_time",
                "address", "recipient_phone", "gamma", "recipient", "card_text",
                "anonymous", "critical_flower", "critical_color",
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
            self.nlu_cache[cache_key] = {"intent": intent, "entities": entities}
            return {"intent": intent, "entities": entities}

        except Exception as exc:
            logger.exception("NLU LLM failed, fallback: %s", exc)
            return {"intent": "unknown", "entities": {}}

    def _fallback_intent_from_rules(self, state: AgentState) -> str:
        text = state["user_input"].lower()

        has_greeting = any(m in text for m in ("привет", "здравствуйте", "добрый", "hello"))
        has_greeting = any(m in text for m in ("привет", "здравствуй", "добрый", "доброе", "хай", "хелло", "hello", "hi"))
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
        has_more = any(m in text for m in ("больше", "ещё", "еще", "показать еще", "покажи еще", "дальше", "следующие"))
        has_agreement = any(m in text for m in ("да", "хочу", "давай", "хорошо", "подходит", "ок", "согласен", "нравит", "го"))
        has_dislike = any(m in text for m in ("не то", "не нравит", "другой", "не подходит", "нет"))

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

        # "от X до Y тыс"
        m = re.search(r'(?:\s|^)от\s+(\d+(?:[.,]\d+)?)\s+до\s+(\d+(?:[.,]\d+)?)\s*тыс(?:яч)?', t)
        if m:
            result["min_price"] = float(m.group(1).replace(",",".")) * 1000
            result["max_price"] = float(m.group(2).replace(",",".")) * 1000
            return result

        # "от X тыс"
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

        # Просто число с "руб" или без — например "20000руб" или "15000"
        if not result.get("min_price") and not result.get("max_price"):
            m = re.search(r'(?:^|\s)(\d{4,6})\s*(?:руб|₽|р\.|rub)?(?:\s|$)', t)
            if m:
                val = float(m.group(1))
                if val >= 1000:
                    result["min_price"] = val

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

            # Общая реакция — спрашиваем пожелания, а не сразу бюджет
            state["phase"] = "N2_brief"
            state["response"] = (
                "Подскажите, какие у Вас пожелания к букету?\n"
                "Например: повод, любимые цветы, цветовая гамма, бюджет."
            )
            state["response"] += " ||phase:N2_brief_budget||"
            logger.info("N0_greet → N2_brief (general wishes) %s", self._user_context(state))
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
            "🌸 Добрый день! Рады приветствовать в нашей цветочной мастерской.\n\n"
            "Подскажите, пожалуйста:\n"
            "• Какой букет/композиция/коробка вас интересует?\n"
            "• Какие будут пожелания?\n\n"
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
            "• Хочу заказать букет — напишите бюджет и пожелания\n"
            "• Узнать о доставке — «доставка»\n"
            "• Посмотреть каталог — «каталог»"
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

        # Обработка общих ответов, которые не содержат полезных данных
        # чтобы не зациклиться на одном вопросе
        vague_words = ("больше", "ещё", "еще", "да", "нет", "ок", "хорошо",
                       "давай", "продолж", "дальше", "не знаю", "посоветуй",
                       "подбери", "на ваш вкус", "любой", "без разницы")
        is_vague = any(w in text for w in vague_words)

        # Если бюджет пропущен — ставим дефолт чтобы не зациклиться
        if collected.get("budget_skipped") and not collected.get("budget_max"):
            collected["budget_min"] = 5000.0
            collected["budget_max"] = 15000.0

        # Если повод пропущен — отмечаем чтобы не спрашивать
        if collected.get("occasion_skipped") and not collected.get("occasion"):
            collected["occasion"] = "не указан"

        # Сохраняем данные из сообщения
        if entities.get("max_price") or entities.get("min_price"):
            bmin, bmax = self._normalize_budget_range(
                entities.get("min_price"), entities.get("max_price")
            )
            collected["budget_min"] = bmin
            collected["budget_max"] = bmax
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
                bmin, bmax = self._normalize_budget_range(
                    pf.get("min_price"), pf.get("max_price")
                )
                collected["budget_min"] = bmin
                collected["budget_max"] = bmax
                if collected.get("budget_min") and collected["budget_min"] < 5000:
                    collected["budget_min"] = 5000

        # Проверка минимального бюджета
        budget = collected.get("budget_max") or collected.get("budget_min") or 0
        budget_ok = budget >= 5000 or (collected.get("budget_min") and collected["budget_min"] >= 5000)

        # Определяем, какой вопрос задать
        # Если ответ размытый ("больше", "ещё", "да") — пропускаем текущий вопрос
        if not collected.get("budget_max") and not collected.get("budget_min"):
            if is_vague:
                # Клиент не хочет называть бюджет — подбираем на наш вкус
                collected["budget_skipped"] = True
                state["response"] = (
                    "Понял! Подберу на наш вкус.\n"
                    "Кому и по какому поводу выбираем?"
                )
                state["phase"] = "N2_brief_occasion"
                state["response"] += " ||phase:N2_brief_occasion||"
                state["collected_data"] = collected
                logger.info("N2_brief: budget skipped (vague), ask occasion %s", self._user_context(state))
                return self._wrap_response(state)
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

        if not collected.get("occasion") and not collected.get("recipient") and not collected.get("budget_skipped"):
            if is_vague:
                collected["occasion_skipped"] = True
            if not collected.get("occasion_skipped"):
                state["phase"] = "N2_brief_occasion"
                state["keyboard"] = ["Девушке", "Маме", "Коллеге", "Свадьба", "День рождения"]
                state["response"] = "К какому поводу и кому выбираем?"
                state["response"] += " ||phase:N2_brief_occasion||"
                logger.info("N2_brief: ask occasion %s", self._user_context(state))
                return self._wrap_response(state)
            # occasion пропущен — идём дальше
            collected["occasion"] = "не указан"

        if not collected.get("product_type"):
            if is_vague:
                collected["product_type"] = "букет"
            else:
                state["phase"] = "N2_brief_format"
                state["keyboard"] = ["Букет", "Композиция (коробка/корзина)", "Кашпо"]
                state["response"] = "Что вам ближе: букет, композиция или кашпо?"
                state["response"] += " ||phase:N2_brief_format||"
                logger.info("N2_brief: ask format %s", self._user_context(state))
                return self._wrap_response(state)

        if not collected.get("gamma"):
            if is_vague:
                collected["gamma"] = "не указан"
            else:
                state["phase"] = "N2_brief_gamma"
                state["keyboard"] = ["Нежно-пастельный", "Яркий/насыщенный"]
                state["response"] = "Какой стиль предпочитаете?"
                state["response"] += " ||phase:N2_brief_gamma||"
                logger.info("N2_brief: ask gamma %s", self._user_context(state))
                return self._wrap_response(state)

        if "restrictions_asked" not in collected:
            if is_vague:
                collected["restrictions_asked"] = True
            else:
                collected["restrictions_asked"] = True
                state["phase"] = "N2_brief_restrictions"
                state["keyboard"] = ["Нет ограничений", "Без лилий", "Без резкого аромата"]
                state["response"] = "Есть ли цветы, которые точно нельзя?\n(аллергии, запах, кошки, лилии и т.п.)"
                state["response"] += " ||phase:N2_brief_restrictions||"
                logger.info("N2_brief: ask restrictions %s", self._user_context(state))
                return self._wrap_response(state)

        if not collected.get("size") and "size_asked" not in collected:
            if is_vague:
                collected["size_asked"] = True
            else:
                collected["size_asked"] = True
                state["phase"] = "N2_brief_restrictions"
                state["keyboard"] = ["Компактный (S)", "Средний (M)", "Пышный (L)"]
                state["response"] = "Какой размер предпочитаете?"
                state["response"] += " ||phase:N2_brief_restrictions||"
                logger.info("N2_brief: ask size %s", self._user_context(state))
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
            lines = ["📋 Категории букетов:\n"]
            for name, items in categories:
                if items:
                    lines.append(f"• {name} — {len(items)} {self._pluralize_variant(len(items))}")
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
                "🏪 Самовывоз\n\n"
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
            response = "📌 Минимальный заказ — 5 000 ₽."
        elif any(w in text for w in ("анонимн", "тайно", "секрет")):
            response = (
                "🤫 Анонимная доставка — возможна. "
                "Можем не указывать отправителя. "
                "Открытка тоже может быть без подписи."
            )
        elif any(w in text for w in ("открытк", "записк")):
            response = (
                "✉️ Открытка — бесплатно!\n"
                "• Рукописная — любой текст\n"
                "• Можно напечатать\n"
                "• Можно анонимно"
            )
        elif any(w in text for w in ("ваз", "вазу")):
            response = "🏺 Вазы: стеклянные, от 1 500 ₽."
        elif any(w in text for w in ("свежест", "гаранти")):
            response = (
                "🌷 Гарантия свежести: цветы закупаем ежедневно.\n"
                "Если завяли за 1–2 дня — разберёмся.\n"
                "Запросим фото и условия хранения.\n"
                "Если наша вина — скидка 50% на следующий заказ."
            )
        elif any(w in text for w in ("замена", "замен")):
            response = (
                "🔄 Замены: в той же гамме и ценовом диапазоне.\n"
                "Незначительные (10–15% состава) — без согласования.\n"
                "Если меняется ключевой цветок/цвет — согласуем с Вами.\n"
                "По референсу — максимально похоже."
            )
        elif any(w in text for w in ("работа", "график", "часы")):
            response = (
                "🕐 Приём заказов: 9:00–23:00.\n"
                "🚚 Доставка: 24/7."
            )
        elif any(w in text for w in ("мкад", "достав", "километр", "за город")):
            response = (
                "🚚 Доставка:\n"
                "• МКАД — от 600 ₽ (бесплатно при заказе от 20 000 ₽)\n"
                "• За МКАД — 1 000 ₽ + 50 ₽/км\n"
                "• Срочная — от 1,5 часов\n"
                "• Ночная (23:00–09:00) — заранее оформленные заказы"
            )
        elif any(w in text for w in ("оплат", "карт", "нал", "перевод")):
            response = (
                "💳 Оплата:\n"
                "• Перевод на карту (Сбер / Тинькофф)\n"
                "• Ссылка на оплату\n"
                "• Наличные при получении\n"
                "• Юрлица — по реквизитам\n\n"
                "Новый заказ запускается в работу после оплаты."
            )
        elif any(w in text for w in ("отмен", "вернуть", "депозит")):
            response = (
                "Перенос до отправки — бесплатно.\n"
                "Отмена после сборки — цветы уже подготовлены, "
                "предлагаем депозит на следующий заказ."
            )
        elif any(w in text for w in ("ожидан", "курьер", "ждать")):
            response = (
                "⏱ Курьер ожидает 15 минут бесплатно.\n"
                "Далее — 100 ₽ за каждые 10 минут."
            )
        elif any(w in text for w in ("повторн", "не дозвон", "не вруч")):
            response = (
                "Если получателя не оказалось — договариваемся о повторной доставке.\n"
                "По тем же тарифам. Если по нашей вине — бесплатно."
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

        # Если клиент выбрал букет по имени или позиции — переходим к доставке
        text_lower = state["user_input"].lower()
        name_query = entities.get("name_query", "")

        # Позиционный выбор: "первый/1", "второй/2", "третий/3/последний"
        positional_match = None
        if state.get("phase") == "N9_offer" and user_id is not None:
            last = self._last_bouquets.get(user_id, [])
            if last:
                pos_map = {
                    0: ("первый", "1 вариант", "вариант 1", "первый вариант"),
                    1: ("второй", "2 вариант", "вариант 2", "второй вариант"),
                    2: ("третий", "3 вариант", "вариант 3", "третий вариант",
                        "последний", "последний вариант"),
                }
                for idx, keywords in pos_map.items():
                    if any(kw in text_lower for kw in keywords) and idx < len(last):
                        positional_match = last[idx]
                        break
                # Просто "3" или "2" или "1" в сообщении
                if not positional_match:
                    for idx in range(min(3, len(last))):
                        if str(idx + 1) in text_lower.split():
                            positional_match = last[idx]
                            break

        chosen = positional_match
        if not chosen and name_query and state.get("phase") == "N9_offer":
            chosen = next(
                (b for b in bouquets if name_query.lower() in b["Название"].lower()),
                None,
            )

        if chosen and state.get("phase") == "N9_offer":
            match = chosen
            collected["bouquet_name"] = match["Название"]
            collected["bouquet_price"] = match["Цена"]
            state["collected_data"] = collected
            state["phase"] = "N10_delivery"
            state["response"] = (
                f"Отлично, {match['Название']} — хороший выбор!\n\n"
                "Теперь нужны данные для доставки.\n"
                "Напишите адрес доставки."
            )
            state["response"] += f" ||phase:N10_delivery|| ||data:{json.dumps(collected, ensure_ascii=False)}||"
            logger.info(
                "N9_offer: bouquet chosen '%s' → N10_delivery %s",
                match["Название"], self._user_context(state),
            )
            return state

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
        labels = ["✅ Вариант 1", "⭐ Вариант 2", "🏆 Вариант 3"]
        for i, b in enumerate(unique_picks[:3]):
            label = labels[i] if i < len(labels) else f"Вариант {i+1}"
            lines.append(
                f"{label}\n"
                f"💐 {b['Название']}\n"
                f"💰 {int(b['Цена'])} ₽\n"
                f"🔗 {b['Ссылка']}\n"
            )

        lines.append("Можно добавить:")
        lines.append("• ✉️ Открытка — бесплатно")
        lines.append("• 🏺 Ваза стеклянная — от 1 500 ₽")
        lines.append("• 🎈 Шарики, сладкое — по запросу")
        lines.append("\nКакой вариант больше нравится? Или хотите что-то поменять?")

        state["response"] = "\n".join(lines)
        state["phase"] = "N9_offer"
        state["collected_data"] = collected
        state["response"] += f" ||phase:N9_offer|| ||data:{json.dumps(collected, ensure_ascii=False)}||"
        logger.info("N9_offer: %d options presented %s", len(unique_picks), self._user_context(state))
        return state

    def _N9_confirm_node(self, state: AgentState) -> AgentState:
        """После выбора букета — спрашиваем открытку и контакт получателя."""
        logger.info("Agent node enter: N9_confirm %s", self._user_context(state))

        collected = state.get("collected_data", {})
        text = state["user_input"].lower()

        # Шаг 1: текст открытки
        if "card_asked" not in collected:
            collected["card_asked"] = True
            state["phase"] = "N9_offer"
            state["keyboard"] = ["Нужна открытка", "Без открытки"]
            state["response"] = (
                "✉️ Открытка — бесплатно, рукописная.\n\n"
                "Нужна? Если да — напишите текст."
            )
            state["response"] += f" ||phase:N9_offer|| ||data:{json.dumps(collected, ensure_ascii=False)}||"
            state["collected_data"] = collected
            logger.info("N9_confirm: ask card %s", self._user_context(state))
            return self._wrap_response(state)

        # Шаг 2: телефон получателя
        if not collected.get("recipient_phone") and "phone_asked" not in collected:
            collected["phone_asked"] = True
            state["phase"] = "N10_delivery"
            state["response"] = (
                "📞 Подскажите контакт получателя (телефон).\n"
                "Нужен для уточнения деталей доставки."
            )
            state["response"] += f" ||phase:N10_delivery|| ||data:{json.dumps(collected, ensure_ascii=False)}||"
            state["collected_data"] = collected
            logger.info("N9_confirm: ask phone %s", self._user_context(state))
            return self._wrap_response(state)

        # Всё собрано — переходим к доставке
        state["phase"] = "N10_delivery"
        return self._N10_delivery_node(state)

    # ──────────────────────────────────────────────
    # N10 — Доставка (сбор данных + цена)
    # ──────────────────────────────────────────────

    def _N10_delivery_node(self, state: AgentState) -> AgentState:
        logger.info("Agent node enter: N10_delivery %s", self._user_context(state))

        collected = state.get("collected_data", {})
        entities = state.get("entities", {})
        text = state["user_input"]
        text_lower = text.lower()

        # ── Извлекаем все поля из entities и текста ──────────────────────────
        if entities.get("address"):
            collected["address"] = entities["address"]
        if not collected.get("address"):
            m = re.search(r'(?:адрес|по адресу|на? )([а-яА-ЯёЁ\s\d.,/-]{5,})', text_lower)
            if m:
                collected["address"] = m.group(1).strip()

        if entities.get("delivery_date"):
            collected["delivery_date"] = entities["delivery_date"]

        if entities.get("delivery_time"):
            collected["delivery_time"] = entities["delivery_time"]
        if not collected.get("delivery_time"):
            m = re.search(r'(?:в\s)?(\d{1,2}[:.]\d{2}|\d{1,2}\s*ч(?:ас)?)', text_lower)
            if m:
                collected["delivery_time"] = m.group(0).strip()

        if entities.get("recipient_phone"):
            collected["recipient_phone"] = entities["recipient_phone"]
        if not collected.get("recipient_phone"):
            phone = self._extract_phone_from_text(text)
            if phone:
                collected["recipient_phone"] = phone

        # ── Шаг 1: адрес ─────────────────────────────────────────────────────
        if not collected.get("address"):
            state["collected_data"] = collected
            state["phase"] = "N10_delivery"
            state["response"] = "Напишите, пожалуйста, адрес доставки полностью."
            return self._wrap_response(state)

        # ── За МКАД: уточняем км ──────────────────────────────────────────────
        beyond_mkad = any(w in text_lower for w in (
            "за мкад", "замкад", "область", "московская область",
            "подмосковье", "за город",
        ))
        if beyond_mkad and not collected.get("beyond_km"):
            state["collected_data"] = collected
            state["phase"] = "N10_delivery"
            state["response"] = (
                "Доставка за МКАД: 1 000 ₽ + 50 ₽/км от МКАД.\n\n"
                "Подскажите примерное расстояние от МКАД в км?\n"
                "Или назовите район — посчитаем."
            )
            return self._wrap_response(state)

        # ── Расчёт стоимости доставки ─────────────────────────────────────────
        if beyond_mkad and collected.get("beyond_km"):
            km = collected["beyond_km"]
            delivery_price = DELIVERY_BEYOND_BASE + DELIVERY_BEYOND_PER_KM * km
            delivery_line = f"🚚 Доставка: {int(delivery_price):,} ₽ (за МКАД, {int(km)} км)"
        else:
            total = collected.get("budget_max") or 0
            delivery_price = 0 if total >= FREE_DELIVERY_THRESHOLD else DELIVERY_MKAD_PRICE
            delivery_line = (
                "🚚 Доставка: бесплатно (заказ от 20 000 ₽)"
                if delivery_price == 0
                else f"🚚 Доставка: {delivery_price} ₽"
            )
        collected["delivery_cost"] = delivery_price

        # ── Шаг 2: дата ──────────────────────────────────────────────────────
        if not collected.get("delivery_date"):
            state["collected_data"] = collected
            state["phase"] = "N10_delivery"
            state["keyboard"] = ["Сегодня", "Завтра", "Другая дата"]
            state["response"] = "На какую дату нужна доставка?"
            return self._wrap_response(state)

        # ── Шаг 3: время ─────────────────────────────────────────────────────
        if not collected.get("delivery_time"):
            state["collected_data"] = collected
            state["phase"] = "N10_delivery"
            state["response"] = (
                "В какое время доставить?\n\n"
                "Доставка работает 24/7. Срочная — от 1,5 часов.\n"
                "Ночная (23:00–09:00) — только заранее оформленные."
            )
            return self._wrap_response(state)

        # ── Шаг 4: телефон получателя ────────────────────────────────────────
        if not collected.get("recipient_phone") and not collected.get("phone_skipped"):
            state["collected_data"] = collected
            state["phase"] = "N10_delivery"
            state["keyboard"] = ["Не знаю телефон"]
            state["response"] = (
                "📞 Телефон получателя?\n\n"
                "Нужен на случай, если курьер не найдёт адрес.\n"
                "Если не знаете — нажмите «Не знаю телефон»."
            )
            return self._wrap_response(state)

        # Если клиент нажал «Не знаю телефон»
        if "не знаю" in text_lower or "не знаю телефон" in text_lower:
            collected["phone_skipped"] = True
            collected["recipient_phone"] = None

        # ── Всё собрано → сводка → payment ───────────────────────────────────
        state["collected_data"] = collected
        state = self._upsert_order(state)
        collected = state["collected_data"]

        order_id = collected.get("order_id", "—")
        bouquet_line = f"💐 {collected['bouquet_name']}" if collected.get("bouquet_name") else "💐 Букет"
        phone_line = collected.get("recipient_phone") or "не указан"

        state["response"] = (
            f"📋 Детали заказа #{order_id}:\n\n"
            f"{bouquet_line}\n"
            f"📍 {collected['address']}\n"
            f"📅 {collected['delivery_date']}\n"
            f"⏰ {collected['delivery_time']}\n"
            f"📞 Получатель: {phone_line}\n"
            f"{delivery_line}\n\n"
            "⏱ Курьер ожидает 15 мин бесплатно, далее 100 ₽/10 мин.\n\n"
            "Всё верно? Переходим к оплате?"
        )
        state["phase"] = "payment"
        state["collected_data"] = collected
        state["response"] += f" ||phase:payment|| ||data:{json.dumps(collected, ensure_ascii=False)}||"
        logger.info("N10_delivery: all data collected → payment %s", self._user_context(state))
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
                "Примите наши извинения.\n\n"
                "Чтобы разобраться, напишите, пожалуйста:\n"
                "• Когда получили букет\n"
                "• Что именно не так (фото приложите)\n"
                "• Условия хранения (в вазе/без, температура)\n\n"
                "Это поможет нам понять причину и предложить решение."
            )
            state["response"] += " ||phase:N11_complaint_facts||"
            state["collected_data"] = collected
            logger.info("N11_complaint: collect facts %s", self._user_context(state))
            return self._wrap_response(state)

        # Факты собраны — даём конкретные обещания из политики
        text = state["user_input"].lower()

        # Проверка на неправильное хранение
        bad_storage = any(w in text for w in (
            "холод", "балкон", "окно", "батарея", "солнц",
            "фрукт", "яблок", "банан",
        ))

        if bad_storage:
            state["response"] = (
                "Понимаю Ваше разочарование. К сожалению, при указанных условиях "
                "хранения компенсация не предусмотрена — цветы чувствительны "
                "к температуре и окружению.\n\n"
                "💡 Рекомендации по уходу:\n"
                "• Вода комнатной температуры, менять раз в 2 дня\n"
                "• Подрезать стебли под углом\n"
                "• Не ставить рядом с фруктами и на прямое солнце\n\n"
                "Если есть вопросы — напишите, постараюсь помочь."
            )
            state["phase"] = "N15_close"
            state["response"] += " ||phase:N15_close||"
            logger.info("N11_complaint: bad storage, no compensation %s", self._user_context(state))
            return state

        # Наша вина — предлагаем компенсацию
        state["response"] = (
            "Спасибо за информацию. Понимаю ситуацию и предлагаю:\n\n"
            "• Скидка 50% на следующий заказ\n"
            "• Или скидка 50% на следующие 2 заказа\n\n"
            "Также можем пересобрать букет. Что Вам будет удобнее?"
        )
        state["phase"] = "N12_escalated"
        collected["escalation_reason"] = "complaint_compensation"
        state["collected_data"] = collected
        state["response"] += " ||phase:N12_escalated||"
        logger.info("N11_complaint → N12 (compensation) %s", self._user_context(state))
        return state

    # ──────────────────────────────────────────────
    # N12 — Эскалация владельцу
    # ──────────────────────────────────────────────

    def _N12_escalation_node(self, state: AgentState) -> AgentState:
        logger.info("Agent node enter: N12_escalation %s", self._user_context(state))

        collected = state.get("collected_data", {})
        reason = collected.get("escalation_reason", "other")

        escalation_payload = {
            "reason": reason,
            "order_id": collected.get("order_id"),
            "bouquet_name": collected.get("bouquet_name"),
            "budget_max": collected.get("budget_max"),
            "address": collected.get("address"),
        }

        state["phase"] = "N12_escalated"
        state["response"] = "📞 Сейчас всё уточню и вернусь к вам."
        state["response"] += (
            f" ||phase:N12_escalated||"
            f" ||escalate:{json.dumps(escalation_payload, ensure_ascii=False)}||"
        )
        logger.info("N12_escalation: reason=%s %s", reason, self._user_context(state))
        return state

    # ──────────────────────────────────────────────
    # N13 — Фото перед доставкой
    # ──────────────────────────────────────────────

    def _N13_photo_node(self, state: AgentState) -> AgentState:
        logger.info("Agent node enter: N13_photo %s", self._user_context(state))

        collected = state.get("collected_data", {})
        is_returning = collected.get("is_returning")

        if "photo_requested" not in collected:
            collected["photo_requested"] = True
            state["phase"] = "N13_photo_approval"

            if is_returning:
                state["response"] = (
                    "📸 Фото букета\n\n"
                    "Постоянным клиентам обычно отправляем фото после доставки. "
                    "Но если хотите увидеть перед отправкой — без проблем, пришлём.\n\n"
                    "Нужно фото до доставки?"
                )
            else:
                state["response"] = (
                    "📸 Перед отправкой пришлём фото букета — "
                    "чтобы Вы убедились, что всё как нужно.\n\n"
                    "Если заказ уже оформлен — напишите, отправлю фото."
                )

            state["response"] += " ||phase:N13_photo_approval||"
            state["collected_data"] = collected
            logger.info("N13_photo: first request (returning=%s) %s", is_returning, self._user_context(state))
            return self._wrap_response(state)

        # Если клиент недоволен фото → пересоберём или эскалация
        text = state["user_input"].lower()
        if any(w in text for w in ("не нравит", "не то", "не так", "плох", "переделай")):
            collected["escalation_reason"] = "photo_dispute"
            state["collected_data"] = collected
            state["phase"] = "N12_escalated"
            state["response"] = (
                "Пересоберём и покажем другие варианты — "
                "мы за результат, который радует."
            )
            state["response"] += " ||phase:N12_escalated||"
            logger.info("N13_photo → N12 (dispute) %s", self._user_context(state))
            return state

        # Фото одобрено → заказ в работу
        self._set_order_status(state, "in_progress")
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

        self._set_order_status(state, "delivery")
        state["phase"] = "N15_close"
        state["response"] = (
            "Спасибо за заказ! Надеемся, букет порадует получателя 🌸\n\n"
            "Если в будущем понадобится:\n"
            "— оформление цветами отеля\n"
            "— встреча с букетом\n"
            "— салют или личные поручения\n"
            "— просто напишите.\n\n"
            "Будем рады помочь в любой ситуации."
        )
        state["response"] += " ||phase:N15_close||"
        logger.info("N14_delivery_exec → N15_close %s", self._user_context(state))
        return state

    # ──────────────────────────────────────────────
    # N15 — Закрытие
    # ──────────────────────────────────────────────

    def _N15_close_node(self, state: AgentState) -> AgentState:
        logger.info("Agent node enter: N15_close %s", self._user_context(state))
        self._set_order_status(state, "delivered")

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

        # Определяем — новый или постоянный клиент
        is_new = not collected.get("is_returning")

        if is_new:
            state["response"] = (
                "💳 Для оформления заказа необходима предоплата.\n\n"
                "Способы оплаты:\n"
                "• 💸 Перевод на карту (Сбер / Тинькофф)\n"
                "• 🔗 Ссылка на оплату\n"
                "• 💵 Наличные при получении — только для постоянных клиентов\n\n"
                "📋 Правила:\n"
                "• Новый заказ запускается в работу после оплаты\n"
                "• Композиция от 15 000 ₽ с самовывозом — возможна 50% предоплата\n"
                "• Юрлица — оплата по реквизитам\n\n"
                "Как вам удобнее оплатить?"
            )
        else:
            state["response"] = (
                "💳 Способы оплаты:\n\n"
                "• 💸 Перевод на карту (Сбер / Тинькофф)\n"
                "• 🔗 Ссылка на оплату\n"
                "• 💵 Наличные при получении\n"
                "• 🏢 Юрлица — оплата по реквизитам\n\n"
                "Как вам удобнее?"
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
        """Добавляет метаданные фазы, данных и клавиатуры в response."""
        phase = state.get("phase", "init")
        data = state.get("collected_data", {})
        keyboard = state.get("keyboard")  # список строк для кнопок-саджестов
        # Всегда перезаписываем метаданные в конце строки
        # (убираем старые, если были)
        resp = state.get("response", "")
        resp = re.sub(r'\s*\|\|phase:[^\|]*\|\|\s*', '', resp)
        resp = re.sub(r'\s*\|\|data:\{[^\}]*\}\|\|\s*', '', resp)
        resp = re.sub(r'\s*\|\|keyboard:\[.*?\]\|\|\s*', '', resp)
        state["response"] = resp.strip() + f" ||phase:{phase}|| ||data:{json.dumps(data, ensure_ascii=False)}||"
        if keyboard:
            state["response"] += f" ||keyboard:{json.dumps(keyboard, ensure_ascii=False)}||"
        return state

    @staticmethod
    def _extract_first_json(text: str) -> dict:
        """Извлекает первый полный JSON-объект из текста, игнорируя всё после него."""
        start = text.find("{")
        if start == -1:
            return {}
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except (json.JSONDecodeError, ValueError):
                        return {}
        return {}

    @staticmethod
    def _extract_phone_from_text(text: str) -> Optional[str]:
        """Ищет телефонный номер в произвольном тексте."""
        m = re.search(r'(\+7|8|7)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}', text)
        return m.group(0).strip() if m else None

    def _upsert_order(self, state: AgentState) -> AgentState:
        """Создаёт заказ в БД при первом вызове, обновляет при повторном.
        order_id сохраняется в collected_data для персистентности."""
        collected = state.get("collected_data", {})
        user_id = state.get("user_id")
        if user_id is None:
            return state
        try:
            order_id = collected.get("order_id")
            if order_id:
                update_order(
                    order_id,
                    bouquet_name=collected.get("bouquet_name"),
                    bouquet_price=collected.get("bouquet_price"),
                    delivery_address=collected.get("address"),
                    delivery_date=collected.get("delivery_date"),
                    delivery_time=collected.get("delivery_time"),
                    recipient_phone=collected.get("recipient_phone"),
                    card_text=collected.get("card_text"),
                    delivery_cost=collected.get("delivery_cost", 0),
                    total_cost=(collected.get("bouquet_price") or 0) + (collected.get("delivery_cost") or 0),
                )
            else:
                order_id = create_order(
                    user_id=user_id,
                    user_name=state.get("user_name"),
                    data=collected,
                )
                collected["order_id"] = order_id
                state["collected_data"] = collected
            logger.info("Order upserted: order_id=%s uid=%s", order_id, user_id)
        except Exception as exc:
            logger.exception("Failed to upsert order: %s", exc)
        return state

    def _set_order_status(self, state: AgentState, status: str) -> None:
        """Обновляет статус заказа в БД. Не прерывает поток при ошибке."""
        order_id = state.get("collected_data", {}).get("order_id")
        if not order_id:
            return
        try:
            update_order(order_id, order_status=status)
            logger.info("Order status → %s: order_id=%s", status, order_id)
        except Exception as exc:
            logger.exception("Failed to update order status: %s", exc)

    @staticmethod
    def _normalize_budget_range(min_price: Optional[float], max_price: Optional[float]):
        """Если указана только одна граница — расширяет до диапазона ±10%.
        Возвращает (budget_min, budget_max)."""
        if min_price and not max_price:
            # Пользователь сказал «от X» или просто «X руб» — даём диапазон ±10%
            return (min_price * 0.9, min_price * 1.1)
        if max_price and not min_price:
            return (max_price * 0.9, max_price * 1.1)
        return (min_price, max_price)

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

        # Limit history to last 10 messages to prevent memory issues
        history_lines = history_lines[-10:]

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

    def handle_photo_message(
        self,
        user_id: Optional[int],
        user_name: Optional[str],
        photo_bytes: bytes,
        caption: Optional[str],
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Обрабатывает входящее фото. Если есть caption — обрабатываем как текст."""
        if caption and not caption.startswith("http"):
            return self.get_bouquet_recommendation(
                user_input=caption,
                user_id=user_id,
                user_name=user_name,
                conversation_history=conversation_history,
            )
        return (
            "Получил фото! Опишите, пожалуйста, что вас привлекло — "
            "цветы, цвета, стиль?\n"
            "Подберу похожий вариант из каталога."
        )

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