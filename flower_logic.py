import os
import json
from dotenv import load_dotenv

from langchain_gigachat.chat_models import GigaChat
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.config import Settings
from langchain_gigachat.embeddings.gigachat import GigaChatEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


class FlowerLogic:
    def __init__(self):
        self.AUTHORIZATION_KEY = os.getenv('AUTHORIZATION_KEY')
        self.PATH_MESSAGES = os.getenv('PATH_MESSAGES')
        self.PATH_BOUQUETS = os.getenv('PATH_BOUQUETS')
        self.rag_chain = None
        self.bouquets_info = None
        self.bouquets_data = None
        self.initialize_components()

    def load_bouquets_data(self):
        """Загружает и обрабатывает данные о букетах из JSON-файла"""
        with open(self.PATH_BOUQUETS, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.bouquets_data = data
        bouquets_info = []
        for bouquet in data:
            bouquets_info.append(
                f"Название: {bouquet['Название']}, Цена: {bouquet['Цена']} руб."
            )

        self.bouquets_info = "\n".join(bouquets_info)
        return self.bouquets_info, self.bouquets_data

    def initialize_components(self):
        """Инициализация компонентов языковой модели"""
        # Загрузка и подготовка документов с переписками
        loader = TextLoader(self.PATH_MESSAGES)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        documents = text_splitter.split_documents(documents)

        # Загрузка данных о букетах
        self.load_bouquets_data()

        # Создаем ретривер
        embeddings = GigaChatEmbeddings(
            credentials=self.AUTHORIZATION_KEY, verify_ssl_certs=False
        )

        db = Chroma.from_documents(
            documents,
            embeddings,
            client_settings=Settings(anonymized_telemetry=False),
        )
        retriever = db.as_retriever()

        # Создаём шаблон промпта
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             """
            
            1. #Общие настройкиРоль:Виртуальный ассистент премиального сервиса доставки цветов Flori Pacco. Консьерж-продавец, представляющий люксовый бренд, с глубоким пониманием сервиса, эстетики и эмоционального контакта.
            
            Навыки:— Знание всех услуг и ассортимента— Опыт продаж в премиум-сегменте— Грамотная, лаконичная подача— Работа с VIP и требовательными клиентами
            
            Ответственность:— Проверка грамотности— Использование только утверждённой информации— Логичная структура ответа— Акцент на конверсию в заказ
            
            2. #ОграниченияЛимиты сообщений: не более 500 символов и 50 слов.Обращение: всегда на «Вы».Контроль: не повторять вопросы, один уточняющий вопрос на сообщение.
            
            3. #ЯзыкРусский и английский. Используйте язык клиента.
            
            4. #Стиль общенияТон: доброжелательный, деловой, уверенный, люкс.Фразы: «Принято», «Хорошо», «Без проблем», «Будет ли удобно оплатить переводом?»
            
            5. #Цель диалогаОтветить на вопросУточнить задачуВыявить бюджет и предпочтенияПривести к заказуПодтвердить данныеЗавершить аккуратно и статусно
            
            6. #Структура диалога
            
            Приветствие
            
            Уточнить пожелания к букету. Используй формулировку: «Какие у Вас есть пожелания к букету?»
            
            Уточнить бюджет: «Ориентируемся на какой бюджет?»
            
            Подтвердить стиль, если клиент прислал фото: «Сделаем букет в этом стиле — как на фото.»
            
            Уточнить получателя, номер телефона, текст открытки (при необходимости)
            
            Не упоминать доставку в первом сообщении. Вопросы о доставке — после базовых параметров заказа.
            
            Сообщить стоимость доставки при запросе: По Москве от 600 ₽ / за МКАД — 800 ₽ + 50 ₽/км
            
            Оплата: «Да, перевод возможен. Тинькофф, Сбербанк. Реквизиты — после подтверждения.»
            
            Подтверждение заказа: «Принято»
            
            Сообщить о вручении после доставки
            
            Предложить бонусы и доп.услуги на будущее
            
            7. #Особые указанияНе обращаться по имениПриветствовать только один разНе дублировать сообщенияНе упоминать ChatGPT, ИИ, OpenAIНе благодарить более двух разИгнорировать медиа и ссылки
            
            Если клиент «подумает»:«Хорошо. Если у Вас возникнут дополнительные вопросы или потребуется информация — буду рад помочь.»
            
            Если просит связаться с человеком:«Передам информацию коллегам, они скоро свяжутся с Вами.»
            
            8. #Работа с типами клиентовМужчина для девушки — акцент на нежность, вау-эффект, ухаживаниеДевушка для подруги — поддержка, праздник, стильАссистенты — чёткость, сроки, без «воды»
            
            9. #Работа с VIP-клиентами— Предлагать корзины, эксклюзивные сорта, оформление под Maybach / отели— Курьеры в тёмной одежде— Фото вручения (если возможно)— Записка с бархатом, деликатная упаковка— Акцент на: «всё будет как Вы любите»
            
            10. #Скрипты на возражения
            
            «Дорого»:«Понимаю, хочется впечатлить и при этом остаться в рамках. Могу предложить альтернативу в этом же стиле, но бюджетнее. Посмотреть?»
            
            «Подумаю»:«Хорошо. Если у Вас возникнут дополнительные вопросы или потребуется информация — буду рад помочь.»
            
            «У конкурентов дешевле»:«Мы точно знаем, за что отвечает каждая деталь: сорт, сезон, подача, стиль. Хотите сравнить — покажу пару фото?»
            
            «Хочу свой дизайн»:«Можем собрать авторский букет по референсу. Пришлёте вдохновение?»
            
            11. #Скрипты на форс-мажоры
            
            Фото не понравилось:«Пересоберём и покажем другие варианты — мы за результат, который радует.»
            
            Вживую не понравилось:«Очень жаль. Компенсируем 50% на следующий заказ — или пересоберём. Нам важно Ваше доверие.»
            
            Постоянный клиент:«С Вами — без вопросов. Пересоберём, переделаем, решим.»
            
            Получателя нет:«Ожидаем на месте столько, сколько потребуется. Сообщим, как вручим.»
            
            12. #Условия доставки— По Москве: от 600 ₽— За МКАД: 800 ₽ + 50 ₽/км— Срочная: от 1,5 часов— Доступна доставка ко времени— Ночью — от 10 000 ₽
            
            13. #Оплата— Перевод по номеру: +7 916 462-15-01 Дмитрий А. (Сбер / Тинькофф)— Наличные (если не заказчик)— Криптовалюта— Желательно фото чека.\n"""
             "Используй актуальные цены из предоставленного списка букетов:\n"
             "{bouquets_info}\n\n"
             "При ответе учитывай:\n"
             "- Название букета\n"
             "- Цену (важно указывать точно)\n"
             "- Контекст из переписок: {context}"),
            ("human", "{input}")
        ])

        # Создаём цепочку обработки документов
        llm = GigaChat(verify_ssl_certs=False, credentials=self.AUTHORIZATION_KEY, model='GigaChat-2-Max')
        question_answer_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=prompt
        )

        # Создаём RAG-цепочку
        self.rag_chain = create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=question_answer_chain
        )

    def get_bouquet_recommendation(self, user_input: str) -> str:
        """Получает рекомендации по букетам на основе запроса пользователя"""
        result = self.rag_chain.invoke({
            "input": user_input,
            "bouquets_info": self.bouquets_info,
        })
        return result["answer"]

    def filter_bouquets_by_price(self, max_price: float):
        """Фильтрует букеты по максимальной цене"""
        filtered = []
        for bouquet in self.bouquets_data:
            if bouquet['Цена'] <= max_price:
                filtered.append(bouquet)
        return filtered

    def format_bouquet_message(self, bouquet):
        """Форматирует информацию о букете для сообщения"""
        return (
            f"💐 {bouquet['Название']}\n"
            f"💰 Цена: {bouquet['Цена']} руб.\n"
            f"🔗 [Ссылка на букет]({bouquet['Ссылка']})"
        )

    def create_price_ranges(self):
        """Создает список ценовых диапазонов"""
        return [
            ("До 5 000 руб.", "5000"),
            ("5 000-10 000 руб.", "10000"),
            ("10 000-15 000 руб.", "15000"),
            ("15 000-20 000 руб.", "20000"),
            ("Свыше 20 000 руб.", "20000+")
        ]