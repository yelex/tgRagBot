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
from langchain.memory import ConversationBufferMemory

from utils.utils import load_text

load_dotenv()


class FlowerLogic:
    def __init__(self):
        self.AUTHORIZATION_KEY = os.getenv('AUTHORIZATION_KEY')
        self.PATH_MESSAGES = os.getenv('PATH_MESSAGES')
        self.PATH_BOUQUETS = os.getenv('PATH_BOUQUETS')
        self.PATH_SYSTEM_PROMPT = os.getenv('PATH_SYSTEM_PROMPT')
        self.rag_chain = None
        self.bouquets_info = None
        self.bouquets_data = None
        self.system_prompt = None
        self.user_memories = {}  # сюда будем складывать memory для каждого пользователя
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

    def set_system_prompt(self):
        self.system_prompt = load_text(self.PATH_SYSTEM_PROMPT)

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

        # Устанавливаем системный промпт
        self.set_system_prompt()

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
             self.system_prompt
             ),
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

    def get_user_memory(self, user_id):
        if user_id not in self.user_memories:
            self.user_memories[user_id] = ConversationBufferMemory(
                return_messages=True
            )
        return self.user_memories[user_id]

    def get_bouquet_recommendation(self, user_input: str, user_id: int) -> str:
        memory = self.get_user_memory(user_id)

        # добавляем текущее сообщение в память
        memory.chat_memory.add_user_message(user_input)

        # делаем запрос с историей
        past_messages = "\n".join(
            [f"{m.type}: {m.content}" for m in memory.chat_memory.messages]
        )

        result = self.rag_chain.invoke({
            "input": f"{past_messages}\n\nПоследнее сообщение пользователя: {user_input}",
            "bouquets_info": self.bouquets_info,
        })

        # сохраняем ответ в память
        memory.chat_memory.add_ai_message(result["answer"])

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