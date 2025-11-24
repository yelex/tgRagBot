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
from langchain_core.documents import Document
from interfaces import mysql_interface

load_dotenv()

class FlowerLogic:
    def __init__(self):
        self.AUTHORIZATION_KEY = os.getenv('AUTHORIZATION_KEY')
        self.PATH_MESSAGES = os.getenv('PATH_MESSAGES')
        self.PATH_BOUQUETS = os.getenv('PATH_BOUQUETS')
        self.PERSIST_DIR = 'chroma_db'
        self.rag_chain = None
        self.bouquets_info = None
        self.bouquets_data = None
        self.embeddings = None
        self.user_memories = {}  # user_id -> conversation memory
        self.initialize_components()

    def load_bouquets_data(self):
        """Загружает данные о букетах из JSON"""
        with open(self.PATH_BOUQUETS, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.bouquets_data = data
        bouquets_info = []
        for bouquet in data:
            bouquets_info.append(
                f"Название: {bouquet['Название']}, Цена: {bouquet['Цена']} руб."
            )
        self.bouquets_info = "\n".join(bouquets_info)

    def initialize_components(self):
        """Создает или загружает векторное хранилище"""
        print("🌸 Инициализация компонентов...")
        self.embeddings = GigaChatEmbeddings(
            credentials=self.AUTHORIZATION_KEY, verify_ssl_certs=False
        )

        # Загружаем данные о букетах
        self.load_bouquets_data()

        # Инициализация Chroma
        client_settings = Settings(
            anonymized_telemetry=False,
            persist_directory=self.PERSIST_DIR,
            is_persistent=True
        )

        if os.path.exists(self.PERSIST_DIR) and os.listdir(self.PERSIST_DIR):
            print("🔄 Найдена сохранённая Chroma-база. Загружаем...")
            self.db = Chroma(
                persist_directory=self.PERSIST_DIR,
                embedding_function=self.embeddings,
                client_settings=client_settings
            )
        else:
            print("⚡️ Индексируем переписки впервые...")
            loader = TextLoader(self.PATH_MESSAGES)
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(documents)

            self.db = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.PERSIST_DIR,
                client_settings=client_settings
            )
            print("✅ Индексация завершена.")

        retriever = self.db.as_retriever()

        # Шаблон для RAG
        system_prompt_text = self.load_system_prompt()

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt_text),
            ("human", "{input}")
        ])

        llm = GigaChat(verify_ssl_certs=False,
                       credentials=self.AUTHORIZATION_KEY,
                       model='GigaChat-2-Max')

        question_answer_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=prompt
        )

        self.rag_chain = create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=question_answer_chain
        )

        print("🌸 RAG-цепочка готова!")

    def load_system_prompt(self):
        with open(os.getenv('PATH_SYSTEM_PROMPT'), 'r', encoding='utf-8') as f:
            return f.read()

    def get_user_memory(self, user_id):
        if user_id not in self.user_memories:
            from langchain.memory import ConversationBufferMemory
            self.user_memories[user_id] = ConversationBufferMemory(return_messages=True)
        return self.user_memories[user_id]

    def get_bouquet_recommendation(self, user_input: str, user_id: int) -> str:
        """Обрабатывает запрос пользователя"""
        # Сохраняем в MySQL
        mysql_interface.save_message(user_id, user_input)

        memory = self.get_user_memory(user_id)
        memory.chat_memory.add_user_message(user_input)

        past_messages = "\n".join(
            [f"{m.type}: {m.content}" for m in memory.chat_memory.messages]
        )

        result = self.rag_chain.invoke({
            "input": f"{past_messages}\n\nПоследнее сообщение пользователя: {user_input}",
            "bouquets_info": self.bouquets_info,
        })

        memory.chat_memory.add_ai_message(result["answer"])
        return result["answer"]

    def filter_bouquets_by_price(self, max_price: float):
        return [b for b in self.bouquets_data if b['Цена'] <= max_price]

    def format_bouquet_message(self, bouquet):
        return (
            f"💐 {bouquet['Название']}\n"
            f"💰 Цена: {bouquet['Цена']} руб.\n"
            f"🔗 [Ссылка на букет]({bouquet['Ссылка']})"
        )

    def create_price_ranges(self):
        return [
            ("До 5 000 руб.", "5000"),
            ("5 000-10 000 руб.", "10000"),
            ("10 000-15 000 руб.", "15000"),
            ("15 000-20 000 руб.", "20000"),
            ("Свыше 20 000 руб.", "20000+")
        ]

    def add_new_messages_to_index(self):
        """Обновляет Chroma из MySQL"""
        print("⚡️ Загружаем все переписки из MySQL...")
        all_texts = mysql_interface.load_all_messages()
        print(f"✅ Найдено сообщений: {len(all_texts)}")

        documents = [Document(page_content=txt) for txt in all_texts]
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)

        # Создаем новую коллекцию с обновленными документами
        self.db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.PERSIST_DIR,
            client_settings=Settings(
                anonymized_telemetry=False,
                persist_directory=self.PERSIST_DIR,
                is_persistent=True
            )
        )
        print("✅ Chroma-индекс обновлён!")