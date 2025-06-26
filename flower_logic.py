import os
import json
import logging
from typing import List, Union
from dotenv import load_dotenv

from langchain_gigachat.chat_models import GigaChat
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.config import Settings
from langchain_gigachat.embeddings.gigachat import GigaChatEmbeddings
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage, BaseMessage

load_dotenv()
logger = logging.getLogger(__name__)


def create_price_ranges():
    return [
        ("До 5 000 руб.", "5000"),
        ("5 000-10 000 руб.", "10000"),
        ("10 000-15 000 руб.", "15000"),
        ("15 000-20 000 руб.", "20000"),
        ("Свыше 20 000 руб.", "20000+")
    ]


def format_bouquet_message(bouquet):
    return (
        f"💐 *{bouquet['Название']}*\n"
        f"💰 Цена: {bouquet['Цена']} руб.\n"
        f"🔗 [Ссылка на букет]({bouquet['Ссылка']})"
    )


class FlowerLogic:
    def __init__(self):
        self.AUTHORIZATION_KEY = os.getenv('AUTHORIZATION_KEY')
        self.PATH_MESSAGES = os.getenv('PATH_MESSAGES')
        self.PATH_BOUQUETS = os.getenv('PATH_BOUQUETS')

        self.bouquets_info, self.bouquets_data = self.load_bouquets_data()

        self.llm = GigaChat(
            credentials=self.AUTHORIZATION_KEY,
            verify_ssl_certs=False,
            model='GigaChat-2-Max',
            temperature=0.3,
            top_p=0.9
        )

        self.embeddings = GigaChatEmbeddings(
            credentials=self.AUTHORIZATION_KEY,
            verify_ssl_certs=False
        )

        loader = TextLoader(self.PATH_MESSAGES)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = splitter.split_documents(documents)

        self.db = Chroma.from_documents(
            documents,
            self.embeddings,
            client_settings=Settings(anonymized_telemetry=False),
        )
        self.retriever = self.db.as_retriever()

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="question"
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Ты - консультант цветочного магазина. Отвечай на вопросы клиента на основе предоставленных переписок и информации о букетах.\n"
             "Всегда учитывай историю диалога и предыдущие вопросы клиента.\n"
             "Актуальные цены на букеты:\n{bouquets_info}\n\n"
             "Контекст из переписок: {context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": self.prompt},
            chain_type="stuff"
        )

    def load_bouquets_data(self):
        with open(self.PATH_BOUQUETS, 'r', encoding='utf-8') as f:
            data = json.load(f)

        bouquets_info = [
            f"Название: {b['Название']}, Цена: {b['Цена']} руб., Ссылка: {b['Ссылка']}"
            for b in data
        ]
        return "\n".join(bouquets_info), data

    def get_bouquet_recommendation(
        self,
        user_input: str,
        conversation_history: List[BaseMessage] = None
    ) -> str:
        """Генерация ответа на основе истории"""
        try:
            self.memory.clear()

            # Вызов с прямой передачей истории в chain (chat_history)
            result = self.qa_chain.invoke({
                "question": user_input,
                "chat_history": conversation_history or [],
                "bouquets_info": self.bouquets_info
            })

            return result["answer"]

        except Exception as e:
            logger.error(f"Ошибка при генерации ответа: {e}")
            return "Извините, произошла ошибка при обработке вашего запроса."

    def filter_bouquets_by_price(self, max_price: float):
        return [b for b in self.bouquets_data if b['Цена'] <= max_price]
