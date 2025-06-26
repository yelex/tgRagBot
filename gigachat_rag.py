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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

load_dotenv()

AUTHORIZATION_KEY = os.getenv('AUTHORIZATION_KEY')
PATH_MESSAGES = os.getenv('PATH_MESSAGES')
PATH_BOUQUETS = os.getenv('PATH_BOUQUETS')


def load_bouquets_data(file_path):
    """Загружает и обрабатывает данные о букетах из JSON-файла"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    bouquets_info = []
    for bouquet in data:
        bouquets_info.append(
            f"Название: {bouquet['Название']}, Цена: {bouquet['Цена']} руб., Ссылка: {bouquet['Ссылка']}"
        )
    return "\n".join(bouquets_info), data  # Возвращаем и данные для фильтрации


def initialize_chatbot():
    # Загрузка и подготовка документов
    loader = TextLoader(PATH_MESSAGES)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(documents)

    # Загрузка данных о букетах
    bouquets_info, bouquets_data = load_bouquets_data(PATH_BOUQUETS)

    # Инициализация embeddings и базы данных
    embeddings = GigaChatEmbeddings(
        credentials=AUTHORIZATION_KEY, verify_ssl_certs=False
    )
    db = Chroma.from_documents(
        documents,
        embeddings,
        client_settings=Settings(anonymized_telemetry=False),
    )
    retriever = db.as_retriever()

    # Инициализация памяти для диалога
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="input"
    )

    # Обновленный промпт с учетом истории диалога
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Ты - консультант цветочного магазина. Отвечай на вопросы клиента на основе предоставленных переписок и информации о букетах.\n"
         "Всегда предлагай клиенту конкретные варианты букетов с учетом его пожеланий и бюджета.\n"
         "Твой стиль должен быть максимально похож на стиль из переписок.\n\n"
         "Актуальные цены на букеты:\n"
         "{bouquets_info}\n\n"
         "Учитывай историю диалога: {chat_history}\n"
         "Контекст из переписок: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    # Инициализация модели GigaChat с оптимизированными параметрами
    llm = GigaChat(
        credentials=AUTHORIZATION_KEY,
        verify_ssl_certs=False,
        model='GigaChat-2-Max',
        temperature=0.3,  # Для более детерминированных ответов
        top_p=0.9
    )

    # Создание цепочек с учетом памяти
    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=question_answer_chain,
        memory=memory
    )

    return rag_chain, bouquets_info, bouquets_data


def chat_loop(rag_chain, bouquets_info):
    print("Чат-бот готов к общению. Введите 'выход' чтобы завершить сеанс.")
    chat_history = []  # Дополнительное хранилище истории

    while True:
        user_input = input("Вы: ")
        if user_input.lower() in ['выход', 'exit', 'quit']:
            print("До свидания!")
            break

        result = rag_chain.invoke({
            "input": user_input,
            "bouquets_info": bouquets_info,
            "chat_history": chat_history
        })

        # Сохраняем обновленную историю
        chat_history.extend([
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": result["answer"]}
        ])

        print("\nБот:", result["answer"], "\n")


if __name__ == '__main__':
    rag_chain, bouquets_info, bouquets_data = initialize_chatbot()
    chat_loop(rag_chain, bouquets_info)