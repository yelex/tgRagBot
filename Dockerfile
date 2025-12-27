FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Копирование requirements и установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода приложения
COPY . .

# Создание директории для логов
RUN mkdir -p /app/logs

# Переменные окружения по умолчанию (будут переопределены через docker-compose)
ENV PYTHONUNBUFFERED=1

# Запуск бота
CMD ["python", "tg_bot.py"]

