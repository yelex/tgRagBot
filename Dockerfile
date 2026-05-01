FROM python:3.11-slim

WORKDIR /app

# Копирование requirements и установка зависимостей
COPY requirements.txt .
RUN pip install --default-timeout=120 --no-cache-dir -r requirements.txt

# Копирование кода приложения
COPY . .

# Создание директории для логов
RUN mkdir -p /app/logs && chmod +x /app/scripts/start.sh

# Переменные окружения по умолчанию (будут переопределены через docker-compose)
ENV PYTHONUNBUFFERED=1

# Запуск бота
CMD ["/app/scripts/start.sh"]