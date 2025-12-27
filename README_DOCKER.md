# Docker Setup для Telegram RAG Bot

## Требования

- Docker
- Docker Compose

## Быстрый старт

1. Скопируйте `.env.example` в `.env` и заполните необходимые переменные:
   ```bash
   cp .env.example .env
   ```

2. Отредактируйте `.env` файл и укажите:
   - `TELEGRAM_TOKEN` - токен вашего Telegram бота
   - `AUTHORIZATION_KEY` - ключ авторизации GigaChat

3. Запустите контейнеры:
   ```bash
   docker-compose up -d
   ```

4. Просмотр логов:
   ```bash
   docker-compose logs -f bot
   ```

5. Остановка:
   ```bash
   docker-compose down
   ```

## Структура

- `Dockerfile` - образ для бота
- `docker-compose.yml` - конфигурация для бота и MySQL
- `init.sql` - SQL скрипт для инициализации базы данных
- `.dockerignore` - файлы, исключаемые из образа

## Volumes

- `chroma_db/` - векторная база данных Chroma (персистентная)
- `data/` - данные (букеты, переписки)
- `prompts/` - системные промпты
- `mysql_data` - данные MySQL (создается автоматически)

## Переменные окружения

Все переменные окружения настраиваются через файл `.env`. См. `.env.example` для примера.

**Важно:** Если на хосте уже запущен MySQL на порту 3306, порт контейнера будет автоматически изменен на 3307. Вы можете задать другой порт через переменную `MYSQL_PORT` в `.env` файле, или полностью убрать проброс порта из `docker-compose.yml`, если внешний доступ к MySQL не нужен.

## Полезные команды

```bash
# Пересборка образа
docker-compose build

# Перезапуск бота
docker-compose restart bot

# Просмотр статуса
docker-compose ps

# Подключение к MySQL
docker-compose exec mysql mysql -u flower_user -p flower_bot

# Очистка всех данных (включая volumes)
docker-compose down -v
```

