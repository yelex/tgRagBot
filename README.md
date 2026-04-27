# Telegram RAG Bot

Telegram-бот для работы с каталогом букетов FloriPacco. Проект запускается через Docker Compose: поднимаются бот и MySQL, каталог обновляется при старте контейнера.

## Требования

- Docker
- Docker Compose

## Быстрый старт

1. Создайте файл `.env` в корне проекта и заполните обязательные переменные:

   ```env
   TELEGRAM_TOKEN=your_telegram_bot_token
   AUTHORIZATION_KEY=your_gigachat_authorization_key
   ```

   При необходимости можно переопределить параметры MySQL:

   ```env
   MYSQL_ROOT_PASSWORD=rootpassword
   MYSQL_DATABASE=flower_bot
   MYSQL_USER=flower_user
   MYSQL_PASSWORD=flower_password
   MYSQL_PORT=3307
   ```

2. Запустите контейнеры:

   ```bash
   docker-compose up -d
   ```

3. Посмотрите логи бота:

   ```bash
   docker-compose logs -f bot
   ```

4. Остановите проект:

   ```bash
   docker-compose down
   ```

## Docker

- `Dockerfile` — образ для бота.
- `docker-compose.yml` — конфигурация бота и MySQL.
- `db/init.sql` — инициализация базы данных.
- `.dockerignore` — исключения для Docker build context.

Персистентные данные:

- `mysql_data` — volume с данными MySQL.
- `chroma_db/` — локальная векторная база Chroma.
- `data/` — данные бота, включая каталог букетов.
- `prompts/` — системные промпты (включая `system_prompt.txt`).
- `logs/` — логи приложения.

Полезные команды:

```bash
# Пересборка образа
docker-compose build

# Перезапуск бота
docker-compose restart bot

# Статус сервисов
docker-compose ps

# Подключение к MySQL
docker-compose exec mysql mysql -u flower_user -p flower_bot

# Очистка контейнеров и volumes
docker-compose down -v
```

## Индексация каталога

Индексатор обновляет JSON-каталог букетов из `https://floripacco.ru/catalog`. При запуске контейнера он выполняется автоматически через `scripts/start.sh`, перед стартом бота.

Ручной запуск:

```bash
python utils/catalog_indexer.py
```

Полезные флаги:

- `--dry-run` — проверить сбор без записи файла.
- `--verbose` — включить подробные debug-логи.
- `--max-pages 120` — увеличить глубину обхода.
- `--output data/bouquets.json` — указать путь к выходному JSON.

Примеры:

```bash
python utils/catalog_indexer.py --dry-run --verbose
python utils/catalog_indexer.py --max-pages 120
```

Формат выходных данных совместим с ботом:

```json
[
  {
    "Название": "Букет ...",
    "Цена": 10900.0,
    "Ссылка": "https://floripacco.ru/..."
  }
]
```