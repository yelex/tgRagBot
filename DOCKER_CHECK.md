# Проверка работы Docker Compose

## Быстрая проверка

### 1. Проверка конфигурации
```bash
docker-compose config
```

### 2. Автоматическая проверка (рекомендуется)
```bash
./check_docker.sh
```

### 3. Ручная проверка

#### Сборка и запуск:
```bash
# Сборка образов
docker-compose build

# Запуск в фоновом режиме
docker-compose up -d

# Просмотр статуса
docker-compose ps
```

#### Проверка логов:
```bash
# Логи бота
docker-compose logs -f bot

# Логи MySQL
docker-compose logs -f mysql

# Все логи
docker-compose logs -f
```

#### Проверка работы сервисов:
```bash
# Проверка MySQL
docker-compose exec mysql mysqladmin ping -h localhost

# Проверка подключения к MySQL из бота
docker-compose exec bot python -c "from interfaces.mysql_interface import get_connection; conn = get_connection(); print('MySQL подключен!')"

# Проверка переменных окружения в контейнере бота
docker-compose exec bot env | grep -E "(TELEGRAM|AUTHORIZATION|PATH_|MYSQL)"
```

#### Остановка и очистка:
```bash
# Остановка контейнеров
docker-compose stop

# Остановка и удаление контейнеров
docker-compose down

# Остановка и удаление с очисткой volumes
docker-compose down -v
```

## Возможные проблемы

### 1. Ошибка "TELEGRAM_TOKEN не задан"
Создайте файл `.env` в корне проекта:
```bash
TELEGRAM_TOKEN=your_token_here
AUTHORIZATION_KEY=your_key_here
```

### 2. Ошибка подключения к MySQL
Проверьте, что MySQL контейнер запущен и здоров:
```bash
docker-compose ps mysql
docker-compose logs mysql
```

### 3. Ошибка ChromaDB
Если база данных ChromaDB повреждена, она будет автоматически пересоздана при запуске.

### 4. Проблемы с правами доступа
```bash
# Исправление прав на директории
sudo chown -R $USER:$USER chroma_db/
sudo chmod -R 755 chroma_db/
```

## Полезные команды

```bash
# Перезапуск бота
docker-compose restart bot

# Пересборка образа бота
docker-compose build --no-cache bot

# Выполнение команды в контейнере бота
docker-compose exec bot bash

# Просмотр использования ресурсов
docker stats

# Очистка неиспользуемых образов
docker system prune -a
```

