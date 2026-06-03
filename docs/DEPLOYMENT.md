# Деплой Floribot v2 на сервер

## Требования к серверу

- Ubuntu 20.04+ / Debian 11+
- 1 CPU, 1 GB RAM (минимум)
- Docker 24+ и Docker Compose v2
- Открытый исходящий HTTPS (порт 443) — для Telegram API и GigaChat

---

## Шаг 1 — Установка Docker (если ещё нет)

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker
docker --version   # проверка
```

---

## Шаг 2 — Клонировать репозиторий

```bash
git clone https://github.com/YOUR_USERNAME/tgRagBot.git
cd tgRagBot
git checkout feature/floribot-v2
```

> Если репо приватное — используй deploy key или Personal Access Token:
> ```bash
> git clone https://YOUR_TOKEN@github.com/YOUR_USERNAME/tgRagBot.git
> ```

---

## Шаг 3 — Создать .env на сервере

Скопируй `.env.example` и заполни реальными значениями:

```bash
cp .env.example .env
nano .env
```

Минимальный `.env`:

```env
TELEGRAM_TOKEN=7905921782:AAFsv_f6udnZ9lz7ufWlbe8Ev_JThBSu6to

GIGACHAT_CREDENTIALS=MGY5ZDg4YzAtYzRhNS00ZGYyLTk5Y2ItZjQ0ZTFjYTBjZGY5OmY1Nzg4MmMzLWVjMGEtNDgzYi1hMDllLTM5MzZhZjE5OTc2Ng==
GIGACHAT_SCOPE=GIGACHAT_API_CORP
GIGACHAT_MODEL=GigaChat-2-Max
GIGACHAT_TIMEOUT=60

OPERATOR_CHAT_ID=7180426531
FEEDBACK_DELAY_HOURS=2

MYSQL_HOST=95.142.42.28
MYSQL_PORT=3306
MYSQL_USER=yelex
MYSQL_PASSWORD=Fokina12
MYSQL_DATABASE=floridb

PATH_BOUQUETS=/app/data/bouquets.json
PATH_SYSTEM_PROMPT=/app/prompts/system_prompt.txt
```

> **Важно:** `.env` не должен попадать в git (он в `.gitignore`).
> Передай файл через `scp` или создай вручную на сервере.

---

## Шаг 4 — Создать папку для логов

```bash
mkdir -p logs
```

---

## Шаг 5 — Запустить

```bash
docker compose up -d --build
```

Проверить что бот поднялся:

```bash
docker compose logs -f bot
```

Нормальный вывод:
```
🌸 Инициализация LangGraph-агента (v2 — многошаговый диалог)...
✅ LangGraph-агент v2 готов!
Routing LLM (structured output) инициализирован
Scheduler started
Application started
```

---

## Управление

```bash
# Остановить
docker compose down

# Перезапустить после обновления кода
git pull
docker compose up -d --build

# Посмотреть логи
docker compose logs -f bot

# Войти внутрь контейнера
docker compose exec bot bash
```

---

## Обновление кода (деплой новой версии)

```bash
cd tgRagBot
git pull origin feature/floribot-v2
docker compose up -d --build
```

Бот перезапускается с даунтаймом ~10–15 секунд.

---

## Миграция БД (если ещё не применена)

Таблица `orders` уже создана скриптом на локальной машине.
Если нужно применить вручную:

```bash
# Войти в MySQL
mysql -h 95.142.42.28 -u yelex -p floridb < db/init.sql
```

Или выполнить внутри контейнера:

```bash
docker compose exec bot python3 -c "
import os; from dotenv import load_dotenv; load_dotenv()
import mysql.connector
conn = mysql.connector.connect(
    host=os.getenv('MYSQL_HOST'), user=os.getenv('MYSQL_USER'),
    password=os.getenv('MYSQL_PASSWORD'), database=os.getenv('MYSQL_DATABASE'),
    port=int(os.getenv('MYSQL_PORT', 3306))
)
with open('db/init.sql') as f:
    for stmt in f.read().split(';'):
        if stmt.strip():
            conn.cursor().execute(stmt)
conn.commit()
print('Done')
"
```

---

## Быстрый деплой одной командой (scp + ssh)

С **локальной машины** (macOS):

```bash
# 1. Передать .env на сервер
scp .env user@YOUR_SERVER_IP:/home/user/tgRagBot/.env

# 2. Зайти по SSH и поднять
ssh user@YOUR_SERVER_IP << 'EOF'
  cd /home/user/tgRagBot
  git pull origin feature/floribot-v2
  docker compose up -d --build
  docker compose logs --tail=20 bot
EOF
```

---

## Troubleshooting

| Симптом | Причина | Решение |
|---------|---------|---------|
| `GigaChat` не отвечает | Нет доступа к `gigachat.devices.sberbank.ru` | Проверить firewall, добавить исходящий HTTPS |
| `Connection refused` к MySQL | Сервер БД не доступен с нового IP | Добавить IP сервера в whitelist MySQL |
| `Scheduler started` дважды | Нормально — ptb запускает планировщик дважды из-за своего event loop | Не баг |
| Бот не отвечает | Проверить `docker compose logs bot` | Обычно ошибка в .env |
