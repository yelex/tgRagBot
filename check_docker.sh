#!/bin/bash

# Скрипт для проверки работы docker-compose

set -e

echo "🔍 Проверка конфигурации Docker Compose..."
docker-compose config > /dev/null && echo "✅ Конфигурация валидна"

echo ""
echo "🏗️  Сборка образов..."
docker-compose build

echo ""
echo "🚀 Запуск сервисов..."
docker-compose up -d

echo ""
echo "⏳ Ожидание готовности MySQL (30 секунд)..."
sleep 30

echo ""
echo "📊 Статус контейнеров:"
docker-compose ps

echo ""
echo "📝 Логи MySQL:"
docker-compose logs --tail=20 mysql

echo ""
echo "📝 Логи бота:"
docker-compose logs --tail=20 bot

echo ""
echo "✅ Проверка завершена!"
echo ""
echo "Полезные команды:"
echo "  docker-compose logs -f bot      # Просмотр логов бота в реальном времени"
echo "  docker-compose logs -f mysql    # Просмотр логов MySQL"
echo "  docker-compose ps               # Статус контейнеров"
echo "  docker-compose stop             # Остановка контейнеров"
echo "  docker-compose down             # Остановка и удаление контейнеров"
echo "  docker-compose restart bot      # Перезапуск бота"

