# Ручной индексатор каталога FloriPacco

Скрипт обновляет `data/bouquets.json` напрямую из `https://floripacco.ru/catalog`.

## Запуск

```bash
cd /home/user/dev/tg_rag_bot
source .venv/bin/activate
python utils/catalog_indexer.py
```

## Полезные флаги

- `--dry-run` - только проверить сбор без записи файла.
- `--verbose` - подробные debug-логи.
- `--max-pages 120` - увеличить глубину обхода.
- `--output data/bouquets.json` - указать путь до выходного JSON.

Примеры:

```bash
python utils/catalog_indexer.py --dry-run --verbose
python utils/catalog_indexer.py --max-pages 120
```

## Что собирается

На выходе формат полностью совместим с ботом:

```json
[
  {
    "Название": "Букет ...",
    "Цена": 10900.0,
    "Ссылка": "https://floripacco.ru/..."
  }
]
```

