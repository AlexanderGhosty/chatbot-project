# Intelligent Furniture-Selling Telegram Chatbot

## Быстрый запуск

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Заполните `TELEGRAM_BOT_TOKEN` в `.env` или экспортируйте переменную окружения:

```bash
export TELEGRAM_BOT_TOKEN=123456:replace_me
python main.py
```

Для голосовых сообщений нужен установленный `ffmpeg`. GigaAM и Silero
подключаются локально. Docker-образ прогревает Silero на этапе сборки, чтобы
бот не скачивал TTS-веса во время ответа пользователю.

Для логирования распознанного текста из голосовых сообщений включите:

```bash
VOICE_LOGGING_ENABLED=true
```

Для отладки выбора маршрута ответа включите:

```bash
DIALOGUE_LOGGING_ENABLED=true
```

`TTS_ALLOW_ESPEAK_FALLBACK` по умолчанию выключен, чтобы production-бот не
подменял Silero синтезом через `espeak-ng`.

## Индексация диалогов

Индекс строится автоматически при первом retrieval-запросе. Его можно
подготовить заранее:

```bash
python scripts/build_vector_db.py
```

Если установлен `chromadb`, будет использована persistent collection. Если нет,
создается локальный fallback-индекс в `data/chromadb`.

## Chitchat fallback

Общие реплики вынесены в отдельный fallback-индекс, чтобы разговорный датасет
не перебивал мебельный FAQ и рекламный сценарий. По умолчанию используется
`data/raw/chitchat_dialogues.txt`, а collection называется
`chitchat_dialogues`.

Для загрузки расширенного набора из `SiberiaSoft/SiberianPersonaChat-2`:

```bash
python scripts/import_siberian_persona_chat.py --max-pairs 20000
python scripts/build_vector_db.py \
  --dialogues data/raw/chitchat_dialogues.txt \
  --collection chitchat_dialogues \
  --no-seed-dialogues \
  --filter-unsafe-pairs
```

Импортёр по умолчанию берёт только класс `chitchat`, фильтрует рискованные и
низкокачественные пары, поэтому датасет используется только для бытового
диалога.

Через Docker Compose то же самое можно запустить так:

```bash
docker compose --profile tools run --rm chitchat-importer
docker compose --profile tools run --rm chitchat-indexer
```

Индексы можно прогревать при старте бота через `PREWARM_VECTOR_INDEXES=true`,
чтобы первый пользовательский ответ не ждал построения retrieval-базы.

## Docker Compose

Контейнерный запуск использует host-сеть, persistent volumes для кэшей моделей
и embedded ChromaDB в `data/chromadb`:

```bash
cp .env.example .env
# заполните TELEGRAM_BOT_TOKEN в .env
docker compose build
docker compose --profile tools run --rm indexer
docker compose up bot
```
