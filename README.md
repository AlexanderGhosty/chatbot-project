# Intelligent Furniture-Selling Telegram Chatbot

Учебный Telegram-бот для продажи мебели. Проект поддерживает текстовый диалог,
семантический поиск по `dialogues.txt`, ML-классификацию намерений,
рекламный сценарий с тремя товарами и голосовой pipeline ASR -> NLP -> TTS.

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
