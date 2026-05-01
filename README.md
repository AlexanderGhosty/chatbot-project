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
подключаются лениво: если модель или зависимость недоступна, бот не падает,
а возвращает понятный fallback.

## Индексация диалогов

Индекс строится автоматически при первом retrieval-запросе. Его можно
подготовить заранее:

```bash
python scripts/build_vector_db.py
```

Если установлен `chromadb`, будет использована persistent collection. Если нет,
создается локальный fallback-индекс в `data/chromadb`.
