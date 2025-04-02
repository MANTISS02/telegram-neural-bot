# Нейросетевой ассистент для Telegram

## 📝 Описание
Многофункциональный Telegram-бот с интеграцией современных языковых моделей и локальным распознаванием голоса. Бот позволяет общаться с различными AI-моделями (DeepSeek, Gemini, Mistral, Llama), отправлять голосовые сообщения и настраивать параметры генерации ответов.

## ✨ Основные возможности
- 🤖 Интеграция с языковыми моделями через REST API
- 🔄 Автоматическое переключение между моделями при недоступности основной
- 🎤 Локальное распознавание голосовых сообщений через Vosk
- 🔒 Приватность: обработка голосовых сообщений на вашем сервере
- ⚙️ Гибкие настройки генерации текста (температура, длина ответа, модель)
- 📚 Сохранение истории диалогов для каждого пользователя
- 💬 Поддержка Markdown форматирования и отображение кода

## 🛠️ Установка и настройка

### Требования к системе
- Python 3.8 или выше
- FFmpeg
- ~2.5 ГБ свободного места для Vosk-модели

### Шаги установки

1. **Клонирование репозитория**
   ```bash
   git clone https://github.com/MANTISS02/telegram-neural-bot.git
   cd telegram-neural-bot
   ```

2. **Установка зависимостей**
   ```bash
   pip install -r requirements.txt
   ```

3. **Установка FFmpeg (обязательно для обработки голосовых сообщений)**
   
   **Для Windows:**
   - Скачайте последнюю версию FFmpeg с официального сайта: https://ffmpeg.org/download.html 
     или с GitHub: https://github.com/BtbN/FFmpeg-Builds/releases (выберите файл ffmpeg-master-latest-win64-gpl.zip)
   - Распакуйте архив
   - Скопируйте файл ffmpeg.exe из папки bin в корневую директорию проекта
   
   **Альтернативный способ (через PATH):**
   - Распакуйте скачанный архив в удобное место
   - Добавьте путь к папке bin в системную переменную PATH
   - Проверьте установку командой: `ffmpeg -version`

4. **Загрузка Vosk-модели**
   
   Скачайте русскоязычную модель Vosk и распакуйте ее в корневую директорию проекта:
   ```bash
   wget https://alphacephei.com/vosk/models/vosk-model-ru-0.22.zip
   unzip vosk-model-ru-0.22.zip
   ```

5. **Создание файла конфигурации**
   
   Создайте файл `.env` в корневой директории проекта со следующими параметрами:
   ```
   BOT_TOKEN=your_telegram_bot_token
   OPENROUTER_API_KEY=your_openrouter_api_key
   TOGETHER_API_KEY=your_together_api_key
   HUGGINGFACE_API_KEY=your_huggingface_api_key
   ```

## 🚀 Запуск бота

```bash
python Bot.py
```

## 📋 Использование

### Команды бота
- `/start` - Начать диалог с ботом
- `/help` - Показать справку и доступные команды
- `/settings` - Настроить параметры генерации
- `/clear` - Очистить историю диалога
- `/info` - Информация о статусе моделей и системе

### Настройка параметров
Через меню настроек (/settings) вы можете изменить:
- Выбор языковой модели
- Уровень "креативности" (температура)
- Длину генерируемых ответов
- Длину сохраняемой истории диалога

### Голосовые сообщения
Бот автоматически распознает и обрабатывает голосовые сообщения на русском языке используя локальную Vosk-модель.

## 🧩 Структура проекта

- `Bot.py` - Основной файл бота
- `vosk-model-ru-0.22/` - Модель для распознавания русской речи
- `.env` - Конфигурационный файл с API ключами
- `requirements.txt` - Список зависимостей

## 📊 Используемые технологии

- [Aiogram 3](https://docs.aiogram.dev/en/latest/) - Фреймворк для создания Telegram-ботов
- [Vosk](https://alphacephei.com/vosk/) - Система распознавания речи офлайн
- [FFmpeg](https://ffmpeg.org/) - Библиотека для обработки аудио и видео
- [OpenRouter](https://openrouter.ai/), [Together AI](https://www.together.ai/), [Hugging Face](https://huggingface.co/) - API для доступа к языковым моделям

## 👨‍💻 Автор

- [MANTISS02](https://github.com/MANTISS02)