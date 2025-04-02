# -*- coding: utf-8 -*-

import os
import logging
import json
import aiohttp
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.filters.command import CommandObject
from aiogram.enums import ParseMode
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
import time
import asyncio
import re
from datetime import datetime
from collections import deque
import speech_recognition as sr
import tempfile
# Добавляем импорты для Vosk
import vosk
import wave
import concurrent.futures
import subprocess
import threading

# Путь к файлу с настройками пользователей
SETTINGS_FILE = "user_settings.json"
HISTORY_FILE = "user_history.json"

# Константы для таймаутов
VOICE_RECOGNITION_TIMEOUT = 120  # Таймаут для распознавания голоса (в секундах)
FFMPEG_CONVERSION_TIMEOUT = 60   # Таймаут для конвертации аудио через ffmpeg (в секундах)
API_CHECK_TIMEOUT = 30          # Таймаут для проверки API моделей (в секундах)
API_MIN_CHECK_TIME = 1         # Минимальное время проверки API (в секундах)

VOSK_MODEL_PATH = "vosk-model-ru-0.22"
vosk_model = None
use_local_recognition = True

recognition_stop_flag = threading.Event()

# Функция для инициализации Vosk модели
def init_vosk_model():
    global vosk_model, use_local_recognition
    try:
        if not os.path.exists(VOSK_MODEL_PATH):
            logging.warning(f"Путь к Vosk модели не найден: {VOSK_MODEL_PATH}. Будет использован Google Speech Recognition.")
            use_local_recognition = False
            return False
            
        required_files = ['am/final.mdl', 'conf/mfcc.conf']
        for file_path in required_files:
            full_path = os.path.join(VOSK_MODEL_PATH, file_path)
            if not os.path.exists(full_path):
                logging.warning(f"Отсутствует необходимый файл модели: {full_path}")
                use_local_recognition = False
                return False
                
        logging.info(f"Инициализация Vosk модели из {VOSK_MODEL_PATH}...")
        
        try:
            vosk_model = vosk.Model(VOSK_MODEL_PATH)
            use_local_recognition = True
            logging.info(f"Vosk модель успешно загружена из {VOSK_MODEL_PATH}")
            
            test_recognizer = vosk.KaldiRecognizer(vosk_model, 16000)
            test_result = json.loads(test_recognizer.FinalResult())
            if "text" in test_result:
                logging.info("Успешная проверка инициализации распознавателя")
            
            return True
        except vosk.KaldiError as e:
            logging.error(f"Ошибка Kaldi при инициализации модели Vosk: {e}")
            use_local_recognition = False
            return False
        
    except Exception as e:
        logging.error(f"Ошибка при инициализации Vosk модели: {e}", exc_info=True)
        use_local_recognition = False
        return False

# Функция для локального распознавания через Vosk
def recognize_with_vosk(audio_file_path):
    global recognition_stop_flag
    recognition_stop_flag.clear()
    
    if vosk_model is None:
        logging.error("Vosk модель не инициализирована при попытке распознавания")
        raise Exception("Vosk модель не инициализирована")
    
    if not os.path.exists(audio_file_path):
        logging.error(f"Аудио файл не существует: {audio_file_path}")
        raise FileNotFoundError(f"Аудио файл не найден: {audio_file_path}")
    
    wf = None
    try:
        file_size = os.path.getsize(audio_file_path)
        logging.info(f"Размер аудио файла для распознавания: {file_size} байт")
        
        if file_size == 0:
            logging.error("Размер аудио файла равен нулю")
            raise Exception("Пустой аудио файл")
            
        wf = wave.open(audio_file_path, "rb")
        
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            logging.error(f"Неподходящий формат аудио: каналов={wf.getnchannels()}, битность={wf.getsampwidth()}, компрессия={wf.getcomptype()}")
            raise Exception("Аудио файл должен быть WAV формата моно PCM")
        
        sample_rate = wf.getframerate()
        logging.info(f"Аудио параметры: частота={sample_rate}, каналов={wf.getnchannels()}, битность={wf.getsampwidth()}")
        
        if sample_rate != 16000:
            logging.warning(f"Частота дискретизации {sample_rate} Гц не оптимальна, рекомендуется 16000 Гц")
        
        rec = vosk.KaldiRecognizer(vosk_model, sample_rate)
        rec.SetWords(True)
        
        result = ""
        chunk_size = 4096  # Размер блока для чтения
        
        # Добавляем отслеживание прогресса распознавания
        total_frames = wf.getnframes()
        processed_frames = 0
        last_progress = 0
        
        while not recognition_stop_flag.is_set():
            data = wf.readframes(chunk_size)
            if len(data) == 0:
                break
            
            processed_frames += chunk_size
            progress = min(100, int((processed_frames / total_frames) * 100))
            
            if progress >= last_progress + 20:
                logging.info(f"Прогресс распознавания: {progress}%")
                last_progress = progress
            
            if rec.AcceptWaveform(data):
                part_result = json.loads(rec.Result())
                if "text" in part_result and part_result["text"].strip():
                    result += part_result["text"] + " "
                    logging.debug(f"Промежуточное распознавание: {part_result['text']}")
        
        if recognition_stop_flag.is_set():
            logging.info("Процесс распознавания был принудительно остановлен")
            return None
            
        final_result = json.loads(rec.FinalResult())
        if "text" in final_result and final_result["text"].strip():
            result += final_result["text"]
            logging.debug(f"Финальное распознавание: {final_result['text']}")
        
        result = result.strip()
        logging.info(f"Результат распознавания: '{result}'")
        
        if not result:
            logging.error("После распознавания получен пустой результат")
            raise sr.UnknownValueError("Не удалось распознать речь")
        
        return result
    except wave.Error as e:
        logging.error(f"Ошибка при открытии WAV файла: {e}")
        raise Exception(f"Ошибка при чтении аудио файла: {e}")
    except json.JSONDecodeError as e:
        logging.error(f"Ошибка при разборе JSON результата Vosk: {e}")
        raise Exception(f"Ошибка при обработке результатов распознавания: {e}")
    except Exception as e:
        logging.error(f"Неизвестная ошибка при распознавании с Vosk: {e}", exc_info=True)
        raise
    finally:
        if wf:
            try:
                wf.close()
            except:
                pass

# Функция для загрузки настроек пользователей из файла
def load_user_settings():
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                settings_data = json.load(f)
                return {int(user_id): settings for user_id, settings in settings_data.items()}
        return {}
    except Exception as e:
        logging.error(f"Ошибка при загрузке настроек пользователей: {e}")
        return {}

# Функция для сохранения настроек пользователей в файл
def save_user_settings(settings_dict):
    try:
        settings_data = {str(user_id): settings for user_id, settings in settings_dict.items()}
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"Ошибка при сохранении настроек пользователей: {e}")

# Функция для загрузки истории сообщений пользователей из файла
def load_user_history():
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
                return {int(user_id): history for user_id, history in history_data.items()}
        return {}
    except Exception as e:
        logging.error(f"Ошибка при загрузке истории сообщений: {e}")
        return {}

# Функция для сохранения истории сообщений пользователей в файл
def save_user_history(history_dict):
    try:
        history_data = {str(user_id): history for user_id, history in history_dict.items()}
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"Ошибка при сохранении истории сообщений: {e}")

# Загружаем переменные окружения из .env файла
load_dotenv()

# Настраиваем логирование
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Получаем токены из переменных окружения
API_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")  # API ключ OpenRouter

bot = Bot(token=API_TOKEN)
dp = Dispatcher()

# Словарь для хранения истории сообщений пользователей
user_message_history = {}

# Словарь для хранения настроек пользователей
user_settings = {}

# Словарь для хранения ID последних сообщений бота для каждого пользователя
user_last_messages = {}

# Настройки по умолчанию
DEFAULT_SETTINGS = {
    "model": "google/gemini-2.0-pro-exp-02-05:free",  # Gemini Pro модель
    "max_tokens": 500,
    "temperature": 0.7,
    "dynamic_chat": False,  # Удалять ли предыдущие сообщения бота
    "history_length": 10,   # Количество пар сообщений (вопрос-ответ) в истории
    "system_message": "Ты дружелюбный ассистент, который помогает пользователям. Отвечай на русском языке, кратко и по делу. При форматировании текста следуй этим правилам: 1) Для блоков кода используй ```язык и ``` (например, ```python для Python кода); 2) Для однострочного кода используй обратные кавычки `код`; 3) Для выделения заголовков используй # для основного заголовка и ## для подзаголовков; 4) Для жирного текста используй **текст**; 5) Для курсива используй *текст*; 6) Для маркированного списка используй звездочку и пробел: * элемент списка. Telegram поддерживает базовое форматирование и корректно отображает код в сообщениях."
}

# Модели
AVAILABLE_MODELS = [
    # OpenRouter модели
    "google/gemini-2.0-pro-exp-02-05:free",  # Модель по умолчанию
    "google/gemini-2.0-flash-lite-preview-02-05:free", 
    "deepseek/deepseek-r1",         
    "deepseek/deepseek-r1-zero:free",
    "deepseek/deepseek-chat-v3-0324:free",
    "perplexity/sonar-reasoning-pro", 
    "perplexity/r1-1776", 
    
    # Together AI
    "together/mixtral-8x7b-instruct",
    "together/mistral-7b-instruct",
    "together/llama-2-13b-chat",
    
    # HuggingFace
    "huggingface/mistralai/Mistral-7B-Instruct-v0.2",
    "huggingface/microsoft/phi-2",
    "huggingface/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "huggingface/facebook/opt-350m",
    "huggingface/facebook/opt-1.3b"
]

# Максимальное количество сообщений в истории
MAX_HISTORY_LENGTH = 10

# Создаем клавиатуру основного меню (постоянно доступна внизу экрана)
main_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="🤖 Новый диалог"), KeyboardButton(text="🧹 Очистить историю")],
        [KeyboardButton(text="⚙️ Настройки"), KeyboardButton(text="ℹ️ Информация")],
        [KeyboardButton(text="❓ Помощь")]
    ],
    resize_keyboard=True,  # Уменьшить размер кнопок
    persistent=True  # Сделать клавиатуру постоянной
)

# Описания моделей для пользовательского интерфейса
MODEL_DESCRIPTIONS = {
    # OpenRouter модели
    "deepseek/deepseek-r1": "Многоязычная модель с расширенным контекстным окном. Оптимизирована для работы с русским языком. Подходит для программирования, анализа и генерации текста.",
    
    "deepseek/deepseek-r1-zero:free": "Базовая версия DeepSeek с ограниченным контекстным окном. Оптимальна для коротких диалогов и простых задач. Бесплатная для использования.",
    
    "deepseek/deepseek-chat-v3-0324:free": "Новейшая версия DeepSeek V3 с расширенным контекстом (131K). Большая модель типа MoE (Mixture-of-Experts) с превосходными способностями к рассуждению и обработке текста. Бесплатная для использования.",
    
    "google/gemini-2.0-pro-exp-02-05:free": "Экспериментальная версия Gemini 2.0 с обновленной архитектурой. Бесплатная для тестирования с ограничениями по запросам.", 
    
    "google/gemini-2.0-flash-lite-preview-02-05:free": "Облегченная версия Gemini 2.0 с оптимизацией скорости. Подходит для быстрых ответов и базового анализа.",
    
    "perplexity/sonar-reasoning-pro": "Модель с фокусом на логический анализ. Расширенные возможности для технических и научных задач. Улучшенная обработка длинных текстов.",
    
    "perplexity/r1-1776": "Универсальная модель с большим контекстным окном. Стабильная производительность в длительных диалогах. Поддержка различных типов задач.",
    
    # Together AI модели
    "together/mixtral-8x7b-instruct": "Мощная смешанная модель экспертов (MoE) от Mistral AI. Отлично работает с русским языком, имеет широкий спектр возможностей и большой контекст.",
    
    "together/mistral-7b-instruct": "Легкая и быстрая модель от Mistral AI. Хорошее соотношение производительности и скорости работы. Подходит для большинства задач.",
    
    "together/llama-2-13b-chat": "Средняя версия Llama 2 от Meta для диалогов. Хороший баланс между скоростью и качеством ответов.",
    
    "together/llama-2-70b-chat": "Большая версия Llama 2 от Meta. Мощная и универсальная модель для сложных запросов и генерации текста.",
    
    "together/qwen-72b-chat": "Мощная китайская модель Qwen с 72 миллиардами параметров. Хорошо работает как с китайским, так и с другими языками.",
    
    "together/codellama-34b-instruct": "Специализированная модель для программирования от Meta. Отлично справляется с генерацией и анализом кода.",
    
    "together/neural-chat-7b-v3-1": "Улучшенная версия нейронного чата. Хорошо работает с диалогами и имеет улучшенное понимание контекста.",
    
    # HuggingFace модели
    "huggingface/mistralai/Mistral-7B-Instruct-v0.2": "Инструктированная версия Mistral 7B. Хорошо следует инструкциям и генерирует качественные ответы. Подходит для широкого круга задач.",
    
    "huggingface/microsoft/phi-2": "Компактная и эффективная модель от Microsoft. Отличная производительность при малом размере, хорошо работает с кодом.",
    
    "huggingface/TinyLlama/TinyLlama-1.1B-Chat-v1.0": "Сверхлегкая модель для быстрых ответов на простые вопросы. Высокая скорость работы.",
    
    "huggingface/facebook/opt-350m": "Маленькая модель OPT от Meta, очень быстрая и компактная. Идеальна для базовых запросов и ситуаций с ограниченными ресурсами.",
    
    "huggingface/facebook/opt-1.3b": "Средняя модель OPT от Meta с хорошим балансом качества и скорости. Универсальная модель для повседневного использования."
}

# Описания для настроек
SETTINGS_DESCRIPTIONS = {
    "model": "Выберите языковую модель, которая будет генерировать ответы. Разные модели имеют различные сильные стороны и особенности.",
    "max_tokens": "Ограничение длины ответа. Больше значение - возможность получать более длинные ответы, но увеличивается время генерации.",
    "temperature": "Параметр, влияющий на креативность ответов. Низкие значения (0.3) дают более предсказуемые ответы, высокие (1.5) - более творческие и разнообразные.",
    "system_message": "Инструкция для модели, определяющая её поведение и стиль ответов. Это сообщение влияет на то, как модель будет отвечать.",
    "dynamic_chat": "Режим динамического чата. Когда включен - предыдущие сообщения бота удаляются при отправке новых, создавая более чистый интерфейс.",
    "history_length": "Максимальное количество пар сообщений (вопрос-ответ) в истории диалога. Больше значение - больше контекста для модели."
}

# Функция для обработки содержимого (удаление тегов думания)
def process_content(content):
    if content is None:
        logger.warning("Получен пустой контент (None) в process_content")
        return ""
        
    content = content.replace('<think>', '').replace('</think>', '')
    content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
    content = re.sub(r'\\boxed\{(.*?)\}', r'\1', content, flags=re.DOTALL)
    
    allowed_tags = ['b', 'i', 'u', 's', 'a', 'code', 'pre']
    for tag in re.findall(r'</?(\w+)[^>]*>', content):
        if tag.lower() not in allowed_tags:
            content = re.sub(r'<' + tag + '[^>]*>', '', content, flags=re.IGNORECASE)
            content = re.sub(r'</' + tag + '>', '', content, flags=re.IGNORECASE)
    
    return content

# Функция для улучшенного форматирования кода в ответах
def format_code_blocks(text):
    if text is None:
        return ""
        
    pattern = r'```(\w+)?\n([\s\S]*?)\n```'
    
    def replace_with_formatted_code(match):
        lang = match.group(1) or ""
        code = match.group(2)
        
        code = code.rstrip()
        
        code = (code.replace('&', '&amp;')
                   .replace('<', '&lt;')
                   .replace('>', '&gt;'))
        
        return f'<pre>{code}</pre>'
    
    formatted_text = re.sub(pattern, replace_with_formatted_code, text)
    
    def replace_inline_code(match):
        code = match.group(1)
        code = (code.replace('&', '&amp;')
                   .replace('<', '&lt;')
                   .replace('>', '&gt;'))
        return f'<code>{code}</code>'
    
    formatted_text = re.sub(r'`([^`]+)`', replace_inline_code, formatted_text)
    
    open_code_pattern = r'```(\w+)?\n([\s\S]*?)$'
    formatted_text = re.sub(open_code_pattern, replace_with_formatted_code, formatted_text)
    
    return formatted_text

# Функция для форматирования Markdown-разметки в HTML
def format_markdown_to_html(text):
    if text is None:
        return ""
        
    code_blocks = {}
    code_block_count = 0
    
    def save_code_block(match):
        nonlocal code_block_count
        placeholder = f'CODEBLOCK{code_block_count}'
        code_blocks[placeholder] = match.group(0)
        code_block_count += 1
        return placeholder
    
    text = re.sub(r'```[\s\S]*?```', save_code_block, text)
    text = re.sub(r'`[^`]+`', save_code_block, text)
    
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    
    text = re.sub(r'^# (.+)$', r'<b>\1</b>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$', r'<b>\1</b>', text, flags=re.MULTILINE)
    text = re.sub(r'^### (.+)$', r'<b>\1</b>', text, flags=re.MULTILINE)
    
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    text = re.sub(r'_(.+?)_', r'<i>\1</i>', text)
    
    lines = text.split('\n')
    for i in range(len(lines)):
        if re.match(r'^\* (.+)$', lines[i]):
            lines[i] = re.sub(r'^\* (.+)$', r'• \1', lines[i])
    text = '\n'.join(lines)
    
    text = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2">\1</a>', text)
    
    for placeholder, code_block in code_blocks.items():
        formatted_code = format_code_blocks(code_block)
        text = text.replace(placeholder, formatted_code)
    
    return text

# Функция для генерации ответа с использованием OpenRouter API
async def generate_response_openrouter(messages, model, max_tokens, temperature, timeout=30):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    formatted_messages = []
    
    system_content = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
    if system_content:
        formatted_messages.append({"role": "system", "content": system_content})
    
    for msg in messages:
        if msg["role"] in ["user", "assistant"]:
            formatted_messages.append({"role": msg["role"], "content": msg["content"]})
    
    data = {
        "model": model,
        "messages": formatted_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Ошибка API: {response.status}, {error_text}")
                    raise Exception(f"API вернул код {response.status}: {error_text}")
                
                result = await response.json()
                
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    return process_content(content)
                else:
                    logger.error(f"API не вернул ожидаемый результат: {result}")
                    if "error" in result:
                        raise Exception(f"API вернул ошибку: {result['error']}")
                    else:
                        raise Exception(f"API не вернул ожидаемый результат: {result}")
    
    except asyncio.TimeoutError:
        logger.error(f"Таймаут при запросе к OpenRouter API для модели {model}")
        raise Exception(f"Таймаут при запросе к API (превышено время ожидания {timeout} сек)")
    except Exception as e:
        logger.error(f"Ошибка при запросе к OpenRouter API: {e}")
        raise

# Функция для генерации ответа с использованием Together AI API
async def generate_response_together(messages, model, max_tokens, temperature, timeout=30):
    headers = {
        "Authorization": f"Bearer {os.getenv('TOGETHER_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    model_mapping = {
        "together/mixtral-8x7b-instruct": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "together/mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.2",
        "together/llama-2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",
        "together/llama-2-70b-chat": "meta-llama/Llama-2-70b-chat-hf",
        "together/qwen-72b-chat": "Qwen/Qwen-72B-Chat",
        "together/codellama-34b-instruct": "codellama/CodeLlama-34b-Instruct-hf"
    }
    
    api_model = model_mapping.get(model)
    if not api_model:
        raise Exception(f"Модель {model} не поддерживается Together AI API")
    
    formatted_messages = []
    
    system_content = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
    if system_content:
        formatted_messages.append({"role": "system", "content": system_content})
    
    for msg in messages:
        if msg["role"] in ["user", "assistant"]:
            formatted_messages.append({"role": msg["role"], "content": msg["content"]})
    
    data = {
        "model": api_model,
        "messages": formatted_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.7,
        "top_k": 50,
        "repetition_penalty": 1.1
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.together.xyz/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Ошибка Together AI API: {response.status}, {error_text}")
                    raise Exception(f"Together AI API вернул код {response.status}: {error_text}")
                
                result = await response.json()
                
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    return process_content(content)
                else:
                    logger.error(f"Together AI API не вернул ожидаемый результат: {result}")
                    if "error" in result:
                        raise Exception(f"Together AI API вернул ошибку: {result['error']}")
                    else:
                        raise Exception(f"Together AI API не вернул ожидаемый результат: {result}")
    
    except asyncio.TimeoutError:
        logger.error(f"Таймаут при запросе к Together AI API для модели {model}")
        raise Exception(f"Таймаут при запросе к API (превышено время ожидания {timeout} сек)")
    except Exception as e:
        logger.error(f"Ошибка при запросе к Together AI API: {e}")
        raise

# Функция для генерации ответа с использованием Hugging Face API
async def generate_response_huggingface(messages, model, max_tokens, temperature, timeout=30):
    headers = {
        "Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    # Маппинг моделей Hugging Face на их правильные названия в API
    model_mapping = {
        "huggingface/mistralai/Mistral-7B-Instruct-v0.2": "mistralai/Mistral-7B-Instruct-v0.2",
        "huggingface/microsoft/phi-2": "microsoft/phi-2",
        "huggingface/TinyLlama/TinyLlama-1.1B-Chat-v1.0": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "huggingface/facebook/opt-350m": "facebook/opt-350m",
        "huggingface/facebook/opt-1.3b": "facebook/opt-1.3b"
    }
    
    api_model = model_mapping.get(model)
    if not api_model:
        raise Exception(f"Модель {model} не поддерживается Hugging Face API")
    
    conversation = ""
    
    system_content = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
    if system_content:
        conversation += f"<|system|>\n{system_content}\n"
    
    for msg in messages:
        if msg["role"] == "user":
            conversation += f"<|user|>\n{msg['content']}\n"
        elif msg["role"] == "assistant":
            conversation += f"<|assistant|>\n{msg['content']}\n"
    
    conversation += "<|assistant|>\n"
    
    data = {
        "inputs": conversation,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "do_sample": True
        }
    }
    
    max_retries = 2
    retry_delay = 1
    last_error = None
    
    api_endpoint = f"https://api-inference.huggingface.co/models/{api_model}"
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Запрос к Hugging Face API (попытка {attempt+1}/{max_retries})")
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    api_endpoint,
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
                            content = result[0]["generated_text"]
                            assistant_parts = content.split("<|assistant|>\n")
                            if len(assistant_parts) > 1:
                                return process_content(assistant_parts[-1])
                            return process_content(content)
                        elif isinstance(result, dict) and "generated_text" in result:
                            content = result["generated_text"]
                            assistant_parts = content.split("<|assistant|>\n")
                            if len(assistant_parts) > 1:
                                return process_content(assistant_parts[-1])
                            return process_content(content)
                        else:
                            logger.error(f"Hugging Face API не вернул ожидаемый результат: {result}")
                            last_error = f"Hugging Face API не вернул ожидаемый результат: {result}"
                            continue
                    
                    elif response.status in [503, 502, 500]:
                        error_text = await response.text()
                        logger.warning(f"Hugging Face API временно недоступен ({response.status}): попытка {attempt+1}/{max_retries}")
                        last_error = f"Hugging Face API вернул код {response.status} (сервис временно недоступен)"
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay * (attempt + 1))
                            continue
                        else:
                            raise Exception(last_error)
                            
                    elif response.status == 429:
                        error_text = await response.text()
                        logger.warning(f"Hugging Face API превышение лимита запросов (429): попытка {attempt+1}/{max_retries}")
                        last_error = f"Hugging Face API вернул код 429 (превышен лимит запросов)"
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay * (attempt + 1))
                            continue
                        else:
                            raise Exception(last_error)
                            
                    else:
                        error_text = await response.text()
                        logger.error(f"Ошибка Hugging Face API: {response.status}, {error_text[:500]}...")
                        last_error = f"Hugging Face API вернул код {response.status}"
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay)
                            continue
                        else:
                            raise Exception(f"Hugging Face API вернул код {response.status}: {error_text[:500]}...")
        
        except asyncio.TimeoutError:
            logger.error(f"Таймаут при запросе к Hugging Face API для модели {model}")
            last_error = f"Таймаут при запросе к API (превышено время ожидания {timeout} сек)"
            if attempt < max_retries - 1:
                logger.warning(f"Повторная попытка {attempt+2}/{max_retries} после таймаута...")
                await asyncio.sleep(retry_delay)
                continue
            else:
                raise Exception(last_error)
                
        except aiohttp.ClientError as e:
            logger.error(f"Ошибка сетевого соединения с Hugging Face API: {e}")
            last_error = f"Ошибка сетевого соединения: {str(e)}"
            if attempt < max_retries - 1:
                logger.warning(f"Повторная попытка {attempt+2}/{max_retries} после ошибки соединения...")
                await asyncio.sleep(retry_delay)
                continue
            else:
                raise Exception(last_error)
                
        except Exception as e:
            logger.error(f"Неожиданная ошибка при запросе к Hugging Face API: {e}")
            last_error = str(e)
            if attempt < max_retries - 1:
                logger.warning(f"Повторная попытка {attempt+2}/{max_retries} после ошибки...")
                await asyncio.sleep(retry_delay)
                continue
            else:
                raise Exception(f"Неожиданная ошибка при запросе к Hugging Face API: {e}")
    
    if last_error:
        raise Exception(f"Все попытки подключения к Hugging Face API не удались. Последняя ошибка: {last_error}")
    
    raise Exception("Непредвиденная ошибка в работе с Hugging Face API")

# Функция для определения и вызова правильного API на основе имени модели
async def generate_response(messages, model, max_tokens, temperature, timeout=30):
    """
    Определяет нужный API на основе имени модели и вызывает соответствующую функцию
    
    Args:
        messages: Список сообщений для отправки в API
        model: Имя модели
        max_tokens: Максимальное количество токенов в ответе
        temperature: Температура (креативность) генерации
        timeout: Время ожидания ответа от API в секундах
        
    Returns:
        str: Сгенерированный ответ
    """
    if messages is None:
        logger.error("Получен пустой список сообщений (None) в generate_response")
        raise ValueError("Список сообщений не может быть пустым")
        
    if model is None:
        logger.error("Получено пустое название модели (None) в generate_response")
        raise ValueError("Название модели не может быть пустым")
        
    if max_tokens is None:
        max_tokens = 500  # Значение по умолчанию
        
    if temperature is None:
        temperature = 0.7  # Значение по умолчанию
        
    if not messages:
        logger.error("Получен пустой список сообщений в generate_response")
        raise ValueError("Список сообщений не может быть пустым")
    
    try:
        if model.startswith('together/'):
            return await generate_response_together(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout
            )
        elif model.startswith('huggingface/'):
            return await generate_response_huggingface(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout
            )
        else:
            return await generate_response_openrouter(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout
            )
    except Exception as e:
        logger.error(f"Ошибка при запросе к API для модели {model}: {e}")
        raise

@dp.message(Command("start"))
async def start(message: types.Message):
    user_id = message.from_user.id
    
    user_message_history[user_id] = []
    
    if user_id in user_last_messages:
        for msg_id in user_last_messages[user_id]:
            try:
                await bot.delete_message(chat_id=message.chat.id, message_id=msg_id)
            except Exception as e:
                logger.error(f"Не удалось удалить сообщение {msg_id}: {e}")
        
        user_last_messages[user_id] = []
    
    save_user_history(user_message_history)
    
    if user_id not in user_settings:
        user_settings[user_id] = DEFAULT_SETTINGS.copy()
        save_user_settings(user_settings)
    
    quick_start_keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🔍 Выбрать модель", callback_data="setting_model")],
        [InlineKeyboardButton(text="🚀 Начать общение", callback_data="start_chatting")]
    ])
    
    welcome_message = (
        "👋 <b>Добро пожаловать в AI Ассистента!</b>\n\n"
        "Я многофункциональный бот, использующий передовые языковые модели: DeepSeek, Gemini, Mistral, Llama и другие через сервисы OpenRouter, Together AI и Hugging Face. "
        "Вы можете:\n\n"
        "• Задавать любые вопросы на русском языке\n"
        "• Получать помощь с кодом и программированием\n"
        "• Генерировать и редактировать тексты\n"
        "• Отправлять <b>голосовые сообщения</b> — я распознаю их с помощью Vosk (без ограничений)\n"
        "• Настраивать параметры бота и выбирать модель ИИ\n\n"
        "<b>Технические особенности:</b>\n"
        "• Автоматическое переключение между моделями при недоступности\n"
        "• Локальное распознавание голоса с помощью Vosk и FFmpeg\n"
        "• Сохранение контекста разговора для умных ответов\n\n"
        "<b>Используйте кнопки меню внизу экрана для быстрого доступа к функциям:</b>\n"
        "🤖 Новый диалог - начать заново\n"
        "🧹 Очистить историю - удалить историю\n"
        "⚙️ Настройки - настроить модель и параметры\n"
        "ℹ️ Информация - текущие настройки\n"
        "❓ Помощь - справка по командам\n\n"
        "<b>Выберите действие для начала:</b>"
    )
    
    await message.answer(
        welcome_message, 
        reply_markup=quick_start_keyboard, 
        parse_mode=ParseMode.HTML
    )

@dp.message(Command("help"))
async def help_command(message: types.Message):
    """Показать информацию о доступных командах"""
    
    # Текст о голосовых сообщениях зависит от того, доступно ли локальное распознавание
    if use_local_recognition:
        voice_feature = "• <b>НОВОЕ!</b> Поддерживаю распознавание голосовых сообщений — отправьте аудио, и я отвечу на ваш вопрос\n  <i>(Безлимитный режим с локальным распознаванием - без ограничений на длину и количество запросов)</i>"
    else:
        voice_feature = "• <b>НОВОЕ!</b> Поддерживаю распознавание голосовых сообщений — отправьте аудио, и я отвечу на ваш вопрос\n  <i>(Лимиты: до 60 секунд, до 50 запросов в день, только русский язык)</i>"
    
    help_message = f"""
Привет! Я – многофункциональный <b>AI-ассистент</b> на базе современных языковых моделей. Готов помочь с ответами на вопросы и решением самых разных задач! 🤖💡

<b>Доступные команды:</b>
/start - начать новый диалог
/help - показать эту справку
/clear - очистить историю диалога
/settings - открыть настройки бота
/info - показать текущие настройки
/menu - показать меню с кнопками

<b>Поддерживаемые модели:</b>
• DeepSeek - многоязычная модель, оптимизированная для русского языка
• Gemini - модель от Google с высокими когнитивными способностями
• Mistral и Mixtral - мощные открытые модели с поддержкой русского языка
• Llama - семейство моделей от Meta для различных задач
• И другие (доступность моделей проверяется автоматически)

<b>Особенности:</b>
• Я помню контекст разговора и отвечаю с учетом предыдущих сообщений
• Автоматически переключаюсь между моделями при проблемах с доступностью
• Поддерживаю форматирование текста с Markdown и подсветку кода
• Можно настроить системное сообщение для особых задач
• Есть режим динамического чата для более чистого интерфейса{voice_feature}
• Использую локальное преобразование голоса в текст с помощью Vosk и FFmpeg

<b>Просто напишите сообщение или отправьте голосовое сообщение, чтобы начать разговор!</b>
"""
    await message.answer(help_message, parse_mode=ParseMode.HTML)

@dp.message(Command("clear"))
async def clear_history(message: types.Message):
    user_id = message.from_user.id
    
    user_message_history[user_id] = []
    
    save_user_history(user_message_history)
    
    await message.answer("История нашего разговора очищена!")

@dp.message(Command("info"))
async def show_info(message: types.Message):
    user_id = message.from_user.id
    
    if user_id not in user_settings:
        user_settings[user_id] = DEFAULT_SETTINGS.copy()
    
    settings = user_settings[user_id]
    dynamic_status = "включен" if settings.get('dynamic_chat', False) else "выключен"
    
    await message.answer(
        f"Ваши текущие настройки:\n"
        f"Модель: {settings['model']}\n"
        f"Максимальное количество токенов: {settings['max_tokens']}\n"
        f"Температура (креативность): {settings['temperature']}\n"
        f"Динамический чат: {dynamic_status}\n"
        f"Системное сообщение: {settings['system_message']}"
    )

@dp.message(Command("settings"))
async def settings(message: types.Message):
    # Создаем более описательное меню настроек с картинками и описаниями
    settings_text = (
        "⚙️ <b>Настройки бота</b>\n\n"
        "Здесь вы можете настроить параметры работы бота под свои предпочтения. "
        "Выберите, что именно хотите изменить:"
    )
    
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🤖 Изменить модель", callback_data='setting_model')],
        [InlineKeyboardButton(text="📏 Подробность ответа", callback_data='setting_max_tokens')],
        [InlineKeyboardButton(text="🎨 Уровень креативности", callback_data='setting_temperature')],
        [InlineKeyboardButton(text="📜 Длина истории", callback_data='setting_history_length')],
        [InlineKeyboardButton(text="✍️ Системное сообщение", callback_data='setting_system_message')],
        [InlineKeyboardButton(text="💬 Режим динамического чата", callback_data='setting_dynamic_chat')],
        [InlineKeyboardButton(text="🔙 Вернуться в главное меню", callback_data='back_to_main')]
    ])
    
    await message.answer(settings_text, reply_markup=keyboard, parse_mode=ParseMode.HTML)

@dp.callback_query(lambda c: c.data == 'back_to_main')
async def back_to_main_menu(callback_query: types.CallbackQuery):
    await callback_query.answer()
    await callback_query.message.edit_text(
        "✅ Вы вернулись в главное меню. Используйте кнопки внизу экрана для основных функций.",
        parse_mode=ParseMode.HTML
    )

@dp.callback_query()
async def button_callback(callback_query: types.CallbackQuery):
    try:
        # Проверяем, не устарел ли callback query
        if callback_query.message.date.timestamp() + 900 < time.time():
            logger.warning("Callback query устарел, игнорируем")
            return
            
        await callback_query.answer()
        
        user_id = callback_query.from_user.id
        # chat_id = callback_query.message.chat.id
        # chat_type = callback_query.message.chat.type
        
        # # Логируем информацию о чате для отладки проблемы с групповыми чатами
        # logger.info(f"Обработка callback от пользователя {user_id} в чате {chat_id} типа {chat_type}")
        
        # Инициализируем настройки пользователя, если их нет
        if user_id not in user_settings:
            user_settings[user_id] = DEFAULT_SETTINGS.copy()
        
        callback_data = callback_query.data
        
        if callback_data == 'setting_model':
            # Добавляем описание и инструкцию
            model_intro = (
                "🤖 <b>Выбор модели</b>\n\n"
                "Выберите модель, которая будет генерировать ответы. "
                "Модели сгруппированы по статусу доступности:\n\n"
                "✅ <b>Полностью рабочие</b> - доступны и дают ответы\n"
                "⚠️ <b>Частично рабочие</b> - могут иметь временные ограничения\n"
                "❌ <b>Недоступные</b> - временно не работают\n\n"
                "Выберите категорию для просмотра моделей:"
            )
            
            # Создаем клавиатуру с категориями
            keyboard = []
            
            # Добавляем кнопки для каждой категории, если в ней есть модели
            if MODEL_STATUSES["fully_working"]:
                keyboard.append([InlineKeyboardButton(
                    text="✅ Полностью рабочие модели",
                    callback_data='show_models_fully_working'
                )])
            
            if MODEL_STATUSES["partially_working"]:
                keyboard.append([InlineKeyboardButton(
                    text="⚠️ Частично рабочие модели",
                    callback_data='show_models_partially_working'
                )])
            
            if MODEL_STATUSES["unavailable"]:
                keyboard.append([InlineKeyboardButton(
                    text="❌ Недоступные модели",
                    callback_data='show_models_unavailable'
                )])
            
            # Добавляем кнопку возврата
            keyboard.append([InlineKeyboardButton(text="🔙 Назад к настройкам", callback_data='back_to_settings')])
            
            markup = InlineKeyboardMarkup(inline_keyboard=keyboard)
            await callback_query.message.edit_text(model_intro, reply_markup=markup, parse_mode=ParseMode.HTML)
        
        elif callback_data.startswith('show_models_'):
            # Категория моделей
            category = callback_data.replace('show_models_', '')
            category_names = {
                'fully_working': '✅ Полностью рабочие модели',
                'partially_working': '⚠️ Частично рабочие модели',
                'unavailable': '❌ Недоступные модели'
            }
            
            # Формируем текст с описанием категории
            category_text = f"<b>{category_names[category]}</b>\n\n"
            
            if category == 'fully_working':
                category_text += "Эти модели полностью работоспособны и готовы к использованию.\n\n"
            elif category == 'partially_working':
                category_text += "Эти модели могут иметь временные ограничения (лимит запросов, недостаточно кредитов) или не всегда возвращать ответ. Они часто работают, но с некоторыми ограничениями.\n\n"
            else:
                category_text += "Эти модели временно недоступны. Рекомендуется выбрать другую модель.\n\n"
            
            # Список моделей
            category_text += "<b>Доступные модели:</b>\n"
            for model in MODEL_STATUSES[category]:
                display_name = model.split('/')[-1]
                description = MODEL_DESCRIPTIONS.get(model, "Нет описания")
                current_indicator = "✅ " if model == user_settings[user_id]["model"] else ""
                category_text += f"\n• <b>{current_indicator}{display_name}</b>\n{description}\n"
            
            # Клавиатура с моделями
            keyboard = []
            for model in MODEL_STATUSES[category]:
                display_name = model.split('/')[-1]
                current_indicator = "✅ " if model == user_settings[user_id]["model"] else ""
                keyboard.append([InlineKeyboardButton(
                    text=f"{current_indicator}{display_name}",
                    callback_data=f'set_model_{model}'
                )])
            
            # Кнопки навигации
            keyboard.append([InlineKeyboardButton(text="🔙 К категориям", callback_data='setting_model')])
            keyboard.append([InlineKeyboardButton(text="🔙 К настройкам", callback_data='back_to_settings')])
            
            markup = InlineKeyboardMarkup(inline_keyboard=keyboard)
            await callback_query.message.edit_text(category_text, reply_markup=markup, parse_mode=ParseMode.HTML)
        
        elif callback_data == 'setting_max_tokens':
            # Описание для настройки максимального количества токенов
            tokens_text = (
                "📏 <b>Подробность ответа</b>\n\n"
                f"{SETTINGS_DESCRIPTIONS['max_tokens']}\n\n"
                f"Текущее значение: <b>{user_settings[user_id]['max_tokens']}</b> токенов"
            )
            
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(text="250 (короткий)", callback_data='set_max_tokens_250'),
                    InlineKeyboardButton(text="500 (средний)", callback_data='set_max_tokens_500')
                ],
                [
                    InlineKeyboardButton(text="1000 (длинный)", callback_data='set_max_tokens_1000'),
                    InlineKeyboardButton(text="2000 (очень длинный)", callback_data='set_max_tokens_2000')
                ],
                [InlineKeyboardButton(text="🔙 Назад к настройкам", callback_data='back_to_settings')]
            ])
            await callback_query.message.edit_text(tokens_text, reply_markup=keyboard, parse_mode=ParseMode.HTML)
        
        elif callback_data == 'setting_temperature':
            # Описание для настройки температуры
            temp_text = (
                "🎨 <b>Уровень креативности (температура)</b>\n\n"
                f"{SETTINGS_DESCRIPTIONS['temperature']}\n\n"
                f"Текущее значение: <b>{user_settings[user_id]['temperature']}</b>"
            )
            
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="0.3 (более точные ответы)", callback_data='set_temperature_0.3')],
                [InlineKeyboardButton(text="0.7 (сбалансировано)", callback_data='set_temperature_0.7')],
                [InlineKeyboardButton(text="1.0 (более творческие)", callback_data='set_temperature_1.0')],
                [InlineKeyboardButton(text="1.5 (максимум креативности)", callback_data='set_temperature_1.5')],
                [InlineKeyboardButton(text="🔙 Назад к настройкам", callback_data='back_to_settings')]
            ])
            await callback_query.message.edit_text(temp_text, reply_markup=keyboard, parse_mode=ParseMode.HTML)
        
        elif callback_data == 'setting_system_message':
            # Описание для настройки системного сообщения
            system_text = (
                "✍️ <b>Системное сообщение</b>\n\n"
                f"{SETTINGS_DESCRIPTIONS['system_message']}\n\n"
                "Текущее сообщение:\n"
                f"<code>{user_settings[user_id]['system_message']}</code>\n\n"
                "Для изменения, введите новое системное сообщение в следующем сообщении."
            )
            
            # Примеры системных сообщений
            system_examples = [
                "Ты дружелюбный ассистент, который помогает пользователям. Отвечай на русском языке, кратко и по делу. При форматировании используй: **жирный текст**, *курсив*, # заголовки, ## подзаголовки, * для списков. Это будет корректно отображаться в Telegram.",
                "Ты эксперт по программированию. Давай подробные технические ответы с примерами кода. Используй ```python\nкод``` для блоков кода Python, ```javascript\nкод``` для JavaScript и т.д. Для однострочного кода используй обратные кавычки `код`. Если нужны заголовки, используй markdown формат: # Главный заголовок, ## Подзаголовок.",
                "Ты креативный писатель. Твои ответы должны быть яркими, образными и увлекательными. Структурируй текст с помощью заголовков (# Заголовок), подзаголовков (## Подзаголовок) и выделяй важные фразы **жирным текстом** или *курсивом*.",
                "Ты научный консультант. Давай точные ответы, основанные на проверенных фактах. Для структурирования информации используй: # Главные разделы, ## Подразделы, * маркированные списки. Выделяй ключевые термины с помощью **жирного** текста или `моноширинного шрифта`."
            ]
            
            example_buttons = []
            for i, example in enumerate(system_examples):
                short_example = example.split('.')[0] + "..."
                example_buttons.append([InlineKeyboardButton(
                    text=f"Пример {i+1}: {short_example}", 
                    callback_data=f'set_system_example_{i}'
                )])
            
            example_buttons.append([InlineKeyboardButton(text="✏️ Написать своё", callback_data='write_system_message')])
            example_buttons.append([InlineKeyboardButton(text="🔙 Назад к настройкам", callback_data='back_to_settings')])
            
            keyboard = InlineKeyboardMarkup(inline_keyboard=example_buttons)
            await callback_query.message.edit_text(system_text, reply_markup=keyboard, parse_mode=ParseMode.HTML)
        
        elif callback_data == 'back_to_settings':
            # Возвращаемся к общему меню настроек
            await settings(callback_query.message)
        
        elif callback_data.startswith('set_model_'):
            model = callback_data.replace('set_model_', '')
            old_model = user_settings[user_id]['model']
            user_settings[user_id]['model'] = model
            
            save_user_settings(user_settings)
            
            loading_message = await callback_query.message.edit_text(
                f"⏳ Меняю модель с {old_model.split('/')[-1]} на {model.split('/')[-1]}...",
                parse_mode=ParseMode.HTML
            )
            
            await asyncio.sleep(1)
            
            model_desc = MODEL_DESCRIPTIONS.get(model, "Нет описания")
            success_text = (
                f"✅ <b>Модель успешно изменена!</b>\n\n"
                f"Выбрана модель: <b>{model.split('/')[-1]}</b>\n"
                f"Описание: {model_desc}\n\n"
                f"Теперь все ваши диалоги будут обрабатываться с помощью этой модели."
            )
            
            back_keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="🔙 Назад к настройкам", callback_data='back_to_settings')]
            ])
            
            await loading_message.edit_text(success_text, reply_markup=back_keyboard, parse_mode=ParseMode.HTML)
        
        elif callback_data.startswith('set_max_tokens_'):
            max_tokens = int(callback_data.replace('set_max_tokens_', ''))
            user_settings[user_id]['max_tokens'] = max_tokens
            
            save_user_settings(user_settings)
            
            success_text = (
                f"✅ <b>Настройка успешно изменена!</b>\n\n"
                f"Подробность ответа: <b>{max_tokens} токенов</b>\n\n"
                f"При значении {max_tokens} токенов "
                f"{'ответы будут довольно короткими.' if max_tokens <= 250 else 'ответы будут средней длины.' if max_tokens <= 500 else 'ответы могут быть достаточно длинными.' if max_tokens <= 1000 else 'ответы могут быть очень подробными.'}"
            )
            
            back_keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="🔙 Назад к настройкам", callback_data='back_to_settings')]
            ])
            
            await callback_query.message.edit_text(success_text, reply_markup=back_keyboard, parse_mode=ParseMode.HTML)
        
        elif callback_data.startswith('set_temperature_'):
            temperature = float(callback_data.replace('set_temperature_', ''))
            user_settings[user_id]['temperature'] = temperature
            
            save_user_settings(user_settings)
            
            creativity_level = "низкая" if temperature <= 0.3 else "средняя" if temperature <= 0.7 else "высокая" if temperature <= 1.0 else "очень высокая"
            
            success_text = (
                f"✅ <b>Настройка успешно изменена!</b>\n\n"
                f"Уровень креативности (температура): <b>{temperature}</b>\n\n"
                f"Креативность: <b>{creativity_level}</b>\n"
                f"{'Ответы будут более предсказуемыми и точными.' if temperature <= 0.3 else 'Ответы будут сбалансированными между точностью и креативностью.' if temperature <= 0.7 else 'Ответы будут более творческими и разнообразными.' if temperature <= 1.0 else 'Ответы будут максимально творческими, но могут быть менее точными.'}"
            )
            
            back_keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="🔙 Назад к настройкам", callback_data='back_to_settings')]
            ])
            
            await callback_query.message.edit_text(success_text, reply_markup=back_keyboard, parse_mode=ParseMode.HTML)
        
        elif callback_data.startswith('set_system_example_'):
            example_index = int(callback_data.replace('set_system_example_', ''))
            system_examples = [
                "Ты дружелюбный ассистент, который помогает пользователям. Отвечай на русском языке, кратко и по делу. При форматировании используй: **жирный текст**, *курсив*, # заголовки, ## подзаголовки, * для списков. Это будет корректно отображаться в Telegram.",
                "Ты эксперт по программированию. Давай подробные технические ответы с примерами кода. Используй ```python\nкод``` для блоков кода Python, ```javascript\nкод``` для JavaScript и т.д. Для однострочного кода используй обратные кавычки `код`. Если нужны заголовки, используй markdown формат: # Главный заголовок, ## Подзаголовок.",
                "Ты креативный писатель. Твои ответы должны быть яркими, образными и увлекательными. Структурируй текст с помощью заголовков (# Заголовок), подзаголовков (## Подзаголовок) и выделяй важные фразы **жирным текстом** или *курсивом*.",
                "Ты научный консультант. Давай точные ответы, основанные на проверенных фактах. Для структурирования информации используй: # Главные разделы, ## Подразделы, * маркированные списки. Выделяй ключевые термины с помощью **жирного** текста или `моноширинного шрифта`."
            ]
            
            user_settings[user_id]['system_message'] = system_examples[example_index]
            
            save_user_settings(user_settings)
            
            success_text = (
                f"✅ <b>Системное сообщение успешно изменено!</b>\n\n"
                f"Новое системное сообщение:\n"
                f"<code>{system_examples[example_index]}</code>\n\n"
                f"Теперь модель будет следовать этой инструкции при генерации ответов."
            )
            
            back_keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="🔙 Назад к настройкам", callback_data='back_to_settings')]
            ])
            
            await callback_query.message.edit_text(success_text, reply_markup=back_keyboard, parse_mode=ParseMode.HTML)
        
        elif callback_data == 'write_system_message':
            # Создаем запись в пользовательских данных
            user_data = {}
            user_data[user_id] = {"waiting_for_system_message": True}
            
            await callback_query.message.edit_text(
                "✏️ <b>Введите новое системное сообщение</b>\n\n"
                "Системное сообщение - это инструкция для модели, определяющая её поведение и стиль ответов.\n\n"
                "Текущее сообщение:\n"
                f"<code>{user_settings[user_id]['system_message']}</code>\n\n"
                "Отправьте ваш вариант в следующем сообщении.",
                parse_mode=ParseMode.HTML
            )
            
            # Сохраняем состояние в глобальной переменной
            globals()['user_waiting'] = user_data
        
        elif callback_data == 'setting_dynamic_chat':
            # Описание для настройки динамического чата
            current_status = "включен" if user_settings[user_id].get('dynamic_chat', False) else "выключен"
            dynamic_text = (
                f"💬 <b>Режим динамического чата</b>\n\n"
                f"{SETTINGS_DESCRIPTIONS['dynamic_chat']}\n\n"
                f"Текущий статус: <b>{current_status}</b>\n\n"
                f"При включенном режиме предыдущие сообщения бота будут автоматически удаляться при отправке новых ответов, "
                f"что создает эффект 'обновления' чата. Это делает интерфейс более чистым и динамичным."
            )
            
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="✅ Включить", callback_data='set_dynamic_chat_true')],
                [InlineKeyboardButton(text="❌ Выключить", callback_data='set_dynamic_chat_false')],
                [InlineKeyboardButton(text="🔙 Назад к настройкам", callback_data='back_to_settings')]
            ])
            await callback_query.message.edit_text(dynamic_text, reply_markup=keyboard, parse_mode=ParseMode.HTML)
        
        elif callback_data.startswith('set_dynamic_chat_'):
            enabled = callback_data == 'set_dynamic_chat_true'
            user_settings[user_id]['dynamic_chat'] = enabled
            
            save_user_settings(user_settings)
            
            status_text = "включен" if enabled else "выключен"
            success_text = (
                f"✅ <b>Настройка успешно изменена!</b>\n\n"
                f"Режим динамического чата: <b>{status_text}</b>\n\n"
                f"{'Теперь предыдущие сообщения бота будут автоматически удаляться при отправке новых ответов.' if enabled else 'Теперь все сообщения бота будут сохраняться в истории чата.'}"
            )
            
            back_keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="🔙 Назад к настройкам", callback_data='back_to_settings')]
            ])
            await callback_query.message.edit_text(success_text, reply_markup=back_keyboard, parse_mode=ParseMode.HTML)
        
        elif callback_data.startswith('set_history_length_'):
            history_length = int(callback_data.replace('set_history_length_', ''))
            user_settings[user_id]['history_length'] = history_length
            
            save_user_settings(user_settings)
            
            history_description = "минимальная" if history_length <= 5 else "небольшая" if history_length <= 10 else "средняя" if history_length <= 20 else "большая" if history_length <= 50 else "максимальная"
            
            success_text = (
                f"✅ <b>Настройка успешно изменена!</b>\n\n"
                f"Длина истории: <b>{history_length} пар сообщений</b>\n\n"
                f"Глубина контекста: <b>{history_description}</b>\n"
                f"{'Минимальный контекст, бот будет помнить только последние сообщения. Экономия токенов.' if history_length <= 5 else 'Небольшой контекст, подходит для большинства диалогов.' if history_length <= 10 else 'Средний контекст, бот будет помнить больше деталей разговора.' if history_length <= 20 else 'Большой контекст, хорошо для сложных и длинных обсуждений.' if history_length <= 50 else 'Максимально возможный контекст, бот запомнит весь возможный диалог. Может потреблять больше токенов.'}"
            )
            
            back_keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="🔙 Назад к настройкам", callback_data='back_to_settings')]
            ])
            
            await callback_query.message.edit_text(success_text, reply_markup=back_keyboard, parse_mode=ParseMode.HTML)
        
        elif callback_data == 'setting_history_length':
            # Создаем клавиатуру для выбора длины истории
            history_keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(text="5 пар", callback_data='set_history_length_5'),
                    InlineKeyboardButton(text="10 пар", callback_data='set_history_length_10'),
                    InlineKeyboardButton(text="15 пар", callback_data='set_history_length_15')
                ],
                [
                    InlineKeyboardButton(text="20 пар", callback_data='set_history_length_20'),
                    InlineKeyboardButton(text="30 пар", callback_data='set_history_length_30'),
                    InlineKeyboardButton(text="50 пар", callback_data='set_history_length_50')
                ],
                [
                    InlineKeyboardButton(text="75 пар", callback_data='set_history_length_75'),
                    InlineKeyboardButton(text="100 пар", callback_data='set_history_length_100')
                ],
                [InlineKeyboardButton(text="🔙 Назад к настройкам", callback_data='back_to_settings')]
            ])
            
            current_history_length = user_settings[user_id].get('history_length', 10)
            
            history_text = (
                f"🔢 <b>Настройка длины истории</b>\n\n"
                f"Текущее значение: <b>{current_history_length} пар сообщений</b>\n\n"
                f"Выберите, сколько пар сообщений (вопрос-ответ) бот будет хранить в истории диалога. "
                f"Большее значение позволяет боту помнить больше контекста, но увеличивает расход токенов.\n\n"
                f"Минимальное значение: 5 пар\n"
                f"Максимальное значение: 100 пар\n"
                f"Рекомендуемое значение: 10-20 пар"
            )
            
            await callback_query.message.edit_text(history_text, reply_markup=history_keyboard, parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.error(f"Ошибка при обработке callback query: {e}")

# Новая команда для вызова меню
@dp.message(Command("menu"))
async def show_menu(message: types.Message):
    await message.answer(
        "Главное меню:",
        reply_markup=main_keyboard
    )

# Добавляем обработчики для кнопок основного меню
@dp.message(lambda message: message.text == "🤖 Новый диалог")
async def new_dialog(message: types.Message):
    user_id = message.from_user.id
    user_message_history[user_id] = []
    
    if user_id in user_last_messages:
        for msg_id in user_last_messages[user_id]:
            try:
                await bot.delete_message(chat_id=message.chat.id, message_id=msg_id)
            except Exception as e:
                logger.error(f"Не удалось удалить сообщение {msg_id}: {e}")
        
        user_last_messages[user_id] = []
    
    save_user_history(user_message_history)

    await start(message)

@dp.message(lambda message: message.text == "🧹 Очистить историю")
async def clear_history_button(message: types.Message):
    await clear_history(message)

@dp.message(lambda message: message.text == "⚙️ Настройки")
async def settings_button(message: types.Message):
    await settings(message)

@dp.message(lambda message: message.text == "ℹ️ Информация")
async def info_button(message: types.Message):
    await show_info(message)

@dp.message(lambda message: message.text == "❓ Помощь")
async def help_button(message: types.Message):
    await help_command(message)

@dp.message(lambda message: message.voice is not None, flags={"priority": 10})
async def handle_voice_message(message: types.Message):
    logger.info(f"Получено голосовое сообщение от пользователя {message.from_user.id}")
    
    global use_local_recognition, vosk_model
    
    try:
        processing_msg = await message.answer("🎤 Распознаю голосовое сообщение...")
        
        voice = message.voice
        file_id = voice.file_id
        logger.info(f"ID голосового файла: {file_id}, длительность: {voice.duration} сек")
        
        if use_local_recognition and vosk_model is None:
            logger.error("Vosk модель помечена как доступная, но не инициализирована. Попытка переинициализации.")
            if not init_vosk_model():
                logger.warning("Не удалось инициализировать Vosk модель. Переключаемся на Google API.")
                use_local_recognition = False
        
        if not use_local_recognition and voice.duration > 60:
            await processing_msg.edit_text("⚠️ Голосовое сообщение слишком длинное для Google API. Максимальная длительность - 60 секунд.")
            return
        
        logger.info("Скачиваю голосовой файл...")
        voice_file = await bot.get_file(file_id)
        voice_data = await bot.download_file(voice_file.file_path)
        
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as temp_voice:
            temp_voice.write(voice_data.read())
            temp_voice_path = temp_voice.name
        logger.info(f"Голосовой файл сохранен во временный файл: {temp_voice_path}")
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_wav_path = temp_wav.name
        logger.info(f"Создан временный WAV файл: {temp_wav_path}")
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ffmpeg_path = os.path.join(current_dir, 'ffmpeg.exe')
        if not os.path.exists(ffmpeg_path):
            ffmpeg_path = 'ffmpeg'
        
        logger.info(f"Используем ffmpeg по пути: {ffmpeg_path}")
        
        logger.info("Начинаю конвертацию OGG в WAV...")
        command = [
            ffmpeg_path,
            '-hide_banner',  # Скрыть баннер
            '-loglevel', 'error',  # Показывать только ошибки
            '-i', temp_voice_path,
            '-ar', '16000',  # Частота дискретизации
            '-ac', '1',      # Моно
            '-acodec', 'pcm_s16le',  # Кодек для WAV
            '-f', 'wav',     # Формат WAV
            '-y',            # Перезаписать файл если существует
            temp_wav_path
        ]
        logger.info(f"Выполняемая команда: {' '.join(command)}")
        
        try:
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=FFMPEG_CONVERSION_TIMEOUT,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            
            if process.returncode != 0:
                logger.error(f"FFmpeg вернул ошибку: {process.stderr}")
                await processing_msg.edit_text("❌ Ошибка при конвертации голосового сообщения")
                return
                
            logger.info("Конвертация завершена успешно")
            
        except subprocess.TimeoutExpired:
            logger.error(f"FFmpeg превысил время ожидания ({FFMPEG_CONVERSION_TIMEOUT} секунд)")
            try:
                process.kill()
            except:
                pass
            await processing_msg.edit_text("❌ Превышено время ожидания при конвертации голосового сообщения")
            return
        
        if not os.path.exists(temp_wav_path):
            logger.error(f"WAV файл не существует: {temp_wav_path}")
            await processing_msg.edit_text("❌ Ошибка при конвертации голосового сообщения. WAV файл не был создан.")
            return
            
        file_size = os.path.getsize(temp_wav_path)
        logger.info(f"Размер WAV файла: {file_size} байт")
        
        if file_size == 0:
            logger.error(f"WAV файл имеет нулевой размер: {temp_wav_path}")
            await processing_msg.edit_text("❌ Ошибка при конвертации голосового сообщения. WAV файл имеет нулевой размер.")
            return
        
        try:
            text = None
            
            if use_local_recognition:
                logger.info("Начинаю локальное распознавание через Vosk...")
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = None
                    try:
                        future = executor.submit(recognize_with_vosk, temp_wav_path)
                        text = future.result(timeout=VOICE_RECOGNITION_TIMEOUT)
                    except concurrent.futures.TimeoutError:
                        logger.error(f"Превышено время ожидания при распознавании через Vosk ({VOICE_RECOGNITION_TIMEOUT} сек)")
                        if future:
                            recognition_stop_flag.set()
                            future.cancel()
                            await asyncio.sleep(1)
                        await processing_msg.edit_text(f"⚠️ Превышено время ожидания при распознавании ({VOICE_RECOGNITION_TIMEOUT} секунд). Пожалуйста, используйте более короткое сообщение.")
                        return
                
                if text:
                    logger.info(f"Распознан текст: '{text}'")
                else:
                    logger.error("Распознанный текст пуст")
                    await processing_msg.edit_text("❌ Не удалось распознать речь. Пожалуйста, попробуйте снова или говорите более отчетливо.")
                    return
                
                await processing_msg.edit_text(
                    f"🎤 <b>Распознано (локально):</b>\n\n"
                    f"{text}\n\n"
                    f"<i>ℹ️ Используется локальное распознавание - без ограничений</i>", 
                    parse_mode=ParseMode.HTML
                )
            else:
                # Используем Google Speech Recognition (с ограничениями)
                try:
                    recognizer = sr.Recognizer()
                    with sr.AudioFile(temp_wav_path) as source:
                        audio_data = recognizer.record(source)
                        text = recognizer.recognize_google(audio_data, language='ru-RU')
                    
                    if not text or not text.strip():
                        logger.error("Google API вернул пустой текст")
                        await processing_msg.edit_text("❌ Не удалось распознать речь. Пожалуйста, попробуйте снова или говорите более отчетливо.")
                        return
                    
                    logger.info(f"Распознан текст: '{text}'")
                    
                    await processing_msg.edit_text(
                        f"🎤 <b>Распознано (Google API):</b>\n\n"
                        f"{text}\n\n"
                        f"<i>ℹ️ Лимиты распознавания: до 60 сек, ~50 запросов в день</i>", 
                        parse_mode=ParseMode.HTML
                    )
                except sr.UnknownValueError:
                    logger.error("Google API не смог распознать речь")
                    await processing_msg.edit_text("❌ Не удалось распознать речь. Пожалуйста, попробуйте снова или говорите более отчетливо.")
                    return
                except sr.RequestError as e:
                    logger.error(f"Ошибка сервиса распознавания Google: {e}")
                    await processing_msg.edit_text(f"❌ Ошибка сервиса распознавания Google: {e}\n\nВозможно, превышен дневной лимит запросов (около 50).")
                    return
            
            if text:
                try:
                    message_dict = {
                        'message_id': message.message_id,
                        'date': message.date,
                        'chat': message.chat,
                        'from_user': message.from_user,
                        'text': text
                    }
                    
                    new_message = types.Message(**message_dict)
                    new_message.as_(bot)
                    
                    logger.info("Передаю распознанный текст в обработчик сообщений")
                    await handle_message(new_message)
                except Exception as e:
                    logger.error(f"Ошибка при создании нового объекта сообщения: {e}")
                    await message.answer(f"❌ Ошибка при обработке распознанного текста: {e}")
        except sr.UnknownValueError:
            logger.error("Не удалось распознать речь")
            await processing_msg.edit_text("❌ Не удалось распознать речь. Пожалуйста, попробуйте снова или говорите более отчетливо.")
        except sr.RequestError as e:
            if use_local_recognition:
                logger.error(f"Ошибка локального сервиса распознавания: {e}")
                await processing_msg.edit_text(f"❌ Ошибка локального сервиса распознавания: {e}")
            else:
                logger.error(f"Ошибка сервиса распознавания Google: {e}")
                await processing_msg.edit_text(f"❌ Ошибка сервиса распознавания Google: {e}\n\nВозможно, превышен дневной лимит запросов (около 50).")
        except Exception as e:
            logger.error(f"Общая ошибка при распознавании: {e}")
            await processing_msg.edit_text(f"❌ Ошибка при распознавании речи: {e}")
        
        try:
            os.unlink(temp_voice_path)
            os.unlink(temp_wav_path)
        except Exception as e:
            logger.error(f"Ошибка при удалении временных файлов: {e}")
            
    except Exception as e:
        logger.error(f"Ошибка при обработке голосового сообщения: {e}", exc_info=True)
        try:
            await message.answer(f"❌ Произошла ошибка при обработке голосового сообщения: {e}")
        except Exception as msg_err:
            logger.error(f"Невозможно отправить сообщение об ошибке: {msg_err}")

@dp.message(flags={"priority": 1})
async def handle_message(message: types.Message):
    if message is None:
        logger.error("Получено пустое сообщение (None) в handle_message")
        return
    
    if message.text and message.text in ["🤖 Новый диалог", "🧹 Очистить историю", "⚙️ Настройки", "ℹ️ Информация", "❓ Помощь"]:
        return
        
    if message.text is None:
        if message.voice is not None:
            logger.debug("Получено голосовое сообщение в handle_message. Будет обработано специальным обработчиком.")
            return
        else:
            logger.error("Получено сообщение с пустым текстом (None) в handle_message")
            return
        
    user_id = message.from_user.id
    if user_id is None:
        logger.error("Получено сообщение с пустым ID пользователя (None) в handle_message")
        return
        
    if user_id not in user_settings:
        logger.warning(f"Настройки не найдены для пользователя {user_id}, создаем новые")
        user_settings[user_id] = DEFAULT_SETTINGS.copy()
        save_user_settings(user_settings)
    
    settings = user_settings[user_id]
    if settings is None:
        logger.error(f"Настройки пользователя {user_id} оказались None")
        settings = DEFAULT_SETTINGS.copy()
        user_settings[user_id] = settings
        save_user_settings(user_settings)
    
    if user_id not in user_message_history:
        logger.info(f"Создаем новую историю для пользователя {user_id}")
        user_message_history[user_id] = []
        save_user_history(user_message_history)
    
    user_message_history[user_id].append({"role": "user", "content": message.text})
    
    history_length = min(settings.get("history_length", 10), 100)
    max_messages = history_length * 2
    if len(user_message_history[user_id]) > max_messages:
        excess = len(user_message_history[user_id]) - max_messages
        user_message_history[user_id] = user_message_history[user_id][excess:]
        logger.info(f"История пользователя {user_id} сокращена до {max_messages} сообщений (макс. {history_length} пар)")
    
    save_user_history(user_message_history)
    
    try:
        dynamic_chat = settings.get('dynamic_chat', False)
        
        if dynamic_chat and user_id in user_last_messages:
            last_messages = user_last_messages[user_id]
            for msg_id in last_messages:
                try:
                    await bot.delete_message(chat_id=message.chat.id, message_id=msg_id)
                except Exception as e:
                    logger.error(f"Не удалось удалить сообщение {msg_id}: {e}")
            user_last_messages[user_id] = []
        
        system_message = settings['system_message']
        if "русском языке" not in system_message:
            system_message = "Ты русскоязычный ассистент. ОБЯЗАТЕЛЬНО отвечай ТОЛЬКО на русском языке, кратко и по делу. " + system_message
        
        user_message = message.text
        if not any(phrase in user_message.lower() for phrase in ["на русском", "по-русски", "русский"]):
            user_message = f"{user_message}\n\nОтветь на русском языке."
        
        messages = [{"role": "system", "content": system_message}] + user_message_history[user_id]
        messages[-1]["content"] = user_message
        
        await bot.send_chat_action(message.chat.id, 'typing')
        
        model_name = settings['model'].split('/')[-1]
        loading_message = await message.answer(
            f"⏳ <i>Генерирую ответ с помощью модели</i> <code>{model_name}</code>...", 
            parse_mode=ParseMode.HTML
        )
        
        start_time = time.time()
        
        model = settings['model']
        bot_response = None
        used_fallback = False
        fallback_model = None
        
        current_model_status = None
        for status, models in MODEL_STATUSES.items():
            if model in models:
                current_model_status = status
                break
        
        if current_model_status == "unavailable":
            fallback_models = get_fallback_models(model)
            
            if fallback_models:
                fallback_model = fallback_models[0]
                logger.info(f"Основная модель {model} недоступна. Используем запасную: {fallback_model}")
                used_fallback = True
            else:
                logger.error("Нет доступных запасных моделей!")
                raise Exception("Нет доступных запасных моделей")
                
        try:
            current_model = fallback_model if fallback_model else model
            
            bot_response = await generate_response(
                messages=messages,
                model=current_model,
                max_tokens=settings['max_tokens'],
                temperature=settings['temperature']
            )
                
            if not bot_response or bot_response.strip() == "":
                logger.warning(f"Получен пустой ответ от модели {current_model}, пробуем запасную модель")
                raise Exception(f"Пустой ответ от модели {current_model}")
                
        except Exception as api_error:
            logger.warning(f"Ошибка при использовании {model if not fallback_model else fallback_model}: {api_error}")
            
            # Получаем приоритетный список запасных моделей
            fallback_models = get_fallback_models(model)
            if fallback_model and fallback_model in fallback_models:
                fallback_models.remove(fallback_model)
            
            if not fallback_models:
                logger.error("Исчерпаны все доступные запасные модели!")
                raise Exception(f"Не удалось получить ответ. Последняя ошибка: {api_error}")
            
            fallback_system_message = (
                "Ты русскоязычный ассистент общего назначения. ВАЖНО: отвечаешь на вопросы по ЛЮБЫМ темам, "
                "Не ограничивай себя какой-то конкретной тематикой. "
                "Ты должен помогать с любыми темами: наукой, искусством, историей, повседневными вопросами - со всем. "
                "НЕ выдавай шаблонные ответы. Каждый ответ должен быть уникальным и соответствовать запросу пользователя. "
                "Обязательно отвечай ТОЛЬКО НА РУССКОМ ЯЗЫКЕ, кратко и по сути вопроса. "
                "Очень внимательно прочитай вопрос пользователя и отвечай ТОЛЬКО на заданный вопрос. "
                "Не давай общую информацию не по теме запроса. "
                "При необходимости используй форматирование markdown: **жирный**, *курсив*, # заголовок, ```код```."
            )

            if user_id in user_message_history and len(user_message_history[user_id]) > 0:
                for i in range(len(user_message_history[user_id]) - 1, -1, -1):
                    if user_message_history[user_id][i]["role"] == "assistant":
                        logger.info(f"Удаляем последний ответ ассистента из истории при переключении на запасную модель")
                        break
            
            # Пробуем использовать запасные модели из списка
            for current_fallback_model in fallback_models:
                try:
                    logger.info(f"Пробуем запасную модель: {current_fallback_model}")
                    fallback_messages = [{"role": "system", "content": fallback_system_message}]
                    
                    if user_id in user_message_history and len(user_message_history[user_id]) > 0:
                        history_to_add = user_message_history[user_id][-3:]
                        if len(history_to_add) > 1:
                            fallback_messages.extend(history_to_add[:-1])
                        
                        last_message = history_to_add[-1]
                        if last_message["role"] == "user":
                            last_content = last_message["content"]
                            enhanced_content = (
                                f"{last_content}\n\n"
                                "ВАЖНО: Отвечай ТОЛЬКО на русском языке. "
                                "Отвечай СТРОГО по теме вопроса, не давай общую информацию не по теме. "
                                "ОТВЕЧАЙ НА ВОПРОСЫ ПО ЛЮБЫМ ТЕМАМ. "
                                "Ты универсальный помощник, который может обсуждать любые темы."
                            )
                            fallback_messages.append({"role": "user", "content": enhanced_content})
                    
                    if not fallback_messages or len(fallback_messages) == 1:
                        fallback_messages.append({
                            "role": "user", 
                            "content": user_message + "\n\nВАЖНО: Отвечай ТОЛЬКО на русском языке и строго по теме вопроса. "
                            "Отвечай на вопросы по ЛЮБЫМ темам."
                        })
                    
                    logger.debug(f"Сообщения для запасной модели {current_fallback_model}: {fallback_messages}")
                    
                    bot_response = await generate_response(
                        messages=fallback_messages,
                        model=current_fallback_model,
                        max_tokens=settings['max_tokens'],
                        temperature=settings['temperature']
                    )
                    
                    if bot_response and bot_response.strip() != "":
                        fallback_model = current_fallback_model
                        used_fallback = True
                        update_model_status(current_fallback_model, "fully_working")
                        
                        if current_model_status != "unavailable":
                            update_model_status(model, "partially_working", f"Заменена на {current_fallback_model}")
                        
                        break
                    else:
                        logger.warning(f"Получен пустой ответ от запасной модели {current_fallback_model}")
                        update_model_status(current_fallback_model, "partially_working", "Пустой ответ")
                        
                except Exception as fallback_error:
                    error_msg = str(fallback_error) if fallback_error is not None else "Неизвестная ошибка"
                    logger.error(f"Запасная модель {current_fallback_model} тоже недоступна: {error_msg}")
                    update_model_status(current_fallback_model, None, error_msg)
            
            # Если ни одна запасная модель не сработала
            if not bot_response or bot_response.strip() == "":
                raise Exception(f"Все модели недоступны. Последняя ошибка: {api_error}")
        
        # Считаем время генерации
        generation_time = time.time() - start_time
        await loading_message.delete()
        
        if bot_response is None:
            logger.error("Получен пустой ответ (None) от модели после всех попыток")
            raise Exception("Все модели вернули пустой ответ")
            
        formatted_response = prepare_response_for_telegram(bot_response)
        new_messages = []
        
        # Проверяем длину ответа и обрезаем при необходимости
        if len(formatted_response) > 4096:
            formatted_response_truncated = formatted_response[:4090] + "..."
            continue_keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="🔄 Продолжить ответ", callback_data='continue_response')]
            ])
            
            if not hasattr(globals(), 'full_responses'):
                globals()['full_responses'] = {}
            globals()['full_responses'][user_id] = formatted_response
            
            try:
                response_msg = await message.answer(formatted_response_truncated, reply_markup=continue_keyboard, parse_mode=ParseMode.HTML)
                new_messages.append(response_msg.message_id)
            except Exception as e:
                logger.error(f"Ошибка при отправке HTML-ответа: {e}")
                # Если ошибка связана с HTML, отправляем текст без форматирования
                if "can't parse entities" in str(e):
                    response_msg = await message.answer(
                        "❗ <b>Не удалось отформатировать ответ.</b> Ниже ответ без форматирования:\n\n" + bot_response,
                        reply_markup=continue_keyboard,
                        parse_mode=None
                    )
                    new_messages.append(response_msg.message_id)
                else:
                    raise
        else:
            try:
                response_msg = await message.answer(formatted_response, parse_mode=ParseMode.HTML)
                new_messages.append(response_msg.message_id)
            except Exception as e:
                logger.error(f"Ошибка при отправке HTML-ответа: {e}")
                if "can't parse entities" in str(e):
                    response_msg = await message.answer(
                        "❗ <b>Не удалось отформатировать ответ.</b> Ниже ответ без форматирования:\n\n" + bot_response,
                        parse_mode=None
                    )
                    new_messages.append(response_msg.message_id)
                else:
                    raise
        
        info_model = fallback_model if fallback_model else model
        if fallback_model:
            info_text = (
                f"<i>Сгенерировано за {generation_time:.2f} сек. | "
                f"Модель: <s>{model.split('/')[-1]}</s> → {info_model.split('/')[-1]} (запасная модель, т.к. используемая недоступна) | "
                f"Темп.: {settings['temperature']} | Макс.токенов: {settings['max_tokens']}</i>"
            )
            used_fallback = True  # Если используется запасная модель, всегда устанавливаем флаг
        else:
            info_text = (
                f"<i>Сгенерировано за {generation_time:.2f} сек. | "
                f"Модель: {info_model.split('/')[-1]} | "
                f"Темп.: {settings['temperature']} | Макс.токенов: {settings['max_tokens']}</i>"
            )
        info_msg = await message.answer(info_text, parse_mode=ParseMode.HTML)
        new_messages.append(info_msg.message_id)
        
        # Если использовалась запасная модель, предложим пользователю переключиться на нее постоянно
        if used_fallback and fallback_model:
            switch_keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text=f"🔄 Переключиться на {fallback_model.split('/')[-1]}", 
                                     callback_data=f'set_model_{fallback_model}')]
            ])
            switch_msg = await message.answer(
                f"⚠️ <b>Внимание:</b> Модель {model} недоступна или вернула пустой ответ. "
                f"Был автоматически использован запасной вариант. Хотите переключиться на эту модель постоянно?",
                reply_markup=switch_keyboard,
                parse_mode=ParseMode.HTML
            )
            new_messages.append(switch_msg.message_id)
        
        user_last_messages[user_id] = new_messages
        
        user_message_history[user_id].append({"role": "assistant", "content": bot_response})
        
        # Каждая пара - это 2 сообщения, поэтому умножаем на 2
        history_length = min(settings.get("history_length", 10), 100)  # Максимум 100 пар
        max_messages = history_length * 2
        if len(user_message_history[user_id]) > max_messages:
            excess = len(user_message_history[user_id]) - max_messages
            user_message_history[user_id] = user_message_history[user_id][excess:]
            logger.info(f"История пользователя {user_id} сокращена до {max_messages} сообщений (макс. {history_length} пар)")
        
        save_user_history(user_message_history)
    
    except Exception as e:
        logger.error(f"Ошибка при запросе к API: {e}")
        
        error_keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="⚙️ Настройки", callback_data='setting_model')],
            [InlineKeyboardButton(text="🔄 Попробовать снова", callback_data='retry_last_message')]
        ])
        
        error_msg = await message.answer(
            f"❌ <b>Произошла ошибка</b>\n\n"
            f"<code>{str(e)}</code>\n\n"
            f"Это может быть связано с перегрузкой API, ограничениями модели или другими техническими проблемами. "
            f"Попробуйте изменить модель в настройках или повторить запрос позже.",
            reply_markup=error_keyboard,
            parse_mode=ParseMode.HTML
        )
        
        if user_id not in user_last_messages:
            user_last_messages[user_id] = []
        user_last_messages[user_id].append(error_msg.message_id)

@dp.callback_query(lambda c: c.data == 'continue_response')
async def continue_response(callback_query: types.CallbackQuery):
    if callback_query is None:
        logger.error("Получен пустой callback_query (None) в continue_response")
        return
        
    if callback_query.message is None:
        logger.error("Получен пустой message в callback_query (None) в continue_response")
        return
        
    await callback_query.answer()
    
    user_id = callback_query.from_user.id
    if user_id is None:
        logger.error("Получен пустой ID пользователя (None) в continue_response")
        return
    
    if not hasattr(globals(), 'full_responses') or user_id not in globals()['full_responses']:
        await callback_query.message.edit_text(
            "❌ К сожалению, продолжение ответа недоступно. Попробуйте задать вопрос заново.",
            reply_markup=None
        )
        return
    
    full_response = globals()['full_responses'][user_id]
    if full_response is None:
        logger.error("Получен пустой полный ответ (None) в continue_response")
        await callback_query.message.edit_text(
            "❌ К сожалению, продолжение ответа недоступно. Попробуйте задать вопрос заново.",
            reply_markup=None
        )
        return
    
    current_text = callback_query.message.text
    if current_text is None:
        logger.error("Получен пустой текущий текст (None) в continue_response")
        current_text = ""
    
    if current_text.endswith("..."):
        current_text = current_text[:-3]
    
    next_part_start = len(current_text)
    
    if next_part_start >= len(full_response):
        await callback_query.message.edit_text(
            current_text, 
            reply_markup=None, 
            parse_mode=ParseMode.HTML
        )
        return
    
    max_len = min(4000, len(full_response) - next_part_start)
    next_part = full_response[next_part_start:next_part_start + max_len]
    
    try:
        if next_part_start + max_len < len(full_response):
            continue_keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="🔄 Продолжить ответ", callback_data='continue_response')]
            ])
            
            await callback_query.message.answer(
                next_part + "...", 
                reply_markup=continue_keyboard, 
                parse_mode=ParseMode.HTML
            )
        else:
            await callback_query.message.answer(
                next_part, 
                reply_markup=None, 
                parse_mode=ParseMode.HTML
            )
    except Exception as e:
        logger.error(f"Ошибка при отправке продолжения HTML-ответа: {e}")
        # Если ошибка связана с HTML, отправляем текст без форматирования
        if "can't parse entities" in str(e):
            await callback_query.message.answer(
                "❗ <b>Не удалось отформатировать ответ.</b> Продолжение без форматирования:\n\n" + next_part,
                parse_mode=None
            )
        else:
            raise

@dp.callback_query(lambda c: c.data == 'retry_last_message')
async def retry_last_message(callback_query: types.CallbackQuery):
    await callback_query.answer()
    
    user_id = callback_query.from_user.id
    
    if user_id in user_message_history and user_message_history[user_id]:
        user_messages = [msg for msg in user_message_history[user_id] if msg["role"] == "user"]
        if user_messages:
            last_user_message = user_messages[-1]["content"]
            
            new_message = types.Message(
                message_id=callback_query.message.message_id,
                date=callback_query.message.date,
                chat=callback_query.message.chat,
                from_user=callback_query.from_user,
                text=last_user_message,
                bot=callback_query.bot,
                via_bot=None
            )
            
            if user_message_history[user_id][-1]["role"] == "user":
                user_message_history[user_id].pop()
            elif len(user_message_history[user_id]) >= 2 and user_message_history[user_id][-2]["role"] == "user":
                user_message_history[user_id].pop(-2)
                user_message_history[user_id].pop(-1)
            
            await callback_query.message.answer(f"🔄 <i>Повторяю запрос:</i>\n{last_user_message}", parse_mode=ParseMode.HTML)
            
            await handle_message(new_message)
        else:
            await callback_query.message.answer("❌ Не удалось найти последний запрос. Пожалуйста, отправьте новый запрос.")
    else:
        await callback_query.message.answer("❌ История сообщений пуста. Пожалуйста, отправьте новый запрос.")

@dp.callback_query(lambda c: c.data in ["setting_model", "start_chatting"])
async def onboarding_callback(callback_query: types.CallbackQuery):
    await callback_query.answer()
    
    if callback_query.data == "setting_model":
        await button_callback(callback_query)
    
    elif callback_query.data == "start_chatting":
        await callback_query.message.answer(
            "🚀 <b>Отлично!</b> Я готов отвечать на ваши вопросы!\n\n"
            f"Сейчас я использую модель <code>{user_settings[callback_query.from_user.id]['model'].split('/')[-1]}</code> для ответов.\n\n"
            "Вы можете:\n"
            "• Задавать вопросы текстом\n"
            "• Отправлять голосовые сообщения\n\n"
            "Если модель будет недоступна, я автоматически переключусь на другую подходящую модель.\n\n"
            "Используйте ⚙️ <b>Настройки</b> в меню внизу для изменения модели, температуры генерации и других параметров.",
            parse_mode=ParseMode.HTML
        )

def sanitize_html_for_telegram(html_text):
    """
    Проверяет и очищает HTML для совместимости с Telegram API.
    Telegram поддерживает только ограниченный набор HTML-тегов:
    <b>, <i>, <u>, <s>, <a>, <code>, <pre>
    """
    if html_text is None:
        logger.warning("Получен пустой HTML-текст (None) в sanitize_html_for_telegram")
        return ""
        
    html_text = re.sub(r'<!--.*?-->', '', html_text, flags=re.DOTALL)
    html_text = re.sub(r'<\?xml.*?\?>', '', html_text, flags=re.DOTALL)
    tag_pattern = re.compile(r'</?(\w+)[^>]*>')
    
    allowed_tags = ['b', 'i', 'u', 's', 'a', 'code', 'pre']
    
    matches = list(tag_pattern.finditer(html_text))
    for match in reversed(matches):
        tag = match.group(1).lower()
        if tag not in allowed_tags:
            start, end = match.span()
            html_text = html_text[:start] + html_text[end:]
    
    html_text = re.sub(r'<\?.*?\?>', '', html_text, flags=re.DOTALL)
    html_text = re.sub(r'<(?!/?(?:b|i|u|s|a|code|pre)(?:\s|>))[^>]*>', '', html_text)
    
    html_text = re.sub(r'<pre><code(.*?)>(.*?)</pre></code>', r'<pre><code\1>\2</code></pre>', html_text, flags=re.DOTALL)
    html_text = re.sub(r'<code><pre>(.*?)</code></pre>', r'<pre><code>\1</code></pre>', html_text, flags=re.DOTALL)
    
    for tag in allowed_tags:
        opening_count = len(re.findall(f'<{tag}[^>]*>', html_text))
        closing_count = len(re.findall(f'</{tag}>', html_text))
        
        if opening_count > closing_count:
            html_text += f'</{tag}>' * (opening_count - closing_count)
    
    html_text = re.sub(r'<code class="[^"]*">', '<code>', html_text)
    
    return html_text

def prepare_response_for_telegram(response_text):
    """
    Подготавливает ответ для отправки в Telegram:
    1. Форматирует Markdown в HTML
    2. Очищает HTML для совместимости с Telegram API
    3. Проверяет корректность вложенности тегов
    
    Args:
        response_text: Исходный текст ответа от модели
        
    Returns:
        str: Подготовленный текст для отправки в Telegram
    """
    if response_text is None:
        logger.warning("Получен пустой текст ответа (None) в prepare_response_for_telegram")
        return ""
        
    formatted_text = format_markdown_to_html(response_text)
    
    cleaned_text = sanitize_html_for_telegram(formatted_text)
    
    cleaned_text = re.sub(r'<pre>([^<]*)<code>', r'<pre><code>\1', cleaned_text)
    cleaned_text = re.sub(r'</code>([^<]*)</pre>', r'\1</code></pre>', cleaned_text)
    
    cleaned_text = re.sub(r'(<pre>.*?)<code>(.*?)</pre>', r'\1<code>\2</code></pre>', cleaned_text, flags=re.DOTALL)
    
    return cleaned_text

# Добавляем структуру для хранения статусов моделей
MODEL_STATUSES = {
    "fully_working": [],  # Доступна и даёт ответ
    "partially_working": [],  # Доступна, но без ответа
    "unavailable": []  # Недоступна
}

# Функция для проверки и переключения моделей при проблемах с API
async def check_api_models(check_timeout, min_check_time):
    """
    Проверяет доступность моделей и группирует их по категориям:
    - fully_working: доступна и даёт ответ
    - partially_working: доступна, но может не давать ответы или есть временные ограничения
    - unavailable: недоступна полностью (не существует, отключена и т.д.)
    
    Args:
        check_timeout: Время ожидания ответа от модели в секундах при проверке
        min_check_time: Минимальное время (в секундах), которое будет затрачено на проверку каждой модели
    """
    global user_settings, MODEL_STATUSES
    
    logger.info(f"Проверка доступности моделей (таймаут: {check_timeout} сек, мин.время: {min_check_time} сек)...")
    
    # Сбрасываем статусы моделей
    MODEL_STATUSES = {
        "fully_working": [],
        "partially_working": [],
        "unavailable": []
    }
    
    # Проверяем доступность моделей с помощью тестового запроса
    test_messages = [
        {"role": "system", "content": "Вы помощник."},
        {"role": "user", "content": "Тестовый запрос."}
    ]
    
    response_times = {}
    
    # Проверяем каждую модель
    for model in AVAILABLE_MODELS:
        start_time = time.time()
        result_status = None
        error_message = None
        
        try:
            logger.info(f"Проверка модели {model}...")
            
            model_check_task = asyncio.create_task(
                generate_response(
                    messages=[
                        {"role": "system", "content": "Дай очень короткий ответ."},
                        {"role": "user", "content": "Привет"}
                    ],
                    model=model,
                    max_tokens=20,  # Небольшое количество токенов для быстрого ответа
                    temperature=0.3,  # Низкая температура для стабильности
                    timeout=check_timeout  # Используем настраиваемый таймаут
                )
            )
            
            try:
                response = await asyncio.wait_for(model_check_task, timeout=check_timeout)
                
                if response and response.strip():
                    result_status = "fully_working"
                else:
                    result_status = "partially_working"
                    error_message = "доступна, но без ответа"
            except asyncio.TimeoutError:
                if not model_check_task.done():
                    model_check_task.cancel()
                
                result_status = "partially_working"
                error_message = f"превышено время ожидания ({check_timeout} сек)"
        except Exception as e:
            result_status = "unavailable"
            error_message = str(e)
            error_str = error_message.lower()
            
            # Классифицируем ошибки по типу API
            if model.startswith('huggingface/'):
                # Особая обработка для Hugging Face API
                
                # Временные ошибки Hugging Face
                if "503" in error_str or "service unavailable" in error_str:
                    result_status = "partially_working"
                    error_message = "сервис временно недоступен (503)"
                
                # Ограничения по запросам или квотам
                elif "429" in error_str or "too many requests" in error_str:
                    result_status = "partially_working"
                    error_message = "превышен лимит запросов (429)"
                
                # Проблемы с доступом к модели
                elif "404" in error_str or "not found" in error_str:
                    result_status = "unavailable"
                    error_message = "модель не найдена (404)"
            
            elif model.startswith('together/'):
                # Особая обработка для Together AI API
                
                # Временные ошибки API
                if "rate limit" in error_str:
                    result_status = "partially_working"
                    error_message = "превышен лимит запросов"
                
                # Ошибки доступности модели
                elif "model_not_available" in error_str or "not supported" in error_str:
                    result_status = "unavailable"
                    error_message = "модель недоступна"
            
            else:
                # Общие правила для любого API
                if any(temp_issue in error_str for temp_issue in [
                    "insufficient credits", 
                    "quota exceeded", 
                    "rate limit", 
                    "timeout", 
                    "429",
                    "402",
                    "503",
                    "connection error",
                    "temporary"
                ]):
                    result_status = "partially_working"
        
        elapsed_time = time.time() - start_time
        if elapsed_time < min_check_time:
            wait_time = min_check_time - elapsed_time
            logger.info(f"Ожидание дополнительно {wait_time:.2f} сек для модели {model} (минимальное время проверки)")
            await asyncio.sleep(wait_time)
        
        response_time = time.time() - start_time
        response_times[model] = response_time
        
        # Добавляем модель в соответствующую категорию
        MODEL_STATUSES[result_status].append(model)
        
        if result_status == "fully_working":
            logger.info(f"✅ Модель {model} полностью рабочая: получен ответ за {response_time:.2f} сек")
        elif result_status == "partially_working" and error_message:
            logger.warning(f"⚠️ Модель {model} частично рабочая: {error_message} (время: {response_time:.2f} сек)")
        elif result_status == "unavailable" and error_message:
            logger.error(f"❌ Модель {model} недоступна: {error_message} (время: {response_time:.2f} сек)")
    
    logger.info("Результаты проверки моделей:")
    logger.info(f"✅ Полностью рабочие: {len(MODEL_STATUSES['fully_working'])}")
    for model in MODEL_STATUSES['fully_working']:
        logger.info(f"  - {model}: {response_times.get(model, 'н/д'):.2f} сек")
        
    logger.info(f"⚠️ Частично рабочие: {len(MODEL_STATUSES['partially_working'])}")
    for model in MODEL_STATUSES['partially_working']:
        logger.info(f"  - {model}: {response_times.get(model, 'н/д'):.2f} сек")
        
    logger.info(f"❌ Недоступные: {len(MODEL_STATUSES['unavailable'])}")
    for model in MODEL_STATUSES['unavailable']:
        logger.info(f"  - {model}: {response_times.get(model, 'н/д'):.2f} сек")

async def main():
    # Загружаем настройки пользователей при запуске
    global user_settings, user_message_history
    
    logger.info("Загружаем сохраненные настройки пользователей...")
    user_settings = load_user_settings()
    logger.info(f"Загружены настройки для {len(user_settings)} пользователей")
    
    # Инициализируем Vosk модель для локального распознавания
    logger.info("Попытка инициализации Vosk модели для локального распознавания...")
    if init_vosk_model():
        logger.info("✅ Локальное распознавание голосовых сообщений активировано (безлимитное)")
    else:
        if os.path.exists(VOSK_MODEL_PATH) and os.path.isdir(VOSK_MODEL_PATH):
            logger.warning(f"Директория модели {VOSK_MODEL_PATH} существует, но инициализация не удалась. Повторная попытка...")
            time.sleep(1)
            if init_vosk_model():
                logger.info("✅ Локальное распознавание голосовых сообщений активировано при повторной попытке")
            else:
                logger.warning("⚠️ Локальное распознавание не доступно даже при повторной попытке, будет использоваться Google API (с ограничениями)")
        else:
            logger.warning("⚠️ Локальное распознавание не доступно, будет использоваться Google API (с ограничениями)")
    
    await check_api_models(check_timeout=API_CHECK_TIMEOUT, min_check_time=API_MIN_CHECK_TIME)
    
    logger.info("Загружаем историю сообщений пользователей...")
    user_message_history = load_user_history()
    logger.info(f"Загружена история для {len(user_message_history)} пользователей")
    
    await bot.set_my_commands([
        types.BotCommand(command="start", description="Начать диалог заново"),
        types.BotCommand(command="help", description="Показать справку"),
        types.BotCommand(command="clear", description="Очистить историю"),
        types.BotCommand(command="settings", description="Настройки бота"),
        types.BotCommand(command="info", description="Показать текущие настройки"),
        types.BotCommand(command="menu", description="Показать меню с кнопками")
    ])
    
    logger.info("Проверка регистрации обработчиков сообщений...")
    
    num_handlers = len(dp.message.handlers)
    logger.info(f"Зарегистрировано обработчиков сообщений: {num_handlers}")
    
    voice_handler_registered = False
    for handler in dp.message.handlers:
        if "voice" in str(handler) or "voice is not None" in str(handler):
            voice_handler_registered = True
            break
    
    if voice_handler_registered:
        logger.info("✅ Обнаружен обработчик голосовых сообщений")
    else:
        logger.warning("⚠️ Обработчик голосовых сообщений НЕ обнаружен!")
    
    asyncio.create_task(periodic_save())
    
    logger.info("Запуск бота...")
    
    await dp.start_polling(bot)

async def periodic_save():
    """Периодически сохраняет настройки и историю сообщений пользователей"""
    while True:
        # Ждем 5 минут
        await asyncio.sleep(5 * 60)
        
        logger.info("Выполняем автоматическое сохранение данных...")
        save_user_settings(user_settings)
        save_user_history(user_message_history)
        logger.info("Автоматическое сохранение выполнено успешно.")

# Функция для получения приоритетного списка запасных моделей
def get_fallback_models(current_model):
    """
    Создает приоритетный список запасных моделей на основе:
    1. Текущего провайдера API (предпочтение отдается моделям того же провайдера)
    2. Статуса работоспособности моделей
    
    Args:
        current_model: Текущая модель, для которой нужны запасные варианты
        
    Returns:
        list: Отсортированный список запасных моделей в порядке приоритета
    """
    current_provider = None
    if current_model.startswith('huggingface/'):
        current_provider = 'huggingface'
    elif current_model.startswith('together/'):
        current_provider = 'together'
    else:
        current_provider = 'openrouter'
    
    fallback_models = []
    
    # 1. Полностью рабочие модели того же провайдера
    for m in MODEL_STATUSES["fully_working"]:
        if m != current_model and ((current_provider == 'huggingface' and m.startswith('huggingface/')) or 
                               (current_provider == 'together' and m.startswith('together/')) or
                               (current_provider == 'openrouter' and not (m.startswith('huggingface/') or m.startswith('together/')))):
            fallback_models.append(m)
    
    # 2. Полностью рабочие модели других провайдеров
    for m in MODEL_STATUSES["fully_working"]:
        if m != current_model and m not in fallback_models:
            fallback_models.append(m)
    
    # 3. Частично рабочие модели того же провайдера (только если полностью рабочих нет)
    if not fallback_models:
        for m in MODEL_STATUSES["partially_working"]:
            if m != current_model and ((current_provider == 'huggingface' and m.startswith('huggingface/')) or 
                                  (current_provider == 'together' and m.startswith('together/')) or
                                  (current_provider == 'openrouter' and not (m.startswith('huggingface/') or m.startswith('together/')))):
                fallback_models.append(m)
    
    # 4. Частично рабочие модели других провайдеров (только если нет других вариантов)
    if not fallback_models:
        for m in MODEL_STATUSES["partially_working"]:
            if m != current_model and m not in fallback_models:
                fallback_models.append(m)
    
    reliable_models = ["google/gemini-2.0-pro-exp-02-05:free", "together/mistral-7b-instruct"]
    for reliable_model in reliable_models:
        if reliable_model not in fallback_models and reliable_model in AVAILABLE_MODELS:
            fallback_models.append(reliable_model)
    
    return fallback_models

# Функция для обновления статуса модели
def update_model_status(model, new_status=None, error_message=None):
    """
    Обновляет статус модели в MODEL_STATUSES на основе текущего опыта использования.
    
    Args:
        model: Название модели
        new_status: Новый статус ('fully_working', 'partially_working', 'unavailable')
        error_message: Сообщение об ошибке для логирования
    """
    global MODEL_STATUSES
    
    if model is None:
        logger.error("Получено пустое название модели (None) в update_model_status")
        return
    
    if new_status is None and error_message is not None:
        if "таймаут" in error_message.lower() or "timeout" in error_message.lower():
            new_status = "partially_working"  # Таймауты обычно означают временные проблемы
        elif "503" in error_message or "недоступен" in error_message.lower():
            new_status = "partially_working"  # Временная недоступность
        elif "404" in error_message or "не найден" in error_message.lower():
            new_status = "unavailable"  # Модель не существует или удалена
        else:
            new_status = "partially_working"  # По умолчанию считаем частично рабочей
    
    if new_status is None:
        return
    
    for status in MODEL_STATUSES:
        if model in MODEL_STATUSES[status]:
            MODEL_STATUSES[status].remove(model)
    
    MODEL_STATUSES[new_status].append(model)
    
    if error_message:
        logger.warning(f"Статус модели {model} изменен на {new_status} из-за ошибки: {error_message}")
    else:
        logger.info(f"Статус модели {model} изменен на {new_status}")

@dp.callback_query(lambda c: c.data == 'back_to_settings')
async def back_to_settings_menu(callback_query: types.CallbackQuery):
    await callback_query.answer()
    
    settings_text = (
        "⚙️ <b>Настройки бота</b>\n\n"
        "Здесь вы можете настроить параметры работы бота под свои предпочтения. "
        "Выберите, что именно хотите изменить:"
    )
    
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🤖 Изменить модель", callback_data='setting_model')],
        [InlineKeyboardButton(text="📏 Подробность ответа", callback_data='setting_max_tokens')],
        [InlineKeyboardButton(text="🎨 Уровень креативности", callback_data='setting_temperature')],
        [InlineKeyboardButton(text="📜 Длина истории", callback_data='setting_history_length')],
        [InlineKeyboardButton(text="✍️ Системное сообщение", callback_data='setting_system_message')],
        [InlineKeyboardButton(text="💬 Режим динамического чата", callback_data='setting_dynamic_chat')],
        [InlineKeyboardButton(text="🔙 Вернуться в главное меню", callback_data='back_to_main')]
    ])
    
    await callback_query.message.edit_text(settings_text, reply_markup=keyboard, parse_mode=ParseMode.HTML)

if __name__ == "__main__":
    asyncio.run(main())