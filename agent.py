"""
AI Agent на базе LangChain.

Агент для управления заметками в Markdown формате.
"""

import os
import sys
import logging
from typing import Type, Optional
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv


# === Настройка логирования ===
# Создаём логгер
logger = logging.getLogger("agent")
logger.setLevel(logging.DEBUG)

# Формат логов
log_formatter = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Хендлер для консоли
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(log_formatter)

# Хендлер для файла
file_handler = logging.FileHandler("agent.log", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(log_formatter)

# Добавляем хендлеры к логгеру
logger.addHandler(console_handler)
logger.addHandler(file_handler)


# === Настройка кодировки для Windows ===
def clean_text(text):
    """Очистка текста от surrogate characters."""
    if text is None:
        return ""
    # Удаляем surrogate pairs
    return text.encode('utf-8', 'ignore').decode('utf-8')


# Настраиваем stdout/stdin для корректной работы с UTF-8
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from langchain.agents import create_agent as create_langchain_agent
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI  # DeepSeek API совместим с OpenAI API
from pydantic import BaseModel, Field

# Загрузка переменных окружения
load_dotenv()

# Импорт векторного хранилища
try:
    from vector_store import get_vector_store, VectorStore
    VECTOR_SEARCH_AVAILABLE = True
except ImportError:
    VECTOR_SEARCH_AVAILABLE = False
    logger.warning("Модуль vector_store не найден. Семантический поиск недоступен.")

# Импорт Yandex Vision
try:
    from yandex_vision import get_vision_client, ImageAnalysisResult
    YANDEX_VISION_AVAILABLE = True
except ImportError:
    YANDEX_VISION_AVAILABLE = False
    logger.warning("Модуль yandex_vision не найден. Обработка изображений недоступна.")


# === Определение инструментов (Tools) ===

class SaveNoteInput(BaseModel):
    """Входные параметры для сохранения заметки."""
    content: str = Field(description="Текст заметки для сохранения в формате Markdown")


class SaveNoteTool(BaseTool):
    """Инструмент для сохранения заметок в Markdown файлы."""
    
    name: str = "save_note"
    description: str = "Сохраняет текст как Markdown файл в директорию notes. Имя файла - дата и время создания."
    args_schema: Type[BaseModel] = SaveNoteInput
    
    def _run(self, content: str) -> str:
        """Сохранить заметку в Markdown файл."""
        logger.info(f"🔧 TOOL CALL: save_note | filename: auto-generated")
        logger.debug(f"save_note | content preview: {content[:100]}...")
        
        # Очищаем текст от surrogate characters
        content = clean_text(content)
        
        # Создаем директорию notes если её нет
        notes_dir = Path("notes")
        notes_dir.mkdir(exist_ok=True)
        
        # Формируем имя файла: YYYY-MM-DD_HH-MM-SS.md
        filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".md"
        filepath = notes_dir / filename
        
        # Записываем содержимое в файл
        try:
            filepath.write_text(content, encoding="utf-8")
            
            # Индексируем в векторное хранилище (если доступно)
            if VECTOR_SEARCH_AVAILABLE:
                try:
                    vector_store = get_vector_store()
                    if vector_store.enabled:
                        note_id = filepath.stem  # Имя файла без расширения
                        vector_store.add_note(
                            note_id=note_id,
                            content=content,
                            filename=filename,
                            preview=content[:200]
                        )
                except Exception as e:
                    logger.warning(f"Не удалось проиндексировать заметку в векторное хранилище: {e}")
            
            result = f"✅ Заметка сохранена: {filepath}"
            logger.info(f"🔧 TOOL RESULT: save_note | {result}")
            return result
        except Exception as e:
            result = f"❌ Ошибка сохранения: {str(e)}"
            logger.error(f"🔧 TOOL RESULT: save_note | {result}")
            return result


class SearchNotesInput(BaseModel):
    """Входные параметры для поиска в заметках."""
    query: str = Field(description="Текст для поиска в файлах заметок")


class SearchNotesTool(BaseTool):
    """Инструмент для поиска текста в файлах директории notes."""
    
    name: str = "search_notes"
    description: str = "Ищет текст в файлах директории notes. Возвращает найденные совпадения с именами файлов."
    args_schema: Type[BaseModel] = SearchNotesInput
    
    def _run(self, query: str) -> str:
        """Поиск текста в файлах notes."""
        logger.info(f"🔧 TOOL CALL: search_notes | query: '{query}'")
        
        # Очищаем запрос от surrogate characters
        query = clean_text(query)
        notes_dir = Path("notes")
        
        if not notes_dir.exists():
            return "❌ Директория notes не существует. Сначала сохраните заметку."
        
        query_lower = query.lower()
        results = []
        
        try:
            for md_file in notes_dir.glob("*.md"):
                try:
                    content = clean_text(md_file.read_text(encoding="utf-8"))
                    if query_lower in content.lower():
                        # Находим контекст вокруг совпадения
                        lines = content.split("\n")
                        matching_lines = []
                        for i, line in enumerate(lines, 1):
                            if query_lower in line.lower():
                                matching_lines.append(f"  Строка {i}: {line.strip()}")
                        
                        if matching_lines:
                            results.append(f"📄 {md_file.name}:\n" + "\n".join(matching_lines))
                except Exception as e:
                    results.append(f"⚠️ {md_file.name}: ошибка чтения ({str(e)})")
            
            if not results:
                result_msg = f"🔍 По запросу '{query}' ничего не найдено."
                logger.info(f"🔧 TOOL RESULT: search_notes | {result_msg}")
                return result_msg
            
            result_msg = f"🔍 Результаты поиска по '{query}':\n\n" + "\n\n".join(results)
            logger.info(f"🔧 TOOL RESULT: search_notes | found {len(results)} file(s)")
            logger.debug(f"search_notes | full result: {result_msg[:200]}...")
            return result_msg
            
        except Exception as e:
            result = f"❌ Ошибка поиска: {str(e)}"
            logger.error(f"🔧 TOOL RESULT: search_notes | {result}")
            return result


class GetNoteInput(BaseModel):
    """Входные параметры для получения содержимого заметки."""
    filename: str = Field(description="Имя файла заметки (например: 2026-02-13_21-54-01.md)")


class GetNoteTool(BaseTool):
    """Инструмент для получения содержимого заметки по имени файла."""
    
    name: str = "get_note"
    description: str = "Возвращает полное содержимое заметки по имени файла. Файл должен быть в директории notes."
    args_schema: Type[BaseModel] = GetNoteInput
    
    def _run(self, filename: str) -> str:
        """Получить содержимое заметки по имени файла."""
        logger.info(f"🔧 TOOL CALL: get_note | filename: '{filename}'")
        
        # Очищаем имя файла
        filename = clean_text(filename).strip()
        
        notes_dir = Path("notes")
        filepath = notes_dir / filename
        
        # Проверяем безопасность пути (чтобы не выйти за пределы notes)
        try:
            filepath.resolve().relative_to(notes_dir.resolve())
        except ValueError:
            result = "❌ Некорректное имя файла."
            logger.error(f"🔧 TOOL RESULT: get_note | {result}")
            return result
        
        if not filepath.exists():
            result = f"❌ Файл '{filename}' не найден в директории notes."
            logger.warning(f"🔧 TOOL RESULT: get_note | {result}")
            return result
        
        if not filepath.is_file():
            result = f"❌ '{filename}' не является файлом."
            logger.warning(f"🔧 TOOL RESULT: get_note | {result}")
            return result
        
        try:
            content = clean_text(filepath.read_text(encoding="utf-8"))
            result = f"📄 **{filename}**\n\n{content}"
            logger.info(f"🔧 TOOL RESULT: get_note | success, content length: {len(content)} chars")
            logger.debug(f"get_note | full content: {content[:200]}...")
            return result
        except Exception as e:
            result = f"❌ Ошибка чтения файла: {str(e)}"
            logger.error(f"🔧 TOOL RESULT: get_note | {result}")
            return result


class SemanticSearchInput(BaseModel):
    """Входные параметры для семантического поиска."""
    query: str = Field(description="Запрос для семантического поиска (смысловой поиск по заметкам)")
    top_k: int = Field(default=5, description="Количество результатов (по умолчанию 5)")


class SemanticSearchTool(BaseTool):
    """Инструмент для семантического (векторного) поиска по заметкам."""
    
    name: str = "semantic_search"
    description: str = "Семантический поиск по заметкам. Ищет заметки по смыслу, даже если используются другие слова. Требует настройки GIGACHAT_CLIENT_CREDENTIALS."
    args_schema: Type[BaseModel] = SemanticSearchInput
    
    def _run(self, query: str, top_k: int = 5) -> str:
        """Выполнить семантический поиск."""
        logger.info(f"🔧 TOOL CALL: semantic_search | query: '{query}', top_k: {top_k}")
        
        if not VECTOR_SEARCH_AVAILABLE:
            result = "❌ Семантический поиск недоступен. Модуль vector_store не установлен."
            logger.error(f"🔧 TOOL RESULT: semantic_search | {result}")
            return result
        
        try:
            vector_store = get_vector_store()
            
            if not vector_store.enabled:
                result = "❌ Семантический поиск не настроен. Добавьте GIGACHAT_CLIENT_CREDENTIALS в файл .env"
                logger.error(f"🔧 TOOL RESULT: semantic_search | {result}")
                return result
            
            # Очищаем запрос
            query = clean_text(query)
            
            # Выполняем поиск
            results = vector_store.search(query, top_k=top_k)
            
            if not results:
                result_msg = f"🔍 По запросу '{query}' ничего не найдено семантически."
                logger.info(f"🔧 TOOL RESULT: semantic_search | {result_msg}")
                return result_msg
            
            # Форматируем результаты
            formatted = [f"🔍 Семантический поиск: '{query}'\n"]
            for i, r in enumerate(results, 1):
                similarity_pct = int(r['similarity'] * 100)
                formatted.append(
                    f"{i}. 📄 {r['filename']} (сходство: {similarity_pct}%)\n"
                    f"   📝 {r['preview'][:150]}..."
                )
            
            result_msg = "\n\n".join(formatted)
            logger.info(f"🔧 TOOL RESULT: semantic_search | found {len(results)} result(s)")
            return result_msg
            
        except Exception as e:
            result = f"❌ Ошибка семантического поиска: {str(e)}"
            logger.error(f"🔧 TOOL RESULT: semantic_search | {result}")
            return result


class UpdateNoteInput(BaseModel):
    """Входные параметры для обновления заметки."""
    filename: str = Field(description="Имя файла заметки для обновления (например: 2026-02-13_21-54-01.md)")
    content: str = Field(description="Новое содержимое заметки в формате Markdown")


class UpdateNoteTool(BaseTool):
    """Инструмент для обновления существующей заметки."""
    
    name: str = "update_note"
    description: str = "Обновляет содержимое существующей заметки по имени файла. Используй для изменения или исправления заметок."
    args_schema: Type[BaseModel] = UpdateNoteInput
    
    def _run(self, filename: str, content: str) -> str:
        """Обновить заметку по имени файла."""
        logger.info(f"🔧 TOOL CALL: update_note | filename: '{filename}'")
        logger.debug(f"update_note | content preview: {content[:100]}...")
        
        # Очищаем текст
        filename = clean_text(filename).strip()
        content = clean_text(content)
        
        notes_dir = Path("notes")
        filepath = notes_dir / filename
        
        # Проверяем безопасность пути
        try:
            filepath.resolve().relative_to(notes_dir.resolve())
        except ValueError:
            result = "❌ Некорректное имя файла."
            logger.error(f"🔧 TOOL RESULT: update_note | {result}")
            return result
        
        if not filepath.exists():
            result = f"❌ Файл '{filename}' не найден. Сначала создайте заметку."
            logger.warning(f"🔧 TOOL RESULT: update_note | {result}")
            return result
        
        try:
            # Записываем новое содержимое
            filepath.write_text(content, encoding="utf-8")
            
            # Обновляем в векторном хранилище
            if VECTOR_SEARCH_AVAILABLE:
                try:
                    vector_store = get_vector_store()
                    if vector_store.enabled:
                        note_id = filepath.stem
                        vector_store.add_note(
                            note_id=note_id,
                            content=content,
                            filename=filename,
                            preview=content[:200]
                        )
                except Exception as e:
                    logger.warning(f"Не удалось обновить заметку в векторном хранилище: {e}")
            
            result = f"✅ Заметка обновлена: {filepath}"
            logger.info(f"🔧 TOOL RESULT: update_note | {result}")
            return result
            
        except Exception as e:
            result = f"❌ Ошибка обновления: {str(e)}"
            logger.error(f"🔧 TOOL RESULT: update_note | {result}")
            return result


class DeleteNoteInput(BaseModel):
    """Входные параметры для удаления заметки."""
    filename: str = Field(description="Имя файла заметки для удаления (например: 2026-02-13_21-54-01.md)")


class DeleteNoteTool(BaseTool):
    """Инструмент для удаления заметки."""
    
    name: str = "delete_note"
    description: str = "Удаляет заметку по имени файла. Операция необратима."
    args_schema: Type[BaseModel] = DeleteNoteInput
    
    def _run(self, filename: str) -> str:
        """Удалить заметку по имени файла."""
        logger.info(f"🔧 TOOL CALL: delete_note | filename: '{filename}'")
        
        # Очищаем имя файла
        filename = clean_text(filename).strip()
        
        notes_dir = Path("notes")
        filepath = notes_dir / filename
        
        # Проверяем безопасность пути
        try:
            filepath.resolve().relative_to(notes_dir.resolve())
        except ValueError:
            result = "❌ Некорректное имя файла."
            logger.error(f"🔧 TOOL RESULT: delete_note | {result}")
            return result
        
        if not filepath.exists():
            result = f"❌ Файл '{filename}' не найден."
            logger.warning(f"🔧 TOOL RESULT: delete_note | {result}")
            return result
        
        try:
            # Удаляем файл
            filepath.unlink()
            
            # Удаляем из векторного хранилища
            if VECTOR_SEARCH_AVAILABLE:
                try:
                    vector_store = get_vector_store()
                    if vector_store.enabled:
                        note_id = filepath.stem
                        vector_store.delete_note(note_id)
                except Exception as e:
                    logger.warning(f"Не удалось удалить заметку из векторного хранилища: {e}")
            
            result = f"🗑️ Заметка удалена: {filename}"
            logger.info(f"🔧 TOOL RESULT: delete_note | {result}")
            return result
            
        except Exception as e:
            result = f"❌ Ошибка удаления: {str(e)}"
            logger.error(f"🔧 TOOL RESULT: delete_note | {result}")
            return result


class WebSearchInput(BaseModel):
    """Входные параметры для web поиска."""
    query: str = Field(description="Поисковый запрос")
    max_results: int = Field(default=5, description="Количество результатов (1-10)")


class WebSearchTool(BaseTool):
    """Инструмент для поиска в интернете через Brave Search API."""
    
    name: str = "web_search"
    description: str = "Поиск актуальной информации в интернете через Brave Search. Используй для получения свежих данных, новостей, фактов, которые могли измениться. Требует BRAVE_API_KEY в .env"
    args_schema: Type[BaseModel] = WebSearchInput
    
    def _run(self, query: str, max_results: int = 5) -> str:
        """Выполнить поиск через Brave Search API."""
        logger.info(f"🔧 TOOL CALL: web_search | query: '{query}', max_results: {max_results}")
        
        # Получаем API ключ
        api_key = os.getenv("BRAVE_API_KEY")
        if not api_key or api_key == "your_brave_api_key_here":
            result = "❌ Brave Search API ключ не настроен. Добавьте BRAVE_API_KEY в файл .env"
            logger.error(f"🔧 TOOL RESULT: web_search | {result}")
            return result
        
        # Ограничиваем количество результатов
        max_results = max(1, min(10, max_results))
        
        # Очищаем запрос
        query = clean_text(query).strip()
        if not query:
            result = "❌ Пустой поисковый запрос"
            logger.error(f"🔧 TOOL RESULT: web_search | {result}")
            return result
        
        try:
            # Выполняем запрос к Brave API
            headers = {
                "X-Subscription-Token": api_key,
                "Accept": "application/json",
            }
            params = {
                "q": query,
                "count": max_results,
                "text_decorations": False,
                "search_lang": "ru",
            }
            
            response = requests.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers=headers,
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            # Извлекаем результаты
            web_results = data.get("web", {}).get("results", [])
            
            if not web_results:
                result_msg = f"🔍 По запросу '{query}' ничего не найдено."
                logger.info(f"🔧 TOOL RESULT: web_search | {result_msg}")
                return result_msg
            
            # Форматируем результаты
            formatted = [f"🔍 Результаты поиска: '{query}'\n"]
            
            for i, r in enumerate(web_results[:max_results], 1):
                title = r.get("title", "Без названия")
                url = r.get("url", "")
                description = r.get("description", "")
                
                # Дополнительные сниппеты если есть
                extra_snippets = r.get("extra_snippets", [])
                if extra_snippets:
                    description += " " + " ".join(extra_snippets[:2])
                
                # Обрезаем описание
                if len(description) > 300:
                    description = description[:300] + "..."
                
                formatted.append(
                    f"{i}. **{title}**\n"
                    f"   🌐 {url}\n"
                    f"   📝 {description}\n"
                )
            
            result_msg = "\n".join(formatted)
            logger.info(f"🔧 TOOL RESULT: web_search | found {len(web_results)} result(s)")
            return result_msg
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                result = "❌ Ошибка авторизации Brave API. Проверьте BRAVE_API_KEY."
            elif e.response.status_code == 429:
                result = "❌ Превышен лимит запросов Brave API. Попробуйте позже."
            else:
                result = f"❌ Ошибка Brave API: {e.response.status_code}"
            logger.error(f"🔧 TOOL RESULT: web_search | {result}")
            return result
            
        except requests.exceptions.RequestException as e:
            result = f"❌ Ошибка сети при поиске: {str(e)}"
            logger.error(f"🔧 TOOL RESULT: web_search | {result}")
            return result
            
        except Exception as e:
            result = f"❌ Ошибка при выполнении поиска: {str(e)}"
            logger.error(f"🔧 TOOL RESULT: web_search | {result}")
            return result


class AnalyzeImageInput(BaseModel):
    """Входные параметры для анализа изображения."""
    image_path: str = Field(description="Путь к файлу изображения для анализа")


class AnalyzeImageTool(BaseTool):
    """Инструмент для анализа изображений с помощью Yandex Vision."""
    
    name: str = "analyze_image"
    description: str = "Анализирует изображение с помощью Yandex Vision. Распознаёт текст (OCR) и описывает содержимое. Требует YANDEX_VISION_FOLDER_ID и авторизацию в .env"
    args_schema: Type[BaseModel] = AnalyzeImageInput
    
    def _run(self, image_path: str) -> str:
        """Проанализировать изображение."""
        logger.info(f"🔧 TOOL CALL: analyze_image | image_path: '{image_path}'")
        
        if not YANDEX_VISION_AVAILABLE:
            result = "❌ Анализ изображений недоступен. Модуль yandex_vision не установлен."
            logger.error(f"🔧 TOOL RESULT: analyze_image | {result}")
            return result
        
        # Проверяем существование файла
        from pathlib import Path
        path = Path(image_path)
        if not path.exists():
            result = f"❌ Файл не найден: {image_path}"
            logger.error(f"🔧 TOOL RESULT: analyze_image | {result}")
            return result
        
        try:
            client = get_vision_client()
            
            if not client.enabled:
                result = "❌ Yandex Vision не настроен. Добавьте YANDEX_VISION_FOLDER_ID и YANDEX_VISION_IAM_TOKEN (или YANDEX_VISION_API_KEY) в файл .env"
                logger.error(f"🔧 TOOL RESULT: analyze_image | {result}")
                return result
            
            # Читаем файл
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            
            # Анализируем изображение
            analysis = client.analyze_image(image_bytes)
            
            if analysis is None:
                result = "❌ Не удалось проанализировать изображение. Проверьте логи."
                logger.error(f"🔧 TOOL RESULT: analyze_image | {result}")
                return result
            
            # Формируем результат
            result_lines = ["📷 **Результат анализа изображения**\n"]
            
            if analysis.text:
                result_lines.append(f"**Распознанный текст:**\n```\n{analysis.text}\n```")
            else:
                result_lines.append("*Текст на изображении не обнаружен.*")
            
            result_msg = "\n\n".join(result_lines)
            logger.info(f"🔧 TOOL RESULT: analyze_image | success, text length: {len(analysis.text) if analysis.text else 0} chars")
            return result_msg
            
        except Exception as e:
            result = f"❌ Ошибка анализа изображения: {str(e)}"
            logger.error(f"🔧 TOOL RESULT: analyze_image | {result}")
            return result


# === Создание агента ===

def create_agent():
    """Создать и настроить AI агента."""
    
    # Проверка наличия API ключа
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key or api_key == "your_deepseek_api_key_here":
        raise ValueError(
            "DEEPSEEK_API_KEY не настроен!\n"
            "1. Скопируйте .env.example в .env: cp .env.example .env\n"
            "2. Добавьте ваш API ключ в файл .env"
        )
    
    # Инициализация языковой модели (DeepSeek)
    model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.7,
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )
    
    # Список инструментов
    tools = [
        SaveNoteTool(),
        SearchNotesTool(),
        GetNoteTool(),
        SemanticSearchTool(),
        UpdateNoteTool(),
        DeleteNoteTool(),
        WebSearchTool(),
        AnalyzeImageTool(),
    ]
    
    # Создание агента с использованием LangGraph
    system_prompt = """Ты — интеллектуальный ассистент, который общается с пользователем и ведёт личную память (заметки).

Твоя задача — обрабатывать каждое входное сообщение пользователя и определить, является ли оно заметкой для сохранения или обычным сообщением для ответа.

🔹 Основные правила

Всегда анализируй вход пользователя.

Если сообщение является заметкой, то:
- НЕ веди полноценный диалог по её содержанию,
- СОХРАНИ её в память,
- добавь подходящие тэги,
- сохрани в формате Markdown,
- подтверди сохранение короткой фразой.

Если сообщение НЕ является заметкой:
- отвечай пользователю обычным образом,
- не сохраняй сообщение в память.

🔹 Как определять, что сообщение — заметка

Считай сообщение заметкой, если оно:
- начинается со слов типа: "запомни", "заметка", "важно", "напомни", "мне нужно", "надо сделать", "идея", "план"
- содержит важные данные пользователя:
  - цели, планы, предпочтения, факты о себе, расписание, проекты
- выглядит как список дел или пунктов
- явно предназначено для сохранения на будущее

НЕ считай заметкой:
- вопросы
- обсуждения
- просьбы что-то объяснить
- обычные разговорные реплики

Если не уверен (заметка или вопрос), выбери режим chat и уточни.

🔹 Определение тэгов (обязательно)

Если сообщение — заметка, ты должен автоматически определить от 2 до 6 релевантных тэгов.

Правила для тэгов:
- Тэги должны быть короткими (1–2 слова).
- На русском языке.
- В формате #тэг
- Без пробелов внутри (используй _ если нужно).
- Тэги должны отражать смысл заметки.

Примеры тэгов:
#работа #идеи #проект #личное #задачи #финансы #здоровье #учёба #покупки #встречи #планы #книги #напоминание

🔹 Формат сохранения заметки (Markdown)

Если сообщение является заметкой — сформируй Markdown-запись по шаблону:

## Заметка
**Тэги:** #тэг1 #тэг2 #тэг3
**Дата создания:** <Дата создания заметки>

<краткий текст заметки>

Если заметка содержит список задач, оформи их списком:

## Заметка
**Тэги:** #задачи #планы

- пункт 1
- пункт 2
- пункт 3

Дополнительные требования
- Не выдумывай информацию.
- Не добавляй лишние детали в заметку.
- Не сохраняй конфиденциальные данные, если пользователь не просил явно.
- Сохраняй заметки кратко, но понятно.
- Если пользователь прислал несколько отдельных фактов — объедини в одну заметку.

У тебя есть доступ к следующим инструментам:
1. save_note - сохранить текст как Markdown файл в директорию notes
2. search_notes - поиск текста в сохранённых заметках (точное совпадение слов)
3. get_note - получить полное содержимое заметки по имени файла
4. semantic_search - семантический (векторный) поиск по смыслу
5. update_note - обновить содержимое существующей заметки по имени файла
6. delete_note - удалить заметку по имени файла (операция необратима)
7. web_search - поиск актуальной информации в интернете через Brave Search
8. analyze_image - анализ изображения с помощью Yandex Vision (OCR и описание)

🔹 Когда использовать web_search:
- Когда нужна актуальная информация из интернета (новости, погода, курсы валют, события)
- Когда пользователь спрашивает о чём-то, что произошло недавно
- Когда нужно проверить факты или получить свежие данные
- НЕ используй для поиска в своих заметках — для этого есть search_notes и semantic_search

🔹 Когда использовать semantic_search:
- Когда пользователь ищет что-то по описанию, а не по точным словам
- Примеры запросов: "что я планировал купить", "мои идеи для проекта", "планы на отпуск"
- Этот поиск понимает смысл и находит релевантные заметки даже с другими словами

🔹 Когда использовать search_notes (обычный поиск):
- Когда нужно найти точное слово или фразу
- Когда нужны конкретные совпадения в тексте

Если готов — начинай работу сразу с первого сообщения пользователя.
"""
    
    agent_executor = create_langchain_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )
    
    return agent_executor


# === Интерактивный режим ===

def main():
    """Запустить интерактивный режим работы с агентом."""
    print("=" * 50)
    print("🤖 AI Agent для работы с заметками")
    print("=" * 50)
    print()
    
    logger.info("=== Запуск AI Agent ===")
    
    try:
        agent = create_agent()
    except ValueError as e:
        print(f"❌ Ошибка: {e}")
        return
    
    print("✅ Агент успешно инициализирован!")
    print("Доступные команды:")
    print("  • 'save' или 'сохрани' - сохранить заметку в Markdown")
    print("  • 'find' или 'найди' - поиск в сохранённых заметках")
    print("  • 'exit' или 'выход' - завершить работу")
    print()
    
    chat_history = []
    
    while True:
        try:
            user_input = clean_text(input("👤 Вы: ")).strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ("exit", "выход", "quit", "q"):
                print("\n👋 До свидания!")
                logger.info("=== Сессия завершена пользователем ===")
                break
            
            logger.info(f"👤 USER INPUT: {user_input[:100]}...")
            
            # Выполнение запроса
            from langchain_core.messages import HumanMessage, AIMessage
            
            # Формируем сообщения с историей
            messages = list(chat_history)
            messages.append(HumanMessage(content=user_input))
            
            result = agent.invoke({
                "messages": messages,
            })
            
            # Получаем ответ (последнее сообщение от AI)
            output = result["messages"][-1].content
            
            logger.info(f"🤖 AGENT OUTPUT: {output[:100]}...")
            
            print(f"\n🤖 Агент: {clean_text(output)}\n")
            
            # Сохранение истории чата
            chat_history.append(HumanMessage(content=clean_text(user_input)))
            chat_history.append(AIMessage(content=clean_text(output)))
            
            # Ограничение истории (последние 10 сообщений)
            chat_history = chat_history[-10:]
            
        except KeyboardInterrupt:
            print("\n\n👋 До свидания!")
            break
        except Exception as e:
            error_msg = clean_text(str(e))
            print(f"\n❌ Ошибка: {error_msg}\n")


if __name__ == "__main__":
    main()
