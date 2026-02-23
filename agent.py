"""
AI Agent на базе LangChain.

Агент для управления заметками в Markdown формате.
"""

import os
import sys
import logging
from typing import Type
from datetime import datetime
from pathlib import Path

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
