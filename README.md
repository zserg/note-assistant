# AI Agent на базе LangChain 🤖

Пример проекта AI агента, построенного с использованием фреймворка [LangChain](https://www.langchain.com/).

## Структура проекта

```
.
├── venv/                 # Виртуальное окружение
├── .env                  # Переменные окружения (создайте из .env.example)
├── .env.example          # Пример конфигурации
├── requirements.txt      # Зависимости проекта
├── agent.py              # Полнофункциональный агент с инструментами
├── simple_agent.py       # Простой пример работы с LangChain
└── README.md             # Этот файл
```

## Установка

### 1. Активация виртуального окружения

```bash
# Активация
source venv/bin/activate

# Деактивация (когда нужно выйти)
deactivate
```

### 2. Настройка API ключа

```bash
# Скопируйте пример конфигурации
cp .env.example .env

# Отредактируйте .env и добавьте ваш OpenAI API ключ
# Получить ключ: https://platform.openai.com/api-keys
```

## Запуск

### Простой агент (без инструментов)

```bash
source venv/bin/activate && python simple_agent.py
```

### Агент с инструментами

```bash
source venv/bin/activate && python agent.py
```

## Возможности агента

### Доступные инструменты

1. **current_time** — получение текущего времени и даты
2. **calculator** — математические вычисления
3. **search** — поиск информации (заглушка, требует интеграции с реальным API)

### Примеры запросов

- "Который сейчас час?"
- "Посчитай 15 * 23 + 7"
- "Найди информацию о Python"
- "Какая погода сегодня?" (агент ответит, что не имеет доступа к погоде)

## Добавление новых инструментов

Чтобы добавить новый инструмент, создайте класс, наследующий `BaseTool`:

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class MyToolInput(BaseModel):
    param: str = Field(description="Описание параметра")

class MyTool(BaseTool):
    name: str = "my_tool"
    description: str = "Описание инструмента"
    args_schema: Type[BaseModel] = MyToolInput
    
    def _run(self, param: str) -> str:
        return f"Результат: {param}"
```

Затем добавьте инструмент в список `tools` в функции `create_agent()`.

## Расширение проекта

### Интеграция с реальным поиском

Замените `MockSearchTool` на интеграцию с:
- [SerpAPI](https://serpapi.com/) (Google Search)
- [Tavily](https://tavily.com/)
- [DuckDuckGo](https://duckduckgo.com/)

### Добавление памяти

Для долгосрочной памяти агента можно использовать:
- `ConversationBufferMemory`
- `ConversationSummaryMemory`
- Векторные базы данных (Chroma, Pinecone, etc.)

### Другие модели

Проект использует OpenAI, но LangChain поддерживает множество других провайдеров:
- Anthropic (Claude)
- Google (Gemini)
- Cohere
- Ollama (локальные модели)
- И многие другие

## Полезные ссылки

- [Документация LangChain](https://python.langchain.com/docs/introduction/)
- [LangChain Hub](https://smith.langchain.com/hub) — готовые промпты
- [OpenAI API](https://platform.openai.com/)

## Лицензия

MIT
