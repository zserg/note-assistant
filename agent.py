"""
AI Agent на базе LangChain.

Пример агента с инструментами (tools) для выполнения различных задач.
"""

import os
from typing import Type
from datetime import datetime

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI  # DeepSeek API совместим с OpenAI API
from pydantic import BaseModel, Field


# Загрузка переменных окружения
load_dotenv()


# === Определение инструментов (Tools) ===

class CurrentTimeInput(BaseModel):
    """Входные параметры для получения текущего времени."""
    timezone: str = Field(default="UTC", description="Часовой пояс (по умолчанию UTC)")


class CurrentTimeTool(BaseTool):
    """Инструмент для получения текущего времени."""
    
    name: str = "current_time"
    description: str = "Возвращает текущее время и дату"
    args_schema: Type[BaseModel] = CurrentTimeInput
    
    def _run(self, timezone: str = "UTC") -> str:
        """Получить текущее время."""
        now = datetime.now()
        return f"Текущее время ({timezone}): {now.strftime('%Y-%m-%d %H:%M:%S')}"


class CalculatorInput(BaseModel):
    """Входные параметры для калькулятора."""
    expression: str = Field(description="Математическое выражение для вычисления (например: 2 + 2 * 3)")


class CalculatorTool(BaseTool):
    """Инструмент-калькулятор для математических вычислений."""
    
    name: str = "calculator"
    description: str = "Выполняет математические вычисления. Используйте для расчетов."
    args_schema: Type[BaseModel] = CalculatorInput
    
    def _run(self, expression: str) -> str:
        """Вычислить математическое выражение."""
        try:
            # Безопасное вычисление с ограниченным набором функций
            allowed_names = {
                "abs": abs,
                "round": round,
                "max": max,
                "min": min,
                "pow": pow,
            }
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return f"Результат: {result}"
        except Exception as e:
            return f"Ошибка вычисления: {str(e)}"


class SearchInput(BaseModel):
    """Входные параметры для поиска."""
    query: str = Field(description="Поисковый запрос")


class MockSearchTool(BaseTool):
    """Заглушка для поиска (в реальном проекте замените на реальный API)."""
    
    name: str = "search"
    description: str = "Ищет информацию в интернете. Используйте для поиска актуальной информации."
    args_schema: Type[BaseModel] = SearchInput
    
    def _run(self, query: str) -> str:
        """Выполнить поиск (заглушка)."""
        # В реальном проекте здесь будет вызов поискового API
        return (
            f"[ЗАГЛУШКА ПОИСКА] Результаты поиска для '{query}':\n"
            "1. Это пример результата поиска\n"
            "2. В реальном проекте подключите Google Search API, Bing API или другой сервис\n"
            "3. Для тестирования агента используются имитационные данные"
        )


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
        CurrentTimeTool(),
        CalculatorTool(),
        MockSearchTool(),
    ]
    
    # Создание промпта
    system_prompt = """Ты — полезный AI ассистент. У тебя есть доступ к следующим инструментам:

1. current_time - получить текущее время
2. calculator - калькулятор для математических вычислений  
3. search - поиск информации в интернете

Используй инструменты, когда это необходимо для ответа на вопрос пользователя.
Если вопрос можно ответить без инструментов — отвечай напрямую.
Всегда отвечай на языке пользователя (русский или английский).
"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Создание агента
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    # Создание executor'а
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )
    
    return agent_executor


# === Интерактивный режим ===

def main():
    """Запустить интерактивный режим работы с агентом."""
    print("=" * 50)
    print("🤖 AI Agent на базе LangChain")
    print("=" * 50)
    print()
    
    try:
        agent = create_agent()
    except ValueError as e:
        print(f"❌ Ошибка: {e}")
        return
    
    print("✅ Агент успешно инициализирован!")
    print("Доступные команды:")
    print("  • 'time' или 'время' - узнать текущее время")
    print("  • 'calc' или 'посчитай' - калькулятор")
    print("  • 'search' или 'найди' - поиск информации")
    print("  • 'exit' или 'выход' - завершить работу")
    print()
    
    chat_history = []
    
    while True:
        try:
            user_input = input("👤 Вы: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ("exit", "выход", "quit", "q"):
                print("\n👋 До свидания!")
                break
            
            # Выполнение запроса
            result = agent.invoke({
                "input": user_input,
                "chat_history": chat_history,
            })
            
            print(f"\n🤖 Агент: {result['output']}\n")
            
            # Сохранение истории чата
            from langchain_core.messages import HumanMessage, AIMessage
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=result['output']))
            
            # Ограничение истории (последние 10 сообщений)
            chat_history = chat_history[-10:]
            
        except KeyboardInterrupt:
            print("\n\n👋 До свидания!")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {str(e)}\n")


if __name__ == "__main__":
    main()
