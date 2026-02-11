"""
Простой AI Agent на базе LangChain.

Упрощенная версия без сложной настройки инструментов.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  # DeepSeek API совместим с OpenAI API
from langchain_core.messages import HumanMessage, SystemMessage


load_dotenv()


def main():
    """Простой пример работы с LangChain."""
    
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key or api_key == "your_deepseek_api_key_here":
        print("❌ DEEPSEEK_API_KEY не настроен!")
        print("1. Скопируйте .env.example в .env: cp .env.example .env")
        print("2. Добавьте ваш API ключ в файл .env")
        return
    
    # Инициализация модели (DeepSeek)
    model = ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        temperature=0.7,
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )
    
    print("=" * 50)
    print("🤖 Простой AI Agent на LangChain")
    print("=" * 50)
    print("Введите 'exit' для выхода\n")
    
    # Системный промпт
    messages = [
        SystemMessage(content="Ты — полезный ассистент. Отвечай кратко и по существу.")
    ]
    
    while True:
        user_input = input("👤 Вы: ").strip()
        
        if user_input.lower() in ("exit", "выход", "quit"):
            print("👋 До свидания!")
            break
        
        if not user_input:
            continue
        
        # Добавляем сообщение пользователя
        messages.append(HumanMessage(content=user_input))
        
        # Получаем ответ
        response = model.invoke(messages)
        
        print(f"🤖 Агент: {response.content}\n")
        
        # Добавляем ответ в историю
        messages.append(response)
        
        # Ограничиваем историю
        if len(messages) > 10:
            messages = [messages[0]] + messages[-8:]


if __name__ == "__main__":
    main()
