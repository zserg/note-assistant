"""
Telegram Bot для AI Agent.

Запуск:
    python telegram_bot.py

Требования:
    - Установленные зависимости из requirements.txt
    - Настроенный .env файл с TELEGRAM_BOT_TOKEN и DEEPSEEK_API_KEY
"""

import os
import sys
import asyncio
from pathlib import Path

from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Импортируем агента и логгер из agent.py
from agent import create_agent, clean_text, logger
from langchain_core.messages import HumanMessage, AIMessage

# Telegram imports
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)


# Хранилище истории чатов для каждого пользователя
# Ключ: chat_id, Значение: список сообщений
chat_histories = {}

# ID разрешённого пользователя (владельца бота)
ALLOWED_USER_ID = os.getenv("TELEGRAM_ALLOWED_USER_ID")
if ALLOWED_USER_ID:
    try:
        ALLOWED_USER_ID = int(ALLOWED_USER_ID)
    except ValueError:
        logger.error("❌ TELEGRAM_ALLOWED_USER_ID должен быть числом!")
        ALLOWED_USER_ID = None


def is_authorized(user_id: int) -> bool:
    """Проверить, авторизован ли пользователь."""
    if ALLOWED_USER_ID is None:
        # Если ID не настроен — разрешаем всем (для отладки)
        return True
    return user_id == ALLOWED_USER_ID


async def unauthorized_message(update: Update) -> None:
    """Отправить сообщение о запрете доступа."""
    await update.message.reply_text(
        "🚫 Доступ запрещён.\n\n"
        "Этот бот работает только для владельца.\n"
        f"Ваш ID: `{update.effective_user.id}`"
    )
    logger.warning(
        f"🚫 Unauthorized access attempt: user_id={update.effective_user.id}, "
        f"username={update.effective_user.username}, "
        f"name={update.effective_user.first_name}"
    )


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start."""
    # Проверка авторизации
    if not is_authorized(update.effective_user.id):
        await unauthorized_message(update)
        return
    
    user = update.effective_user
    welcome_text = (
        f"👋 Привет, {user.first_name}!\n\n"
        "Я — AI Agent для работы с заметками.\n"
        "Я могу:\n"
        "• 💾 Сохранять заметки в Markdown\n"
        "• 🔍 Искать по сохранённым заметкам\n"
        "• 📄 Показывать содержимое заметок\n\n"
        "Просто напиши мне что угодно!\n\n"
        "Команды:\n"
        "/start — начать работу\n"
        "/clear — очистить историю чата\n"
        "/help — помощь"
    )
    await update.message.reply_text(welcome_text)
    logger.info(f"👤 New user started: {user.id} ({user.username or user.first_name})")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /help."""
    # Проверка авторизации
    if not is_authorized(update.effective_user.id):
        await unauthorized_message(update)
        return
    
    help_text = (
        "📖 *Помощь по использованию*\n\n"
        "*Сохранение заметок:*\n"
        "Просто напиши сообщение, начинающееся со слов:\n"
        "• запомни, заметка, важно\n"
        "• напомни, мне нужно, идея\n"
        "• план, список дел\n\n"
        "*Примеры:*\n"
        "`запомни купить молоко`\n"
        "`идея: написать статью про Python`\n\n"
        "*Обычный поиск (по точным словам):*\n"
        "Напиши: `найди <текст>` или `поиск <текст>`\n\n"
        "*Семантический поиск (по смыслу):*\n"
        "Напиши: `семантический поиск <описание>`\n"
        "Ищет по смыслу, даже если слова другие\n"
        "Пример: `что я планировал купить`\n\n"
        "*Команды бота:*\n"
        "/start — начать работу\n"
        "/clear — очистить историю чата\n"
        "/help — показать эту помощь"
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /clear — очищает историю чата."""
    # Проверка авторизации
    if not is_authorized(update.effective_user.id):
        await unauthorized_message(update)
        return
    
    chat_id = update.effective_chat.id
    if chat_id in chat_histories:
        chat_histories[chat_id] = []
    await update.message.reply_text("🗑️ История чата очищена!")
    logger.info(f"🗑️ Chat history cleared for chat_id: {chat_id}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик текстовых сообщений."""
    # Проверка авторизации
    if not is_authorized(update.effective_user.id):
        await unauthorized_message(update)
        return
    
    chat_id = update.effective_chat.id
    user = update.effective_user
    
    # Получаем текст сообщения
    user_input = clean_text(update.message.text).strip()
    
    if not user_input:
        return
    
    logger.info(f"👤 Telegram user {user.id} ({user.username or user.first_name}): {user_input[:100]}...")
    
    # Показываем статус "печатает"
    await update.message.chat.send_action(action="typing")
    
    try:
        # Получаем или создаём историю чата для этого пользователя
        if chat_id not in chat_histories:
            chat_histories[chat_id] = []
        
        chat_history = chat_histories[chat_id]
        
        # Создаём агента (singleton, можно оптимизировать)
        # Для production лучше создавать агента один раз при старте
        if not hasattr(context.application, "agent"):
            context.application.agent = create_agent()
        agent = context.application.agent
        
        # Формируем сообщения с историей
        messages = list(chat_history)
        messages.append(HumanMessage(content=user_input))
        
        # Вызываем агента
        result = await asyncio.get_event_loop().run_in_executor(
            None, 
            lambda: agent.invoke({"messages": messages})
        )
        
        # Получаем ответ
        output = result["messages"][-1].content
        output = clean_text(output)
        
        logger.info(f"🤖 Agent response to {user.id}: {output[:100]}...")
        
        # Отправляем ответ пользователю
        # Telegram имеет ограничение на длину сообщения (4096 символов)
        MAX_MESSAGE_LENGTH = 4000
        
        if len(output) <= MAX_MESSAGE_LENGTH:
            await update.message.reply_text(output)
        else:
            # Разбиваем длинное сообщение на части
            for i in range(0, len(output), MAX_MESSAGE_LENGTH):
                chunk = output[i:i + MAX_MESSAGE_LENGTH]
                await update.message.reply_text(chunk)
        
        # Сохраняем историю чата
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=output))
        
        # Ограничение истории (последние 10 сообщений)
        chat_histories[chat_id] = chat_history[-10:]
        
    except Exception as e:
        error_msg = clean_text(str(e))
        logger.error(f"❌ Error processing message from {user.id}: {error_msg}")
        await update.message.reply_text(
            f"❌ Произошла ошибка при обработке сообщения.\n"
            f"Попробуйте позже или обратитесь к администратору."
        )


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик ошибок."""
    logger.error(f"⚠️ Telegram error: {context.error}")
    if update and update.effective_message:
        await update.effective_message.reply_text(
            "⚠️ Произошла ошибка. Попробуйте ещё раз позже."
        )


def main() -> None:
    """Запустить Telegram бота."""
    # Проверяем наличие токена
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token or token == "your_telegram_bot_token_here":
        print("❌ TELEGRAM_BOT_TOKEN не настроен!")
        print("1. Скопируйте .env.example в .env: cp .env.example .env")
        print("2. Получите токен у @BotFather в Telegram")
        print("3. Добавьте токен в файл .env")
        sys.exit(1)
    
    # Проверяем API ключ DeepSeek
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key or api_key == "your_deepseek_api_key_here":
        print("❌ DEEPSEEK_API_KEY не настроен!")
        print("Добавьте ваш API ключ в файл .env")
        sys.exit(1)
    
    print("=" * 50)
    print("🤖 AI Agent Telegram Bot")
    print("=" * 50)
    print()
    
    # Проверяем настройку доступа
    if ALLOWED_USER_ID:
        print(f"🔒 Доступ ограничен: только пользователь с ID {ALLOWED_USER_ID}")
    else:
        print("⚠️  ВНИМАНИЕ: Доступ не ограничен! Любой может использовать бота.")
        print("   Добавьте TELEGRAM_ALLOWED_USER_ID в .env для ограничения доступа.")
    print()
    
    # Создаём приложение
    application = Application.builder().token(token).build()
    
    # Создаём агента заранее (при первом запуске)
    print("⏳ Инициализация агента...")
    try:
        application.agent = create_agent()
        print("✅ Агент успешно инициализирован!")
    except Exception as e:
        print(f"❌ Ошибка инициализации агента: {e}")
        sys.exit(1)
    
    print()
    print("🚀 Запуск бота...")
    print("Нажмите Ctrl+C для остановки")
    print()
    
    # Регистрируем обработчики
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("clear", clear_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Регистрируем обработчик ошибок
    application.add_error_handler(error_handler)
    
    # Запускаем бота
    logger.info("=== Telegram Bot запущен ===")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
