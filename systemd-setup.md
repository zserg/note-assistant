# Настройка systemd для AI Agent Bot

Это руководство поможет настроить автоматический запуск Telegram бота через systemd.

## Быстрая установка

### 1. Скопируйте service-файл

```bash
sudo cp /home/zserg/projects3/agent-simple/agent-bot.service /etc/systemd/system/
```

### 2. Перезагрузите systemd

```bash
sudo systemctl daemon-reload
```

### 3. Включите автозапуск

```bash
sudo systemctl enable agent-bot
```

### 4. Запустите сервис

```bash
sudo systemctl start agent-bot
```

## Проверка статуса

```bash
# Проверить статус
sudo systemctl status agent-bot

# Посмотреть логи
sudo journalctl -u agent-bot -f

# Или последние 50 строк
sudo journalctl -u agent-bot -n 50
```

## Управление сервисом

```bash
# Запустить
sudo systemctl start agent-bot

# Остановить
sudo systemctl stop agent-bot

# Перезапустить
sudo systemctl restart agent-bot

# Отключить автозапуск
sudo systemctl disable agent-bot
```

## Обновление сервиса

После изменения `.env` или кода:

```bash
sudo systemctl restart agent-bot
sudo journalctl -u agent-bot -f
```

## Настройка (опционально)

Если нужно изменить пути, отредактируйте `agent-bot.service`:

```bash
sudo nano /etc/systemd/system/agent-bot.service
```

Основные параметры:
- `User=` — пользователь, от которого запускается бот
- `Group=` — группа пользователя
- `WorkingDirectory=` — рабочая директория проекта
- `ExecStart=` — полный путь к Python и скрипту

После изменения:
```bash
sudo systemctl daemon-reload
sudo systemctl restart agent-bot
```

## Решение проблем

### Сервис не запускается

Проверьте права доступа:
```bash
ls -la /home/zserg/projects3/agent-simple/
```

Проверьте логи:
```bash
sudo journalctl -u agent-bot --no-pager
```

### Проверьте виртуальное окружение

```bash
ls -la /home/zserg/projects3/agent-simple/venv/bin/python
```

### Проверьте файл .env

```bash
cat /home/zserg/projects3/agent-simple/.env
```

Все переменные должны быть корректно настроены:
- `DEEPSEEK_API_KEY`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_ALLOWED_USER_ID` (опционально)
