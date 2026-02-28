"""
Модуль для работы с Yandex Vision OCR API.

Поддерживает:
- Распознавание текста (OCR) с изображений через новый OCR API
- Автоматическое определение языка
- Поддержка моделей: page, line, table

Требования:
- YANDEX_VISION_FOLDER_ID - ID каталога в Yandex Cloud
- YANDEX_VISION_IAM_TOKEN или YANDEX_VISION_API_KEY - для авторизации

Документация: https://yandex.cloud/en/docs/vision/operations/ocr/text-detection-image
"""

import os
import base64
import logging
import subprocess
import threading
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

import requests
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

logger = logging.getLogger("agent")


@dataclass
class OCRResult:
    """Результат распознавания текста."""
    text: str
    full_text: str  # Полный текст из textAnnotation.fullText
    markdown: Optional[str]  # Markdown версия (если доступна)
    blocks: List[Dict[str, Any]]
    width: int
    height: int
    rotate: str  # Угол поворота изображения


@dataclass
class ImageAnalysisResult:
    """Результат анализа изображения."""
    description: str
    labels: List[str]
    text: Optional[str]


class YandexCLITokenManager:
    """
    Менеджер для автоматического обновления IAM токена через YC CLI.
    
    Обновляет токен каждый час (токен живёт 12 часов, но обновляем чаще для надёжности).
    """
    
    def __init__(self, refresh_interval: int = 3600):
        """
        Args:
            refresh_interval: Интервал обновления токена в секундах (по умолчанию 1 час)
        """
        self.refresh_interval = refresh_interval
        self._token: Optional[str] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._refresh_thread: Optional[threading.Thread] = None
        self._last_refresh: float = 0
        
        # Пробуем получить токен сразу при инициализации
        self._refresh_token()
        
        # Запускаем фоновый поток для обновления
        self._start_refresh_thread()
    
    def _start_refresh_thread(self):
        """Запустить фоновый поток для периодического обновления токена."""
        if self._refresh_thread is None or not self._refresh_thread.is_alive():
            self._stop_event.clear()
            self._refresh_thread = threading.Thread(target=self._refresh_loop, daemon=True)
            self._refresh_thread.start()
            logger.info("🔄 Запущен фоновый поток обновления IAM токена через YC CLI")
    
    def _refresh_loop(self):
        """Цикл периодического обновления токена."""
        while not self._stop_event.is_set():
            time.sleep(10)  # Проверяем каждые 10 секунд
            
            if self._stop_event.is_set():
                break
                
            # Проверяем, пора ли обновлять токен
            if time.time() - self._last_refresh >= self.refresh_interval:
                self._refresh_token()
    
    def _refresh_token(self) -> bool:
        """
        Получить новый IAM токен через YC CLI.
        
        Returns:
            True если токен успешно получен, иначе False
        """
        try:
            logger.info("Запрос нового IAM токена через YC CLI...")
            
            result = subprocess.run(
                ["yc", "iam", "create-token"],
                capture_output=True,
                text=True,
                timeout=30,
                check=True
            )
            
            token = result.stdout.strip()
            if not token:
                logger.error("YC CLI вернул пустой токен")
                return False
            
            with self._lock:
                self._token = token
                self._last_refresh = time.time()
            
            logger.info("✅ IAM токен успешно обновлён через YC CLI")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Ошибка YC CLI при получении токена: {e.stderr}")
            return False
        except subprocess.TimeoutExpired:
            logger.error("Таймаут при выполнении YC CLI")
            return False
        except FileNotFoundError:
            logger.error("YC CLI не найден. Убедитесь, что 'yc' установлен и доступен в PATH")
            return False
        except Exception as e:
            logger.error(f"Неожиданная ошибка при получении токена через CLI: {e}")
            return False
    
    def get_token(self) -> Optional[str]:
        """
        Получить текущий актуальный IAM токен.
        
        Returns:
            IAM токен или None если не удалось получить
        """
        with self._lock:
            # Если токена нет или прошло больше часа с последнего обновления,
            # пробуем обновить синхронно
            if self._token is None or (time.time() - self._last_refresh >= self.refresh_interval):
                pass  # Выйдем из lock и обновим снаружи
            else:
                return self._token
        
        # Пробуем обновить синхронно если токен устарел или отсутствует
        self._refresh_token()
        
        with self._lock:
            return self._token
    
    def stop(self):
        """Остановить фоновый поток обновления токена."""
        self._stop_event.set()
        if self._refresh_thread and self._refresh_thread.is_alive():
            self._refresh_thread.join(timeout=5)
            logger.info("🛑 Фоновый поток обновления IAM токена остановлен")


class YandexVisionClient:
    """Клиент для работы с Yandex Vision OCR API."""
    
    # Новый OCR API endpoint (вместо старого vision.api.cloud.yandex.net)
    BASE_URL = "https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText"
    
    # Альтернативный хост (если основной не работает)
    ALT_URL = "https://ocr.{{ api-host }}/ocr/v1/recognizeText"
    
    def __init__(self, use_cli_auth: bool = True, cli_refresh_interval: int = 3600):
        """
        Args:
            use_cli_auth: Использовать YC CLI для получения IAM токена, если не указан API Key
            cli_refresh_interval: Интервал обновления токена через CLI в секундах (по умолчанию 1 час)
        """
        self.folder_id = os.getenv("YANDEX_VISION_FOLDER_ID")
        self.iam_token = os.getenv("YANDEX_VISION_IAM_TOKEN")
        self.api_key = os.getenv("YANDEX_VISION_API_KEY")
        
        # Менеджер токенов через CLI (используется если нет API Key и статического IAM токена)
        self._token_manager: Optional[YandexCLITokenManager] = None
        
        # Приоритет авторизации:
        # 1. API Key (самый простой, не требует обновления)
        # 2. Статический IAM токен (из env)
        # 3. YC CLI (динамическое обновление)
        if not self.api_key and not self.iam_token and use_cli_auth:
            logger.info("API Key и IAM токен не настроены, пробуем использовать YC CLI...")
            try:
                self._token_manager = YandexCLITokenManager(refresh_interval=cli_refresh_interval)
                # Проверяем, удалось ли получить токен
                if self._token_manager.get_token() is None:
                    logger.warning("Не удалось получить IAM токен через YC CLI")
                    self._token_manager = None
            except Exception as e:
                logger.error(f"Ошибка инициализации YC CLI менеджера: {e}")
                self._token_manager = None
        
        self.enabled = self._check_configuration()
    
    def _check_configuration(self) -> bool:
        """Проверить настройку API."""
        if not self.folder_id:
            logger.warning("YANDEX_VISION_FOLDER_ID не настроен. Yandex Vision OCR недоступен.")
            return False
        
        # Проверяем наличие любого метода авторизации
        has_static_auth = bool(self.iam_token or self.api_key)
        has_cli_auth = self._token_manager is not None and self._token_manager.get_token() is not None
        
        if not has_static_auth and not has_cli_auth:
            logger.warning(
                "Авторизация не настроена. Установите YANDEX_VISION_API_KEY, "
                "YANDEX_VISION_IAM_TOKEN или убедитесь, что YC CLI настроен и работает."
            )
            return False
        
        auth_method = []
        if self.api_key:
            auth_method.append("API Key")
        if self.iam_token:
            auth_method.append("Static IAM Token")
        if has_cli_auth:
            auth_method.append("YC CLI (auto-refresh)")
        
        logger.info(f"✅ Yandex Vision OCR API настроен (авторизация: {', '.join(auth_method)})")
        return True
    
    def _get_headers(self) -> Dict[str, str]:
        """Получить заголовки для запроса."""
        headers = {
            "Content-Type": "application/json",
            "x-folder-id": self.folder_id,  # Обязательный заголовок для OCR API
        }
        
        # Приоритет: API Key > Static IAM Token > YC CLI
        if self.api_key:
            headers["Authorization"] = f"Api-Key {self.api_key}"
        elif self.iam_token:
            headers["Authorization"] = f"Bearer {self.iam_token}"
        elif self._token_manager:
            token = self._token_manager.get_token()
            if token:
                headers["Authorization"] = f"Bearer {token}"
        
        return headers
    
    def _detect_mime_type(self, image_bytes: bytes) -> str:
        """Определить MIME-тип изображения по сигнатуре."""
        if image_bytes.startswith(b'\xff\xd8'):
            return "JPEG"
        elif image_bytes.startswith(b'\x89PNG'):
            return "PNG"
        elif image_bytes.startswith(b'%PDF'):
            return "PDF"
        else:
            # По умолчанию JPEG
            return "JPEG"
    
    def recognize_text(
        self, 
        image_bytes: bytes, 
        language_codes: List[str] = None,
        model: str = "page"
    ) -> Optional[OCRResult]:
        """
        Распознать текст на изображении.
        
        Args:
            image_bytes: Байты изображения
            language_codes: Список кодов языков (по умолчанию ["*"] для автоопределения)
            model: Модель распознавания - "page" (по умолчанию), "line", "table"
        
        Returns:
            OCRResult с распознанным текстом или None при ошибке
        """
        if not self.enabled:
            return None
        
        if language_codes is None:
            language_codes = ["*"]  # Автоопределение языка
        
        # Кодируем изображение в base64
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        mime_type = self._detect_mime_type(image_bytes)
        
        # Формируем запрос по новому API
        body = {
            "content": image_base64,
            "mimeType": mime_type,
            "languageCodes": language_codes,
            "model": model
        }
        
        try:
            logger.debug(f"Отправка запроса в Yandex OCR API: mimeType={mime_type}, model={model}")
            
            response = requests.post(
                self.BASE_URL,
                headers=self._get_headers(),
                json=body,
                timeout=60
            )
            
            # Если не удалось, попробуем альтернативный URL
            if response.status_code in [404, 500]:
                logger.warning(f"Основной URL вернул {response.status_code}, пробуем альтернативный...")
                response = requests.post(
                    self.ALT_URL.replace("{{ api-host }}", "api.cloud.yandex.net"),
                    headers=self._get_headers(),
                    json=body,
                    timeout=60
                )
            
            response.raise_for_status()
            
            data = response.json()
            
            # Извлекаем результаты из нового формата ответа
            # Ответ обёрнут в поле "result"
            result = data.get("result", {})
            text_annotation = result.get("textAnnotation", {}) if result else {}
            
            if not text_annotation:
                logger.warning(f"Yandex OCR API: textAnnotation отсутствует в ответе. Response keys: {list(data.keys())}")
                return None
            
            # Получаем полный текст (fullText)
            full_text = text_annotation.get("fullText", "")
            
            # Получаем markdown (если использовалась соответствующая модель)
            markdown = text_annotation.get("markdown")
            
            # Получаем блоки
            blocks = text_annotation.get("blocks", [])
            width = int(text_annotation.get("width", 0))
            height = int(text_annotation.get("height", 0))
            rotate = text_annotation.get("rotate", "ANGLE_UNSPECIFIED")
            
            # Формируем структурированный текст из блоков (для совместимости)
            structured_text = self._extract_structured_text(blocks)
            
            logger.info(f"Yandex Vision OCR: распознано {len(blocks)} блоков, "
                       f"полный текст: {len(full_text)} символов")
            
            return OCRResult(
                text=structured_text or full_text,
                full_text=full_text,
                markdown=markdown,
                blocks=blocks,
                width=width,
                height=height,
                rotate=rotate
            )
            
        except requests.exceptions.HTTPError as e:
            error_text = ""
            try:
                error_data = e.response.json()
                error_text = str(error_data)
            except:
                error_text = e.response.text
            logger.error(f"Yandex OCR API ошибка: {e.response.status_code} - {error_text}")
            return None
        except Exception as e:
            logger.error(f"Ошибка при распознавании текста: {e}")
            return None
    
    def _extract_structured_text(self, blocks: List[Dict]) -> str:
        """Извлечь структурированный текст из блоков."""
        all_text = []
        
        for block in blocks:
            block_text = []
            for line in block.get("lines", []):
                line_text = line.get("text", "").strip()
                if line_text:
                    block_text.append(line_text)
            
            if block_text:
                all_text.append("\n".join(block_text))
        
        return "\n\n".join(all_text)
    
    def analyze_image(
        self, 
        image_bytes: bytes, 
        language_codes: List[str] = None,
        model: str = "page"
    ) -> Optional[ImageAnalysisResult]:
        """
        Проанализировать изображение: распознать текст.
        
        Args:
            image_bytes: Байты изображения
            language_codes: Список кодов языков для OCR
            model: Модель распознавания
        
        Returns:
            ImageAnalysisResult с результатами анализа
        """
        if not self.enabled:
            return None
        
        # Распознаём текст
        ocr_result = self.recognize_text(image_bytes, language_codes, model)
        
        if ocr_result is None:
            return None
        
        text = ocr_result.full_text or ""
        
        # Формируем описание на основе распознанного текста
        if text.strip():
            description = f"На изображении обнаружен текст:\n{text[:1000]}"
            if len(text) > 1000:
                description += "\n... (текст обрезан)"
            labels = ["текст", "документ"] if len(text) > 50 else ["текст"]
        else:
            description = "Изображение не содержит распознаваемого текста."
            labels = ["изображение"]
        
        return ImageAnalysisResult(
            description=description,
            labels=labels,
            text=text if text.strip() else None
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Получить статус конфигурации."""
        has_cli = self._token_manager is not None
        cli_token_valid = has_cli and self._token_manager.get_token() is not None
        
        auth_methods = []
        if self.api_key:
            auth_methods.append("api_key")
        if self.iam_token:
            auth_methods.append("static_iam_token")
        if has_cli:
            auth_methods.append("yc_cli_auto_refresh")
        
        return {
            "enabled": self.enabled,
            "folder_id_configured": bool(self.folder_id),
            "auth_configured": bool(self.iam_token or self.api_key or cli_token_valid),
            "auth_methods": auth_methods,
            "primary_auth_method": (
                "api_key" if self.api_key 
                else ("static_iam_token" if self.iam_token 
                else ("yc_cli" if cli_token_valid else None))
            ),
            "cli_token_valid": cli_token_valid,
            "cli_last_refresh": (
                self._token_manager._last_refresh if has_cli else None
            )
        }
    
    def stop(self):
        """Остановить фоновые процессы (например, поток обновления токена)."""
        if self._token_manager:
            self._token_manager.stop()
            self._token_manager = None


# Глобальный экземпляр клиента
_vision_client: Optional[YandexVisionClient] = None


def get_vision_client() -> YandexVisionClient:
    """Получить экземпляр клиента Yandex Vision OCR (singleton)."""
    global _vision_client
    if _vision_client is None:
        _vision_client = YandexVisionClient()
    return _vision_client


def analyze_image_file(image_path: str, model: str = "page") -> Optional[ImageAnalysisResult]:
    """
    Проанализировать изображение из файла.
    
    Args:
        image_path: Путь к файлу изображения
        model: Модель распознавания ("page", "line", "table")
    
    Returns:
        ImageAnalysisResult или None при ошибке
    """
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        client = get_vision_client()
        return client.analyze_image(image_bytes, model=model)
    except Exception as e:
        logger.error(f"Ошибка при чтении файла {image_path}: {e}")
        return None


def recognize_text_file(
    image_path: str, 
    language_codes: List[str] = None,
    model: str = "page"
) -> Optional[OCRResult]:
    """
    Распознать текст на изображении из файла.
    
    Args:
        image_path: Путь к файлу изображения
        language_codes: Список кодов языков
        model: Модель распознавания
    
    Returns:
        OCRResult или None при ошибке
    """
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        client = get_vision_client()
        return client.recognize_text(image_bytes, language_codes, model)
    except Exception as e:
        logger.error(f"Ошибка при чтении файла {image_path}: {e}")
        return None
