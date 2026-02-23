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


class YandexVisionClient:
    """Клиент для работы с Yandex Vision OCR API."""
    
    # Новый OCR API endpoint (вместо старого vision.api.cloud.yandex.net)
    BASE_URL = "https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText"
    
    # Альтернативный хост (если основной не работает)
    ALT_URL = "https://ocr.{{ api-host }}/ocr/v1/recognizeText"
    
    def __init__(self):
        self.folder_id = os.getenv("YANDEX_VISION_FOLDER_ID")
        self.iam_token = os.getenv("YANDEX_VISION_IAM_TOKEN")
        self.api_key = os.getenv("YANDEX_VISION_API_KEY")
        
        self.enabled = self._check_configuration()
    
    def _check_configuration(self) -> bool:
        """Проверить настройку API."""
        if not self.folder_id:
            logger.warning("YANDEX_VISION_FOLDER_ID не настроен. Yandex Vision OCR недоступен.")
            return False
        
        if not self.iam_token and not self.api_key:
            logger.warning("YANDEX_VISION_IAM_TOKEN или YANDEX_VISION_API_KEY не настроены. Yandex Vision OCR недоступен.")
            return False
        
        logger.info("✅ Yandex Vision OCR API настроен")
        return True
    
    def _get_headers(self) -> Dict[str, str]:
        """Получить заголовки для запроса."""
        headers = {
            "Content-Type": "application/json",
            "x-folder-id": self.folder_id,  # Обязательный заголовок для OCR API
        }
        
        if self.iam_token:
            headers["Authorization"] = f"Bearer {self.iam_token}"
        elif self.api_key:
            headers["Authorization"] = f"Api-Key {self.api_key}"
        
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
        return {
            "enabled": self.enabled,
            "folder_id_configured": bool(self.folder_id),
            "auth_configured": bool(self.iam_token or self.api_key),
            "auth_method": "iam_token" if self.iam_token else ("api_key" if self.api_key else None)
        }


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
