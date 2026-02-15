"""
Модуль для семантического (векторного) поиска.

Использует:
- GigaChat API для получения эмбеддингов
- sqlite-vec для хранения и поиска векторов в SQLite
"""

import os
import json
import uuid
import logging
import sqlite3
import requests
import base64
import urllib3
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import sqlite_vec

# Отключаем предупреждения SSL (для тестового API Сбера)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger("agent")


class GigaChatEmbeddings:
    """Клиент для получения эмбеддингов через GigaChat API."""
    
    def __init__(self):
        self.client_id = os.getenv("GIGACHAT_CLIENT_ID")
        self.client_secret = os.getenv("GIGACHAT_CLIENT_SECRET")
        self.access_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None
        self.base_url = "https://gigachat.devices.sberbank.ru/api/v1"
        self.auth_url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        
        if not self.client_id or self.client_id == "your_gigachat_client_id_here":
            logger.warning("GIGACHAT_CLIENT_ID не настроен. Семантический поиск недоступен.")
            self.enabled = False
        elif not self.client_secret or self.client_secret == "your_gigachat_client_secret_here":
            logger.warning("GIGACHAT_CLIENT_SECRET не настроен. Семантический поиск недоступен.")
            self.enabled = False
        else:
            self.enabled = True
    
    def _get_access_token(self) -> str:
        """Получить или обновить OAuth токен."""
        # Проверяем, не истёк ли текущий токен
        if self.access_token and self.token_expires_at:
            if datetime.now() < self.token_expires_at:
                return self.access_token
        
        logger.debug("Запрос нового access token для GigaChat...")
        
        credentials = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode()
        ).decode()
        
        try:
            response = requests.post(
                self.auth_url,
                headers={
                    "Authorization": f"Basic {credentials}",
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                    "RqUID": str(uuid.uuid4())
                },
                data={"scope": "GIGACHAT_API_PERS"},
                verify=False,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            self.access_token = data["access_token"]
            # Токен живёт ~30 минут, ставим запас 5 минут
            expires_in = data.get("expires_at", 1800) - 300
            self.token_expires_at = datetime.now().timestamp() + expires_in
            
            logger.debug("Access token получен успешно")
            return self.access_token
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка получения токена GigaChat: {e}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Получить эмбеддинги для списка документов."""
        if not self.enabled:
            raise ValueError("GigaChat API не настроен. Проверьте GIGACHAT_CLIENT_ID и GIGACHAT_CLIENT_SECRET")
        
        if not texts:
            return []
        
        # Ограничение API: максимум 100 текстов за раз
        if len(texts) > 100:
            logger.warning(f"Слишком много текстов ({len(texts)}), разбиваем на батчи")
            results = []
            for i in range(0, len(texts), 100):
                batch = texts[i:i+100]
                results.extend(self.embed_documents(batch))
            return results
        
        token = self._get_access_token()
        
        try:
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "Embeddings",
                    "input": texts
                },
                verify=False,
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            
            # Сортируем по индексу, так как API может вернуть в другом порядке
            embeddings = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in embeddings]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка получения эмбеддингов: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Получить эмбеддинг для одного запроса."""
        return self.embed_documents([text])[0]


class VectorStore:
    """Векторное хранилище на базе sqlite-vec."""
    
    def __init__(self, db_path: str = "vector_store.db"):
        self.db_path = db_path
        self.embeddings = GigaChatEmbeddings()
        self.enabled = self.embeddings.enabled
        self.dimension = 1024  # Размерность эмбеддингов GigaChat
        
        if self.enabled:
            self._init_db()
    
    def _init_db(self):
        """Инициализировать базу данных."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            
            # Создаём виртуальную таблицу для векторов
            conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS note_vectors USING vec0(
                    note_id TEXT PRIMARY KEY,
                    embedding FLOAT[{self.dimension}],
                    +filename TEXT,
                    +created_at TIMESTAMP,
                    +preview TEXT
                )
            """)
            
            conn.commit()
            conn.close()
            logger.info(f"Векторное хранилище инициализировано: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Ошибка инициализации векторного хранилища: {e}")
            self.enabled = False
    
    def add_note(self, note_id: str, content: str, filename: str, preview: Optional[str] = None):
        """Добавить или обновить заметку в векторном хранилище."""
        if not self.enabled:
            logger.warning("Векторное хранилище не инициализировано")
            return False
        
        try:
            # Получаем эмбеддинг
            embedding = self.embeddings.embed_query(content)
            
            # Формируем preview (первые 200 символов)
            if preview is None:
                preview = content[:200].replace('\n', ' ')
            
            conn = sqlite3.connect(self.db_path)
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            
            # Сериализуем вектор в JSON для sqlite-vec
            embedding_json = json.dumps(embedding)
            
            # Удаляем старую запись если есть
            conn.execute("DELETE FROM note_vectors WHERE note_id = ?", (note_id,))
            
            # Добавляем новую
            conn.execute("""
                INSERT INTO note_vectors (note_id, embedding, filename, created_at, preview)
                VALUES (?, vec_f32(?), ?, ?, ?)
            """, (note_id, embedding_json, filename, datetime.now().isoformat(), preview))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Заметка {note_id} добавлена в векторное хранилище")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка добавления заметки в векторное хранилище: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Семантический поиск по запросу."""
        if not self.enabled:
            return []
        
        try:
            # Получаем эмбеддинг запроса
            query_embedding = self.embeddings.embed_query(query)
            query_json = json.dumps(query_embedding)
            
            conn = sqlite3.connect(self.db_path)
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            
            # Выполняем векторный поиск
            results = conn.execute("""
                SELECT 
                    note_id,
                    filename,
                    preview,
                    created_at,
                    vec_distance_cosine(embedding, vec_f32(?)) as distance
                FROM note_vectors
                ORDER BY distance
                LIMIT ?
            """, (query_json, top_k)).fetchall()
            
            conn.close()
            
            # Форматируем результаты
            formatted_results = []
            for row in results:
                note_id, filename, preview, created_at, distance = row
                # distance = 1 - cosine_similarity, поэтому similarity = 1 - distance
                similarity = 1 - distance
                formatted_results.append({
                    "note_id": note_id,
                    "filename": filename,
                    "preview": preview,
                    "created_at": created_at,
                    "similarity": round(similarity, 4)
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Ошибка семантического поиска: {e}")
            return []
    
    def delete_note(self, note_id: str) -> bool:
        """Удалить заметку из векторного хранилища."""
        if not self.enabled:
            return False
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            
            conn.execute("DELETE FROM note_vectors WHERE note_id = ?", (note_id,))
            conn.commit()
            conn.close()
            
            logger.info(f"Заметка {note_id} удалена из векторного хранилища")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка удаления заметки из векторного хранилища: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Получить статистику хранилища."""
        if not self.enabled:
            return {"enabled": False, "count": 0}
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            
            count = conn.execute("SELECT COUNT(*) FROM note_vectors").fetchone()[0]
            conn.close()
            
            return {"enabled": True, "count": count}
            
        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            return {"enabled": False, "count": 0}


# Глобальный экземпляр для использования в проекте
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Получить или создать экземпляр VectorStore."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
