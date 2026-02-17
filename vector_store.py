"""
Модуль для семантического (векторного) поиска.

Использует:
- GigaChat SDK (gigachat) для получения эмбеддингов
- sqlite-vec для хранения и поиска векторов в SQLite
"""

import os
import json
import logging
import sqlite3
from typing import List, Dict, Optional

import sqlite_vec
from gigachat import GigaChat
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger("agent")


class GigaChatEmbeddings:
    """Клиент для получения эмбеддингов через GigaChat SDK."""
    
    def __init__(self):
        self.credentials = os.getenv("GIGACHAT_CLIENT_CREDENTIALS")
        
        if not self.credentials or self.credentials == "your_gigachat_client_credentials_here":
            logger.warning("GIGACHAT_CLIENT_CREDENTIALS не настроен. Семантический поиск недоступен.")
            self.enabled = False
        else:
            self.enabled = True
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def _get_embeddings_with_retry(self, texts: List[str]) -> List[List[float]]:
        """Внутренний метод с retry для получения эмбеддингов."""
        with GigaChat(
            credentials=self.credentials,
            verify_ssl_certs=False,  # Для тестового API Сбера
        ) as client:
            result = client.embeddings(texts)
            
            # Сортируем по индексу, так как API может вернуть в другом порядке
            embeddings = sorted(result.data, key=lambda x: x.index)
            return [item.embedding for item in embeddings]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Получить эмбеддинги для списка документов."""
        if not self.enabled:
            raise ValueError("GigaChat API не настроен. Проверьте GIGACHAT_CLIENT_CREDENTIALS")
        
        if not texts:
            return []
        
        try:
            return self._get_embeddings_with_retry(texts)
        except Exception as e:
            logger.error(f"Ошибка получения эмбеддингов после всех попыток: {e}")
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
            logger.info(f"sqlite start")
            conn = sqlite3.connect(self.db_path)
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            
            logger.info(f"sqlite start cp0")
            # Создаём виртуальную таблицу для векторов
            conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS note_vectors USING vec0(
                    embedding FLOAT[{self.dimension}],
                    +note_id TEXT,
                    +filename TEXT,
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
            logger.info(f"embedding cp0")
            # Получаем эмбеддинг через GigaChat SDK
            embedding = self.embeddings.embed_query(content)
            logger.info(f"embedding cp1")
            
            # Формируем preview (первые 200 символов)
            if preview is None:
                preview = content[:200].replace('\n', ' ')
            
            conn = sqlite3.connect(self.db_path)
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            logger.info(f"embedding cp2")
            
            # Сериализуем вектор в JSON для sqlite-vec
            embedding_json = json.dumps(embedding)
            
            # Удаляем старую запись если есть
            conn.execute("DELETE FROM note_vectors WHERE note_id = ?", (note_id,))
            logger.info(f"embedding cp3")
            
            # Добавляем новую
            conn.execute("""
                INSERT INTO note_vectors (embedding, note_id, filename, preview)
                VALUES (vec_f32(?), ?, ?, ?)
            """, (embedding_json, note_id, filename, preview))
            
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
            # Получаем эмбеддинг запроса через GigaChat SDK
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
                    vec_distance_cosine(embedding, vec_f32(?)) as distance
                FROM note_vectors
                ORDER BY distance
                LIMIT ?
            """, (query_json, top_k)).fetchall()
            
            conn.close()
            
            # Форматируем результаты
            formatted_results = []
            for row in results:
                note_id, filename, preview, distance = row
                # distance = 1 - cosine_similarity, поэтому similarity = 1 - distance
                similarity = 1 - distance
                formatted_results.append({
                    "note_id": note_id,
                    "filename": filename,
                    "preview": preview,
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
