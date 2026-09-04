import os
import json
import logging
import secrets
import asyncio
from typing import Dict, Any, List, Optional

# Configure logger
logger = logging.getLogger("regression_studio.database")

class DatabaseAdapter:
    def __init__(self):
        self.fallback_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
        self.fallback_path = os.path.join(self.fallback_dir, "db.json")
        self.use_fallback = True
        self.client = None
        self.db = None
        self._fallback_cache: Optional[Dict[str, List[Dict[str, Any]]]] = None
        self._lock = asyncio.Lock()

        # Ensure fallback directory exists
        os.makedirs(self.fallback_dir, exist_ok=True)
        if not os.path.exists(self.fallback_path):
            with open(self.fallback_path, "w") as f:
                json.dump({"users": [], "datasets": [], "models": [], "reports": []}, f, indent=4)
                
    async def connect(self):
        """Initializes and logs local file database usage."""
        logger.info(f"Using local file database: {self.fallback_path}")

    def _read_fallback(self) -> Dict[str, List[Dict[str, Any]]]:
        if self._fallback_cache is not None:
            return self._fallback_cache
        try:
            with open(self.fallback_path, "r") as f:
                self._fallback_cache = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read database file: {e}")
            self._fallback_cache = {"users": [], "datasets": [], "models": [], "reports": []}
        return self._fallback_cache

    def _write_fallback(self, data: Dict[str, List[Dict[str, Any]]]):
        self._fallback_cache = data
        try:
            with open(self.fallback_path, "w") as f:
                json.dump(data, f, separators=(",", ":"))
        except Exception as e:
            logger.error(f"Failed to write database file: {e}")

    # Generic CRUD operations
    async def insert_one(self, collection_name: str, document: Dict[str, Any]) -> str:
        """Inserts a document into the specified collection and returns its ID."""
        async with self._lock:
            data = self._read_fallback()
            if collection_name not in data:
                data[collection_name] = []
                
            # Ensure document has a string id
            if "_id" not in document:
                document["_id"] = secrets.token_hex(12)
            else:
                document["_id"] = str(document["_id"])
                
            data[collection_name].append(document)
            self._write_fallback(data)
            return document["_id"]

    async def find_one(self, collection_name: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Finds a single document matching the query."""
        async with self._lock:
            data = self._read_fallback()
            collection = data.get(collection_name, [])
            for doc in collection:
                match = True
                for k, v in query.items():
                    if doc.get(k) != v:
                        match = False
                        break
                if match:
                    return doc
            return None

    async def find_many(self, collection_name: str, query: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Finds multiple documents matching the query."""
        if query is None:
            query = {}
            
        async with self._lock:
            data = self._read_fallback()
            collection = data.get(collection_name, [])
            if not query:
                return collection
                
            results = []
            for doc in collection:
                match = True
                for k, v in query.items():
                    if doc.get(k) != v:
                        match = False
                        break
                if match:
                    results.append(doc)
            return results

    async def update_one(self, collection_name: str, query: Dict[str, Any], update: Dict[str, Any]) -> bool:
        """Updates a single document matching the query."""
        async with self._lock:
            data = self._read_fallback()
            collection = data.get(collection_name, [])
            updated = False
            for doc in collection:
                match = True
                for k, v in query.items():
                    if doc.get(k) != v:
                        match = False
                        break
                if match:
                    # Apply update operators like $set
                    if "$set" in update:
                        for uk, uv in update["$set"].items():
                            doc[uk] = uv
                        updated = True
                    else:
                        # Direct update replacement
                        for uk, uv in update.items():
                            doc[uk] = uv
                        updated = True
                    break
            if updated:
                self._write_fallback(data)
            return updated

    async def delete_one(self, collection_name: str, query: Dict[str, Any]) -> bool:
        """Deletes a single document matching the query."""
        async with self._lock:
            data = self._read_fallback()
            collection = data.get(collection_name, [])
            
            filtered_collection = []
            deleted = False
            for doc in collection:
                match = True
                for k, v in query.items():
                    if doc.get(k) != v:
                        match = False
                        break
                if match and not deleted:
                    deleted = True
                    continue
                filtered_collection.append(doc)
                
            data[collection_name] = filtered_collection
            if deleted:
                self._write_fallback(data)
            return deleted

# Singleton instance
db_client = DatabaseAdapter()
