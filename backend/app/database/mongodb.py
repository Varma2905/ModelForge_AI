import os
import json
import logging
from typing import Dict, Any, List, Optional
import asyncio
from bson import ObjectId

# Configure logger
logger = logging.getLogger("regression_studio.database")
logging.basicConfig(level=logging.INFO)

# Try importing motor/pymongo; fall back if imports fail completely
try:
    from motor.motor_asyncio import AsyncIOMotorClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    MOTOR_AVAILABLE = True
except ImportError:
    MOTOR_AVAILABLE = False
    logger.warning("motor or pymongo is not installed. Database will use JSON file fallback.")

class DatabaseAdapter:
    def __init__(self):
        self.mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        self.db_name = os.getenv("DATABASE_NAME", "regression_studio")
        self.fallback_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
        self.fallback_path = os.path.join(self.fallback_dir, "db.json")
        self.use_fallback = True
        self.client = None
        self.db = None
        # In-memory cache of the fallback JSON store, populated on first
        # access and kept in sync on every write. Before this, every single
        # fallback CRUD call — including the per-request current-user lookup
        # every authenticated endpoint does — re-read and json.load()'d the
        # *entire* file from disk, so cost scaled with total accumulated
        # history (all datasets/models/reports ever saved), not with the
        # current request. On a ~65MB dev store that was ~1s per read, on
        # top of a full pretty-printed rewrite (~7s) per write — dominating
        # total request time regardless of how small the current dataset
        # was. Safe for the single-process dev use this fallback exists for;
        # an external process editing db.json while the server is running
        # (e.g. running tests/test_pipeline.py against a live server) won't
        # be picked up until restart, same caveat as any in-memory cache.
        self._fallback_cache: Optional[Dict[str, List[Dict[str, Any]]]] = None

        # Ensure fallback directory exists
        os.makedirs(self.fallback_dir, exist_ok=True)
        if not os.path.exists(self.fallback_path):
            with open(self.fallback_path, "w") as f:
                json.dump({"users": [], "datasets": [], "models": [], "reports": []}, f, indent=4)
                
    async def connect(self):
        """Attempts to connect to MongoDB. Falls back to file-based JSON database if connection fails."""
        if not MOTOR_AVAILABLE:
            logger.warning("Motor is unavailable. Operating in JSON fallback mode.")
            self.use_fallback = True
            return
            
        try:
            logger.info(f"Connecting to MongoDB at {self.mongodb_uri}...")
            # Try to connect with a short 2-second timeout
            self.client = AsyncIOMotorClient(self.mongodb_uri, serverSelectionTimeoutMS=2000)
            # Force a connection check
            await self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            self.use_fallback = False
            logger.info("Successfully connected to MongoDB!")
        except Exception as e:
            logger.warning(f"MongoDB connection failed: {e}. Falling back to local file database: {self.fallback_path}")
            self.use_fallback = True
            self.client = None
            self.db = None

    def _read_fallback(self) -> Dict[str, List[Dict[str, Any]]]:
        if self._fallback_cache is not None:
            return self._fallback_cache
        try:
            with open(self.fallback_path, "r") as f:
                self._fallback_cache = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read fallback file: {e}")
            self._fallback_cache = {"users": [], "datasets": [], "models": [], "reports": []}
        return self._fallback_cache

    def _write_fallback(self, data: Dict[str, List[Dict[str, Any]]]):
        self._fallback_cache = data
        try:
            # Compact (no indent) — serializing with indent=4 is meaningfully
            # slower on a large store and produces a bigger file, which then
            # makes every future read/write slower too. Formatting only ever
            # mattered for manually inspecting db.json; nothing parses it
            # expecting pretty-printing.
            with open(self.fallback_path, "w") as f:
                json.dump(data, f, separators=(",", ":"))
        except Exception as e:
            logger.error(f"Failed to write fallback file: {e}")

    # Generic CRUD operations
    async def insert_one(self, collection_name: str, document: Dict[str, Any]) -> str:
        """Inserts a document into the specified collection and returns its ID."""
        if not self.use_fallback and self.db is not None:
            try:
                result = await self.db[collection_name].insert_one(document)
                return str(result.inserted_id)
            except Exception as e:
                logger.error(f"MongoDB insert error: {e}. Attempting fallback.")
                
        # Fallback implementation
        data = self._read_fallback()
        if collection_name not in data:
            data[collection_name] = []
            
        # Ensure document has a string id
        if "_id" not in document:
            document["_id"] = str(ObjectId())
        elif isinstance(document["_id"], ObjectId):
            document["_id"] = str(document["_id"])
            
        data[collection_name].append(document)
        self._write_fallback(data)
        return document["_id"]

    async def find_one(self, collection_name: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Finds a single document matching the query."""
        if not self.use_fallback and self.db is not None:
            try:
                # Handle string to ObjectId conversion if _id is queried
                mongo_query = dict(query)
                if "_id" in mongo_query and isinstance(mongo_query["_id"], str):
                    try:
                        mongo_query["_id"] = ObjectId(mongo_query["_id"])
                    except Exception:
                        pass # Keep as string if it's not a valid ObjectId
                doc = await self.db[collection_name].find_one(mongo_query)
                if doc:
                    doc["_id"] = str(doc["_id"])
                    return doc
                return None
            except Exception as e:
                logger.error(f"MongoDB find_one error: {e}. Attempting fallback.")

        # Fallback implementation
        data = self._read_fallback()
        collection = data.get(collection_name, [])
        for doc in collection:
            match = True
            for k, v in query.items():
                # Handle matching of _id
                doc_val = doc.get(k)
                if doc_val != v:
                    match = False
                    break
            if match:
                return doc
        return None

    async def find_many(self, collection_name: str, query: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Finds multiple documents matching the query."""
        if query is None:
            query = {}
            
        if not self.use_fallback and self.db is not None:
            try:
                mongo_query = dict(query)
                if "_id" in mongo_query and isinstance(mongo_query["_id"], str):
                    try:
                        mongo_query["_id"] = ObjectId(mongo_query["_id"])
                    except Exception:
                        pass
                cursor = self.db[collection_name].find(mongo_query)
                docs = await cursor.to_list(length=1000)
                for d in docs:
                    d["_id"] = str(d["_id"])
                return docs
            except Exception as e:
                logger.error(f"MongoDB find_many error: {e}. Attempting fallback.")

        # Fallback implementation
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
        if not self.use_fallback and self.db is not None:
            try:
                mongo_query = dict(query)
                if "_id" in mongo_query and isinstance(mongo_query["_id"], str):
                    try:
                        mongo_query["_id"] = ObjectId(mongo_query["_id"])
                    except Exception:
                        pass
                result = await self.db[collection_name].update_one(mongo_query, update)
                return result.modified_count > 0
            except Exception as e:
                logger.error(f"MongoDB update_one error: {e}. Attempting fallback.")

        # Fallback implementation
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
        if not self.use_fallback and self.db is not None:
            try:
                mongo_query = dict(query)
                if "_id" in mongo_query and isinstance(mongo_query["_id"], str):
                    try:
                        mongo_query["_id"] = ObjectId(mongo_query["_id"])
                    except Exception:
                        pass
                result = await self.db[collection_name].delete_one(mongo_query)
                return result.deleted_count > 0
            except Exception as e:
                logger.error(f"MongoDB delete_one error: {e}. Attempting fallback.")

        # Fallback implementation
        data = self._read_fallback()
        collection = data.get(collection_name, [])
        initial_len = len(collection)
        
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
                continue # Skip this document (delete it)
            filtered_collection.append(doc)
            
        data[collection_name] = filtered_collection
        if deleted:
            self._write_fallback(data)
        return deleted

# Singleton instance
db_client = DatabaseAdapter()
