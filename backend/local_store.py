import asyncio
import json
import math
import uuid
import hashlib
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.documents import Document


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _serialize_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return {"__type__": "datetime", "value": value.isoformat()}
    if isinstance(value, dict):
        return {key: _serialize_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    return value


def _deserialize_value(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get("__type__") == "datetime":
            return datetime.fromisoformat(value["value"])
        return {key: _deserialize_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_deserialize_value(item) for item in value]
    return value


def _normalize_query_value(value: Any) -> Any:
    return str(value) if value is not None else None


def _get_nested_value(document: Dict[str, Any], key: str) -> Any:
    current = document
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _matches_query(document: Dict[str, Any], query: Optional[Dict[str, Any]]) -> bool:
    if not query:
        return True

    for key, condition in query.items():
        if key == "$or":
            return any(_matches_query(document, clause) for clause in condition)

        value = _get_nested_value(document, key)
        if isinstance(condition, dict):
            if "$in" in condition:
                if _normalize_query_value(value) not in {_normalize_query_value(item) for item in condition["$in"]}:
                    return False
                continue
        if _normalize_query_value(value) != _normalize_query_value(condition):
            return False

    return True


def _apply_projection(document: Dict[str, Any], projection: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not projection:
        return deepcopy(document)

    projected = {}
    include_id = projection.get("_id", 1) not in (0, False)
    for key, enabled in projection.items():
        if key == "_id" or not enabled:
            continue
        value = _get_nested_value(document, key)
        if value is not None:
            projected[key] = deepcopy(value)

    if include_id and "_id" in document:
        projected["_id"] = deepcopy(document["_id"])
    return projected


def _sort_documents(documents: List[Dict[str, Any]], sort_spec: List[tuple[str, int]]) -> List[Dict[str, Any]]:
    ordered = list(documents)

    def sort_key(document: Dict[str, Any]):
        keys = []
        for field, direction in sort_spec:
            value = _get_nested_value(document, field)
            if isinstance(value, datetime):
                value = value.timestamp()
            keys.append(value)
        return tuple(keys)

    for index in reversed(range(len(sort_spec))):
        field, direction = sort_spec[index]

        def key_fn(document: Dict[str, Any], field_name=field):
            value = _get_nested_value(document, field_name)
            if isinstance(value, datetime):
                return value.timestamp()
            return (value is None, value)

        ordered.sort(key=key_fn, reverse=direction < 0)

    return ordered


class LocalCursor:
    def __init__(self, documents: List[Dict[str, Any]]):
        self._documents = documents

    def sort(self, sort_spec: List[tuple[str, int]]):
        self._documents = _sort_documents(self._documents, sort_spec)
        return self

    async def to_list(self, length: Optional[int] = None):
        if length is None:
            return [deepcopy(doc) for doc in self._documents]
        return [deepcopy(doc) for doc in self._documents[:length]]


class LocalInsertOneResult:
    def __init__(self, inserted_id: str):
        self.inserted_id = inserted_id


class LocalInsertManyResult:
    def __init__(self, inserted_ids: List[str]):
        self.inserted_ids = inserted_ids


class LocalUpdateResult:
    def __init__(self, modified_count: int, upserted_id: Optional[str] = None):
        self.modified_count = modified_count
        self.upserted_id = upserted_id


class LocalDeleteResult:
    def __init__(self, deleted_count: int):
        self.deleted_count = deleted_count


class LocalCollection:
    def __init__(self, database: "LocalDatabase", name: str):
        self.database = database
        self.name = name

    def _documents(self) -> List[Dict[str, Any]]:
        return self.database._data.setdefault(self.name, [])

    def _persist(self) -> None:
        self.database._persist()

    async def insert_one(self, document: Dict[str, Any]):
        doc = deepcopy(document)
        doc.setdefault("_id", uuid.uuid4().hex)
        self._documents().append(doc)
        self._persist()
        return LocalInsertOneResult(doc["_id"])

    async def insert_many(self, documents: Iterable[Dict[str, Any]]):
        inserted_ids = []
        for document in documents:
            doc = deepcopy(document)
            doc.setdefault("_id", uuid.uuid4().hex)
            self._documents().append(doc)
            inserted_ids.append(doc["_id"])
        self._persist()
        return LocalInsertManyResult(inserted_ids)

    async def find_one(self, query: Optional[Dict[str, Any]] = None, projection: Optional[Dict[str, Any]] = None):
        for document in self._documents():
            if _matches_query(document, query):
                return _apply_projection(document, projection)
        return None

    def find(self, query: Optional[Dict[str, Any]] = None, projection: Optional[Dict[str, Any]] = None):
        matches = [
            _apply_projection(document, projection)
            for document in self._documents()
            if _matches_query(document, query)
        ]
        return LocalCursor(matches)

    async def update_one(self, query: Dict[str, Any], update: Dict[str, Any], upsert: bool = False):
        for document in self._documents():
            if _matches_query(document, query):
                modified = 0
                if "$set" in update:
                    document.update(deepcopy(update["$set"]))
                    modified = 1
                self._persist()
                return LocalUpdateResult(modified)

        if not upsert:
            return LocalUpdateResult(0)

        new_document = deepcopy(query) if isinstance(query, dict) else {}
        if "$set" in update:
            new_document.update(deepcopy(update["$set"]))
        new_document.setdefault("_id", uuid.uuid4().hex)
        self._documents().append(new_document)
        self._persist()
        return LocalUpdateResult(1, upserted_id=new_document["_id"])

    async def delete_many(self, query: Optional[Dict[str, Any]] = None):
        original = self._documents()
        remaining = [document for document in original if not _matches_query(document, query)]
        deleted_count = len(original) - len(remaining)
        self.database._data[self.name] = remaining
        self._persist()
        return LocalDeleteResult(deleted_count)

    async def delete_one(self, query: Optional[Dict[str, Any]] = None):
        documents = self._documents()
        for index, document in enumerate(documents):
            if _matches_query(document, query):
                del documents[index]
                self._persist()
                return LocalDeleteResult(1)
        return LocalDeleteResult(0)

    async def create_index(self, *args, **kwargs):
        return None


class LocalDatabase:
    def __init__(self, name: str, storage_path: Path):
        self.name = name
        self.storage_path = storage_path
        self._data: Dict[str, List[Dict[str, Any]]] = {}
        self._load()

    def _load(self) -> None:
        if not self.storage_path.exists():
            self._data = {}
            return
        try:
            raw = json.loads(self.storage_path.read_text())
            self._data = {
                collection: [_deserialize_value(doc) for doc in documents]
                for collection, documents in raw.get("collections", {}).items()
            }
        except Exception:
            self._data = {}

    def _persist(self) -> None:
        payload = {
            "collections": {
                collection: [_serialize_value(doc) for doc in documents]
                for collection, documents in self._data.items()
            }
        }
        self.storage_path.write_text(json.dumps(payload, indent=2, default=str))

    def __getitem__(self, name: str) -> LocalCollection:
        return LocalCollection(self, name)

    def __getattr__(self, name: str) -> LocalCollection:
        if name.startswith("_"):
            raise AttributeError(name)
        return self[name]


class LocalAdmin:
    async def command(self, command_name: str):
        if command_name != "ping":
            return {"ok": 1}
        return {"ok": 1}


class LocalMongoClient:
    def __init__(self, db_name: str, storage_path: Path):
        self.db_name = db_name
        self.storage_path = storage_path
        self._database = LocalDatabase(db_name, storage_path)
        self.admin = LocalAdmin()

    def __getitem__(self, name: str) -> LocalDatabase:
        if name != self.db_name:
            self.db_name = name
            self._database = LocalDatabase(name, self.storage_path)
        return self._database

    def close(self) -> None:
        return None


class LocalVectorRetriever:
    def __init__(self, store: "LocalVectorStore", k: int = 5):
        self.store = store
        self.k = k

    def invoke(self, query: str):
        return self.store.similarity_search(query, self.k)

    async def ainvoke(self, query: str):
        return await asyncio.to_thread(self.invoke, query)


class LocalVectorStore:
    def __init__(self, collection: LocalCollection, embedding):
        self.collection = collection
        self.embedding = embedding

    def add_documents(self, documents: List[Document]):
        texts = [document.page_content for document in documents]
        embeddings = self.embedding.embed_documents(texts)
        payloads = []
        for document, embedding in zip(documents, embeddings):
            metadata = deepcopy(document.metadata or {})
            source = metadata.get("source")
            payloads.append({
                "_id": uuid.uuid4().hex,
                "content": document.page_content,
                "metadata": metadata,
                "source": source,
                "embedding": embedding,
                "created_at": _now_iso(),
            })
        for payload in payloads:
            self.collection._documents().append(payload)
        self.collection._persist()

    def delete(self, ids: List[str]):
        documents = self.collection._documents()
        remaining = [doc for doc in documents if str(doc.get("_id")) not in {str(item) for item in ids}]
        self.collection.database._data[self.collection.name] = remaining
        self.collection._persist()

    def similarity_search(self, query: str, k: int = 5):
        docs = self.collection._documents()
        if not docs:
            return []
        query_embedding = self.embedding.embed_query(query)
        scored = []
        for document in docs:
            embedding = document.get("embedding")
            if not embedding:
                continue
            score = self._cosine_similarity(query_embedding, embedding)
            scored.append((score, document))
        scored.sort(key=lambda item: item[0], reverse=True)
        results = []
        for _, document in scored[:k]:
            results.append(Document(page_content=document.get("content", ""), metadata=document.get("metadata", {})))
        return results

    def as_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        k = (search_kwargs or {}).get("k", 5)
        return LocalVectorRetriever(self, k=k)

    @staticmethod
    def _cosine_similarity(left: List[float], right: List[float]) -> float:
        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if not left_norm or not right_norm:
            return 0.0
        return numerator / (left_norm * right_norm)


class LocalHashEmbeddings:
    """Offline embedding fallback that maps tokens into a stable hashed vector."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def _embed(self, text: str) -> List[float]:
        vector = [0.0] * self.dimension
        tokens = _tokenize(text)
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self.dimension
            vector[index] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]
