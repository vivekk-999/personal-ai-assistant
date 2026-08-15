
import os
from pathlib import Path
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BACKEND_DIR / ".env", override=False)

DEFAULT_MONGODB_URI = "mongodb://localhost:27017"


def get_env_value(name, default=None):
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().strip('"').strip("'")


UPLOAD_FOLDER = get_env_value("UPLOAD_FOLDER", str(BACKEND_DIR / "uploads"))
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def get_mongodb_uri() -> str:
    for key in ("MONGODB_URI", "MONGO_URI"):
        value = get_env_value(key)
        if value:
            return value
    return DEFAULT_MONGODB_URI


def is_placeholder_mongodb_uri(uri: str) -> bool:
    if not uri:
        return True
    normalized = uri.strip().lower()
    placeholder_fragments = (
        "<db_password>",
        "<password>",
        "your_mongodb_atlas_connection_string",
        "your_password",
        "replace_me",
    )
    return any(fragment in normalized for fragment in placeholder_fragments) or "<" in normalized or ">" in normalized


class DatabaseUnavailableError(Exception):
    """Raised when MongoDB connection cannot be established or configured."""
    pass


class Database:
    def __init__(self):
        self.uri = get_mongodb_uri()
        self.db_name = get_env_value("DB_NAME", "rag_assistant")
        self.client = None
        self.db = None
        self.sync_client = None
        self.sync_db = None
        self.mode = "uninitialized"

    def _ensure_valid_uri(self):
        if is_placeholder_mongodb_uri(self.uri):
            raise DatabaseUnavailableError(
                "MONGODB_URI still contains a placeholder value. "
                "Update backend/.env with your real MongoDB Atlas credentials, or set a working local MongoDB URI."
            )

    async def connect(self):
        self._ensure_valid_uri()

        if self.client is None or self.db is None:
            client = AsyncIOMotorClient(self.uri, serverSelectionTimeoutMS=5000)
            try:
                await client.admin.command("ping")
            except Exception as exc:
                client.close()
                self.client = None
                self.db = None
                raise DatabaseUnavailableError(
                    "MongoDB connection failed. Verify MONGODB_URI, network access, and Atlas IP allow-list."
                ) from exc

            self.client = client
            self.db = client[self.db_name]
            self.mode = "mongo"

            try:
                indexes = await self.db["documents"].index_information()
                if "filename_1" in indexes and indexes["filename_1"].get("unique"):
                    await self.db["documents"].drop_index("filename_1")
                    await self.db["documents"].create_index("filename", background=True)
                if "document_id_1" not in indexes:
                    await self.db["documents"].create_index("document_id", unique=True, sparse=True, background=True)
                
                # Page Store Indexes
                page_indexes = await self.db["pages"].index_information()
                if "doc_page_idx" not in page_indexes:
                    await self.db["pages"].create_index(
                        [("document_id", 1), ("page_number", 1)],
                        name="doc_page_idx",
                        unique=True,
                        background=True
                    )
                if "document_id_1" not in page_indexes:
                    await self.db["pages"].create_index("document_id", background=True)

                # Chunk Store Indexes
                chunk_indexes = await self.db["chunks"].index_information()
                if "chunk_doc_idx" not in chunk_indexes:
                    await self.db["chunks"].create_index("document_id", name="chunk_doc_idx", background=True)
                if "chunk_page_idx" not in chunk_indexes:
                    await self.db["chunks"].create_index("page_number", name="chunk_page_idx", background=True)
                if "chunk_id_1" not in chunk_indexes:
                    await self.db["chunks"].create_index("chunk_id", background=True)
            except Exception as ie:
                print(f"[DB NOTICE] Index setup: {ie}")

        return self.db


    def get_sync_db(self):
        self._ensure_valid_uri()
        if self.sync_client is None or self.sync_db is None:
            client = MongoClient(self.uri, serverSelectionTimeoutMS=5000)
            try:
                client.admin.command("ping")
            except Exception as exc:
                client.close()
                self.sync_client = None
                self.sync_db = None
                raise DatabaseUnavailableError(
                    "MongoDB connection failed. Verify MONGODB_URI, network access, and Atlas IP allow-list."
                ) from exc

            self.sync_client = client
            self.sync_db = client[self.db_name]
            self.mode = "mongo"
        return self.sync_db

    async def disconnect(self):
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
        if self.sync_client:
            self.sync_client.close()
            self.sync_client = None
            self.sync_db = None

    @property
    def is_local(self) -> bool:
        """True when the URI points at a local MongoDB (not Atlas)."""
        return "mongodb+srv" not in self.uri and "mongodb.net" not in self.uri

    def get_sync_collection(self, name):
        """Return a synchronous PyMongo collection.

        ``MongoDBAtlasVectorSearch`` requires a synchronous ``pymongo.Collection``,
        not a Motor async collection.  This helper lazily initialises the sync
        client so callers can simply ask for the collection by name.
        """
        sync_db = self.get_sync_db()
        return sync_db[name]

    def get_collection(self, name):
        if self.db is None:
            raise Exception("Database not connected. Call connect() first.")
        return self.db[name]

# Singleton instance
db_instance = Database()

async def get_db():
    return await db_instance.connect()
