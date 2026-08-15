import warnings
warnings.filterwarnings("ignore")

import os
import uuid
import hashlib
import shutil
import json
import base64
import binascii
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Dict, Optional, Any, Union

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from bson import ObjectId
import re
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage

try:
    from .database import db_instance, get_db, get_env_value, UPLOAD_FOLDER
    from .rag_engine import rag_engine, format_page_ranges
except ImportError:
    from database import db_instance, get_db, get_env_value, UPLOAD_FOLDER
    from rag_engine import rag_engine, format_page_ranges
from langchain_groq import ChatGroq

# Load environment variables
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)
#--------------------------------------------------
os.environ["USER_AGENT"] = "Mozilla/5.0"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    print("Starting RAG Engine...")
    import traceback
    try:
        db = await db_instance.connect()
        if db_instance.mode == "mongo":
            try:
                await db["documents"].create_index("document_id", unique=True, sparse=True)
                await db["documents"].create_index("filename")
                await db["history"].create_index([("conversation_id", 1), ("created_at", 1)])
                await db["chunks"].create_index([("document_id", 1), ("source", 1)])
            except Exception as ie:
                print(f"[STARTUP] MongoDB index setup notice: {ie}")


        print("MongoDB connected.")
        await rag_engine.initialize()
        print("RAG Engine initialized.")
    except Exception as e:
        traceback.print_exc()
        print(f"Initialization failed: {e}")
        raise
    yield
    # Shutdown logic
    await db_instance.disconnect()
    print("Backend stopped.")

app = FastAPI(title="Personal AI", lifespan=lifespan)

# CORS Configuration — support local development across localhost, 127.0.0.1, and custom origins
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174,http://localhost:4173,http://127.0.0.1:4173,http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".xlsx", ".xls", ".md", ".csv"}
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
MAX_IMAGE_SIZE_BYTES = 8 * 1024 * 1024  # 8 MB before base64 encoding
ALLOWED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}

class ImageInput(BaseModel):
    data: str
    mime_type: str
    name: Optional[str] = None


class ChatRequest(BaseModel):
    message: str = ""
    conversation_id: str
    image: Optional[ImageInput] = None
    document_ids: Optional[List[str]] = None
    document_id: Optional[str] = None
    document_name: Optional[str] = None
    document_names: Optional[List[str]] = None
    response_mode: Optional[str] = "balanced"  # fast, balanced, deep
    edit_turn_id: Optional[str] = None


class ConversationUpdate(BaseModel):
    title: Optional[str] = None
    pinned: Optional[bool] = None
    archived: Optional[bool] = None
    selected_document_ids: Optional[List[str]] = None


class ConversationCreate(BaseModel):
    title: Optional[str] = None
    selected_document_ids: Optional[List[str]] = None
    document_name: Optional[str] = None


def conversation_filter(conversation_id: str) -> Dict:
    """Build lookup query for a persisted MongoDB conversation by ObjectId or string ID."""
    if not conversation_id:
        raise HTTPException(status_code=400, detail="A conversation_id is required.")
    
    if ObjectId.is_valid(conversation_id):
        return {"$or": [
            {"_id": ObjectId(conversation_id)},
            {"_id": conversation_id},
            {"session_id": conversation_id},
            {"conversation_id": conversation_id}
        ]}
    else:
        return {"$or": [
            {"_id": conversation_id},
            {"session_id": conversation_id},
            {"conversation_id": conversation_id}
        ]}


def sanitize_document_identifier(ident: Any) -> str:
    """Strip trailing page annotations (e.g. '(p. 4, 13-16, 19)'), quotes, and whitespace.
    
    Guarantees clean, canonical document identifiers for database lookups and routing.
    """
    if not ident or not isinstance(ident, str):
        return ""
    cleaned = ident.strip().strip("'\"`")
    # Remove parenthetical page range indicators like (p. 1-4), (p. 4, 13-16, 19), (pages 2-3), (page 5)
    cleaned = re.sub(r'\s*\(\s*(?:pages?|p\.)\s*[\d\s,–-]+(?:\s*total\s*:\s*\d+\s*pages?)?\s*\)', '', cleaned, flags=re.IGNORECASE).strip()
    return cleaned


def get_document_storage_path(doc: Optional[Dict]) -> Optional[str]:
    """Resolve physical document file path on disk across storage_path, file_path, disk_filename, and filename variants."""
    if not doc or not isinstance(doc, dict):
        return None
    candidates = [
        doc.get("storage_path"),
        doc.get("file_path"),
        os.path.join(UPLOAD_FOLDER, doc.get("disk_filename", "")) if doc.get("disk_filename") else None,
        os.path.join(UPLOAD_FOLDER, f"{doc.get('document_id', '')}_{doc.get('filename', '')}") if doc.get("document_id") and doc.get("filename") else None,
        os.path.join(UPLOAD_FOLDER, f"{doc.get('document_id', '')}_{re.sub(r'[^a-zA-Z0-9_.-]', '_', os.path.basename(doc.get('filename', '')))}") if doc.get("document_id") and doc.get("filename") else None,
        os.path.join(UPLOAD_FOLDER, doc.get("filename", "")) if doc.get("filename") else None,
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return os.path.abspath(c)
    return None


async def require_conversation(conversation_id: str) -> Dict:
    db = await get_db()
    conversation = await db["conversations"].find_one(conversation_filter(conversation_id))
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found. Create a new chat before sending a message.")
    return conversation


async def require_ready_documents(document_identifiers: Optional[List[str]]) -> List[Dict]:
    """Verify all requested documents exist, are ready, have chunks/pages, and auto-reindex if needed."""
    if not document_identifiers:
        return []
    db = await get_db()
    resolved_documents = []
    seen_doc_ids = set()

    for raw_ident in document_identifiers:
        ident = sanitize_document_identifier(raw_ident)
        if not ident:
            continue
        
        # Comprehensive lookup query supporting UUID, clean filename, disk filename, and case-insensitive matching
        doc_q = {"$or": [
            {"document_id": ident},
            {"filename": ident},
            {"disk_filename": ident},
            {"filename": {"$regex": f"^{re.escape(ident)}$", "$options": "i"}}
        ]}
        if ObjectId.is_valid(ident):
            doc_q["$or"].append({"_id": ObjectId(ident)})
        doc_q["$or"].append({"_id": ident})

        document = await db["documents"].find_one(doc_q)

        # If missing in DB, check if file exists on disk and auto-register / index it
        if not document:
            possible_path = os.path.join(UPLOAD_FOLDER, ident)
            if os.path.isfile(possible_path):
                print(f"[RETRIEVAL] [SELF_HEAL] Document record missing for '{ident}', but physical file exists. Auto-registering and indexing...")
                doc_id = str(uuid.uuid4())
                file_size = os.path.getsize(possible_path)
                ext_clean = os.path.splitext(ident)[1].replace('.', '').upper() or "FILE"
                now = datetime.now()
                new_record = {
                    "_id": doc_id,
                    "document_id": doc_id,
                    "filename": ident,
                    "disk_filename": ident,
                    "storage_path": possible_path,
                    "file_path": possible_path,
                    "file_type": ext_clean,
                    "size_bytes": file_size,
                    "status": "PROCESSING",
                    "stage": "EXTRACTING",
                    "created_at": now,
                    "uploaded_at": now,
                    "updated_at": now
                }
                await db["documents"].insert_one(new_record)
                await rag_engine.process_file(possible_path, ident, document_id=doc_id, rebuild=True)
                document = await db["documents"].find_one({"document_id": doc_id})

        if not document:
            print(f"[RETRIEVAL] [NOT_FOUND] Document '{raw_ident}' (sanitized: '{ident}') not found in database or storage.")
            raise HTTPException(
                status_code=404,
                detail=f"Document '{ident}' was not found in the database. Please upload it first."
            )

        doc_id = document.get("document_id") or str(document.get("_id"))
        fn = document.get("filename", ident)

        if doc_id in seen_doc_ids:
            continue

        status = str(document.get("status") or "").upper()
        stage = document.get("stage") or document.get("status") or "Processing"

        print(f"[DOCUMENT_STATUS] document_id={doc_id} filename='{fn}' status={status} stage={stage}")

        if status == "FAILED":
            err = document.get("error") or "Document indexing failed."
            print(f"[RETRIEVAL] [FAILED_DOC] document_id={doc_id} filename='{fn}' error='{err}'")
            raise HTTPException(
                status_code=409,
                detail=f"Document '{fn}' indexing failed: {err}. Please reprocess or re-upload the document."
            )

        if status in ("PROCESSING", "UPLOADING"):
            print(f"[RETRIEVAL] [PROCESSING_DOC] document_id={doc_id} filename='{fn}' stage='{stage}'")
            raise HTTPException(
                status_code=409,
                detail=f"Document '{fn}' is currently being indexed (Stage: {stage}). Please wait a few seconds and retry."
            )

        # Check physical file existence
        storage_path = get_document_storage_path(document)
        if not storage_path or not os.path.isfile(storage_path):
            print(f"[RETRIEVAL] [MISSING_FILE] document_id={doc_id} filename='{fn}' storage_path='{storage_path}'")
            raise HTTPException(
                status_code=409,
                detail=f"Document '{fn}' physical file is missing from storage. Please re-upload the document."
            )

        # Check indexed chunks and pages parity
        chunk_count = await db["chunks"].count_documents({"$or": [{"document_id": doc_id}, {"metadata.document_id": doc_id}]})
        page_count = await db["pages"].count_documents({"document_id": doc_id})

        # Self-healing: If status is READY but chunks or pages are missing, auto-reindex idempotently
        if chunk_count == 0 or page_count == 0:
            print(f"[RETRIEVAL] [SELF_HEAL] Document {doc_id} ('{fn}') marked READY but has 0 chunks/pages in MongoDB. Auto-rebuilding index idempotently...")
            await rag_engine.process_file(storage_path, fn, document_id=doc_id, rebuild=True)
            document = await db["documents"].find_one({"document_id": doc_id})
            chunk_count = await db["chunks"].count_documents({"$or": [{"document_id": doc_id}, {"metadata.document_id": doc_id}]})

        # Ensure in-memory BM25 cache has the document's chunks
        if not any(d.metadata.get("document_id") == doc_id for d in rag_engine.all_docs):
            print(f"[RETRIEVAL] [CACHE_SYNC] Refreshing RAG Engine in-memory cache for document_id={doc_id} ('{fn}')...")
            await rag_engine.rebuild_chain()

        seen_doc_ids.add(doc_id)
        resolved_documents.append(document)

    return resolved_documents


plain_chat_llm = None
vision_chat_llm = None
GROQ_MODEL = get_env_value("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_VISION_MODEL = get_env_value("GROQ_VISION_MODEL", "llama-3.2-11b-vision-preview")

SYSTEM_PROMPT = """You are Personal AI — an elite, highly capable, and exceptionally accurate personal assistant.

Your core principles:
- Extreme accuracy: Never guess or invent information. If you are unsure, clearly say so.
- Professional excellence: Respond with clarity, precision, and high-quality structure.
- Helpfulness: Always aim to give the most useful and complete answer possible.
- Memory: Carefully remember important details the user shares within the conversation and use them when relevant.
- Honesty: Do not hallucinate files, previous conversations, or facts.
- Tone: Professional, calm, intelligent, and slightly warm. Avoid being robotic or overly casual.

When answering:
- Structure complex answers clearly (use headings, bullet points, or numbered lists when helpful).
- Prioritize correctness over speed or length.
- If a question is ambiguous, ask a clarifying question instead of assuming.

You are designed to be one of the most reliable and high-performing personal AI assistants available."""

THINKING_START = "<" + "think" + ">"
THINKING_END = "<" + "/" + "think" + ">"


class ThinkingTokenFilter:
    """Strip reasoning-model `` blocks from streamed tokens."""

    def __init__(self):
        self.inside_thinking = False
        self.pending = ""

    def process(self, token: str) -> str:
        if not token:
            return ""

        self.pending += token
        output = []

        while self.pending:
            if self.inside_thinking:
                end_idx = self.pending.find(THINKING_END)
                if end_idx == -1:
                    break
                self.pending = self.pending[end_idx + len(THINKING_END):]
                self.inside_thinking = False
                continue

            start_idx = self.pending.find(THINKING_START)
            if start_idx == -1:
                keep_back = len(THINKING_START) - 1
                if len(self.pending) <= keep_back:
                    break
                output.append(self.pending[:-keep_back])
                self.pending = self.pending[-keep_back:]
                break

            if start_idx > 0:
                output.append(self.pending[:start_idx])
            self.pending = self.pending[start_idx + len(THINKING_START):]
            self.inside_thinking = True

        return "".join(output)

    def flush(self) -> str:
        if self.inside_thinking:
            self.pending = ""
            return ""
        remaining = self.pending
        self.pending = ""
        return remaining


def strip_thinking_content(text: str) -> str:
    if not text:
        return text
    cleaned = re.sub(
        rf"{re.escape(THINKING_START)}.*?{re.escape(THINKING_END)}",
        "",
        text,
        flags=re.DOTALL,
    )
    return cleaned.strip()


def get_plain_chat_llm():
    global plain_chat_llm
    if plain_chat_llm:
        return plain_chat_llm

    api_key = get_env_value("GROQ_API_KEY")
    if not api_key:
        return None
    plain_chat_llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=api_key,
        temperature=0.6,
        timeout=60,
        max_retries=2,
    )
    return plain_chat_llm


async def invoke_llm_with_fallback(llm, prompt):
    """Invoke LLM with automatic fallback to llama-3.1-8b-instant if 429 RateLimit occurs."""
    try:
        print(f"[LLM] Provider: Groq | Model: {GROQ_MODEL} | API key variable: GROQ_API_KEY")
        return await llm.ainvoke(prompt)
    except Exception as error:
        error_str = str(error).lower()
        if '429' in error_str or 'rate_limit' in error_str or 'rate limit' in error_str or 'tokens' in error_str:
            print("[LLM FALLBACK NOTICE] Primary Groq model hit rate limit. Falling back to llama-3.1-8b-instant...")
            api_key = get_env_value("GROQ_API_KEY")
            fallback_llm = ChatGroq(model="llama-3.1-8b-instant", api_key=api_key, temperature=0.6, timeout=30, max_retries=1)
            return await fallback_llm.ainvoke(prompt)
        raise


def parse_llm_exception(error: Exception) -> tuple[int, str]:
    """Classify exceptions accurately into (status_code, user_friendly_message) without masking exact causes."""
    error_str = str(error)
    error_lower = error_str.lower()
    err_type = type(error).__name__.lower()

    # 1. 401/403 Authentication / Invalid API Key
    if '401' in error_lower or '403' in error_lower or 'invalid api key' in error_lower or 'invalid_api_key' in error_lower or 'authenticationerror' in err_type or 'unauthorized' in error_lower or 'authentication failed' in error_lower:
        return 401, "AI provider authentication failed. Check your GROQ_API_KEY."

    # 2. 404 Model Unavailable / Model Not Found
    if '404' in error_lower or 'model_not_found' in error_lower or 'does not exist' in error_lower or 'unknown model' in error_lower:
        return 404, "Configured AI model is unavailable. Check the model configuration."

    # 3. Quota Exceeded
    if 'quota' in error_lower or 'insufficient_quota' in error_lower or 'credit' in error_lower:
        return 429, "AI API quota has been exceeded. Check your provider billing/usage."

    # 4. 429 Rate Limit
    if '429' in error_lower or 'rate_limit' in error_lower or 'rate limit' in error_lower or 'ratelimit' in err_type:
        return 429, "AI provider rate limit reached. Please try again shortly."

    # 5. Timeout / 504
    if 'timeout' in error_lower or 'timed out' in error_lower or 'timeouterror' in err_type or '504' in error_lower:
        return 504, "The AI request timed out. Please try again."

    # 6. 413 Payload Too Large / Context Limit
    if '413' in error_lower or 'too large' in error_lower or 'context_length_exceeded' in error_lower:
        return 413, "That request is too large or exceeds the model context limit."

    # 7. 400 Bad Request / Invalid Parameters
    if '400' in error_lower or 'bad_request' in error_lower or 'invalid_request' in error_lower:
        return 400, "Invalid request parameters or model payload."

    # 8. 5xx Provider Error
    if '500' in error_lower or '502' in error_lower or '503' in error_lower or 'service unavailable' in error_lower or 'apiconnectionerror' in err_type or 'internal server error' in error_lower:
        return 503, "AI provider is temporarily unavailable."

    # 9. Database / RAG error
    if 'pymongo' in err_type or 'mongo' in error_lower or 'database' in error_lower:
        return 500, "Document retrieval failed. Please check the RAG service."

    # Default fallback
    return 500, f"Something went wrong while generating the response: {error_str[:120]}"


async def stream_plain_reply(message: str, chat_history: Optional[List[BaseMessage]] = None, response_mode: str = "balanced"):
    """Stream plain chat response token-by-token with response mode temperature and token configuration."""
    api_key = get_env_value("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to backend/.env and restart the backend.")

    mode = (response_mode or "balanced").lower()
    temp = 0.3 if mode == "fast" else (0.7 if mode == "deep" else 0.6)
    max_tok = 1024 if mode == "fast" else (4096 if mode == "deep" else 2048)

    prompt_extra = ""
    if mode == "fast":
        prompt_extra = "\n\nProvide a quick, concise, and direct response focusing on essential facts."
    elif mode == "deep":
        prompt_extra = "\n\nEngage in deep, comprehensive analytical reasoning. Break down complex concepts step-by-step and provide thorough explanations."

    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=api_key,
        temperature=temp,
        max_tokens=max_tok,
        timeout=60,
        max_retries=2,
    )

    messages = [SystemMessage(content=f"{SYSTEM_PROMPT}{prompt_extra}")]
    if chat_history:
        messages.extend(chat_history)
    messages.append(HumanMessage(content=message))

    async for chunk in llm.astream(messages):
        token = normalize_model_content(chunk.content)
        if token:
            yield token


async def generate_plain_reply(message: str, chat_history: Optional[List[BaseMessage]] = None) -> str:
    llm = rag_engine.llm or get_plain_chat_llm()
    if not llm:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to backend/.env and restart the backend.")

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    if chat_history:
        messages.extend(chat_history)
    messages.append(HumanMessage(content=message))

    try:
        response = await invoke_llm_with_fallback(llm, messages)
        return strip_thinking_content(response.content)
    except Exception as error:
        print(f"Groq text generation error [provider=groq model={GROQ_MODEL}]: {type(error).__name__}: {error}")
        status_code, friendly_msg = parse_llm_exception(error)
        raise RuntimeError(friendly_msg) from error


def get_vision_chat_llm():
    """Create a dedicated Groq vision client for image requests."""
    global vision_chat_llm
    if vision_chat_llm:
        return vision_chat_llm

    api_key = get_env_value("GROQ_API_KEY")
    if not api_key:
        return None

    vision_chat_llm = ChatGroq(
        model=GROQ_VISION_MODEL,
        api_key=api_key,
        temperature=0.3,
        timeout=60,
        max_retries=2,
    )
    return vision_chat_llm


def build_image_data_url(image: ImageInput) -> str:
    """Validate a browser-provided image and return a Groq-compatible data URL."""
    mime_type = (image.mime_type or "").lower().strip()
    if mime_type not in ALLOWED_IMAGE_MIME_TYPES:
        raise ValueError("Please upload a JPG, PNG, or WEBP image.")

    encoded_image = (image.data or "").strip()
    if encoded_image.startswith("data:"):
        encoded_image = encoded_image.split(",", 1)[-1]

    try:
        raw_image = base64.b64decode(encoded_image, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("The uploaded image could not be read. Please choose another file.") from None

    if not raw_image:
        raise ValueError("The uploaded image is empty.")
    if len(raw_image) > MAX_IMAGE_SIZE_BYTES:
        raise ValueError("Images must be 8 MB or smaller.")
    if mime_type == "image/jpeg" and not raw_image.startswith(b"\xff\xd8\xff"):
        raise ValueError("The file does not appear to be a valid JPG image.")
    if mime_type == "image/png" and not raw_image.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("The file does not appear to be a valid PNG image.")
    if mime_type == "image/webp" and not (raw_image.startswith(b"RIFF") and raw_image[8:12] == b"WEBP"):
        raise ValueError("The file does not appear to be a valid WEBP image.")

    return f"data:{mime_type};base64,{encoded_image}"


def vision_prompt(message: str) -> str:
    question = message.strip()
    if question:
        return question
    return "Analyze this image. Describe the important details clearly, then mention anything notable or useful."


def normalize_model_content(content) -> str:
    """Return text from OpenAI-compatible model chunks and full responses."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
        return "".join(text_parts)
    return ""


async def generate_vision_reply(message: str, image_data_url: str, chat_history: Optional[List[BaseMessage]] = None) -> str:
    llm = get_vision_chat_llm()
    if not llm:
        raise RuntimeError("Image analysis is unavailable because GROQ_API_KEY or GROQ_VISION_MODEL is not configured.")

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    if chat_history:
        messages.extend(chat_history)
    messages.append(HumanMessage(content=[
        {"type": "text", "text": vision_prompt(message)},
        {"type": "image_url", "image_url": {"url": image_data_url}},
    ]))

    try:
        response = await llm.ainvoke(messages)
        return strip_thinking_content(normalize_model_content(response.content))
    except Exception as error:
        error_str = str(error).lower()
        print(f"Groq vision error [provider=groq model={GROQ_VISION_MODEL}]: {type(error).__name__}: {error}")
        if '429' in error_str or 'rate_limit' in error_str or 'rate limit' in error_str:
            raise RuntimeError("Personal AI has temporarily reached its AI usage limit. Please try again shortly.") from error
        raise RuntimeError("Personal AI is temporarily unable to analyze this image. Please try again.") from error


async def stream_vision_reply(message: str, image_data_url: str, chat_history: Optional[List[BaseMessage]] = None):
    """Yield visible response tokens from the configured Groq vision model."""
    llm = get_vision_chat_llm()
    if not llm:
        raise RuntimeError("Image analysis is unavailable because GROQ_API_KEY or GROQ_VISION_MODEL is not configured.")

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    if chat_history:
        messages.extend(chat_history)
    messages.append(HumanMessage(content=[
        {"type": "text", "text": vision_prompt(message)},
        {"type": "image_url", "image_url": {"url": image_data_url}},
    ]))

    async for chunk in llm.astream(messages):
        token = normalize_model_content(chunk.content)
        if token:
            yield token


async def load_chat_history_messages(conversation: Dict, limit: Optional[int] = 6):
    """Rebuild LangChain chat history messages from persisted conversation turns with strict budget bounding."""
    try:
        db = await get_db()
        effective_limit = limit if (limit is not None and limit > 0) else 6
        cursor = db.history.find({"conversation_id": str(conversation["_id"])}).sort("created_at", -1)
        history_items = await cursor.to_list(length=effective_limit)
        history_items.reverse()
    except Exception as e:
        print(f"Chat history load error: {e}")
        return []

    messages = []
    for item in history_items:
        question = item.get("question")
        answer = item.get("answer")
        if question:
            messages.append(HumanMessage(content=question))
        if answer:
            messages.append(AIMessage(content=answer))
    return messages


async def save_history_entry(
    conversation: Dict,
    question: str,
    answer: str,
    sources: Optional[List[str]] = None,
    document_ids: Optional[List[str]] = None,
    image_name: Optional[str] = None,
    source_details: Optional[List[Dict]] = None,
    turn_id: Optional[str] = None,
):
    """Persist or update a conversation turn for the History page and future restores."""
    try:
        db = await get_db()
        now = datetime.now()
        conversation_id = str(conversation["_id"])

        # Sanitize document IDs to ensure no citation strings or page numbers are saved as identifiers
        clean_doc_ids = []
        for d in (document_ids or []):
            cd = sanitize_document_identifier(d)
            if cd and cd not in clean_doc_ids:
                clean_doc_ids.append(cd)

        if turn_id and ObjectId.is_valid(turn_id):
            existing_turn = await db.history.find_one({"_id": ObjectId(turn_id)})
            if existing_turn:
                await db.history.update_one(
                    {"_id": ObjectId(turn_id)},
                    {"$set": {
                        "question": question,
                        "answer": answer,
                        "preview": answer[:150] + "..." if answer else "No response",
                        "sources": sources or [],
                        "source_details": source_details or [],
                        "document_ids": clean_doc_ids,
                        "image_name": image_name,
                        "updated_at": now,
                    }}
                )
                turn_time = existing_turn.get("created_at")
                if turn_time:
                    await db.history.delete_many({
                        "conversation_id": conversation_id,
                        "created_at": {"$gt": turn_time}
                    })
                await db["conversations"].update_one({"_id": conversation["_id"]}, {"$set": {"updated_at": now}})
                return {"_id": turn_id, "conversation_id": conversation_id, "question": question, "answer": answer}

        payload = {
            "conversation_id": conversation_id,
            "session_id": conversation.get("session_id", conversation_id),
            "question": question,
            "answer": answer,
            "preview": answer[:150] + "..." if answer else "No response",
            "sources": sources or [],
            "source_details": source_details or [],
            "document_ids": clean_doc_ids,
            "image_name": image_name,
            "time": now.strftime("%b %d, %Y %I:%M %p"),
            "created_at": now,
            "updated_at": now,
        }
        await db.history.insert_one(payload)

        current_title = conversation.get("title")
        if not current_title or current_title.strip() in ("", "New chat"):
            new_title = await generate_smart_title(question)
        else:
            new_title = current_title

        updates = {"updated_at": now}
        if clean_doc_ids:
            updates["document_name"] = clean_doc_ids[0]

        if new_title and new_title != "New chat" and current_title in (None, "", "New chat"):
            updates["title"] = new_title

        await db["conversations"].update_one({"_id": conversation["_id"]}, {"$set": updates})
        return payload
    except Exception as e:
        print(f"History save error: {type(e).__name__}: {e}")
        raise HTTPException(status_code=503, detail=f"Conversation could not be saved: {e}") from e



def format_display_time(value):
    if isinstance(value, datetime):
        return value.strftime("%b %d, %Y %I:%M %p")
    return value or datetime.now().strftime("%b %d, %Y %I:%M %p")


async def generate_smart_title(question: str) -> str:
    """Generate a short 2-6 word conversation title based on the user's first message."""
    if not question or not question.strip():
        return "New chat"

    clean_q = question.strip()
    lower_q = clean_q.lower().strip("!.,? ")

    # Special cases: trivial greetings get clean titles
    if lower_q in ["hi", "hello", "hey", "good morning", "good afternoon"]:
        return "Greeting"
    if lower_q in ["thanks", "thank you", "thx"]:
        return "Thank You"

    try:
        llm = get_plain_chat_llm() or rag_engine.llm
        if llm:
            prompt = (
                "You are a conversation title generator.\n"
                "Generate a short, natural 2 to 6 word title summarizing the user's question.\n"
                "Rules:\n"
                "- Do NOT answer the question.\n"
                "- Preserve exact technical terms (Python, MongoDB, RAG, AI, API, FastAPI, React, JavaScript, GitHub, etc.).\n"
                "- Maximum length: 40 characters.\n"
                "- Return ONLY the raw title string, with no quotes, punctuation, or preamble.\n\n"
                f"User Question: {clean_q}\nTitle:"
            )
            res = await llm.ainvoke(prompt)
            raw_title = strip_thinking_content(str(res.content)).strip().strip('"\'`')
            cleaned_title = re.sub(r'^(title:\s*|summary:\s*)', '', raw_title, flags=re.IGNORECASE).strip()
            cleaned_title = re.sub(r'[\r\n]+', ' ', cleaned_title).strip()

            if cleaned_title and cleaned_title.lower() not in ["new chat", "chat", "conversation", "question", "help", "user request", "ai conversation"]:
                title = cleaned_title
                if len(title) > 40:
                    shortened = title[:40].rsplit(' ', 1)[0]
                    title = shortened if len(shortened) >= 10 else title[:40]
                return title
    except Exception as e:
        print(f"[TITLE WARN] Fast LLM title generation fallback triggered: {type(e).__name__}: {e}")

    return make_conversation_title(clean_q)


def make_conversation_title(question: str) -> str:
    if not question or not question.strip():
        return "New chat"
    text = question.strip()
    lower = text.lower().strip("!.,? ")
    if lower in ["hi", "hello", "thanks", "thank you", "hey", "thx"]:
        return "New chat"

    prefixes = [
        r"^(can\s+you\s+)?(please\s+)?(explain|tell\s+me\s+about|how\s+to|how\s+do\s+i|how\s+can\s+i|write\s+a|create\s+a|build\s+a|help\s+me\s+with|what\s+is|what\s+are|show\s+me|give\s+me|give\s+me\s+a|draft\s+a)\s+",
        r"^(i\s+want\s+to\s+)?(how\s+do\s+i\s+)?",
    ]
    cleaned = text
    for p in prefixes:
        cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE).strip()

    words = re.sub(r"[^\w\s-]", "", cleaned or text).split()
    if not words:
        words = re.sub(r"[^\w\s-]", "", text).split()
    if not words:
        return "New chat"

    title_words = words[:6]
    title = " ".join(title_words)
    title = title.title() if len(title) <= 40 else title.capitalize()
    if len(title) > 40:
        shortened = title[:40].rsplit(' ', 1)[0]
        title = shortened if len(shortened) >= 10 else title[:40]
    return title


def build_chat_summaries(history_items: list) -> list:
    chats = {}

    for item in history_items:
        session_id = item.get("session_id")
        if not session_id:
            continue

        created_at = item.get("created_at") or item.get("updated_at")
        updated_at = item.get("updated_at") or item.get("created_at") or created_at

        if session_id not in chats:
            chats[session_id] = {
                "id": session_id,
                "title": item.get("question") or "New chat",
                "preview": item.get("preview") or item.get("question") or "No messages yet",
                "message_count": 0,
                "created_at": created_at,
                "updated_at": updated_at,
            }

        chat = chats[session_id]
        chat["message_count"] += 1

        if created_at and (chat["created_at"] is None or created_at < chat["created_at"]):
            chat["created_at"] = created_at
            if item.get("question"):
                chat["title"] = item["question"]

        if updated_at and (chat["updated_at"] is None or updated_at > chat["updated_at"]):
            chat["updated_at"] = updated_at
            if item.get("preview"):
                chat["preview"] = item["preview"]

        if chat["title"] == "New chat" and item.get("question"):
            chat["title"] = item["question"]

    chat_list = list(chats.values())
    chat_list.sort(key=lambda chat: chat["updated_at"] or chat["created_at"] or datetime.min, reverse=True)

    for chat in chat_list:
        chat["time"] = format_display_time(chat["updated_at"] or chat["created_at"])
        chat.pop("created_at", None)
        chat.pop("updated_at", None)

    return chat_list


def serialize_conversation(conversation: dict) -> dict:
    conv_id = str(conversation.get("_id"))
    return {
        "id": conv_id,
        "conversation_id": conv_id,
        "session_id": conversation.get("session_id") or conv_id,
        "title": conversation.get("title") or "New chat",
        "created_at": conversation.get("created_at"),
        "updated_at": conversation.get("updated_at"),
        "document_name": conversation.get("document_name"),
        "pinned": bool(conversation.get("pinned", False)),
        "archived": bool(conversation.get("archived", False)),
        "selected_document_ids": conversation.get("selected_document_ids") or [],
    }



def extract_source_names(documents) -> List[str]:
    """Return citation strings with document name and formatted pages used, e.g. ['Document.pdf (p. 4-8)']."""
    grouped: Dict[str, set] = {}
    for doc in documents or []:
        metadata = getattr(doc, "metadata", {}) or {}
        source = metadata.get("source")
        if not source:
            continue
        name = os.path.basename(source)
        pages = grouped.setdefault(name, set())
        page_num = metadata.get("page_number") or metadata.get("page")
        if isinstance(page_num, int) and page_num > 0:
            pages.add(page_num)
        elif isinstance(page_num, str) and page_num.isdigit() and int(page_num) > 0:
            pages.add(int(page_num))

    result = []
    for name, pages in sorted(grouped.items()):
        formatted = format_page_ranges(list(pages))
        if formatted:
            result.append(f"{name} ({formatted})")
        else:
            result.append(name)
    return result


def extract_source_details(documents) -> List[Dict]:
    """Return citation-ready source names and actual page metadata from retrieved chunks."""
    grouped: Dict[str, set] = {}
    for doc in documents or []:
        metadata = getattr(doc, "metadata", {}) or {}
        source = metadata.get("source")
        if not source:
            continue
        name = os.path.basename(source)
        pages = grouped.setdefault(name, set())
        page_num = metadata.get("page_number") or metadata.get("page")
        if isinstance(page_num, int) and page_num > 0:
            pages.add(page_num)
        elif isinstance(page_num, str) and page_num.isdigit() and int(page_num) > 0:
            pages.add(int(page_num))

    return [
        {
            "name": name,
            "pages": sorted(pages),
            "formatted_pages": format_page_ranges(list(pages))
        }
        for name, pages in sorted(grouped.items())
    ]


def selected_summary_filename(request: ChatRequest, available_files: List[Dict]) -> Optional[str]:
    """Prefer the explicitly selected document for a supported summary request."""
    available_names = {item.get("source") for item in available_files if item.get("source")}
    selected_docs = request.document_names or ([request.document_name] if request.document_name else [])
    for doc in selected_docs:
        if doc in available_names:
            return doc
    user_input = request.message.lower()
    return next((name for name in available_names if name.lower() in user_input), None)


@app.get("/status")
@app.get("/health")
async def system_status():
    """Return backend health and live RAG engine indexing statistics from MongoDB."""
    db_ok = False
    indexed_docs = 0
    indexed_pages = 0
    indexed_chunks = 0
    try:
        db = await get_db()
        await db.command("ping")
        db_ok = True

        cursor = db["documents"].find({"status": {"$in": ["READY", "Ready"]}})
        ready_doc_records = await cursor.to_list(length=None)
        
        valid_doc_ids = []
        valid_doc_names = []
        for d in ready_doc_records:
            storage_path = get_document_storage_path(d)
            if storage_path and os.path.isfile(storage_path):
                doc_id = d.get("document_id") or str(d.get("_id"))
                valid_doc_ids.append(doc_id)
                if d.get("filename"):
                    valid_doc_names.append(d["filename"])

        indexed_docs = len(valid_doc_ids)

        if valid_doc_ids or valid_doc_names:
            indexed_chunks = await db["chunks"].count_documents({
                "$or": [
                    {"document_id": {"$in": valid_doc_ids}},
                    {"metadata.document_id": {"$in": valid_doc_ids}},
                    {"source": {"$in": valid_doc_names}},
                    {"metadata.source": {"$in": valid_doc_names}}
                ]
            })
            indexed_pages = await db["pages"].count_documents({
                "$or": [
                    {"document_id": {"$in": valid_doc_ids}},
                    {"filename": {"$in": valid_doc_names}}
                ]
            })
        else:
            indexed_chunks = 0
            indexed_pages = 0

        # Keep RAG engine memory cache in sync with actual valid chunks
        if valid_doc_ids or valid_doc_names:
            chunks_cursor = db["chunks"].find({
                "$or": [
                    {"document_id": {"$in": valid_doc_ids}},
                    {"metadata.document_id": {"$in": valid_doc_ids}},
                    {"source": {"$in": valid_doc_names}},
                    {"metadata.source": {"$in": valid_doc_names}}
                ]
            })
            all_chunks_data = await chunks_cursor.to_list(length=None)
            from langchain_core.documents import Document
            rag_engine.all_docs = [
                Document(page_content=c.get("content") or c.get("text", ""), metadata=c.get("metadata", {})) 
                for c in all_chunks_data
            ]
        else:
            rag_engine.all_docs = []

    except Exception as exc:
        print(f"[STATUS CHECK ERROR] {exc}")
        db_ok = False

    storage_ok = os.path.exists(UPLOAD_FOLDER) and os.access(UPLOAD_FOLDER, os.W_OK)

    return {
        "status": "healthy" if (db_ok and storage_ok) else "degraded",
        "database": {
            "connected": db_ok,
            "mode": db_instance.mode,
            "is_local": db_instance.is_local,
            "database_name": db_instance.db_name,
        },
        "storage": {
            "available": storage_ok,
            "status": "Available" if storage_ok else "Error",
            "path": str(UPLOAD_FOLDER),
        },
        "rag_engine": {
            "initialized": db_ok and (rag_engine.rag_chain is not None or indexed_chunks > 0 or indexed_docs > 0),
            "status": "Ready" if (rag_engine.rag_chain is not None or indexed_chunks > 0 or db_ok) else "Error",
            "total_documents_indexed": indexed_docs,
            "total_pages_indexed": indexed_pages,
            "total_chunks_indexed": indexed_chunks,
            "total_documents_cached": indexed_docs,
            "total_chunks_cached": indexed_chunks,
            "groq_model": getattr(rag_engine, "groq_model", GROQ_MODEL),
            "vector_dimensions": getattr(rag_engine, "vector_dimensions", 384),
        }
    }


@app.post("/system/cleanup_orphaned")
@app.post("/documents/cleanup_orphaned")
async def cleanup_orphaned():
    """Scan for and purge orphaned documents, chunks, vectors, and metadata."""
    try:
        report = await rag_engine.cleanup_orphaned_documents()
        return report
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Cleanup API ERROR] {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}") from e


@app.get("/files")
@app.get("/documents")
async def list_files():
    files_metadata = []
    try:
        db = await get_db()
        cursor = db["documents"].find({}).sort([("uploaded_at", -1), ("created_at", -1)])
        doc_records = await cursor.to_list(length=1000)
    except Exception as error:
        print(f"Document records unavailable: {type(error).__name__}: {error}")
        raise HTTPException(status_code=503, detail=f"Document storage is unavailable: {error}") from error

    for record in doc_records:
        doc_id = record.get("document_id") or str(record.get("_id"))
        fn = record.get("filename") or "Document"
        storage_path = get_document_storage_path(record)
        size_bytes = record.get("size_bytes") or record.get("file_size") or (os.path.getsize(storage_path) if storage_path and os.path.isfile(storage_path) else 0)

        raw_status = (record.get("status") or "READY").upper()
        stage = record.get("stage") or ("Ready" if raw_status == "READY" else raw_status)

        chunk_count = await db["chunks"].count_documents({
            "$or": [
                {"document_id": doc_id},
                {"metadata.document_id": doc_id},
                {"source": fn},
                {"metadata.source": fn}
            ]
        })

        ext = os.path.splitext(fn)[1].lower().replace('.', '').upper() or record.get("file_type") or "FILE"

        files_metadata.append({
            "document_id": doc_id,
            "id": doc_id,
            "filename": fn,
            "name": fn,
            "file_type": ext,
            "size": f"{size_bytes / (1024 * 1024):.2f} MB" if size_bytes > 1024*1024 else f"{size_bytes / 1024:.2f} KB",
            "size_bytes": size_bytes,
            "file_size": size_bytes,
            "uploaded_on": format_display_time(record.get("uploaded_at") or record.get("created_at")),
            "status": raw_status,
            "stage": stage,
            "status_details": record.get("status_details"),
            "error": record.get("error"),
            "upload_status": record.get("upload_status", "uploaded"),
            "processing_status": "ready" if raw_status == "READY" else ("processing" if raw_status in ("PROCESSING", "UPLOADING") else "failed"),
            "summary_status": record.get("summary_status", "completed"),
            "chunk_count": chunk_count,
            "page_count": record.get("total_pages") or record.get("page_count", 0),
            "total_pages": record.get("total_pages") or record.get("page_count", 0),
            "extracted_page_count": record.get("extracted_page_count", 0),
            "toc": record.get("toc", []),
        })
    return {"files": files_metadata}




@app.get("/documents/{identifier}")
@app.get("/files/{identifier}")
@app.get("/files/{identifier}/details")
async def file_details(identifier: str):
    """Return comprehensive document metadata, status, chunk counts, and summary."""
    db = await get_db()
    query = {"$or": [
        {"document_id": identifier},
        {"filename": identifier}
    ]}
    if ObjectId.is_valid(identifier):
        query["$or"].append({"_id": ObjectId(identifier)})
    query["$or"].append({"_id": identifier})

    record = await db["documents"].find_one(query)
    if not record:
        raise HTTPException(status_code=404, detail="Document metadata not found")

    doc_id = record.get("document_id") or str(record.get("_id"))
    filename = record.get("filename")

    summary_doc = await db["summaries"].find_one({"$or": [{"document_id": doc_id}, {"source": filename}]})
    chunk_count = await db["chunks"].count_documents({"$or": [{"document_id": doc_id}, {"metadata.document_id": doc_id}, {"source": filename}]})

    storage_path = record.get("storage_path") or record.get("file_path") or os.path.join(UPLOAD_FOLDER, filename)
    size_bytes = record.get("size_bytes") or (os.path.getsize(storage_path) if os.path.isfile(storage_path) else 0)

    return {
        "filename": filename,
        "document_id": doc_id,
        "status": record.get("status"),
        "stage": record.get("stage"),
        "status_details": record.get("status_details"),
        "error": record.get("error"),
        "chunk_count": chunk_count,
        "page_count": record.get("total_pages") or record.get("page_count", 0),
        "total_pages": record.get("total_pages") or record.get("page_count", 0),
        "extracted_page_count": record.get("extracted_page_count", 0),
        "toc": record.get("toc", []),
        "summary": summary_doc.get("summary") if summary_doc else None,
        "size_bytes": size_bytes,
        "updated_at": format_display_time(record.get("updated_at")),
    }


@app.get("/documents/{identifier}/status")
@app.get("/files/{identifier}/status")
async def get_document_status(identifier: str):
    """Return persistent indexing status, chunk counts, and error details for a document."""
    db = await get_db()
    ident = sanitize_document_identifier(identifier)
    query = {"$or": [
        {"document_id": ident},
        {"filename": ident},
        {"disk_filename": ident},
        {"filename": {"$regex": f"^{re.escape(ident)}$", "$options": "i"}}
    ]}
    if ObjectId.is_valid(ident):
        query["$or"].append({"_id": ObjectId(ident)})
    query["$or"].append({"_id": ident})

    doc = await db["documents"].find_one(query)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{identifier}' not found.")

    doc_id = doc.get("document_id") or str(doc.get("_id"))
    fn = doc.get("filename") or ident
    raw_status = (doc.get("status") or "READY").upper()
    stage = doc.get("stage") or ("Ready" if raw_status == "READY" else raw_status)

    chunk_count = await db["chunks"].count_documents({
        "$or": [
            {"document_id": doc_id},
            {"metadata.document_id": doc_id},
            {"source": fn},
            {"metadata.source": fn}
        ]
    })
    page_count = await db["pages"].count_documents({"document_id": doc_id})

    print(f"[DOCUMENT_STATUS] document_id={doc_id} filename='{fn}' status={raw_status} chunks={chunk_count}")

    return {
        "document_id": doc_id,
        "id": doc_id,
        "filename": fn,
        "name": fn,
        "status": raw_status,
        "stage": stage,
        "chunk_count": chunk_count,
        "embedding_count": chunk_count,
        "page_count": doc.get("total_pages") or page_count or doc.get("page_count", 0),
        "total_pages": doc.get("total_pages") or page_count or doc.get("page_count", 0),
        "error": doc.get("error"),
        "created_at": format_display_time(doc.get("created_at") or doc.get("uploaded_at")),
        "updated_at": format_display_time(doc.get("updated_at") or doc.get("created_at")),
        "upload_status": doc.get("upload_status", "uploaded"),
        "processing_status": "ready" if raw_status == "READY" else ("processing" if raw_status in ("PROCESSING", "UPLOADING") else "failed"),
    }



@app.get("/files/{identifier}/download")
@app.get("/documents/{identifier}/download")
async def download_file(identifier: str):
    """Serve an uploaded document for download."""
    clean_id = sanitize_document_identifier(identifier)
    db = await get_db()
    query = {"$or": [
        {"document_id": clean_id},
        {"filename": clean_id}
    ]}
    if ObjectId.is_valid(clean_id):
        query["$or"].append({"_id": ObjectId(clean_id)})
    query["$or"].append({"_id": clean_id})

    record = await db["documents"].find_one(query)
    storage_path = None
    filename = clean_id
    if record:
        storage_path = record.get("storage_path") or record.get("file_path")
        filename = record.get("filename") or clean_id

    if not storage_path or not os.path.isfile(storage_path):
        candidate = os.path.join(UPLOAD_FOLDER, clean_id)
        if os.path.isfile(candidate):
            storage_path = candidate

    if not storage_path or not os.path.isfile(storage_path):
        raise HTTPException(status_code=404, detail="Document not found")
    return FileResponse(storage_path, filename=filename)


@app.get("/files/{identifier}/view")
@app.get("/documents/{identifier}/view")
async def view_file_inline(identifier: str):
    """Serve an uploaded document inline for browser previewing, iframe viewing, and page jumping."""
    clean_id = sanitize_document_identifier(identifier)
    db = await get_db()
    query = {"$or": [
        {"document_id": clean_id},
        {"filename": clean_id}
    ]}
    if ObjectId.is_valid(clean_id):
        query["$or"].append({"_id": ObjectId(clean_id)})
    query["$or"].append({"_id": clean_id})

    record = await db["documents"].find_one(query)
    storage_path = None
    filename = clean_id
    if record:
        storage_path = record.get("storage_path") or record.get("file_path")
        filename = record.get("filename") or clean_id

    if not storage_path or not os.path.isfile(storage_path):
        candidate = os.path.join(UPLOAD_FOLDER, clean_id)
        if os.path.isfile(candidate):
            storage_path = candidate

    if not storage_path or not os.path.isfile(storage_path):
        raise HTTPException(status_code=404, detail="Document not found")

    ext = os.path.splitext(filename)[1].lower()
    media_type = "application/pdf" if ext == ".pdf" else "application/octet-stream"
    headers = {
        "Content-Disposition": f'inline; filename="{filename}"',
        "Access-Control-Allow-Origin": "*",
    }
    return FileResponse(storage_path, media_type=media_type, headers=headers)



@app.get("/history")
async def get_history(conversation_id: Optional[str] = None):
    try:
        db = await get_db()
        query = {"conversation_id": conversation_id} if conversation_id else {}
        cursor = db.history.find(query).sort([("updated_at", -1), ("created_at", -1)])
        history_list = await cursor.to_list(length=100)
        valid_conversation_ids = {
            str(item["_id"])
            for item in await db["conversations"].find({}, {"_id": 1}).to_list(length=1000)
        }

        for item in history_list:
            item["id"] = str(item.pop("_id"))
        return {"history": [item for item in history_list if item.get("conversation_id") in valid_conversation_ids]}
    except Exception as e:
        print(f"History fetch error: {type(e).__name__}: {e}")
        raise HTTPException(status_code=503, detail=f"History storage is unavailable: {e}") from e


@app.get("/chats")
async def get_chats(include_archived: bool = False, query: Optional[str] = None):
    try:
        db = await get_db()
    except Exception as e:
        print(f"Chats unavailable: {e}")
        raise HTTPException(status_code=503, detail=f"Conversation storage is unavailable: {e}") from e

    mongo_filter = {}
    if not include_archived:
        mongo_filter["archived"] = {"$ne": True}
    if query and query.strip():
        q_regex = re.escape(query.strip())
        matching_turns = await db.history.find({"question": {"$regex": q_regex, "$options": "i"}}, {"conversation_id": 1, "session_id": 1}).to_list(length=500)
        matching_conv_ids = set()
        for t in matching_turns:
            if t.get("conversation_id"):
                matching_conv_ids.add(t["conversation_id"])
            if t.get("session_id"):
                matching_conv_ids.add(t["session_id"])

        or_conditions = [{"title": {"$regex": q_regex, "$options": "i"}}]
        for cid in matching_conv_ids:
            if ObjectId.is_valid(cid):
                or_conditions.append({"_id": ObjectId(cid)})
            or_conditions.append({"session_id": cid})
            or_conditions.append({"conversation_id": cid})

        mongo_filter["$or"] = or_conditions

    conversations = await db["conversations"].find(mongo_filter).sort([("pinned", -1), ("updated_at", -1), ("created_at", -1)]).to_list(length=1000)
    chats = []
    for conversation in conversations:
        conversation_id = str(conversation["_id"])
        message_count = await db.history.count_documents({"conversation_id": conversation_id})

        if conversation.get("title") in (None, "", "New chat") and message_count > 0:
            first_turns = await db.history.find({"conversation_id": conversation_id}).sort([("created_at", 1)]).to_list(length=1)
            if first_turns and first_turns[0].get("question"):
                migrated_title = await generate_smart_title(first_turns[0]["question"])
                if migrated_title and migrated_title != "New chat":
                    conversation["title"] = migrated_title
                    await db["conversations"].update_one({"_id": conversation["_id"]}, {"$set": {"title": migrated_title}})

        latest_turn = await db.history.find({"conversation_id": conversation_id}).sort([("updated_at", -1)]).to_list(length=1)
        turn = latest_turn[0] if latest_turn else {}
        chats.append({
            **serialize_conversation(conversation),
            "id": conversation_id,
            "message_count": message_count,
            "preview": turn.get("preview", "No messages yet"),
            "time": format_display_time(conversation.get("updated_at") or conversation.get("created_at")),
        })
    return {"chats": chats}


@app.post("/chats")
@app.post("/conversations")
@app.post("/conversations/new")
async def create_chat(update: Optional[ConversationCreate] = None):
    db = await get_db()
    now = datetime.now()
    title = update.title.strip() if update and update.title and update.title.strip() else "New chat"
    conversation_id = ObjectId()
    session_id = str(conversation_id)

    docs = []
    if update and update.selected_document_ids:
        docs.extend(update.selected_document_ids)
    if update and update.document_name and update.document_name not in docs:
        docs.append(update.document_name)

    conversation = {
        "_id": conversation_id,
        "session_id": session_id,
        "title": title,
        "created_at": now,
        "updated_at": now,
        "document_name": update.document_name if update else None,
        "pinned": False,
        "archived": False,
        "selected_document_ids": list(set(docs)),
    }
    result = await db["conversations"].insert_one(conversation)
    if result.inserted_id != conversation_id:
        raise HTTPException(status_code=500, detail="MongoDB did not return the created conversation ID.")
    return serialize_conversation(conversation)


@app.patch("/chats/{chat_id}")
async def update_chat(chat_id: str, update: ConversationUpdate):
    db = await get_db()
    existing = await db["conversations"].find_one(conversation_filter(chat_id))
    if not existing:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    updates = {}
    if update.title is not None:
        title = update.title.strip()
        if not title or len(title) > 80:
            raise HTTPException(status_code=400, detail="Conversation title must be between 1 and 80 characters.")
        updates["title"] = title
    if update.pinned is not None:
        updates["pinned"] = bool(update.pinned)
    if update.archived is not None:
        updates["archived"] = bool(update.archived)
    if update.selected_document_ids is not None:
        updates["selected_document_ids"] = list(update.selected_document_ids)

    if updates:
        updates["updated_at"] = datetime.now()
        await db["conversations"].update_one(
            conversation_filter(chat_id),
            {"$set": updates},
        )

    updated_conv = await db["conversations"].find_one(conversation_filter(chat_id))
    return serialize_conversation(updated_conv)


@app.get("/chats/{chat_id}/files")
async def get_chat_files(chat_id: str):
    db = await get_db()
    conversation = await require_conversation(chat_id)

    selected_docs = list(conversation.get("selected_document_ids") or [])
    if conversation.get("document_name") and conversation.get("document_name") not in selected_docs:
        selected_docs.append(conversation.get("document_name"))

    files_result = []
    seen_ids = set()

    for ident in selected_docs:
        if not ident:
            continue
        query = {"$or": [
            {"document_id": ident},
            {"filename": ident}
        ]}
        if ObjectId.is_valid(ident):
            query["$or"].append({"_id": ObjectId(ident)})
        query["$or"].append({"_id": ident})

        doc = await db["documents"].find_one(query)
        if not doc:
            continue

        doc_id = doc.get("document_id") or str(doc.get("_id"))
        if doc_id in seen_ids:
            continue
        seen_ids.add(doc_id)

        filename = doc.get("filename") or ident
        storage_path = doc.get("storage_path") or doc.get("file_path") or os.path.join(UPLOAD_FOLDER, filename)
        file_size = doc.get("size_bytes") or doc.get("file_size") or (os.path.getsize(storage_path) if os.path.isfile(storage_path) else 0)
        has_chunks = await db["chunks"].find_one({"$or": [{"document_id": doc_id}, {"source": filename}]})

        raw_status = (doc.get("status") or "READY").upper()
        if raw_status == "PROCESSING":
            status = "PROCESSING"
            stage = doc.get("stage", "Processing")
        elif raw_status == "FAILED" and not has_chunks:
            status = "FAILED"
            stage = "Failed"
        else:
            status = "READY"
            stage = doc.get("stage", "Ready")

        error = doc.get("error") if status == "FAILED" else None
        sum_status = doc.get("summary_status", "completed")
        uploaded_at = doc.get("uploaded_at") or doc.get("created_at")

        ext = os.path.splitext(filename)[1].lower().replace('.', '').upper() or doc.get("file_type", "FILE")
        files_result.append({
            "document_id": doc_id,
            "id": doc_id,
            "filename": filename,
            "name": filename,
            "chat_id": chat_id,
            "file_type": ext,
            "size_bytes": file_size,
            "file_size": file_size,
            "status": status,
            "stage": stage,
            "error": error,
            "upload_status": doc.get("upload_status", "uploaded"),
            "processing_status": "ready" if status == "READY" else ("processing" if status == "PROCESSING" else "failed"),
            "summary_status": sum_status,
            "total_pages": doc.get("total_pages") or doc.get("page_count", 0),
            "page_count": doc.get("total_pages") or doc.get("page_count", 0),
            "chunk_count": doc.get("chunk_count", 0),
            "uploaded_at": format_display_time(uploaded_at),
        })

    return {"chat_id": chat_id, "files": files_result}


class FileAttachRequest(BaseModel):
    document_id: Optional[str] = None
    filename: Optional[str] = None


@app.post("/chats/{chat_id}/files/attach")
async def attach_file_to_chat(chat_id: str, request: FileAttachRequest):
    db = await get_db()
    conversation = await require_conversation(chat_id)
    ident = (request.document_id or request.filename or "").strip()
    if not ident:
        raise HTTPException(status_code=400, detail="Document identifier is required.")

    query = {"$or": [
        {"document_id": ident},
        {"filename": ident}
    ]}
    if ObjectId.is_valid(ident):
        query["$or"].append({"_id": ObjectId(ident)})
    query["$or"].append({"_id": ident})

    doc = await db["documents"].find_one(query)
    target_id = doc.get("document_id") if doc else ident
    target_fn = doc.get("filename") if doc else ident

    current_files = list(conversation.get("selected_document_ids") or [])
    if target_id not in current_files:
        current_files.append(target_id)

    await db["conversations"].update_one(
        conversation_filter(chat_id),
        {"$set": {"selected_document_ids": current_files, "updated_at": datetime.now()}},
    )
    return {
        "status": "attached",
        "chat_id": chat_id,
        "document_id": target_id,
        "filename": target_fn,
        "attached_files": current_files
    }


@app.post("/chats/{chat_id}/files/detach")
async def detach_file_from_chat(chat_id: str, request: FileAttachRequest):
    db = await get_db()
    conversation = await require_conversation(chat_id)
    ident = (request.document_id or request.filename or "").strip()

    query = {"$or": [
        {"document_id": ident},
        {"filename": ident}
    ]}
    if ObjectId.is_valid(ident):
        query["$or"].append({"_id": ObjectId(ident)})
    query["$or"].append({"_id": ident})

    doc = await db["documents"].find_one(query)
    target_id = doc.get("document_id") if doc else ident
    target_fn = doc.get("filename") if doc else ident

    current_files = [
        f for f in (conversation.get("selected_document_ids") or [])
        if f not in (target_id, target_fn, ident)
    ]

    updates = {"selected_document_ids": current_files, "updated_at": datetime.now()}
    if conversation.get("document_name") in (target_id, target_fn, ident):
        updates["document_name"] = None

    await db["conversations"].update_one(
        conversation_filter(chat_id),
        {"$set": updates},
    )
    return {
        "status": "detached",
        "chat_id": chat_id,
        "document_id": target_id,
        "filename": target_fn,
        "attached_files": current_files
    }




@app.get("/chat_messages")
async def get_chat_messages(conversation_id: str):
    try:
        messages = []
        await require_conversation(conversation_id)
        db = await get_db()
        cursor = db.history.find({
            "$or": [
                {"conversation_id": conversation_id},
                {"session_id": conversation_id}
            ]
        }).sort([("created_at", 1), ("updated_at", 1)])
        history_items = await cursor.to_list(length=200)
        for item in history_items:
            item_id = str(item.get("_id", uuid.uuid4()))
            messages.append({
                "id": f"user_{item_id}",
                "turn_id": item_id,
                "text": item.get("question", ""),
                "sender": "user",
                "image_name": item.get("image_name"),
                "time": format_display_time(item.get("created_at")),
            })
            messages.append({
                "id": f"ai_{item_id}",
                "turn_id": item_id,
                "text": item.get("answer", ""),
                "sender": "ai",
                "sources": item.get("sources", []),
                "sourceDetails": item.get("source_details", []),
                "time": format_display_time(item.get("updated_at") or item.get("created_at")),
            })
        return {"messages": messages}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Chat messages unavailable: {e}")
        raise HTTPException(status_code=503, detail=f"Conversation messages could not be loaded: {e}") from e


class MessageEditRequest(BaseModel):
    text: str


@app.patch("/chats/{chat_id}/messages/{turn_id}")
async def edit_chat_message(chat_id: str, turn_id: str, request: MessageEditRequest):
    """Edit a user message text in history."""
    db = await get_db()
    await require_conversation(chat_id)
    if not ObjectId.is_valid(turn_id):
        raise HTTPException(status_code=400, detail="Invalid turn ID")

    new_text = request.text.strip()
    if not new_text:
        raise HTTPException(status_code=400, detail="Message text cannot be empty")

    res = await db.history.update_one(
        {"_id": ObjectId(turn_id)},
        {"$set": {"question": new_text, "updated_at": datetime.now()}}
    )
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Message turn not found")

    return {"success": True, "turn_id": turn_id, "text": new_text}


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        conversation = await require_conversation(request.conversation_id)
        doc_names = request.document_names or ([request.document_name] if request.document_name else [])
        if not doc_names and conversation.get("selected_document_ids"):
            doc_names = list(conversation.get("selected_document_ids"))
        await require_ready_documents(doc_names)

        chat_history = await load_chat_history_messages(conversation)

        if request.image:
            try:
                image_data_url = build_image_data_url(request.image)
            except ValueError as error:
                raise HTTPException(status_code=400, detail=str(error)) from error
            answer = await generate_vision_reply(request.message, image_data_url, chat_history=chat_history)
            question = request.message.strip() or "Analyze this image"
            await save_history_entry(
                conversation,
                question,
                answer,
                image_name=request.image.name,
                turn_id=request.edit_turn_id,
            )
            return {
                "answer": answer,
                "sources": [],
                "time": datetime.now().strftime("%b %d, %Y %I:%M %p"),
            }

        user_input = request.message.lower()
        if not re.search(r'\bpages?\s*#?\s*\d+', user_input) and any(keyword in user_input for keyword in ["summarize the document", "summarize entire document", "summarize whole document", "overall summary", "document summary", "tl;dr of this document"]):
            db = await get_db()
            files_cursor = db["summaries"].find({}, {"source": 1})
            available_files = await files_cursor.to_list(length=None)

            filename = selected_summary_filename(request, available_files)
            if filename:
                summary = await rag_engine.get_summary(filename)
                if summary:
                    answer = f"### Summary of {filename}\n\n{summary}"
                    source_details = [{"name": filename, "pages": []}]
                    await save_history_entry(
                        conversation,
                        request.message,
                        answer,
                        sources=[filename],
                        document_ids=[filename],
                        source_details=source_details,
                        turn_id=request.edit_turn_id,
                    )
                    return {
                        "answer": answer,
                        "sources": [filename],
                        "source_details": source_details,
                        "time": datetime.now().strftime("%b %d, %Y %I:%M %p")
                    }

        if not doc_names or not rag_engine.rag_chain:
            answer = await generate_plain_reply(request.message, chat_history=chat_history)
            await save_history_entry(conversation, request.message, answer, sources=[], document_ids=[], turn_id=request.edit_turn_id)
            return {
                "answer": answer,
                "sources": [],
                "time": datetime.now().strftime("%b %d, %Y %I:%M %p")
            }

        response = await rag_engine.rag_chain.ainvoke({
            "input": request.message,
            "chat_history": chat_history,
            "document_names": doc_names,
            "document_name": doc_names[0] if doc_names else None,
            "response_mode": request.response_mode or "balanced",
        })

        answer = strip_thinking_content(response["answer"])
        sources = extract_source_names(response.get("context", []))
        source_details = extract_source_details(response.get("context", []))
        document_ids = sources[:]
        await save_history_entry(conversation, request.message, answer, sources=sources, document_ids=document_ids, source_details=source_details, turn_id=request.edit_turn_id)

        return {
            "answer": answer,
            "sources": sources,
            "source_details": source_details,
            "time": datetime.now().strftime("%b %d, %Y %I:%M %p")
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[CHAT ERROR] {type(e).__name__}: {e}")
        status_code, friendly = parse_llm_exception(e)
        raise HTTPException(status_code=status_code, detail=friendly) from e


@app.get("/system/diagnostics")
async def system_diagnostics():
    """Return diagnostic status of database, LLM provider, API key, model, RAG, and vector store without exposing secrets."""
    db_ok = False
    indexed_docs = 0
    indexed_pages = 0
    indexed_chunks = 0
    try:
        db = await get_db()
        await db.command("ping")
        db_ok = True

        cursor = db["documents"].find({"status": {"$in": ["READY", "Ready"]}})
        ready_doc_records = await cursor.to_list(length=None)
        valid_doc_ids = []
        valid_doc_names = []
        for d in ready_doc_records:
            storage_p = get_document_storage_path(d)
            if storage_p and os.path.isfile(storage_p):
                valid_doc_ids.append(d.get("document_id") or str(d.get("_id")))
                if d.get("filename"):
                    valid_doc_names.append(d["filename"])
        indexed_docs = len(valid_doc_ids)

        if valid_doc_ids or valid_doc_names:
            indexed_chunks = await db["chunks"].count_documents({
                "$or": [
                    {"document_id": {"$in": valid_doc_ids}},
                    {"metadata.document_id": {"$in": valid_doc_ids}},
                    {"source": {"$in": valid_doc_names}},
                    {"metadata.source": {"$in": valid_doc_names}}
                ]
            })
            indexed_pages = await db["pages"].count_documents({
                "$or": [
                    {"document_id": {"$in": valid_doc_ids}},
                    {"filename": {"$in": valid_doc_names}}
                ]
            })
        else:
            indexed_chunks = 0
            indexed_pages = 0
    except Exception as exc:
        print(f"[DIAGNOSTICS ERROR] {exc}")
        db_ok = False
        indexed_pages = 0

    api_key = get_env_value("GROQ_API_KEY")
    has_api_key = bool(api_key and len(api_key.strip()) > 5)
    masked_key = f"{api_key[:6]}...{api_key[-4:]}" if (api_key and len(api_key) > 10) else ("Configured" if has_api_key else "Missing")
    storage_ok = os.path.exists(UPLOAD_FOLDER) and os.access(UPLOAD_FOLDER, os.W_OK)

    return {
        "database": "Connected" if db_ok else "Disconnected",
        "storage": "Available" if storage_ok else "Error",
        "storage_path": str(UPLOAD_FOLDER),
        "llm_provider": "Groq",
        "llm_model": getattr(rag_engine, "groq_model", GROQ_MODEL),
        "api_key_variable": "GROQ_API_KEY",
        "api_key_configured": has_api_key,
        "api_key_masked": masked_key,
        "rag_status": "Available" if (rag_engine.rag_chain is not None or indexed_chunks > 0 or indexed_docs > 0) else "Ready",
        "vector_store_status": "Available" if db_ok else "Unavailable",
        "indexed_documents": indexed_docs,
        "indexed_pages": indexed_pages,
        "indexed_chunks": indexed_chunks,
    }




@app.post("/chat_stream")
async def chat_stream(request: ChatRequest):
    conversation = await require_conversation(request.conversation_id)
    raw_doc_identifiers = (
        request.document_ids
        or ([request.document_id] if request.document_id else None)
        or request.document_names
        or ([request.document_name] if request.document_name else None)
        or (list(conversation.get("selected_document_ids")) if conversation.get("selected_document_ids") else [])
    )
    doc_identifiers = [sanitize_document_identifier(x) for x in (raw_doc_identifiers or []) if sanitize_document_identifier(x)]
    resolved_docs = await require_ready_documents(doc_identifiers)
    canonical_doc_ids = [d.get("document_id") for d in resolved_docs if d.get("document_id")]
    canonical_filenames = [d.get("filename") for d in resolved_docs if d.get("filename")]
    doc_names = canonical_filenames or canonical_doc_ids or doc_identifiers or []

    print(f"[RETRIEVAL] [CHAT_INIT] conversation_id={request.conversation_id} attached_docs={len(resolved_docs)} ({canonical_filenames})")

    image_data_url = None
    if request.image:
        try:
            image_data_url = build_image_data_url(request.image)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    async def event_generator():
        try:
            print(f"[CHAT] Request started | Conversation ID: {request.conversation_id} | Turn: {request.edit_turn_id or 'new'}")
            chat_history = await load_chat_history_messages(conversation)

            if image_data_url:
                full_response = ""
                thinking_filter = ThinkingTokenFilter()
                async for token in stream_vision_reply(request.message, image_data_url, chat_history=chat_history):
                    full_response += token
                    visible_token = thinking_filter.process(token)
                    if visible_token:
                        yield f"data: {json.dumps({'token': visible_token})}\n\n"

                trailing_token = thinking_filter.flush()
                if trailing_token:
                    yield f"data: {json.dumps({'token': trailing_token})}\n\n"

                full_response = strip_thinking_content(full_response)
                question = request.message.strip() or "Analyze this image"
                await save_history_entry(
                    conversation,
                    question,
                    full_response,
                    image_name=request.image.name,
                    turn_id=request.edit_turn_id,
                )
                yield f"data: {json.dumps({'done': True, 'sources': []})}\n\n"
                print("[CHAT] Vision LLM request completed successfully.")
                return

            user_input = request.message.lower()
            if not re.search(r'\bpages?\s*#?\s*\d+', user_input) and any(keyword in user_input for keyword in ["summarize the document", "summarize entire document", "summarize whole document", "overall summary", "document summary", "tl;dr of this document"]):
                db = await get_db()
                files_cursor = db["summaries"].find({}, {"source": 1})
                available_files = await files_cursor.to_list(length=None)

                filename = selected_summary_filename(request, available_files)
                if filename:
                    summary = await rag_engine.get_summary(filename)
                    if summary:
                        full_response = f"### Summary of {filename}\n\n{summary}"
                        source_details = [{"name": filename, "pages": []}]
                        yield f"data: {json.dumps({'token': full_response})}\n\n"
                        await save_history_entry(
                            conversation,
                            request.message,
                            full_response,
                            sources=[filename],
                            document_ids=[filename],
                            source_details=source_details,
                            turn_id=request.edit_turn_id,
                        )
                        yield f"data: {json.dumps({'done': True, 'sources': [filename], 'source_details': source_details})}\n\n"
                        print("[CHAT] Summary request completed successfully.")
                        return

            print(f"[CHAT] LLM streaming request started (Mode: {request.response_mode or 'balanced'})")

            has_attached_docs = bool(resolved_docs and len(resolved_docs) > 0)

            if has_attached_docs:
                full_response = ""
                final_sources = []
                final_source_details = []
                thinking_filter = ThinkingTokenFilter()

                async for token, src_names, src_details in rag_engine.stream_rag_reply(
                    query=request.message,
                    chat_history=chat_history,
                    doc_names=doc_names,
                    response_mode=request.response_mode or "balanced"
                ):
                    full_response += token
                    final_sources = src_names
                    final_source_details = src_details
                    visible_token = thinking_filter.process(token)
                    if visible_token:
                        yield f"data: {json.dumps({'token': visible_token})}\n\n"

                trailing_token = thinking_filter.flush()
                if trailing_token:
                    yield f"data: {json.dumps({'token': trailing_token})}\n\n"

                full_response = strip_thinking_content(full_response)
                await save_history_entry(
                    conversation,
                    request.message,
                    full_response,
                    sources=final_sources,
                    document_ids=canonical_filenames or canonical_doc_ids or doc_names,
                    source_details=final_source_details,
                    turn_id=request.edit_turn_id,
                )
                yield f"data: {json.dumps({'done': True, 'sources': final_sources, 'source_details': final_source_details})}\n\n"
                print(f"[RETRIEVAL] [CHAT_DONE] RAG stream completed successfully with {len(final_sources)} source citations.")
                return
            else:
                # Plain chat true streaming
                full_response = ""
                thinking_filter = ThinkingTokenFilter()
                async for token in stream_plain_reply(
                    message=request.message,
                    chat_history=chat_history,
                    response_mode=request.response_mode or "balanced"
                ):
                    full_response += token
                    visible_token = thinking_filter.process(token)
                    if visible_token:
                        yield f"data: {json.dumps({'token': visible_token})}\n\n"

                trailing_token = thinking_filter.flush()
                if trailing_token:
                    yield f"data: {json.dumps({'token': trailing_token})}\n\n"

                full_response = strip_thinking_content(full_response)
                await save_history_entry(
                    conversation,
                    request.message,
                    full_response,
                    sources=[],
                    document_ids=[],
                    turn_id=request.edit_turn_id,
                )
                yield f"data: {json.dumps({'done': True, 'sources': []})}\n\n"
                print("[CHAT] Plain chat stream completed successfully.")
                return

        except Exception as e:
            import traceback
            traceback.print_exc()
            status_code, friendly = parse_llm_exception(e)
            print(f"[CHAT STREAM ERROR] [HTTP {status_code}] {type(e).__name__}: {e}")
            yield f"data: {json.dumps({'error': friendly, 'status_code': status_code})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...), chat_id: Optional[str] = Form(None)):
    original_filename = file.filename or "uploaded_file"
    ext = os.path.splitext(original_filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{ext}' not supported. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty (0 bytes). Please upload a valid document."
        )
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE_BYTES // (1024*1024)} MB."
        )

    # 1. Generate canonical permanent document_id
    document_id = str(uuid.uuid4())

    # 2. Collision-safe storage path on disk
    ascii_safe = re.sub(r'[^a-zA-Z0-9_.-]', '_', os.path.basename(original_filename))
    disk_filename = f"{document_id}_{ascii_safe}" if ascii_safe else f"{document_id}{ext}"
    file_path = os.path.join(UPLOAD_FOLDER, disk_filename)

    with open(file_path, "wb") as buffer:
        buffer.write(contents)

    content_hash = hashlib.sha256(contents).hexdigest()
    db = await get_db()
    now = datetime.now()
    ext_clean = ext.replace('.', '').upper() if ext else "FILE"

    # Clean up any older orphaned / failed record with the exact same filename
    try:
        await db["documents"].delete_many({
            "filename": original_filename,
            "$or": [
                {"status": "FAILED"},
                {"status": "PROCESSING", "size_bytes": 0},
                {"size_bytes": {"$lte": 0}}
            ]
        })
    except Exception:
        pass

    # 3. Create document record with full original filename (UTF-8 / Unicode / emojis fully preserved!)
    doc_record = {
        "_id": document_id,
        "document_id": document_id,
        "filename": original_filename,
        "disk_filename": disk_filename,
        "storage_path": file_path,
        "file_path": file_path,
        "file_type": ext_clean,
        "mime_type": file.content_type or f"application/{ext_clean.lower()}",
        "size_bytes": len(contents),
        "file_size": len(contents),
        "status": "PROCESSING",
        "stage": "Uploading",
        "upload_status": "uploaded",
        "processing_status": "processing",
        "summary_status": "pending",
        "total_pages": None,
        "page_count": 0,
        "chunk_count": 0,
        "conversation_id": chat_id,
        "content_hash": content_hash,
        "created_at": now,
        "uploaded_at": now,
        "updated_at": now,
        "error": None
    }

    await db["documents"].insert_one(doc_record)
    print(f"[UPLOAD] Received file '{original_filename}' ({len(contents)} bytes) -> assigned document_id={document_id} storage_path='{file_path}'")

    # 4. Auto-attach canonical document_id to chat session if chat_id provided
    if chat_id:
        try:
            conversation = await db["conversations"].find_one(conversation_filter(chat_id))
            if conversation:
                current_files = list(conversation.get("selected_document_ids") or [])
                if document_id not in current_files:
                    current_files.append(document_id)
                await db["conversations"].update_one(
                    conversation_filter(chat_id),
                    {"$set": {"selected_document_ids": current_files, "updated_at": now}},
                )
        except Exception as attach_err:
            print(f"[DOC NOTICE] Auto-attach document {document_id} to chat {chat_id} failed: {attach_err}")

    background_tasks.add_task(rag_engine.process_file, file_path, original_filename, document_id, chat_id)
    return {
        "message": f"File '{original_filename}' uploaded and is being processed",
        "status": "Processing",
        "filename": original_filename,
        "name": original_filename,
        "document_id": document_id,
        "chat_id": chat_id,
        "size_bytes": len(contents),
        "file_type": ext_clean
    }


async def reprocess_uploaded_file(file_path: str, filename: str, document_id: Optional[str] = None):
    """Rebuild one document's index without touching the uploaded original."""
    await rag_engine.remove_file(filename=filename, document_id=document_id)
    await rag_engine.process_file(file_path, filename, document_id=document_id)


@app.post("/files/{identifier}/reprocess")
@app.post("/documents/{identifier}/reprocess")
async def reprocess_file(background_tasks: BackgroundTasks, identifier: str):
    db = await get_db()
    ident = sanitize_document_identifier(identifier)
    query = {"$or": [
        {"document_id": ident},
        {"filename": ident},
        {"disk_filename": ident},
        {"filename": {"$regex": f"^{re.escape(ident)}$", "$options": "i"}}
    ]}
    if ObjectId.is_valid(ident):
        query["$or"].append({"_id": ObjectId(ident)})
    query["$or"].append({"_id": ident})

    record = await db["documents"].find_one(query)
    if not record:
        raise HTTPException(status_code=404, detail="Document not found")

    storage_path = get_document_storage_path(record)
    if not storage_path or not os.path.isfile(storage_path):
        raise HTTPException(status_code=404, detail="Document physical file not found on disk")

    doc_id = record.get("document_id") or str(record.get("_id"))
    filename = record.get("filename")

    await db["documents"].update_one(
        {"_id": record["_id"]},
        {"$set": {"status": "PROCESSING", "stage": "Processing", "updated_at": datetime.now(), "error": None}},
    )
    background_tasks.add_task(reprocess_uploaded_file, storage_path, filename, doc_id)
    return {"message": f"Reprocessing {filename} has started", "document_id": doc_id}


@app.delete("/api/documents/{identifier}")
@app.delete("/documents/{identifier}")
@app.delete("/files/{identifier}")
async def delete_file(identifier: Optional[str] = None, filename: Optional[str] = None, document_id: Optional[str] = None):
    """Permanently delete a document, its chunks, vector embeddings, storage file, and disassociate from all conversations."""
    raw_target = identifier or document_id or filename
    if not raw_target:
        raise HTTPException(status_code=400, detail="Document identifier is required.")

    target = sanitize_document_identifier(raw_target)
    db = await get_db()
    query = {"$or": [
        {"document_id": target},
        {"filename": target},
        {"disk_filename": target},
        {"filename": {"$regex": f"^{re.escape(target)}$", "$options": "i"}}
    ]}
    if ObjectId.is_valid(target):
        query["$or"].append({"_id": ObjectId(target)})
    query["$or"].append({"_id": target})

    doc_record = await db["documents"].find_one(query)
    actual_filename = doc_record.get("filename") if doc_record else target
    actual_doc_id = doc_record.get("document_id") if doc_record else target

    # Execute full cascade deletion via rag_engine
    await rag_engine.remove_file(filename=actual_filename, document_id=actual_doc_id)

    # Extra safety cleanup on MongoDB collections
    try:
        if actual_doc_id:
            del_filter = {"$or": [{"document_id": actual_doc_id}, {"_id": actual_doc_id}]}
            chunk_filter = {"$or": [{"document_id": actual_doc_id}, {"metadata.document_id": actual_doc_id}]}
        else:
            del_filter = {"filename": actual_filename}
            chunk_filter = {"$or": [{"source": actual_filename}, {"metadata.source": actual_filename}]}

        await db["documents"].delete_many(del_filter)
        await db["chunks"].delete_many(chunk_filter)
        await db["vector_store"].delete_many(chunk_filter)
        await db["summaries"].delete_many(chunk_filter)
        
        pull_targets = [actual_doc_id, target] if actual_doc_id else [actual_filename, target]
        await db["conversations"].update_many(
            {},
            {"$pull": {"selected_document_ids": {"$in": pull_targets}}}
        )
        if not actual_doc_id or (await db["documents"].count_documents({"filename": actual_filename})) == 0:
            await db["conversations"].update_many(
                {"document_name": actual_filename},
                {"$set": {"document_name": None}}
            )
    except Exception as e:
        print(f"Error executing DB cleanup for deleted file {actual_filename}: {e}")


    return {
        "success": True,
        "message": f"Document '{actual_filename}' has been permanently deleted",
        "status": "deleted",
        "filename": actual_filename,
        "document_id": actual_doc_id
    }



class ClearSessionRequest(BaseModel):
    conversation_id: str



@app.post("/clear_session")
async def clear_session(request: ClearSessionRequest):
    try:
        db = await get_db()
        conversation = await require_conversation(request.conversation_id)
        await db.history.delete_many({"conversation_id": str(conversation["_id"])})
    except Exception as e:
        print(f"Clear session unavailable: {e}")
    return {"message": "Chat session cleared"}

@app.post("/clear_history")
async def clear_history():
    try:
        db = await get_db()
    except Exception as e:
        print(f"Clear history unavailable: {e}")
        raise HTTPException(status_code=503, detail="History storage is unavailable.") from e
    await db.history.delete_many({})
    return {"message": "All history cleared"}

@app.delete("/history/{history_id}")
async def delete_history_item(history_id: str):
    try:
        db = await get_db()
    except Exception as e:
        print(f"Delete history unavailable: {e}")
        raise HTTPException(status_code=503, detail="History storage is unavailable.") from e

    try:
        result = await db.history.delete_one({"_id": ObjectId(history_id)})
        if result.deleted_count == 1:
            return {"message": "Item deleted"}
        else:
            raise HTTPException(status_code=404, detail="Item not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str):
    try:
        db = await get_db()
    except Exception as e:
        print(f"DELETE /chats/{chat_id} database error: {type(e).__name__}: {e}")
        raise HTTPException(status_code=503, detail=f"Conversation storage is unavailable: {e}") from e

    try:
        query = conversation_filter(chat_id)
        conversation = await db["conversations"].find_one(query)
        session_id = conversation.get("session_id") if conversation else chat_id
        
        # Delete related chat messages from history
        history_result = await db.history.delete_many({
            "$or": [
                {"conversation_id": chat_id},
                {"session_id": chat_id},
                {"conversation_id": session_id},
                {"session_id": session_id}
            ]
        })

        deleted_conv_count = 0
        if conversation:
            conv_del_res = await db["conversations"].delete_one({"_id": conversation["_id"]})
            deleted_conv_count = conv_del_res.deleted_count
        else:
            conv_del_res = await db["conversations"].delete_many(query)
            deleted_conv_count = conv_del_res.deleted_count

        if not conversation and history_result.deleted_count == 0 and deleted_conv_count == 0:
            raise HTTPException(status_code=404, detail="Conversation not found.")

        print(f"DELETE /chats/{chat_id} status=200 messages_deleted={history_result.deleted_count} conv_deleted={deleted_conv_count}")
        return {
            "success": True,
            "message": "Conversation deleted successfully",
            "conversation_id": chat_id,
            "messages_deleted": history_result.deleted_count
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"DELETE /chats/{chat_id} error: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"Unable to delete conversation: {e}") from e

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
