import os
import io
import re
import uuid
import asyncio
import time
from typing import List, Dict, Optional
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader, 
    UnstructuredExcelLoader, UnstructuredMarkdownLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
try:
    from .database import get_db, db_instance, get_env_value, UPLOAD_FOLDER
    from .local_store import LocalHashEmbeddings
except ImportError:
    from database import get_db, db_instance, get_env_value, UPLOAD_FOLDER
    from local_store import LocalHashEmbeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    HuggingFaceEmbeddings = None


def format_page_ranges(pages: List[int]) -> str:
    """Format a list of page numbers (e.g. [4, 5, 6, 7, 8, 11, 13]) into formatted ranges ('p. 4-8, 11, 13')."""
    if not pages:
        return ""
    sorted_pages = sorted(list(set(p for p in pages if isinstance(p, int) and p > 0)))
    if not sorted_pages:
        return ""

    ranges = []
    range_start = sorted_pages[0]
    prev = sorted_pages[0]

    for page in sorted_pages[1:]:
        if page == prev + 1:
            prev = page
        else:
            if range_start == prev:
                ranges.append(str(range_start))
            else:
                ranges.append(f"{range_start}-{prev}")
            range_start = page
            prev = page

    if range_start == prev:
        ranges.append(str(range_start))
    else:
        ranges.append(f"{range_start}-{prev}")

    formatted = ", ".join(ranges)
    return f"p. {formatted}" if formatted else ""


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


import hashlib


class QueryIntent:
    MODE_METADATA = "METADATA"
    MODE_TOTAL_PAGES = "TOTAL_PAGES"
    MODE_TOC = "TOC"
    MODE_CHAPTER = "CHAPTER"
    MODE_SECTION = "SECTION"
    MODE_PAGE = "PAGE"
    MODE_RANGE = "RANGE"
    MODE_COMPARE_PAGES = "COMPARE_PAGES"
    MODE_BLANK_PAGES = "BLANK_PAGES"
    MODE_TABLE = "TABLE"
    MODE_IMAGE = "IMAGE"
    MODE_WHICH_PAGE = "WHICH_PAGE"
    MODE_DOCUMENT = "DOCUMENT"
    MODE_SEMANTIC = "SEMANTIC"
    MODE_FOLLOW_UP = "FOLLOW_UP"
    MODE_MULTI_PDF = "MULTI_PDF"


# ── Context Budget & Token Constants (Prevents Groq 413 "Request too large") ──
MAX_RETRIEVED_CONTEXT_TOKENS = 2500  # Strict context token budget
MAX_CHAT_HISTORY_TOKENS = 800       # Strict chat history token budget
MAX_CHAT_HISTORY_MESSAGES = 6       # Max recent conversation turns
MAX_RESPONSE_TOKENS = 1200          # Output headroom


def estimate_tokens(text: str) -> int:
    """Conservative token estimator (~3.5 chars per token for English & code)."""
    if not text:
        return 0
    return max(1, int(len(text) / 3.5))


def truncate_context_to_budget(docs: List[Document], max_tokens: int = MAX_RETRIEVED_CONTEXT_TOKENS) -> List[Document]:
    """Strictly enforce maximum retrieved context tokens to prevent LLM request overflow."""
    selected = []
    current_tokens = 0
    trunc_notice = "\n... [Context truncated to stay within token budget]"
    trunc_notice_tokens = estimate_tokens(trunc_notice)

    for doc in docs:
        t_count = estimate_tokens(doc.page_content)
        if current_tokens + t_count <= max_tokens:
            selected.append(doc)
            current_tokens += t_count
        elif current_tokens < max_tokens:
            remaining = max_tokens - current_tokens - trunc_notice_tokens
            if remaining > 50:
                allowed_chars = int(remaining * 3.3)
                trunc_txt = doc.page_content[:allowed_chars]
                if '\n' in trunc_txt:
                    trunc_txt = trunc_txt.rsplit('\n', 1)[0]
                doc_copy = Document(
                    page_content=trunc_txt + trunc_notice,
                    metadata=doc.metadata
                )
                if estimate_tokens(doc_copy.page_content) + current_tokens <= max_tokens:
                    selected.append(doc_copy)
            break
        else:
            break
    return selected


def trim_chat_history_to_budget(messages: Optional[List[BaseMessage]], max_messages: int = MAX_CHAT_HISTORY_MESSAGES, max_tokens: int = MAX_CHAT_HISTORY_TOKENS) -> List[BaseMessage]:
    """Bound chat history to recent messages within a strict token budget."""
    if not messages:
        return []
    recent = list(messages[-max_messages:])
    bounded = []
    curr_tokens = 0
    for msg in reversed(recent):
        content_str = msg.content if isinstance(msg.content, str) else str(msg.content)
        msg_tokens = estimate_tokens(content_str)
        if curr_tokens + msg_tokens <= max_tokens:
            bounded.insert(0, msg)
            curr_tokens += msg_tokens
        else:
            break
    return bounded


# ── Structured Debug Logging Helpers ─────────────────────────────────────────
def log_upload(doc_id: str, filename: str, size: int):
    print(f"[UPLOAD] document_id={doc_id} filename='{filename}' size_bytes={size}")

def log_pdf_parse(doc_id: str, filename: str, pages: int, text_pages: int, toc_items: int):
    print(f"[PDF_PARSE] document_id={doc_id} filename='{filename}' total_pages={pages} text_pages={text_pages} toc_items={toc_items}")

def log_chunk(doc_id: str, num_chunks: int, avg_size: float):
    print(f"[CHUNK] document_id={doc_id} chunks_created={num_chunks} avg_chars={avg_size:.1f} est_tokens_per_chunk={int(avg_size//3.5)}")

def log_embedding(doc_id: str, num_chunks: int, dim: int):
    print(f"[EMBEDDING] document_id={doc_id} chunks={num_chunks} embedding_dim={dim}")

def log_vector_store(doc_id: str, inserted: int):
    print(f"[VECTOR_STORE] document_id={doc_id} vectors_stored={inserted}")

def log_retrieval(doc_id: str, query: str, num_chunks: int, pages: List[int], scores: List[float], context_tokens: int):
    formatted_scores = [round(s, 3) for s in scores] if scores else []
    print(f"[RETRIEVAL] document_id={doc_id} query='{query[:50]}' chunks={num_chunks} pages={pages} similarity_scores={formatted_scores} context_token_estimate={context_tokens}")

def log_llm(model: str, prompt_tokens: int, max_tokens: int):
    print(f"[LLM] model={model} prompt_token_est={prompt_tokens} max_response_tokens={max_tokens} total_request_est={prompt_tokens + max_tokens}")


def calculate_file_hash(filepath: str) -> str:
    """Compute SHA-256 hash for document integrity and deduplication."""
    if not os.path.isfile(filepath):
        return ""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def get_document_storage_path(doc: Optional[Dict], upload_folder: str) -> Optional[str]:
    """Resolve physical document file path on disk across storage_path, file_path, disk_filename, and filename variants."""
    if not doc or not isinstance(doc, dict):
        return None
    candidates = [
        doc.get("storage_path"),
        doc.get("file_path"),
        os.path.join(upload_folder, doc.get("disk_filename", "")) if doc.get("disk_filename") else None,
        os.path.join(upload_folder, f"{doc.get('document_id', '')}_{doc.get('filename', '')}") if doc.get("document_id") and doc.get("filename") else None,
        os.path.join(upload_folder, f"{doc.get('document_id', '')}_{re.sub(r'[^a-zA-Z0-9_.-]', '_', os.path.basename(doc.get('filename', '')))}") if doc.get("document_id") and doc.get("filename") else None,
        os.path.join(upload_folder, doc.get("filename", "")) if doc.get("filename") else None,
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return os.path.abspath(c)
    return None


def extract_table_structures(text: str, page_num: int) -> List[Dict]:
    """Detect and parse tabular content from page text into structured objects and Markdown."""
    tables = []
    if not text:
        return tables
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    table_lines = []

    def build_table(t_lines):
        if len(t_lines) < 2:
            return None
        # Parse header
        headers = [h.strip() for h in re.split(r'\||\s{2,}|\t', t_lines[0]) if h.strip()]
        if not headers or len(headers) < 2:
            return None
        rows = []
        for r_line in t_lines[1:]:
            if re.match(r'^[-|\s:]+$', r_line):
                continue
            cells = [c.strip() for c in re.split(r'\||\s{2,}|\t', r_line) if c.strip()]
            if cells:
                rows.append(cells)
        if not rows:
            return None
        max_cols = max(len(headers), max(len(r) for r in rows))
        padded_headers = headers + [f"Col {i+1}" for i in range(len(headers), max_cols)]
        md_rows = [
            f"| {' | '.join(padded_headers)} |",
            f"| {' | '.join(['---'] * max_cols)} |"
        ]
        for r in rows:
            padded_row = r + [""] * (max_cols - len(r))
            md_rows.append(f"| {' | '.join(padded_row)} |")
        return {
            "table_index": len(tables) + 1,
            "page_number": page_num,
            "headers": padded_headers,
            "rows": rows,
            "markdown": "\n".join(md_rows)
        }

    for line in lines:
        if "|" in line or (re.search(r'\b\S+\s{2,}\S+\b', line) and len(line.split()) >= 3):
            table_lines.append(line)
        else:
            if len(table_lines) >= 2:
                tbl = build_table(table_lines)
                if tbl:
                    tables.append(tbl)
            table_lines = []

    if len(table_lines) >= 2:
        tbl = build_table(table_lines)
        if tbl:
            tables.append(tbl)

    return tables


def extract_table_of_contents_and_chapters(reader, pages_data: List[Dict], total_pages: int) -> List[Dict]:
    """Extract structured Table of Contents with chapters, titles, start_page, and end_page.
    Combines:
    1. PyPDF reader outline/bookmarks (if present)
    2. In-text Table of Contents scanner across initial pages
    3. Document-wide chapter heading scanner across all pages
    """
    toc_entries = []
    seen_chapters = set()

    # 1. Digital Outline / Bookmarks (if present in PDF metadata)
    if hasattr(reader, "outline") and reader.outline:
        def traverse(items):
            for item in items:
                if isinstance(item, list):
                    traverse(item)
                elif hasattr(item, "title"):
                    try:
                        title = str(item.title).strip()
                        page_num = reader.get_destination_page_number(item) + 1
                        ch_m = re.search(r'\b(?:chapter|chap\.?)\s*(\d+|[ivxlcdm]+)\b', title, re.I)
                        ch_num = None
                        if ch_m:
                            try:
                                ch_num = int(ch_m.group(1))
                            except Exception:
                                pass
                        toc_entries.append({
                            "chapter": ch_num,
                            "title": title,
                            "start_page": page_num
                        })
                    except Exception:
                        pass
        try:
            traverse(reader.outline)
        except Exception:
            pass

    # 2. In-Text Table of Contents scanning across initial pages
    toc_page_patterns = [
        r'\b(?:table\s+of\s+contents|contents|brief\s+contents)\b',
    ]
    for p_entry in pages_data[:min(30, len(pages_data))]:
        p_num = p_entry.get("page_number", 1)
        p_txt = p_entry.get("text", "")
        if not p_txt:
            continue
        
        lines = [ln.strip() for ln in p_txt.splitlines() if ln.strip()]
        for line in lines:
            # Pattern: "Chapter 2 — Lists and Tuples ... 25" or "Chapter 2: Lists and Tuples 25"
            ch_match = re.search(
                r'^(?:Chapter\s*(\d+|[IVXLCDM]+)\s*[:—–-]?\s*)?([A-Za-z0-9\s,\'\"\-—–()]+?)\s*[.\s_…-]{2,}\s*(\d+)\s*$',
                line,
                re.I
            ) or re.search(
                r'^Chapter\s*(\d+)\s*[:—–-]?\s*(.+?)\s+(\d+)$',
                line,
                re.I
            )
            if ch_match:
                try:
                    c_num_str = ch_match.group(1)
                    title_str = ch_match.group(2).strip(" ._-—–…")
                    target_p_str = ch_match.group(3) if len(ch_match.groups()) >= 3 else ch_match.group(2)
                    
                    c_num = int(c_num_str) if (c_num_str and c_num_str.isdigit()) else None
                    target_p = int(target_p_str) if (target_p_str and target_p_str.isdigit()) else p_num
                    
                    if 1 <= target_p <= total_pages and title_str and len(title_str) > 2:
                        key = (c_num, title_str.lower())
                        if key not in seen_chapters:
                            seen_chapters.add(key)
                            full_title = f"Chapter {c_num}: {title_str}" if c_num else title_str
                            toc_entries.append({
                                "chapter": c_num,
                                "title": full_title,
                                "raw_title": title_str,
                                "start_page": target_p
                            })
                except Exception:
                    pass

    # 3. Document-wide Chapter Heading Scanning (scans all pages for chapter headers)
    for p_entry in pages_data:
        p_num = p_entry.get("page_number", 1)
        p_txt = p_entry.get("text", "")
        if not p_txt:
            continue
        lines = [ln.strip() for ln in p_txt.splitlines() if ln.strip()]
        for idx, line in enumerate(lines[:8]):
            # Header like: "Chapter 6 — The sympy Library" or "Chapter 6: The sympy Library"
            head_m = re.match(r'^Chapter\s*(\d+)\s*[:—–-]?\s*(.*)$', line, re.I)
            if head_m:
                try:
                    c_num = int(head_m.group(1))
                    c_title = head_m.group(2).strip(" :—–-")
                    if not c_title and idx + 1 < len(lines):
                        next_l = lines[idx + 1].strip()
                        if len(next_l) < 80 and not next_l.lower().startswith("chapter"):
                            c_title = next_l
                    if c_num and c_num not in [e.get("chapter") for e in toc_entries if e.get("chapter")]:
                        clean_title = f"Chapter {c_num}: {c_title}" if c_title else f"Chapter {c_num}"
                        toc_entries.append({
                            "chapter": c_num,
                            "title": clean_title,
                            "raw_title": c_title or f"Chapter {c_num}",
                            "start_page": p_num
                        })
                except Exception:
                    pass

    # Sort TOC entries by start_page and chapter number
    toc_entries = sorted(
        toc_entries,
        key=lambda x: (x.get("start_page", 1), x.get("chapter") or 999)
    )

    # Deduplicate entries with the same chapter number
    deduped = []
    seen_ch_nums = set()
    for entry in toc_entries:
        ch = entry.get("chapter")
        if ch is not None:
            if ch in seen_ch_nums:
                continue
            seen_ch_nums.add(ch)
        deduped.append(entry)
    toc_entries = deduped

    # Compute end_page for each section
    for i in range(len(toc_entries)):
        if i + 1 < len(toc_entries):
            next_start = toc_entries[i + 1]["start_page"]
            toc_entries[i]["end_page"] = max(toc_entries[i]["start_page"], next_start - 1)
        else:
            toc_entries[i]["end_page"] = total_pages

    return toc_entries


def parse_pdf_outline(reader) -> List[Dict]:
    """Compatibility alias for TOC parser."""
    return extract_table_of_contents_and_chapters(reader, [], len(reader.pages) if hasattr(reader, "pages") else 1)


def parse_query_intent(query: str, chat_history: Optional[List[BaseMessage]] = None) -> tuple[str, dict]:
    """Classify user query into (intent_mode, metadata_kwargs) with contextual follow-up resolution."""
    if not query:
        return QueryIntent.MODE_SEMANTIC, {}
    q_lower = query.lower().strip()

    # ── 1. FOLLOW-UP PRONOUN RESOLUTION (e.g. "explain that page", "compare it with page 15", "now explain that in simple language")
    follow_up_patterns = [
        r'\b(?:explain|summarize|simplify|describe|elaborate\s+on|detail)\s+(?:that|this|the)\s+page\b',
        r'\b(?:what\s+about|tell\s+me\s+more\s+about)\s+(?:that|this|the)\s+page\b',
        r'\b(?:explain|summarize|simplify)\s+(?:it|that)\s+in\s+simple\s+(?:language|terms|words)\b',
        r'\bcompare\s+(?:it|that)\s+(?:with|to)\s+pages?\s*#?\s*(\d+)\b',
        r'\bwhat\s+else\s+is\s+(?:on|in)\s+(?:that|this)\s+page\b',
        r'\bnow\s+explain\s+that\b',
        r'\bexplain\s+that\s+in\s+simple\b',
    ]
    if any(re.search(p, q_lower) for p in follow_up_patterns) and chat_history:
        recent_pages = []
        for msg in reversed(chat_history[-6:]):
            text_val = msg.content if hasattr(msg, "content") else str(msg)
            if isinstance(text_val, str):
                p_matches = re.findall(r'(?:page|p\.)\s*#?\s*(\d+)', text_val, re.I)
                for p_str in p_matches:
                    try:
                        p_val = int(p_str)
                        if p_val not in recent_pages:
                            recent_pages.append(p_val)
                    except ValueError:
                        pass
        compare_it_match = re.search(r'\bcompare\s+(?:it|that)\s+(?:with|to)\s+pages?\s*#?\s*(\d+)\b', q_lower)
        if compare_it_match and recent_pages:
            p2 = int(compare_it_match.group(1))
            p1 = recent_pages[0]
            return QueryIntent.MODE_COMPARE_PAGES, {"pages": [p1, p2], "page1": p1, "page2": p2, "is_follow_up": True}
        if recent_pages:
            target_p = recent_pages[0]
            return QueryIntent.MODE_PAGE, {"target_page": target_p, "pages": [target_p], "is_follow_up": True}

    # ── 2. TOTAL PAGES INQUIRY
    total_pages_patterns = [
        r'\bhow\s+many\s+(?:total\s+)?pages\b',
        r'\bhow\s+many\s+(?:total\s+)?pages\s+(?:are\s+there|are\s+in|in|does|is|has)\b',
        r'\btotal\s+(?:number\s+of\s+)?pages\b',
        r'\bpage\s+count\b',
        r'\bnumber\s+of\s+pages\b',
        r'\bhow\s+long\s+is\s+.*in\s+pages\b',
        r'\bhow\s+many\s+pages\s+total\b',
        r'\bcount\s+of\s+pages\b',
    ]
    if any(re.search(pat, q_lower) for pat in total_pages_patterns):
        return QueryIntent.MODE_TOTAL_PAGES, {}

    # ── 3. DOCUMENT METADATA INQUIRY
    metadata_patterns = [
        r'\bwhat\s+is\s+the\s+filename\b',
        r'\bfile\s+size\b',
        r'\bdocument\s+metadata\b',
        r'\bdocument\s+properties\b',
        r'\bfile\s+properties\b',
        r'\bwhen\s+was\s+this\s+uploaded\b',
        r'\bwhat\s+type\s+of\s+(?:file|document)\b',
        r'\bfile\s+hash\b',
        r'\btell\s+me\s+about\s+this\s+(?:file|document|pdf)\b',
    ]
    if any(re.search(pat, q_lower) for pat in metadata_patterns):
        return QueryIntent.MODE_METADATA, {}

    # ── 4. CHAPTER SPECIFIC QUERY (e.g. "What is Chapter 6?", "What is Chapter 7 of this PDF?", "What is Chapter 8?")
    chapter_patterns = [
        r'\b(?:what\s+is\s+|tell\s+me\s+about\s+|explain\s+)?chapter\s*(\d+|[ivxlcdm]+)\b',
        r'\bchapter\s*(\d+|[ivxlcdm]+)\s+(?:title|summary|overview|topics?|page|start|range)\b',
        r'\bwhat\s+(?:page\s+does\s+)?chapter\s*(\d+|[ivxlcdm]+)\b',
        r'\bchapter\s*(\d+|[ivxlcdm]+)\b',
    ]
    for c_pat in chapter_patterns:
        ch_m = re.search(c_pat, q_lower)
        if ch_m:
            ch_raw = ch_m.group(1)
            roman_map = {'i': 1, 'ii': 2, 'iii': 3, 'iv': 4, 'v': 5, 'vi': 6, 'vii': 7, 'viii': 8, 'ix': 9, 'x': 10}
            try:
                ch_num = int(ch_raw) if ch_raw.isdigit() else roman_map.get(ch_raw.lower(), 1)
            except Exception:
                ch_num = 1
            # Avoid triggering if query is asking general chapters list
            if not any(k in q_lower for k in ["what chapters", "which chapters", "all chapters", "list chapters", "chapter list", "chapter structure"]):
                return QueryIntent.MODE_CHAPTER, {"chapter_num": ch_num, "chapter_raw": ch_raw}

    # ── 5. TABLE OF CONTENTS / CHAPTER STRUCTURE
    toc_patterns = [
        r'\btable\s+of\s+contents\b',
        r'\bshow\s+(?:me\s+)?(?:the\s+)?(?:toc|table\s+of\s+contents|chapters?|outline)\b',
        r'\bwhat\s+chapters?\s+(?:are\s+there|exist|are\s+in|in)\b',
        r'\bchapter\s+(?:structure|list|outline|breakdown)\b',
        r'\blist\s+(?:the\s+)?(?:chapters?|sections?|toc)\b',
        r'\bwhat\s+sections?\s+are\s+(?:there|in)\b',
        r'\boverall\s+structure\b',
        r'\bstructure\s+of\s+(?:this|the)\s+document\b',
        r'\bchapter[\s-]wise\s+topics\b',
    ]
    if any(re.search(pat, q_lower) for pat in toc_patterns):
        return QueryIntent.MODE_TOC, {}

    # ── 5. TABLE SPECIFIC QUERY
    table_patterns = [
        r'\btables?\b',
        r'\bwhat\s+(?:is|are)\s+(?:the\s+)?(?:values?|numbers?|rows?|columns?)\s+in\s+the\s+table\b',
        r'\bhighest\s+value\s+in\s+the\s+table\b',
        r'\blowest\s+value\s+in\s+the\s+table\b',
        r'\bextract\s+(?:the\s+)?table\b',
        r'\bshow\s+(?:all\s+)?tables\b',
    ]
    if any(re.search(pat, q_lower) for pat in table_patterns):
        p_match = re.search(r'\bpages?\s*#?\s*(\d+)\b', q_lower)
        target_p = int(p_match.group(1)) if p_match else None
        return QueryIntent.MODE_TABLE, {"target_page": target_p}

    # ── 6. IMAGE / DIAGRAM SPECIFIC QUERY
    image_patterns = [
        r'\bdiagrams?\b',
        r'\barchitecture\s+diagram\b',
        r'\bcharts?\b',
        r'\bfigures?\b',
        r'\bimages?\b',
        r'\billustrations?\b',
    ]
    if any(re.search(pat, q_lower) for pat in image_patterns):
        p_match = re.search(r'\bpages?\s*#?\s*(\d+)\b', q_lower)
        target_p = int(p_match.group(1)) if p_match else None
        return QueryIntent.MODE_IMAGE, {"target_page": target_p}

    # ── 7. MULTI-PAGE SYNTHESIS (e.g. "based on pages 3, 7 and 15", "pages 2, 4, 6")
    multi_p_match = re.findall(r'\bpages?\s*#?\s*(\d+)', q_lower)
    if len(multi_p_match) >= 3:
        p_list = sorted(list(set(int(x) for x in multi_p_match)))
        return QueryIntent.MODE_COMPARE_PAGES, {"pages": p_list, "multi_pages": p_list}

    # ── 8. COMPARE PAGES / CROSS-PAGE
    compare_match = re.search(r'\bcompare\s+(?:the\s+)?(?:discussion\s+on\s+)?pages?\s*#?\s*(\d+)\s*(?:and|with|to|vs\.?)\s*(?:pages?\s*#?\s*)?(\d+)\b', q_lower) or \
                    re.search(r'\bdifference\s+between\s+pages?\s*#?\s*(\d+)\s*and\s*(?:pages?\s*#?\s*)?(\d+)\b', q_lower) or \
                    re.search(r'\bpages?\s*#?\s*(\d+)\s*(?:vs\.?|versus)\s*pages?\s*#?\s*(\d+)\b', q_lower) or \
                    re.search(r'\bhow\s+does\s+pages?\s*#?\s*(\d+)\s+relate\s+to\s+pages?\s*#?\s*(\d+)\b', q_lower)
    if compare_match:
        p1 = int(compare_match.group(1))
        p2 = int(compare_match.group(2))
        return QueryIntent.MODE_COMPARE_PAGES, {"pages": [p1, p2], "page1": p1, "page2": p2}

    if re.search(r'\bcross\s*(?:-| )?pages?\b', q_lower) or "across multiple pages" in q_lower or "across the whole document" in q_lower:
        if len(multi_p_match) >= 2:
            p_list = sorted(list(set(int(x) for x in multi_p_match)))
            return QueryIntent.MODE_COMPARE_PAGES, {"pages": p_list, "multi_pages": p_list}
        return QueryIntent.MODE_DOCUMENT, {"is_cross_page": True}

    # ── 9. PAGE RANGE (e.g. "pages 3 to 5", "pages 7-10", "p. 3-5")
    range_match = re.search(r'\bpages?\s*#?\s*(\d+)\s*(?:to|-|through)\s*(?:pages?\s*#?\s*)?(\d+)\b', q_lower) or \
                  re.search(r'\bbetween\s+pages?\s*#?\s*(\d+)\s*and\s*(?:pages?\s*#?\s*)?(\d+)\b', q_lower) or \
                  re.search(r'\bp\.\s*(\d+)\s*-\s*(\d+)\b', q_lower)
    if range_match:
        start_p = int(range_match.group(1))
        end_p = int(range_match.group(2))
        if start_p <= end_p and (end_p - start_p) <= 50:
            return QueryIntent.MODE_RANGE, {"start_page": start_p, "end_page": end_p, "pages": list(range(start_p, end_p + 1))}

    # ── 10. BLANK / EMPTY / SCANNED PAGES INQUIRY (e.g. "What are the remaining blank pages?", "which pages are blank", "any empty pages")
    blank_page_patterns = [
        r'\b(?:which|what|any|find|list|show)\s+(?:are\s+the\s+)?(?:remaining\s+)?(?:blank|empty|unreadable|scanned|image[\s-]only)\s+pages?\b',
        r'\b(?:remaining\s+)?blank\s+pages?\b',
        r'\bempty\s+pages?\b',
        r'\bpages?\s+with\s+no\s+(?:text|content)\b',
        r'\bare\s+there\s+(?:any\s+)?blank\s+pages\b',
    ]
    if any(re.search(pat, q_lower) for pat in blank_page_patterns):
        return QueryIntent.MODE_BLANK_PAGES, {}

    # ── 11. WHICH PAGE / TOPIC LOCATION INQUIRY
    which_patterns = [
        r'\bwhich\s+pages?\b',
        r'\bwhat\s+pages?\s+(?:is|are|does|contains?|discusses?|talks?\s+about|mentions?)\b',
        r'\bwhere\s+is\s+.*(?:discussed|mentioned|explained|defined|found|located|detailed)\b',
        r'\bwhere\s+does\s+(?:the\s+)?(?:pdf|document)\s+discuss\b',
        r'\bon\s+which\s+pages?\b',
        r'\bfind\s+(?:the\s+)?page\b',
        r'\bwhich\s+pages?\s+(?:contain|contains|discuss|discusses|mention|mentions)\b',
    ]
    if any(re.search(pat, q_lower) for pat in which_patterns):
        return QueryIntent.MODE_WHICH_PAGE, {}

    # ── 12. SINGLE PAGE INQUIRY (e.g. "What is on page 3?", "explain page 1", "middle page", "last page")
    if re.search(r'\b(?:middle|center)\s*(?:-| )?pages?\b', q_lower):
        return QueryIntent.MODE_PAGE, {"target_page": "MIDDLE", "pages": ["MIDDLE"]}

    if re.search(r'\b(?:last|final|ending)\s*(?:-| )?pages?\b', q_lower):
        return QueryIntent.MODE_PAGE, {"target_page": "LAST", "pages": ["LAST"]}

    if re.search(r'\b(?:first|opening|initial)\s*(?:-| )?pages?\b', q_lower):
        return QueryIntent.MODE_PAGE, {"target_page": 1, "pages": [1]}

    single_match = re.search(r'\bpages?\s*#?\s*(\d+)\b', q_lower) or re.search(r'\bp\.?\s*(\d+)\b', q_lower)
    if single_match:
        page_num = int(single_match.group(1))
        return QueryIntent.MODE_PAGE, {"target_page": page_num, "pages": [page_num]}

    # ── 12. WHOLE DOCUMENT / PAGE-BY-PAGE / SUMMARY INQUIRY
    page_by_page_patterns = [
        r'\b(?:main\s+)?topics?\s+of\s+(?:every|each|all)\s+pages?\b',
        r'\b(?:main\s+)?topics?\s+for\s+(?:every|each|all)\s+pages?\b',
        r'\bgive\s+(?:the\s+)?(?:main\s+)?topics?\s+of\s+(?:every|each|all)\s+pages?\b',
        r'\b(?:every|each)\s+page\s+(?:summary|topic|content|breakdown)\b',
        r'\bpage[\s-]by[\s-]page\b',
        r'\bbreakdown\s+of\s+(?:every|each|all)\s+pages?\b',
        r'\ball\s+pages?\s+(?:summary|topics?|overview|breakdown)\b',
        r'\btopic\s+of\s+every\s+page\b',
        r'\btopic\s+of\s+each\s+page\b',
    ]
    if any(re.search(pat, q_lower) for pat in page_by_page_patterns):
        return QueryIntent.MODE_DOCUMENT, {"is_page_by_page": True}

    doc_keywords = [
        "summarize this document", "summarize the document", "summarize entire document",
        "summarize the pdf", "summarize this pdf", "summarize whole pdf", "explain the whole pdf",
        "analyze the whole pdf", "analyze the entire document", "analyze whole pdf", "complete pdf summary",
        "page-by-page summary", "page by page summary", "content of every page", "main content of every page",
        "complete summary", "whole document", "entire document", "what is this pdf about",
        "what is this document about", "overview of this pdf", "overview of this document",
        "tl;dr of this document", "give me a summary of the document",
        "most important concepts", "three most important concepts", "key takeaways",
        "why can't access remaining pages", "can you read the rest", "remaining pages",
        "rest of the pages", "all pages", "full pdf summary", "full document summary"
    ]
    if any(kw in q_lower for kw in doc_keywords):
        is_remaining = "remaining" in q_lower or "rest of" in q_lower or "why can't access" in q_lower
        is_page_by_page = "page-by-page" in q_lower or "page by page" in q_lower or "every page" in q_lower or "each page" in q_lower
        return QueryIntent.MODE_DOCUMENT, {"is_remaining_query": is_remaining, "is_page_by_page": is_page_by_page}

    # ── 13. MULTI-PDF QUERY
    multi_pdf_patterns = [
        r'\bboth\s+(?:pdfs|documents|files)\b',
        r'\ball\s+(?:pdfs|documents|files)\b',
        r'\bcompare\s+(?:the\s+)?(?:pdfs|documents)\b',
    ]
    if any(re.search(pat, q_lower) for pat in multi_pdf_patterns):
        return QueryIntent.MODE_MULTI_PDF, {}

    # ── 14. DEFAULT: SEMANTIC / HYBRID SEARCH
    return QueryIntent.MODE_SEMANTIC, {}





class RAGEngine:
    def __init__(self):
        self.llm = None
        self.embeddings = None
        self.vector_db = None
        self.rag_chain = None
        self.bm25_retriever = None
        self.all_docs = []  # Memory cache for BM25 docs
        self.upload_folder = UPLOAD_FOLDER
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        self.groq_model = get_env_value("GROQ_MODEL", "llama-3.1-8b-instant")
        self.vision_model = get_env_value("GROQ_VISION_MODEL", "qwen/qwen3.6-27b")
        self.vector_dimensions = None
        self._groq_async_client = None

    async def extract_page_vision_ocr(self, page_image_bytes: bytes, page_num: int = 1) -> str:
        """Extract high-precision text, exact code blocks, and diagrams from a rendered PDF page image using Groq Vision."""
        groq_key = get_env_value("GROQ_API_KEY")
        if not groq_key or len(groq_key) < 5:
            return ""

        try:
            from groq import AsyncGroq
            import base64
            img_b64 = base64.b64encode(page_image_bytes).decode("utf-8")
            if self._groq_async_client is None:
                self._groq_async_client = AsyncGroq(api_key=groq_key, timeout=12.0, max_retries=2)

            prompt = (
                "You are an expert OCR and code document transcription engine.\n"
                "Extract all text, code snippets, tables, and diagrams visible in this document/slide image.\n"
                "CRITICAL REQUIREMENTS:\n"
                "1. If there is code in an editor or image, transcribe the EXACT code into appropriate markdown code blocks (e.g. ```python ... ```), preserving exact syntax, indentation, variable names, and comments.\n"
                "2. Transcribe all headings, subheadings, labels, bullet points, and text.\n"
                "3. If there are tables, format them as clean Markdown tables.\n"
                "4. Output only the extracted content directly without conversational greetings or chat commentary."
            )
            response = await self._groq_async_client.chat.completions.create(
                model=self.vision_model or "qwen/qwen3.6-27b",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                        ]
                    }
                ],
                temperature=0.0,
                max_tokens=1500,
            )
            raw_content = response.choices[0].message.content or ""
            clean_text = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
            print(f"[PARSE] [OCR_VISION] Page {page_num}: Extracted {len(clean_text)} chars of vision/code content.")
            return clean_text
        except Exception as ocr_err:
            print(f"[PARSE] [OCR_VISION_WARN] Page {page_num} vision extraction notice: {ocr_err}")
            return ""

    def _get_vector_store(self, sync_collection):
        """Return a MongoDB Atlas vector store.

        ``sync_collection`` **must** be a synchronous ``pymongo.Collection``;
        Motor async collections are not supported by ``MongoDBAtlasVectorSearch``.

        Atlas-only features (``auto_create_index``, ``dimensions``) are only
        passed when the project is connected to Atlas; they cause errors on a
        local ``mongod`` instance.
        """
        kwargs = {
            "collection": sync_collection,
            "embedding": self.embeddings,
            "index_name": "vector_index",
        }
        if self.vector_dimensions and not db_instance.is_local:
            kwargs.update({
                "auto_create_index": True,
                "dimensions": self.vector_dimensions,
                "relevance_score_fn": "cosine",
            })
        return MongoDBAtlasVectorSearch(**kwargs)

    async def initialize(self, sync: bool = True):
        """Initialize models and RAG chain on startup."""
        print("Initializing RAG Engine...")
        groq_key = get_env_value("GROQ_API_KEY")
        has_key = bool(groq_key and len(groq_key) > 5)
        print(f"Groq API key configured: {has_key}")
        if not has_key:
            raise RuntimeError("GROQ_API_KEY is missing. Add it to backend/.env and restart the backend.")
        self.llm = ChatGroq(
            model=self.groq_model,
            api_key=groq_key,
            temperature=0,
            max_retries=2,
        )
        print(f"LLM (Groq) configured with model: {self.groq_model}.")

        # Load Embeddings
        embedding_model_name = get_env_value(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        try:
            if HuggingFaceEmbeddings is None:
                raise RuntimeError("langchain_huggingface is unavailable")
            self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
            print("Embeddings model loaded.")
            self.vector_dimensions = len(self.embeddings.embed_query("dimension probe"))
        except Exception as e:
            print(f"Hugging Face embeddings unavailable, using offline fallback: {e}")
            self.embeddings = LocalHashEmbeddings(dimension=384)
            self.vector_dimensions = self.embeddings.dimension
        print(f"Embedding dimension set to {self.vector_dimensions}.")

        if sync:
            await self.sync_with_uploads()
        await self.rebuild_chain()

    async def sync_with_uploads(self):
        """Ensure all files in the upload folder are indexed and summarized.
        Processes un-indexed files WITHOUT rebuilding the chain each time;
        a single rebuild happens in initialize() after sync completes.

        IMPORTANT: If chunks exist but summary is missing, only retry summary —
        NEVER delete existing chunks/vectors just because summary failed.
        Individual file failures are logged but do not prevent startup.
        """
        # First, perform orphan detection and cleanup for deleted/missing physical files
        await self.cleanup_orphaned_documents()

        if not os.path.exists(self.upload_folder):
            return

        db = await get_db()
        chunks_collection = db["chunks"]
        summaries_collection = db["summaries"]

        for filename in os.listdir(self.upload_folder):
            file_path = os.path.join(self.upload_folder, filename)
            ext = os.path.splitext(filename)[1].lower()
            if ext not in ('.pdf', '.txt', '.docx', '.xlsx', '.xls', '.md', '.csv'):
                continue

            existing_chunks = await chunks_collection.find_one({"source": filename})
            existing_summary = await summaries_collection.find_one({"source": filename})

            if existing_chunks and existing_summary:
                # Fully indexed and summarized — verify total_pages and ensure status is READY
                doc = await db["documents"].find_one({"filename": filename})
                if doc:
                    needs_update = False
                    update_payload = {}
                    if not doc.get("total_pages") or not doc.get("page_count"):
                        actual_pages = 1
                        if ext == ".pdf":
                            try:
                                import pypdf
                                actual_pages = len(pypdf.PdfReader(file_path).pages)
                            except Exception:
                                actual_pages = 1
                        update_payload["total_pages"] = actual_pages
                        update_payload["page_count"] = actual_pages
                        needs_update = True
                    
                    if doc.get("status") not in ("READY", "Ready"):
                        print(f"[SYNC] Fixing stale status for {filename} (had {doc.get('status')}, has chunks+summary)")
                        update_payload["status"] = "READY"
                        update_payload["stage"] = "Ready"
                        update_payload["processing_status"] = "ready"
                        update_payload["summary_status"] = "completed"
                        needs_update = True

                    if needs_update:
                        await db["documents"].update_one({"filename": filename}, {"$set": update_payload})
                continue


            if existing_chunks and not existing_summary:
                # Chunks exist but summary is missing — retry summary ONLY, never delete chunks
                print(f"[SYNC] Retrying summary for {filename} (chunks exist, summary missing)")
                try:
                    from langchain_community.document_loaders import PyPDFLoader
                    loader = PyPDFLoader(file_path)
                    docs = await asyncio.to_thread(loader.load)
                    docs = [d for d in docs if d.page_content and d.page_content.strip()]
                    if docs:
                        summary = await self.generate_summary(docs)
                        await db["summaries"].update_one(
                            {"source": filename},
                            {"$set": {"summary": summary, "source": filename, "status": "COMPLETED"}},
                            upsert=True
                        )
                        await self.update_doc_status(db, filename, status="READY", stage="Ready",
                                                     processing_status="ready", summary_status="completed")
                    else:
                        # Can't read file for summary — mark summary failed but keep document READY
                        await self.update_doc_status(db, filename, status="READY", stage="Ready",
                                                     processing_status="ready", summary_status="failed")
                except Exception as exc:
                    print(f"[SYNC] Summary retry failed for {filename}: {exc}")
                    await self.update_doc_status(db, filename, status="READY", stage="Ready",
                                                 processing_status="ready", summary_status="failed")
                continue

            # No chunks at all — full processing needed
            print(f"[SYNC] Full processing needed for {filename}")
            try:
                await self.process_file(file_path, filename, rebuild=False)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                print(f"WARNING: Failed to process '{filename}' during startup sync — skipping. Error: {exc}")

    # ── Summarization constants ──────────────────────────────────────────
    _SUMMARY_CHUNK_SIZE = 12_000

    async def generate_summary(self, documents: List[Document]) -> str:
        """Generate a concise summary efficiently with representative sampling.

        To prevent long processing delays on large PDFs (e.g. 1.7 MB text with 200+ chunks),
        we sample representative text from the document (beginning, middle, and end) up to 
        36,000 characters maximum. This reduces 100+ LLM API calls down to 1-3 calls.
        """
        try:
            full_text = "\n".join(doc.page_content for doc in documents if doc.page_content and doc.page_content.strip())
            if not full_text.strip():
                return "The document appears to be empty or contain no readable text."

            # Representative text sampling for large documents
            MAX_SUMMARY_TEXT_LENGTH = 36_000
            if len(full_text) > MAX_SUMMARY_TEXT_LENGTH:
                first_part = full_text[:16_000]
                mid_point = len(full_text) // 2
                middle_part = full_text[mid_point - 5_000 : mid_point + 5_000]
                last_part = full_text[-10_000:]
                sampled_text = first_part + "\n\n[...]\n\n" + middle_part + "\n\n[...]\n\n" + last_part
            else:
                sampled_text = full_text

            chunks = [
                sampled_text[i : i + self._SUMMARY_CHUNK_SIZE]
                for i in range(0, len(sampled_text), self._SUMMARY_CHUNK_SIZE)
            ]

            if len(chunks) == 1:
                return await self._summarise_text(chunks[0])

            print(f"  [RAG TIMING] Map-reduce summarisation: {len(chunks)} sampled chunks (total {len(sampled_text)} chars out of {len(full_text)})")
            intermediate_summaries: List[str] = []
            for idx, chunk in enumerate(chunks, 1):
                summary = await self._summarise_text(chunk)
                intermediate_summaries.append(summary)

            combined = "\n\n".join(
                f"[Section {i}] {s}" for i, s in enumerate(intermediate_summaries, 1)
            )
            final_prompt = (
                "You are given partial summaries of different sections of a document. "
                "Combine them into one clear, executive summary capturing the main "
                "objectives, key findings, and important details in 2-3 concise paragraphs.\n\n"
                f"{combined}\n\nFinal Summary:"
            )
            response = await self.llm.ainvoke(final_prompt)
            return response.content

        except Exception as e:
            print(f"[RAG SUMMARY NOTICE] Summary generation fallback activated: {e}")
            preview = (documents[0].page_content.strip()[:300].replace('\n', ' ') + "...") if (documents and documents[0].page_content) else "Document ready for Q&A."
            return f"Summary preview: {preview}"

    async def _summarise_text(self, text: str) -> str:
        """Summarise a single text chunk with instant timeout fallback for rate limits."""
        prompt = (
            "Summarize the following document section in a clear, professional way. "
            "Focus on the main objectives, key findings, and important details. "
            "Keep it to 1-2 paragraphs maximum.\n\n"
            f"Document Content:\n{text}\n\nSummary:"
        )
        try:
            response = await asyncio.wait_for(self.llm.ainvoke(prompt), timeout=20.0)
            return response.content
        except Exception as e:
            print(f"[RAG SUMMARY NOTICE] Summary LLM skipped ({e}). Using text snippet preview.")
            clean_snippet = text.strip()[:300].replace('\n', ' ')
            return f"Extract: {clean_snippet}..."

    async def get_summary(self, filename: str) -> Optional[str]:
        """Fetch a pre-computed summary from MongoDB."""
        db = await get_db()
        summary_doc = await db["summaries"].find_one({"source": filename})
        return summary_doc["summary"] if summary_doc else None

    async def process_entire_document(self, query: str, scoped_docs: List[Document], is_remaining: bool = False) -> dict:
        """Process an entire document in batches using a map-reduce strategy or pre-computed summaries.
        
        Prevents LLM token overflow and rate-limit spikes by utilizing stored document summaries
        or grouping chunks into batches with delay backoffs.
        """
        if not scoped_docs:
            return {"answer": "No indexed document chunks found to process.", "context": []}

        sorted_chunks = sorted(
            scoped_docs,
            key=lambda d: (d.metadata.get("page_number", 0) or d.metadata.get("page", 0), d.metadata.get("chunk_number", 0))
        )
        
        total_chunks = len(sorted_chunks)
        print(f"[RAG] [DOCUMENT MODE] Processing whole document ({total_chunks} total chunks)")

        def format_doc_chunks(chunks):
            formatted = []
            for doc in chunks:
                source = doc.metadata.get('source', 'Unknown')
                page_num = doc.metadata.get('page_number') or doc.metadata.get('page')
                page_str = f", Page {page_num}" if page_num is not None else ""
                formatted.append(f"[Document: {source}{page_str}]\nContent:\n{doc.page_content}")
            return "\n\n---\n\n".join(formatted)

        prefix = ""
        if is_remaining:
            prefix = "I can access the entire indexed document. My previous answer used only the top relevant pages. Here is the comprehensive analysis across all remaining pages:\n\n"

        if total_chunks <= 12:
            formatted_context = format_doc_chunks(sorted_chunks)
            prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"User Request: {query}\n\n"
                f"Document Context (All {total_chunks} Chunks):\n{formatted_context}\n\n"
                "Provide a comprehensive, accurate response grounded strictly in the document context above."
            )
            response = await self.llm.ainvoke(prompt)
            return {"answer": prefix + response.content, "context": sorted_chunks}

        # Check for pre-computed document summaries in MongoDB to avoid high TPM map-reduce bursts
        try:
            db = await get_db()
            sources = list(set(d.metadata.get("source") for d in scoped_docs if d.metadata.get("source")))
            if sources:
                summaries_cursor = db["summaries"].find({"source": {"$in": sources}})
                stored_summaries = await summaries_cursor.to_list(length=len(sources))
                valid_summaries = [s["summary"] for s in stored_summaries if s.get("summary") and s.get("status") == "COMPLETED"]
                if valid_summaries and len(valid_summaries) == len(sources):
                    print(f"[RAG] [DOCUMENT MODE] Using {len(valid_summaries)} pre-computed document summary(ies)...")
                    combined_pre_summary = "\n\n".join([f"=== Summary for {s['source']} ===\n{s['summary']}" for s in stored_summaries if s.get("summary")])
                    
                    final_prompt = (
                        f"{SYSTEM_PROMPT}\n\n"
                        f"User Request: {query}\n\n"
                        f"Pre-computed Summary of Entire Document(s):\n{combined_pre_summary}\n\n"
                        "Provide a comprehensive, clear, and structured response addressing the user's request based on the complete document summary above."
                    )
                    final_res = await self.llm.ainvoke(final_prompt)
                    return {"answer": prefix + final_res.content, "context": sorted_chunks}
        except Exception as summary_err:
            print(f"[RAG WARNING] Failed to fetch pre-computed document summary: {summary_err}")

        # Fallback to batch map-reduce strategy with rate-limit delays
        BATCH_SIZE = 10
        batches = [sorted_chunks[i : i + BATCH_SIZE] for i in range(0, total_chunks, BATCH_SIZE)]
        print(f"[RAG] [DOCUMENT MODE] Batching into {len(batches)} groups for map-reduce...")

        batch_summaries = []
        for idx, batch in enumerate(batches, start=1):
            batch_text = format_doc_chunks(batch)
            batch_prompt = (
                f"Extract and summarize all key information, facts, findings, and details from this section of the document "
                f"(Batch {idx}/{len(batches)}):\n\n{batch_text}\n\nSummary of Batch {idx}:"
            )
            try:
                summary_res = await asyncio.wait_for(self.llm.ainvoke(batch_prompt), timeout=30.0)
                batch_summaries.append(f"### Section Batch {idx}\n{summary_res.content}")
                await asyncio.sleep(0.5) # Gentle pause to preserve TPM quota
            except Exception as exc:
                print(f"[RAG WARNING] Batch {idx} summary failed: {exc}")
                snippet = batch_text[:300].replace('\n', ' ')
                batch_summaries.append(f"### Section Batch {idx}\nExtract: {snippet}...")

        combined_batch_summaries = "\n\n".join(batch_summaries)
        final_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"User Request: {query}\n\n"
            f"Synthesized Summaries from All Document Sections:\n{combined_batch_summaries}\n\n"
            "Combine these section analyses into a clear, structured, and comprehensive answer."
        )
        final_res = await self.llm.ainvoke(final_prompt)
        return {"answer": prefix + final_res.content, "context": sorted_chunks}

    async def rebuild_chain(self, force_refresh_docs: bool = False):
        """Rebuild retrieval components and multi-document hybrid retriever."""
        db = await get_db()
        
        print("Refreshing document cache from MongoDB...")
        chunks_collection = db["chunks"]
        cursor = chunks_collection.find({})
        all_chunks_data = await cursor.to_list(length=None)
        self.all_docs = [
            Document(page_content=c["content"], metadata=c["metadata"]) 
            for c in all_chunks_data
        ]

        if not self.all_docs:
            print("No documents found in database. RAG chain inactive.")
            self.rag_chain = None
            return

        # Initialize Vector Store
        sync_vector_collection = db_instance.get_sync_collection("vector_store")
        self.vector_db = self._get_vector_store(sync_vector_collection)

        # Base BM25 Retriever
        self.bm25_retriever = BM25Retriever.from_documents(self.all_docs)
        self.bm25_retriever.k = int(get_env_value("RAG_TOP_K", "8"))

        # Hybrid Reciprocal Rank Fusion (RRF) Retriever supporting multi-document filtering & Query Intent Routing
        async def rrf_retriever(retriever_input) -> List[Document]:
            if isinstance(retriever_input, dict):
                query = retriever_input.get("query", "")
                doc_names = retriever_input.get("document_names") or []
                single_doc = retriever_input.get("document_name")
                if single_doc and single_doc not in doc_names:
                    doc_names.append(single_doc)
                top_k = retriever_input.get("top_k", int(get_env_value("RAG_TOP_K", "8")))
            else:
                query = retriever_input
                doc_names = []
                top_k = int(get_env_value("RAG_TOP_K", "8"))

            intent_mode, intent_meta = parse_query_intent(query)

            # Scope documents by selected names or canonical document_ids
            scoped_docs = self.all_docs
            if doc_names:
                doc_identifiers_set = set(doc_names)
                scoped_docs = [
                    doc for doc in self.all_docs
                    if (doc.metadata.get("document_id") in doc_identifiers_set)
                    or (doc.metadata.get("source") in doc_identifiers_set)
                    or (doc.metadata.get("filename") in doc_identifiers_set)
                ]
                if not scoped_docs:
                    return []

            # 1. PAGE Mode
            if intent_mode == QueryIntent.MODE_PAGE:
                target_p = intent_meta["target_page"]
                page_docs = [
                    doc for doc in scoped_docs
                    if (doc.metadata.get("page_number") == target_p) or (doc.metadata.get("page") == target_p)
                ]
                print(f"[RAG] [PAGE INTENT] Document: {doc_names or 'all'} | Target Page: {target_p} | Chunks Found: {len(page_docs)}")
                if page_docs:
                    return page_docs
                print(f"[RAG WARNING] Target page {target_p} not found in scope.")

            # 2. COMPARE_PAGES Mode
            elif intent_mode == QueryIntent.MODE_COMPARE_PAGES:
                p1 = intent_meta["page1"]
                p2 = intent_meta["page2"]
                compare_docs = [
                    doc for doc in scoped_docs
                    if (doc.metadata.get("page_number") in (p1, p2)) or (doc.metadata.get("page") in (p1, p2))
                ]
                print(f"[RAG] [COMPARE INTENT] Document: {doc_names or 'all'} | Pages: [{p1}, {p2}] | Chunks Found: {len(compare_docs)}")
                if compare_docs:
                    return compare_docs

            # 3. RANGE Mode
            elif intent_mode == QueryIntent.MODE_RANGE:
                target_pages = set(intent_meta["pages"])
                range_docs = [
                    doc for doc in scoped_docs
                    if (doc.metadata.get("page_number") in target_pages) or (doc.metadata.get("page") in target_pages)
                ]
                print(f"[RAG] [RANGE INTENT] Document: {doc_names or 'all'} | Target Range: {sorted(target_pages)} | Chunks Found: {len(range_docs)}")
                if range_docs:
                    return range_docs
                print(f"[RAG WARNING] Target range {sorted(target_pages)} not found in scope.")

            # 4. DOCUMENT Mode
            elif intent_mode == QueryIntent.MODE_DOCUMENT:
                print(f"[RAG] [DOCUMENT INTENT] Document: {doc_names or 'all'} | Total Scoped Chunks: {len(scoped_docs)}")
                return sorted(scoped_docs, key=lambda d: (d.metadata.get("page_number", 0) or d.metadata.get("page", 0), d.metadata.get("chunk_number", 0)))


            # 5. SEMANTIC Mode (BM25 + Vector Hybrid Search via Reciprocal Rank Fusion)
            scoped_bm25 = BM25Retriever.from_documents(scoped_docs) if doc_names else self.bm25_retriever
            scoped_bm25.k = top_k * 2

            vector_retriever = self.vector_db.as_retriever(search_kwargs={"k": top_k * 2})

            try:
                bm25_results, vector_results = await asyncio.gather(
                    scoped_bm25.ainvoke(query),
                    vector_retriever.ainvoke(query)
                )
            except Exception as vector_error:
                print(f"Vector retrieval fallback triggered: {vector_error}")
                bm25_results = await scoped_bm25.ainvoke(query)
                vector_results = []

            if doc_names:
                doc_identifiers_set = set(doc_names)
                vector_results = [
                    doc for doc in vector_results
                    if (doc.metadata.get("document_id") in doc_identifiers_set)
                    or (doc.metadata.get("source") in doc_identifiers_set)
                    or (doc.metadata.get("filename") in doc_identifiers_set)
                ]


            rrf_k = 60
            scores: Dict[str, float] = {}
            doc_map: Dict[str, Document] = {}

            for rank, doc in enumerate(bm25_results):
                key = f"{doc.metadata.get('source')}_{doc.metadata.get('chunk_number')}_{doc.page_content[:100]}"
                scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
                doc_map[key] = doc

            for rank, doc in enumerate(vector_results):
                key = f"{doc.metadata.get('source')}_{doc.metadata.get('chunk_number')}_{doc.page_content[:100]}"
                scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
                doc_map[key] = doc

            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            if not ranked:
                return bm25_results[:top_k] if isinstance(bm25_results, list) else []

            retrieved_docs = []
            current_tokens = 0
            max_context_tokens = retriever_input.get("max_context_tokens", 4500) if isinstance(retriever_input, dict) else 4500

            for key, _ in ranked:
                doc = doc_map[key]
                doc_token_est = len(doc.page_content) // 3.5
                if current_tokens + doc_token_est > max_context_tokens and retrieved_docs:
                    break
                retrieved_docs.append(doc)
                current_tokens += doc_token_est
                if len(retrieved_docs) >= top_k:
                    break

            retrieved_pages = sorted(list(set(d.metadata.get("page_number") or d.metadata.get("page") for d in retrieved_docs if d.metadata.get("page_number") or d.metadata.get("page"))))
            print(f"[RAG] [SEMANTIC INTENT] Query: '{query[:40]}' | Top K: {top_k} | Chunks Retrieved: {len(retrieved_docs)} | Pages: {retrieved_pages}")
            return retrieved_docs

        ensemble_retriever = RunnableLambda(rrf_retriever)

        contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

        async def route_retriever(inputs: dict):
            chat_history = inputs.get("chat_history", [])
            query = inputs.get("input", "")
            mode = (inputs.get("response_mode") or "balanced").lower()

            mode_configs = {
                "fast": {"top_k": 4, "max_tokens": 2500},
                "balanced": {"top_k": 8, "max_tokens": 4500},
                "deep": {"top_k": 12, "max_tokens": 7000},
            }
            config = mode_configs.get(mode, mode_configs["balanced"])

            if not chat_history:
                rephrased_query = query
            else:
                rephrase_prompt = ChatPromptTemplate.from_messages([
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ])
                rephrase_chain = rephrase_prompt | self.llm | StrOutputParser()
                rephrased_query = await rephrase_chain.ainvoke({
                    "input": query,
                    "chat_history": chat_history
                })

            return await ensemble_retriever.ainvoke({
                "query": rephrased_query,
                "document_names": inputs.get("document_names"),
                "document_name": inputs.get("document_name"),
                "top_k": config["top_k"],
                "max_context_tokens": config["max_tokens"],
            })

        history_aware_retriever = RunnableLambda(route_retriever)

        system_prompt = f"""{SYSTEM_PROMPT}

Treat the retrieved Context as the ONLY authority for factual claims from documents. Do NOT use outside knowledge, assumptions, or unverified claims for document questions.
When stating facts or claims derived from the context, cite the exact source document and page number in brackets (e.g., [Document: filename.pdf, Page X]).
If the retrieved Context does not contain enough evidence to answer the question, reply EXACTLY:
"I couldn't find this information in your uploaded documents."

Context:
{{context}}"""

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        def format_docs(docs):
            formatted = []
            grouped_by_source = {}
            for doc in docs:
                source = doc.metadata.get('source', 'Unknown')
                grouped_by_source.setdefault(source, []).append(doc)
            
            is_multi_doc = len(grouped_by_source) > 1
            for doc_idx, (source, doc_list) in enumerate(grouped_by_source.items(), start=1):
                header = f"DOCUMENT {chr(64 + doc_idx)}: {source}" if is_multi_doc else f"DOCUMENT: {source}"
                formatted.append(f"=== {header} ===")
                for doc in doc_list:
                    page_num = doc.metadata.get('page_number') or doc.metadata.get('page')
                    page_str = f", Page {page_num}" if page_num is not None else ""
                    formatted.append(f"[Document: {source}{page_str}]\nContent:\n{doc.page_content}")
            
            result = "\n\n---\n\n".join(formatted)
            retrieved_pages = sorted(list(set(d.metadata.get("page_number") or d.metadata.get("page") for d in docs if d.metadata.get("page_number") or d.metadata.get("page"))))
            print(f"[RAG]\nRetrieved chunks: {len(docs)}\nRetrieved pages: {retrieved_pages}\nContext characters/tokens: {len(result)} (~{len(result)//4} tokens)\nContext passed to LLM: YES")
            return result

        async def rag_chain_invoker(inputs: dict):
            query = inputs.get("input", "")
            intent_mode, intent_meta = parse_query_intent(query)
            
            doc_names = inputs.get("document_names") or []
            single_doc = inputs.get("document_name")
            if single_doc and single_doc not in doc_names:
                doc_names.append(single_doc)
                
            scoped_docs = self.all_docs
            if doc_names:
                doc_names_set = set(doc_names)
                scoped_docs = [doc for doc in self.all_docs if doc.metadata.get("source") in doc_names_set]

            if intent_mode == QueryIntent.MODE_TOTAL_PAGES:
                db = await get_db()
                active_sources = list(set(d.metadata.get("source") for d in scoped_docs if d.metadata.get("source")))
                if not active_sources and doc_names:
                    active_sources = doc_names
                docs_cursor = db["documents"].find({"filename": {"$in": active_sources}})
                db_docs = await docs_cursor.to_list(length=None)
                doc_info_map = {d["filename"]: d for d in db_docs}

                lines = []
                for src in active_sources:
                    info = doc_info_map.get(src, {})
                    t_pages = info.get("total_pages") or info.get("page_count")
                    if not t_pages:
                        pages_in_scope = [d.metadata.get("page_number") or d.metadata.get("page") for d in scoped_docs if d.metadata.get("source") == src]
                        valid_p = [p for p in pages_in_scope if isinstance(p, int) and p > 0]
                        t_pages = max(valid_p) if valid_p else 1
                    if len(active_sources) == 1:
                        lines.append(f"The PDF \"{src}\" has {t_pages} pages.")
                    else:
                        lines.append(f"• {src}: {t_pages} pages")
                return {"answer": "\n".join(lines), "context": []}

            if intent_mode == QueryIntent.MODE_DOCUMENT and len(scoped_docs) > 12:
                is_rem = intent_meta.get("is_remaining_query", False)
                return await self.process_entire_document(query, scoped_docs, is_remaining=is_rem)

            retrieved_context = await history_aware_retriever.ainvoke(inputs)
            formatted_ctx = format_docs(retrieved_context)
            chat_hist = inputs.get("chat_history", [])
            
    async def stream_rag_reply(
        self,
        query: str,
        chat_history: Optional[List[BaseMessage]] = None,
        doc_names: Optional[List[str]] = None,
        response_mode: str = "balanced",
    ):
        """Stream RAG response token-by-token with hybrid retrieval, Page Store lookups, follow-up resolution, and 10-type routing."""
        await self.ensure_initialized()

        mode = (response_mode or "balanced").lower()

        mode_configs = {
            "fast": {"top_k": 4, "max_tokens": 1024, "temp": 0.3, "prompt_extra": "Provide a quick, concise, and direct response focusing on essential facts."},
            "balanced": {"top_k": 8, "max_tokens": 2048, "temp": 0.6, "prompt_extra": "Provide a clear, balanced, and well-structured response with thorough explanations and precise citations."},
            "deep": {"top_k": 14, "max_tokens": 4096, "temp": 0.7, "prompt_extra": "Engage in deep, comprehensive analytical reasoning. Break down complex concepts step-by-step, synthesize multiple pieces of evidence, and provide exhaustive citations and insights."},
        }
        config = mode_configs.get(mode, mode_configs["balanced"])

        db = await get_db()
        scoped_docs = []
        target_doc_ids = set()
        target_filenames = set()

        if doc_names:
            doc_query = {"$or": [
                {"document_id": {"$in": list(doc_names)}},
                {"filename": {"$in": list(doc_names)}},
                {"disk_filename": {"$in": list(doc_names)}}
            ]}
            matched_docs = await db["documents"].find(doc_query).to_list(length=None)
            for d in matched_docs:
                if d.get("document_id"):
                    target_doc_ids.add(d["document_id"])
                if d.get("filename"):
                    target_filenames.add(d["filename"])

            target_doc_ids.update(doc_names)
            target_filenames.update(doc_names)

            if self.all_docs:
                scoped_docs = [
                    doc for doc in self.all_docs
                    if (doc.metadata.get("document_id") in target_doc_ids)
                    or (doc.metadata.get("source") in target_filenames)
                    or (doc.metadata.get("filename") in target_filenames)
                ]

            # If scoped_docs is empty in memory, load directly from db["chunks"]
            if not scoped_docs and target_doc_ids:
                chunks_from_db = await db["chunks"].find({
                    "$or": [
                        {"document_id": {"$in": list(target_doc_ids)}},
                        {"metadata.document_id": {"$in": list(target_doc_ids)}},
                        {"source": {"$in": list(target_filenames)}},
                        {"metadata.source": {"$in": list(target_filenames)}},
                        {"filename": {"$in": list(target_filenames)}},
                        {"metadata.filename": {"$in": list(target_filenames)}},
                    ]
                }).to_list(length=None)
                if chunks_from_db:
                    scoped_docs = [
                        Document(page_content=c.get("content") or c.get("text", ""), metadata=c.get("metadata", {}))
                        for c in chunks_from_db
                    ]
                    self.all_docs = [d for d in self.all_docs if d.metadata.get("document_id") not in target_doc_ids]
                    self.all_docs.extend(scoped_docs)
        else:
            scoped_docs = self.all_docs

        if not scoped_docs and not doc_names:
            yield "I couldn't find any relevant information in the selected document(s).", [], []
            return

        intent_mode, intent_meta = parse_query_intent(query, chat_history=chat_history)

        # ── INTENT 1: TOTAL PAGES INQUIRY ────────────────────────────────────
        if intent_mode == QueryIntent.MODE_TOTAL_PAGES:
            docs_cursor = db["documents"].find({
                "$or": [
                    {"document_id": {"$in": list(doc_names or [])}},
                    {"filename": {"$in": list(doc_names or [])}}
                ]
            })
            db_docs = await docs_cursor.to_list(length=None)
            if not db_docs:
                unique_srcs = list(set(d.metadata.get("source", "Document") for d in scoped_docs))
                db_docs = [{"filename": s, "total_pages": 1} for s in unique_srcs] or [{"filename": "Document", "total_pages": 1}]

            lines = []
            source_names = []
            source_details = []

            for info in db_docs:
                src_name = info.get("filename", "Document")
                t_pages = info.get("total_pages") or info.get("page_count") or 1
                if len(db_docs) == 1:
                    lines.append(f"The PDF \"{src_name}\" has **{t_pages}** pages.")
                else:
                    lines.append(f"• **{src_name}**: {t_pages} pages")

                source_names.append(src_name)
                source_details.append({"name": src_name, "pages": [t_pages], "formatted_pages": f"Total: {t_pages} pages"})

            if len(db_docs) > 1:
                total_all = sum(d.get("total_pages") or d.get("page_count") or 1 for d in db_docs)
                lines.append(f"\n**Total across all {len(db_docs)} documents:** {total_all} pages.")

            reply_text = "\n".join(lines)
            yield reply_text, source_names, source_details
            return

        # ── INTENT 2: METADATA INQUIRY ───────────────────────────────────────
        if intent_mode == QueryIntent.MODE_METADATA:
            docs_cursor = db["documents"].find({
                "$or": [
                    {"document_id": {"$in": list(doc_names or [])}},
                    {"filename": {"$in": list(doc_names or [])}}
                ]
            })
            db_docs = await docs_cursor.to_list(length=None)
            if not db_docs:
                unique_srcs = list(set(d.metadata.get("source", "Document") for d in scoped_docs))
                db_docs = [{"filename": s, "total_pages": 1, "file_type": "PDF"} for s in unique_srcs]

            lines = []
            source_names = []
            source_details = []

            for doc in db_docs:
                src_name = doc.get("filename", "Document")
                t_pages = doc.get("total_pages") or doc.get("page_count") or 1
                f_size = doc.get("size_bytes") or doc.get("file_size") or 0
                if f_size > 1024 * 1024:
                    size_str = f"{f_size / (1024 * 1024):.1f} MB"
                elif f_size > 1024:
                    size_str = f"{f_size / 1024:.1f} KB"
                else:
                    size_str = f"{f_size} bytes"
                f_type = doc.get("file_type", "PDF")
                status = doc.get("status", "READY")
                stage = doc.get("stage", "READY")
                f_hash = doc.get("file_hash", "N/A")

                lines.append(f"### Document Details: {src_name}\n"
                             f"• **Filename**: `{src_name}`\n"
                             f"• **Total Pages**: **{t_pages}** pages\n"
                             f"• **File Size**: {size_str}\n"
                             f"• **Document Type**: {f_type}\n"
                             f"• **File Hash (SHA-256)**: `{f_hash[:16]}...`\n"
                             f"• **Status**: {status} ({stage})")

                source_names.append(src_name)
                source_details.append({"name": src_name, "pages": [t_pages], "formatted_pages": f"Total: {t_pages} pages"})

            reply_text = "\n\n".join(lines)
            yield reply_text, source_names, source_details
            return

        # ── INTENT 3: TABLE OF CONTENTS / CHAPTER STRUCTURE ─────────────────
        if intent_mode == QueryIntent.MODE_TOC:
            docs_cursor = db["documents"].find({
                "$or": [
                    {"document_id": {"$in": list(doc_names or [])}},
                    {"filename": {"$in": list(doc_names or [])}}
                ]
            })
            db_docs = await docs_cursor.to_list(length=None)
            lines = []
            source_names = []
            source_details = []

            for doc in db_docs:
                src_name = doc.get("filename", "Document")
                total_p = doc.get("total_pages") or doc.get("page_count") or 1
                toc = doc.get("toc") or []
                if toc:
                    toc_lines = [f"### Table of Contents for {src_name} ({total_p} pages)\n"]
                    for idx, entry in enumerate(toc, start=1):
                        title = entry.get("title", f"Section {idx}")
                        s_page = entry.get("start_page", 1)
                        e_page = min(entry.get("end_page", s_page), total_p)
                        p_range = f"Page {s_page}" if s_page == e_page else f"Pages {s_page}–{e_page}"
                        ch_prefix = f"Chapter {entry['chapter']}: " if entry.get("chapter") and not title.lower().startswith("chapter") else ""
                        toc_lines.append(f"{idx}. **{ch_prefix}{title}** ({p_range})")
                    lines.append("\n".join(toc_lines))
                    source_names.append(src_name)
                    source_details.append({"name": src_name, "pages": [e.get("start_page", 1) for e in toc], "formatted_pages": f"TOC: {len(toc)} chapters/sections"})
                else:
                    lines.append(f"No formal Table of Contents was detected in `{src_name}`. Here is the overview based on extracted document content:")
                    stored_sum = await db["summaries"].find_one({"$or": [{"document_id": doc.get("document_id")}, {"source": src_name}]})
                    if stored_sum and stored_sum.get("summary"):
                        lines.append(stored_sum["summary"])
                    source_names.append(src_name)
                    source_details.append({"name": src_name, "pages": [1], "formatted_pages": "Structure"})

            reply_text = "\n\n".join(lines)
            yield reply_text, source_names, source_details
            return

        # ── INTENT 3.5: SPECIFIC CHAPTER INQUIRY ────────────────────────────
        if intent_mode == QueryIntent.MODE_CHAPTER:
            target_ch = intent_meta.get("chapter_num", 1)
            docs_cursor = db["documents"].find({
                "$or": [
                    {"document_id": {"$in": list(doc_names or [])}},
                    {"filename": {"$in": list(doc_names or [])}}
                ]
            })
            db_docs = await docs_cursor.to_list(length=None)
            if not db_docs:
                unique_srcs = list(set(d.metadata.get("source", "Document") for d in scoped_docs))
                db_docs = [{"filename": s, "total_pages": 1} for s in unique_srcs]

            retrieved_chunks = []
            for doc in db_docs:
                src_name = doc.get("filename", "Document")
                doc_id = doc.get("document_id") or str(doc.get("_id"))
                toc = doc.get("toc") or []
                total_p = doc.get("total_pages") or doc.get("page_count") or 1

                matched_entry = None
                for entry in toc:
                    if entry.get("chapter") == target_ch:
                        matched_entry = entry
                        break
                    if f"chapter {target_ch}" in (entry.get("title") or "").lower():
                        matched_entry = entry
                        break

                if matched_entry:
                    ch_title = matched_entry.get("title") or f"Chapter {target_ch}"
                    start_p = matched_entry.get("start_page", 1)
                    end_p = min(matched_entry.get("end_page", start_p), total_p)
                    p_range_str = f"Page {start_p}" if start_p == end_p else f"Pages {start_p}–{end_p}"

                    sample_pages = [start_p]
                    if start_p + 1 <= end_p:
                        sample_pages.append(start_p + 1)

                    chapter_pages = await db["pages"].find({
                        "$or": [
                            {"document_id": doc_id, "page_number": {"$in": sample_pages}},
                            {"filename": src_name, "page_number": {"$in": sample_pages}}
                        ]
                    }).sort("page_number", 1).to_list(length=3)

                    chapter_intro_text = ""
                    for cp in chapter_pages:
                        p_txt = cp.get("text", "")
                        if p_txt:
                            chapter_intro_text += f"\n\n[Page {cp.get('page_number')} Content]:\n{p_txt[:2500]}"

                    toc_context = (
                        f"=== TABLE OF CONTENTS METADATA ===\n"
                        f"Document: {src_name}\n"
                        f"Target Chapter: Chapter {target_ch}\n"
                        f"Exact Title: {ch_title}\n"
                        f"Exact Page Range: {p_range_str} (Starts on Page {start_p}, Ends on Page {end_p}, Total Document Pages: {total_p})\n\n"
                        f"=== CHAPTER {target_ch} OPENING CONTENT ({p_range_str}) ==={chapter_intro_text}"
                    )

                    retrieved_chunks.append(Document(
                        page_content=toc_context,
                        metadata={
                            "source": src_name,
                            "filename": src_name,
                            "document_id": doc_id,
                            "page_number": start_p,
                            "chapter": target_ch,
                            "chapter_title": ch_title,
                            "start_page": start_p,
                            "end_page": end_p
                        }
                    ))
                else:
                    ch_pages = await db["pages"].find({
                        "$or": [
                            {"document_id": doc_id, "text": {"$regex": f"Chapter\\s*{target_ch}\\b", "$options": "i"}},
                            {"filename": src_name, "text": {"$regex": f"Chapter\\s*{target_ch}\\b", "$options": "i"}}
                        ]
                    }).sort("page_number", 1).to_list(length=3)

                    for cp in ch_pages:
                        retrieved_chunks.append(Document(
                            page_content=f"[Page {cp.get('page_number')} Content]:\n{cp.get('text', '')[:2500]}",
                            metadata={
                                "source": src_name,
                                "filename": src_name,
                                "document_id": doc_id,
                                "page_number": cp.get("page_number", 1)
                            }
                        ))

        # ── INTENT 4: MULTI-PDF QUERY / DISAMBIGUATION ───────────────────────
        elif intent_mode == QueryIntent.MODE_MULTI_PDF:
            docs_cursor = db["documents"].find({
                "$or": [
                    {"document_id": {"$in": list(doc_names or [])}},
                    {"filename": {"$in": list(doc_names or [])}}
                ]
            })
            db_docs = await docs_cursor.to_list(length=None)
            if len(db_docs) > 1 and not any(k in query.lower() for k in ["both", "compare", "all", "difference"]):
                doc_list_str = " or ".join(f"**{d.get('filename')}**" for d in db_docs)
                clarification = f"You have multiple documents attached to this chat. Which document do you mean: {doc_list_str}?"
                yield clarification, [d.get("filename") for d in db_docs], [{"name": d.get("filename"), "pages": [], "formatted_pages": ""} for d in db_docs]
                return

        # ── INTENT 4.5: BLANK / SCANNED PAGES INQUIRY ────────────────────────
        elif intent_mode == QueryIntent.MODE_BLANK_PAGES:
            active_source = scoped_docs[0].metadata.get("source") if scoped_docs else (doc_names[0] if doc_names else "Document")
            active_doc_id = scoped_docs[0].metadata.get("document_id") if scoped_docs else (doc_names[0] if doc_names else None)

            doc_meta = await db["documents"].find_one({"$or": [{"document_id": active_doc_id}, {"filename": active_source}]})
            actual_doc_id = (doc_meta or {}).get("document_id") or active_doc_id
            actual_filename = (doc_meta or {}).get("filename") or active_source
            total_p = (doc_meta or {}).get("total_pages") or (doc_meta or {}).get("page_count") or len(scoped_docs) or 1

            pages_records = await db["pages"].find({
                "$or": [
                    {"document_id": actual_doc_id},
                    {"filename": actual_filename}
                ]
            }).sort("page_number", 1).to_list(length=None)

            blank_pages = []
            if pages_records:
                for p in pages_records:
                    txt = (p.get("text") or "").strip()
                    p_num = p.get("page_number", 1)
                    if not txt or p.get("is_scanned") or not p.get("has_text"):
                        blank_pages.append(p_num)
            else:
                covered_pages = set()
                for d in scoped_docs:
                    p_val = d.metadata.get("page_number") or d.metadata.get("page")
                    if p_val:
                        covered_pages.add(int(p_val))
                if total_p > 0:
                    for p_num in range(1, total_p + 1):
                        if p_num not in covered_pages:
                            blank_pages.append(p_num)

            if blank_pages:
                formatted_p = format_page_ranges(blank_pages)
                msg = f"In **{actual_filename}** (total {total_p} pages), the following {len(blank_pages)} page(s) contain no extractable text layer or are blank/image-only:\n\n- **Blank / Non-text Pages:** {formatted_p}\n\nAll remaining pages contain readable text and are fully indexed for queries."
                yield msg, [f"{actual_filename} ({formatted_p})"], [{"name": actual_filename, "pages": blank_pages, "formatted_pages": formatted_p}]
                return
            else:
                msg = f"There are no blank or unscanned pages in **{actual_filename}**. All {total_p} pages contain extractable text and are fully indexed."
                yield msg, [actual_filename], [{"name": actual_filename, "pages": [], "formatted_pages": f"Total {total_p} pages"}]
                return

        # ── INTENT 5: SINGLE PAGE INQUIRY (Direct Page Store Lookup) ─────────
        elif intent_mode == QueryIntent.MODE_PAGE:
            active_source = scoped_docs[0].metadata.get("source") if scoped_docs else (doc_names[0] if doc_names else "Document")
            active_doc_id = scoped_docs[0].metadata.get("document_id") if scoped_docs else (doc_names[0] if doc_names else None)

            doc_meta = await db["documents"].find_one({"$or": [{"document_id": active_doc_id}, {"filename": active_source}]})
            total_p = (doc_meta or {}).get("total_pages") or (doc_meta or {}).get("page_count") or len(scoped_docs) or 1
            actual_doc_id = (doc_meta or {}).get("document_id") or active_doc_id
            actual_filename = (doc_meta or {}).get("filename") or active_source

            target_p = intent_meta.get("target_page")
            if target_p == "LAST":
                target_p = total_p
            elif target_p == "MIDDLE":
                target_p = max(1, (total_p + 1) // 2)
            elif not isinstance(target_p, int):
                try:
                    target_p = int(target_p)
                except Exception:
                    target_p = 1
            intent_meta["target_page"] = target_p

            if isinstance(target_p, int) and total_p > 0 and target_p > total_p:
                msg = f"Page {target_p} does not exist in '{actual_filename}'. The document has a total of {total_p} pages."
                yield msg, [actual_filename], [{"name": actual_filename, "pages": [], "formatted_pages": f"Total: {total_p} pages"}]
                return

            # Direct lookup in db["pages"]
            page_record = await db["pages"].find_one({
                "$or": [
                    {"document_id": actual_doc_id, "page_number": target_p},
                    {"filename": actual_filename, "page_number": target_p}
                ]
            })

            if not page_record:
                page_chunks = [d for d in scoped_docs if (d.metadata.get("page_number") == target_p or d.metadata.get("page") == target_p)]
                if page_chunks:
                    page_text = "\n\n".join(d.page_content for d in page_chunks)
                    page_record = {"text": page_text, "tables": [], "page_number": target_p, "is_scanned": False}

            if not page_record or (not page_record.get("text") and not page_record.get("tables")):
                msg = f"Page {target_p} in '{actual_filename}' contains scanned or image content with no extractable text layer."
                yield msg, [f"{actual_filename} (p. {target_p})"], [{"name": actual_filename, "pages": [target_p], "formatted_pages": f"p. {target_p}"}]
                return

            page_content = page_record.get("text", "")
            tables = page_record.get("tables", [])
            if tables:
                table_md = "\n\n".join(t.get("markdown", "") for t in tables if t.get("markdown"))
                if table_md:
                    page_content += f"\n\n### Extracted Tables on Page {target_p}:\n{table_md}"

            retrieved_chunks = [
                Document(
                    page_content=page_content,
                    metadata={"source": actual_filename, "filename": actual_filename, "document_id": actual_doc_id, "page_number": target_p}
                )
            ]

        # ── INTENT 6: COMPARE / MULTI-PAGE INQUIRY (Direct Page Store Lookup) ─
        elif intent_mode == QueryIntent.MODE_COMPARE_PAGES:
            active_source = scoped_docs[0].metadata.get("source") if scoped_docs else (doc_names[0] if doc_names else "Document")
            active_doc_id = scoped_docs[0].metadata.get("document_id") if scoped_docs else (doc_names[0] if doc_names else None)
            doc_meta = await db["documents"].find_one({"$or": [{"document_id": active_doc_id}, {"filename": active_source}]})
            actual_doc_id = (doc_meta or {}).get("document_id") or active_doc_id
            actual_filename = (doc_meta or {}).get("filename") or active_source

            target_pages = intent_meta.get("pages", [])
            pages_records = await db["pages"].find({
                "$or": [
                    {"document_id": actual_doc_id, "page_number": {"$in": target_pages}},
                    {"filename": actual_filename, "page_number": {"$in": target_pages}}
                ]
            }).to_list(length=len(target_pages) + 5)

            page_map = {p["page_number"]: p for p in pages_records}
            for p_num in target_pages:
                rec = page_map.get(p_num)
                if rec and rec.get("text"):
                    t_text = rec["text"]
                    if rec.get("tables"):
                        tbl_md = "\n\n".join(t["markdown"] for t in rec["tables"] if t.get("markdown"))
                        if tbl_md:
                            t_text += f"\n\nTables on Page {p_num}:\n{tbl_md}"
                    retrieved_chunks.append(Document(
                        page_content=t_text,
                        metadata={"source": actual_filename, "filename": actual_filename, "document_id": actual_doc_id, "page_number": p_num}
                    ))

            if not retrieved_chunks:
                retrieved_chunks = [d for d in scoped_docs if (d.metadata.get("page_number") in set(target_pages) or d.metadata.get("page") in set(target_pages))]

        # ── INTENT 7: PAGE RANGE INQUIRY ─────────────────────────────────────
        elif intent_mode == QueryIntent.MODE_RANGE:
            active_source = scoped_docs[0].metadata.get("source") if scoped_docs else (doc_names[0] if doc_names else "Document")
            active_doc_id = scoped_docs[0].metadata.get("document_id") if scoped_docs else (doc_names[0] if doc_names else None)
            doc_meta = await db["documents"].find_one({"$or": [{"document_id": active_doc_id}, {"filename": active_source}]})
            actual_doc_id = (doc_meta or {}).get("document_id") or active_doc_id
            actual_filename = (doc_meta or {}).get("filename") or active_source

            target_pages = intent_meta.get("pages", [])
            pages_records = await db["pages"].find({
                "$or": [
                    {"document_id": actual_doc_id, "page_number": {"$in": target_pages}},
                    {"filename": actual_filename, "page_number": {"$in": target_pages}}
                ]
            }).sort("page_number", 1).to_list(length=len(target_pages) + 5)

            if pages_records:
                for p in pages_records:
                    p_num = p.get("page_number", 1)
                    p_txt = p.get("text", "")
                    if p.get("tables"):
                        tbl_md = "\n\n".join(t.get("markdown", "") for t in p["tables"] if t.get("markdown"))
                        if tbl_md:
                            p_txt += f"\n\nTables on Page {p_num}:\n{tbl_md}"
                    retrieved_chunks.append(Document(
                        page_content=p_txt,
                        metadata={"source": actual_filename, "filename": actual_filename, "document_id": actual_doc_id, "page_number": p_num}
                    ))
            else:
                retrieved_chunks = [
                    doc for doc in scoped_docs
                    if (doc.metadata.get("page_number") in set(target_pages) or doc.metadata.get("page") in set(target_pages))
                ]

        # ── INTENT 8: TABLE SPECIFIC INQUIRY ─────────────────────────────────
        elif intent_mode == QueryIntent.MODE_TABLE:
            active_source = scoped_docs[0].metadata.get("source") if scoped_docs else (doc_names[0] if doc_names else "Document")
            active_doc_id = scoped_docs[0].metadata.get("document_id") if scoped_docs else (doc_names[0] if doc_names else None)
            doc_meta = await db["documents"].find_one({"$or": [{"document_id": active_doc_id}, {"filename": active_source}]})
            actual_doc_id = (doc_meta or {}).get("document_id") or active_doc_id
            actual_filename = (doc_meta or {}).get("filename") or active_source

            target_p = intent_meta.get("target_page")
            if target_p:
                page_rec = await db["pages"].find_one({
                    "$or": [
                        {"document_id": actual_doc_id, "page_number": target_p},
                        {"filename": actual_filename, "page_number": target_p}
                    ]
                })
                tables = (page_rec or {}).get("tables", [])
                if tables:
                    table_md = "\n\n".join(t.get("markdown", "") for t in tables if t.get("markdown"))
                    retrieved_chunks = [Document(page_content=f"Tables on Page {target_p}:\n{table_md}\n\nPage Context:\n{page_rec.get('text', '')}", metadata={"source": actual_filename, "filename": actual_filename, "document_id": actual_doc_id, "page_number": target_p})]
            
            if not retrieved_chunks:
                table_pages = await db["pages"].find({
                    "$or": [
                        {"document_id": actual_doc_id, "tables.0": {"$exists": True}},
                        {"filename": actual_filename, "tables.0": {"$exists": True}}
                    ]
                }).to_list(length=10)
                if table_pages:
                    for tp in table_pages:
                        p_n = tp.get("page_number", 1)
                        t_md = "\n\n".join(t.get("markdown", "") for t in tp.get("tables", []) if t.get("markdown"))
                        retrieved_chunks.append(Document(page_content=f"Page {p_n} Tables:\n{t_md}", metadata={"source": actual_filename, "filename": actual_filename, "document_id": actual_doc_id, "page_number": p_n}))
                else:
                    retrieved_chunks = scoped_docs[:config["top_k"]]

        # ── INTENT 9: WHOLE DOCUMENT SUMMARY INQUIRY ─────────────────────────
        elif intent_mode == QueryIntent.MODE_DOCUMENT:
            active_source = scoped_docs[0].metadata.get("source") if scoped_docs else (doc_names[0] if doc_names else "Document")
            active_doc_id = scoped_docs[0].metadata.get("document_id") if scoped_docs else (doc_names[0] if doc_names else None)
            doc_meta = await db["documents"].find_one({"$or": [{"document_id": active_doc_id}, {"filename": active_source}]})
            actual_doc_id = (doc_meta or {}).get("document_id") or active_doc_id
            actual_filename = (doc_meta or {}).get("filename") or active_source
            total_p = (doc_meta or {}).get("total_pages") or (doc_meta or {}).get("page_count") or len(scoped_docs) or 1

            is_page_by_page = intent_meta.get("is_page_by_page", False)
            
            # Fetch representative pages from Page Store
            all_pages = await db["pages"].find({
                "$or": [
                    {"document_id": actual_doc_id},
                    {"filename": actual_filename}
                ]
            }).sort("page_number", 1).to_list(length=min(total_p + 5, 20))

            if all_pages and is_page_by_page and total_p <= 20:
                retrieved_chunks = []
                for p in all_pages:
                    p_num = p.get("page_number", 1)
                    p_txt = p.get("text", "")
                    if p_txt and p_txt.strip():
                        retrieved_chunks.append(
                            Document(
                                page_content=f"[Page {p_num} Content]:\n{p_txt[:1200]}",
                                metadata={
                                    "source": actual_filename,
                                    "filename": actual_filename,
                                    "document_id": actual_doc_id,
                                    "page_number": p_num
                                }
                            )
                        )

            if not retrieved_chunks:
                stored_sum = await db["summaries"].find_one({"$or": [{"document_id": actual_doc_id}, {"source": actual_filename}]})
                if stored_sum and stored_sum.get("summary") and stored_sum.get("status") == "COMPLETED":
                    sum_text = f"### Document Summary for {actual_filename} ({total_p} pages):\n\n{stored_sum['summary']}"
                    yield sum_text, [actual_filename], [{"name": actual_filename, "pages": list(range(1, min(total_p + 1, 10))), "formatted_pages": f"Pages 1-{total_p}"}]
                    return
                retrieved_chunks = sorted(
                    scoped_docs,
                    key=lambda d: (d.metadata.get("page_number", 0) or d.metadata.get("page", 0), d.metadata.get("chunk_number", 0))
                )[:config["top_k"]]

        # ── INTENT 10: WHICH PAGE / SEMANTIC HYBRID RETRIEVAL ────────────────
        else:
            scoped_bm25 = BM25Retriever.from_documents(scoped_docs) if doc_names else self.bm25_retriever
            if scoped_bm25:
                scoped_bm25.k = config["top_k"] * 2

            sync_vector_collection = db_instance.get_sync_collection("vector_store")
            v_db = self._get_vector_store(sync_vector_collection)
            v_retriever = v_db.as_retriever(search_kwargs={"k": config["top_k"] * 2})

            try:
                bm25_res, vec_res = await asyncio.gather(
                    scoped_bm25.ainvoke(query) if scoped_bm25 else asyncio.sleep(0, result=[]),
                    v_retriever.ainvoke(query),
                    return_exceptions=True
                )
                if isinstance(bm25_res, Exception): bm25_res = []
                if isinstance(vec_res, Exception): vec_res = []
            except Exception:
                bm25_res, vec_res = [], []

            if doc_names:
                doc_identifiers_set = set(doc_names)
                vec_res = [
                    d for d in vec_res
                    if (d.metadata.get("document_id") in doc_identifiers_set)
                    or (d.metadata.get("source") in doc_identifiers_set)
                    or (d.metadata.get("filename") in doc_identifiers_set)
                ]

            rrf_k = 60
            scores: Dict[str, float] = {}
            doc_map: Dict[str, Document] = {}
            for rank, doc in enumerate(bm25_res):
                key = f"{doc.metadata.get('source')}_{doc.metadata.get('chunk_number')}_{doc.page_content[:100]}"
                scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
                doc_map[key] = doc
            for rank, doc in enumerate(vec_res):
                key = f"{doc.metadata.get('source')}_{doc.metadata.get('chunk_number')}_{doc.page_content[:100]}"
                scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
                doc_map[key] = doc

            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            retrieved_chunks = [doc_map[k] for k, _ in ranked[:config["top_k"]]] if ranked else bm25_res[:config["top_k"]]

        if not retrieved_chunks:
            yield "I couldn't find any relevant information in the selected document(s).", [], []
            return

        # ── STRICT CONTEXT TOKEN TRUNCATION (Max 2,500 tokens) ───────────────
        retrieved_chunks = truncate_context_to_budget(retrieved_chunks, max_tokens=MAX_RETRIEVED_CONTEXT_TOKENS)

        # Format context
        formatted_context = []
        for doc in retrieved_chunks:
            source = doc.metadata.get('source') or doc.metadata.get('filename') or 'Unknown'
            page_num = doc.metadata.get('page_number') or doc.metadata.get('page')
            p_str = f", Page {page_num}" if page_num is not None else ""
            formatted_context.append(f"[Document: {source}{p_str}]\nContent:\n{doc.page_content}")
        context_str = "\n\n---\n\n".join(formatted_context)

        # Strict chat history trimming (Max 6 messages, 800 tokens)
        bounded_history = trim_chat_history_to_budget(chat_history, max_messages=MAX_CHAT_HISTORY_MESSAGES, max_tokens=MAX_CHAT_HISTORY_TOKENS)

        # Source citations
        source_names = []
        grouped_sources: Dict[str, set] = {}
        for doc in retrieved_chunks:
            s_name = doc.metadata.get('source') or doc.metadata.get('filename') or 'Document'
            p_num = doc.metadata.get('page_number') or doc.metadata.get('page')
            grouped_sources.setdefault(s_name, set())
            if isinstance(p_num, int) and p_num > 0:
                grouped_sources[s_name].add(p_num)

        for s_name, p_set in sorted(grouped_sources.items()):
            p_range = format_page_ranges(list(p_set))
            source_names.append(f"{s_name} ({p_range})" if p_range else s_name)

        source_details = [
            {"name": s_name, "pages": sorted(p_set), "formatted_pages": format_page_ranges(list(p_set))}
            for s_name, p_set in sorted(grouped_sources.items())
        ]

        extra_instruction = config['prompt_extra']
        if intent_mode == QueryIntent.MODE_CHAPTER:
            target_ch = intent_meta.get("chapter_num", 1)
            extra_instruction += f"\nIMPORTANT: The user is asking about Chapter {target_ch}. Answer strictly using the exact Chapter title, page range, and content provided in the Context. If citing, cite the exact source and starting page [Document: filename, Page X]."
        elif intent_mode == QueryIntent.MODE_WHICH_PAGE:
            extra_instruction += "\nIMPORTANT: The user is asking which page contains a specific topic. Review the '[Document: filename, Page X]' headers in the Context and state clearly: '[Topic] is discussed on page X.' Cite the exact page [Document: filename, Page X]."
        elif intent_mode == QueryIntent.MODE_PAGE:
            target_page_num = intent_meta.get('target_page')
            extra_instruction += f"\nIMPORTANT: Answer the question strictly using all text, code snippets, and diagrams from Page {target_page_num} provided in Context. If the user asks for code, output the exact code blocks from Page {target_page_num} in clean markdown code blocks with syntax highlighting. Cite the source: [Document: filename, Page {target_page_num}]."
        elif intent_mode == QueryIntent.MODE_COMPARE_PAGES:
            pages_str = ", ".join(str(p) for p in intent_meta.get('pages', []))
            extra_instruction += f"\nIMPORTANT: Synthesize and compare the discussions and code across Pages {pages_str} based strictly on the provided Context. Cite exact page numbers."
        elif intent_mode == QueryIntent.MODE_TABLE:
            extra_instruction += "\nIMPORTANT: The user is asking about tabular data. Use the structured Markdown tables provided in Context to answer with exact numerical values, rows, and columns."
        elif intent_mode == QueryIntent.MODE_DOCUMENT:
            extra_instruction += "\nIMPORTANT: Provide a clear, structured summary covering the document sections in Context. Cite exact page numbers."

        system_prompt_text = (
            f"{SYSTEM_PROMPT}\n\n"
            f"{extra_instruction}\n\n"
            "CRITICAL RULES FOR DOCUMENT Q&A:\n"
            "1. Answer using the retrieved document context. Treat the retrieved Context as the ONLY authority for factual claims from documents.\n"
            "2. Do NOT invent information or page ranges that are not supported by the retrieved context. If the answer cannot be found in the retrieved document, clearly state: \"I couldn't find that information in the document.\"\n"
            "3. When stating facts derived from the context, ALWAYS cite the source document and page number in brackets (e.g. [Document: file.pdf, Page X]).\n"
            "4. NEVER invent chapter titles, page numbers, or guess future pages.\n\n"
            f"Context:\n{context_str}"
        )

        messages = [SystemMessage(content=system_prompt_text)]
        if bounded_history:
            messages.extend(bounded_history)
        messages.append(HumanMessage(content=query))

        prompt_tokens_est = estimate_tokens(system_prompt_text) + sum(estimate_tokens(m.content if isinstance(m.content, str) else str(m.content)) for m in bounded_history) + estimate_tokens(query)
        resp_max_tokens = min(config["max_tokens"], MAX_RESPONSE_TOKENS)

        log_retrieval(
            doc_id=",".join(target_doc_ids) if target_doc_ids else "all",
            query=query,
            num_chunks=len(retrieved_chunks),
            pages=sorted(list(set(d.metadata.get("page_number") for d in retrieved_chunks if d.metadata.get("page_number")))),
            scores=[],
            context_tokens=estimate_tokens(context_str)
        )
        log_llm(model=self.groq_model, prompt_tokens=prompt_tokens_est, max_tokens=resp_max_tokens)

        api_key = get_env_value("GROQ_API_KEY")
        stream_llm = ChatGroq(
            model=self.groq_model,
            api_key=api_key,
            temperature=config["temp"],
            max_tokens=resp_max_tokens,
            timeout=60,
            max_retries=2,
        )

        try:
            async for chunk in stream_llm.astream(messages):
                content = chunk.content
                if isinstance(content, str) and content:
                    yield content, source_names, source_details
                elif isinstance(content, list):
                    parts = []
                    for p in content:
                        if isinstance(p, str):
                            parts.append(p)
                        elif isinstance(p, dict) and p.get("type") == "text":
                            parts.append(p.get("text", ""))
                    text_out = "".join(parts)
                    if text_out:
                        yield text_out, source_names, source_details
        except Exception as stream_err:
            err_msg = str(stream_err)
            print(f"[RAG STREAM ERROR] {err_msg}")
            if "429" in err_msg or "rate limit" in err_msg.lower():
                yield "I am currently experiencing high request volume with the LLM API (Rate limit reached). Please wait a moment and retry.", source_names, source_details
            else:
                yield f"An error occurred while streaming response: {err_msg}", source_names, source_details


    async def update_doc_status(self, db, filename: str, status: str, stage: str, error: Optional[str] = None,
                                chunk_count: Optional[int] = None, page_count: Optional[int] = None,
                                upload_status: Optional[str] = None, processing_status: Optional[str] = None,
                                summary_status: Optional[str] = None, document_id: Optional[str] = None,
                                conversation_id: Optional[str] = None, file_size: Optional[int] = None,
                                file_type: Optional[str] = None, file_path: Optional[str] = None,
                                file_hash: Optional[str] = None, tables_count: Optional[int] = None,
                                images_count: Optional[int] = None,
                                toc: Optional[List[Dict]] = None, extracted_page_count: Optional[int] = None,
                                embedding_count: Optional[int] = None, status_details: Optional[str] = None):
        """Update document lifecycle with full 3-tier schema tracking."""
        now = __import__("datetime").datetime.now()
        update_fields = {
            "filename": filename,
            "status": status,
            "stage": stage,
            "updated_at": now,
            "error": error,
        }
        if document_id is not None:
            update_fields["document_id"] = document_id
        if conversation_id is not None:
            update_fields["conversation_id"] = conversation_id
        if file_size is not None:
            update_fields["size_bytes"] = file_size
            update_fields["file_size"] = file_size
        if file_type is not None:
            update_fields["file_type"] = file_type
        if file_path is not None:
            update_fields["file_path"] = file_path
            update_fields["storage_path"] = file_path
        if file_hash is not None:
            update_fields["file_hash"] = file_hash
        if tables_count is not None:
            update_fields["tables_count"] = tables_count
        if images_count is not None:
            update_fields["images_count"] = images_count
        if upload_status is not None:
            update_fields["upload_status"] = upload_status
        if processing_status is not None:
            update_fields["processing_status"] = processing_status
        if summary_status is not None:
            update_fields["summary_status"] = summary_status
        if chunk_count is not None:
            update_fields["chunk_count"] = chunk_count
        if page_count is not None:
            update_fields["page_count"] = page_count
            update_fields["total_pages"] = page_count
        if extracted_page_count is not None:
            update_fields["extracted_page_count"] = extracted_page_count
        if embedding_count is not None:
            update_fields["embedding_count"] = embedding_count
        if toc is not None:
            update_fields["toc"] = toc
        if status_details is not None:
            update_fields["status_details"] = status_details

        lookup_query = {"document_id": document_id} if document_id else {"filename": filename}
        await db["documents"].update_one(
            lookup_query,
            {
                "$set": update_fields,
                "$setOnInsert": {"created_at": now, "uploaded_at": now}
            },
            upsert=True,
        )

    async def ensure_initialized(self):
        """Ensure models and embeddings are loaded before processing documents."""
        if self.llm is None or self.embeddings is None:
            await self.initialize()

    async def process_file(self, file_path: str, filename: str, document_id: Optional[str] = None, conversation_id: Optional[str] = None, rebuild: bool = True):
        """Process a single file: extract → Page Store → chunk → embed/index → 6-Point VERIFY → READY → summary.
        Maintains 3-tier architecture: Original Document + Page Store + Semantic Chunk Index.
        """
        await self.ensure_initialized()
        t_pipeline_start = time.perf_counter()
        ext = os.path.splitext(filename)[1].lower()
        loaders = {
            ".pdf": PyPDFLoader, ".txt": TextLoader, ".docx": Docx2txtLoader,
            ".xlsx": UnstructuredExcelLoader, ".xls": UnstructuredExcelLoader,
            ".md": UnstructuredMarkdownLoader, ".csv": TextLoader,
        }

        if ext not in loaders:
            return False

        db = await get_db()
        document = await db["documents"].find_one({"$or": [{"document_id": document_id}, {"filename": filename}]})
        if not document_id:
            document_id = (document or {}).get("document_id", str(uuid.uuid4()))
        if not conversation_id and document:
            conversation_id = document.get("conversation_id")

        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        file_type = ext.replace(".", "").upper() if ext else "FILE"
        file_hash = calculate_file_hash(file_path)

        # ── Stage 1: EXTRACTING & PAGE STORE CREATION ─────────────────────
        t_extract_start = time.perf_counter()
        await self.update_doc_status(
            db, filename, status="PROCESSING", stage="EXTRACTING",
            upload_status="uploaded", processing_status="extracting", summary_status="pending",
            document_id=document_id, conversation_id=conversation_id, file_size=file_size,
            file_type=file_type, file_path=file_path, file_hash=file_hash,
            status_details="Extracting page-by-page content and structure..."
        )
        print(f"[PARSE] document_id={document_id} filename='{filename}' format={file_type} size={file_size} hash={file_hash[:8]}...")

        try:
            documents = []
            pages_to_store = []
            page_count = 0
            extracted_count = 0
            total_tables_count = 0
            total_images_count = 0
            toc_data = []
            now = __import__("datetime").datetime.now()

            if ext in [".xlsx", ".xls", ".csv"]:
                import pandas as pd
                if ext == ".csv":
                    df = await asyncio.to_thread(pd.read_csv, file_path)
                else:
                    df = await asyncio.to_thread(pd.read_excel, file_path)
                df = df.fillna("").astype(str)
                columns = list(df.columns)
                header_context = "Table columns: " + ", ".join(columns) + "\n"
                
                table_md_rows = [f"| {' | '.join(columns)} |", f"| {' | '.join(['---'] * len(columns))} |"]
                for row_idx, row in df.iterrows():
                    fields = [f"{col}: {val}" for col, val in row.items() if val.strip()]
                    if not fields:
                        continue
                    row_text = header_context + f"Row {row_idx + 1}: " + " | ".join(fields)
                    documents.append(Document(
                        page_content=row_text,
                        metadata={
                            "source": filename,
                            "filename": filename,
                            "document_id": document_id,
                            "conversation_id": conversation_id,
                            "chunk_number": row_idx + 1,
                            "page": row_idx + 1,
                            "page_number": row_idx + 1,
                            "page_start": row_idx + 1,
                            "page_end": row_idx + 1,
                        }
                    ))
                    table_md_rows.append(f"| {' | '.join([str(val) for val in row.values])} |")

                page_count = len(df)
                extracted_count = len(documents)
                total_tables_count = 1

                for row_idx, doc_item in enumerate(documents, start=1):
                    pages_to_store.append({
                        "document_id": document_id,
                        "filename": filename,
                        "page_number": row_idx,
                        "pdf_page_number": row_idx,
                        "printed_page_number": row_idx,
                        "text": doc_item.page_content,
                        "tables": [{"table_index": 1, "page_number": row_idx, "headers": columns, "markdown": "\n".join(table_md_rows[:20])}],
                        "images": [],
                        "has_text": True,
                        "ocr_used": False,
                        "is_scanned": False,
                        "summary": f"Table Row {row_idx} data",
                        "created_at": now,
                    })

            elif ext == ".pdf":
                def extract_pdf():
                    import pypdf
                    reader = pypdf.PdfReader(file_path)
                    total_p = len(reader.pages)
                    pdf_docs = []
                    pdf_pages = []
                    extracted_p = 0
                    missing_p = []
                    t_count = 0
                    img_count = 0
                    
                    parsed_toc = parse_pdf_outline(reader)

                    for page_num, page in enumerate(reader.pages, start=1):
                        try:
                            text = (page.extract_text() or "").strip()
                            has_images = bool(hasattr(page, "images") and len(page.images) > 0)
                            num_page_images = len(page.images) if hasattr(page, "images") else 0
                            img_count += num_page_images
                            
                            printed_p = page_num
                            if text:
                                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                                if lines:
                                    first_line = lines[0]
                                    last_line = lines[-1]
                                    p_match = re.search(r'(?:page|p\.)\s*([0-9]+)', f"{first_line} {last_line}", re.I)
                                    if p_match:
                                        try:
                                            printed_p = int(p_match.group(1))
                                        except Exception:
                                            pass

                            page_tables = extract_table_structures(text, page_num)
                            t_count += len(page_tables)

                            is_scanned = not (text and len(text) >= 15)
                            if not is_scanned:
                                extracted_p += 1
                                doc_content = text
                                if page_tables:
                                    tbl_md = "\n\n".join(t["markdown"] for t in page_tables)
                                    doc_content += f"\n\n[Structured Tables]:\n{tbl_md}"

                                pdf_docs.append(Document(
                                    page_content=doc_content,
                                    metadata={
                                        "source": filename,
                                        "filename": filename,
                                        "document_id": document_id,
                                        "conversation_id": conversation_id,
                                        "page": page_num,
                                        "page_number": page_num,
                                        "pdf_page_number": page_num,
                                        "printed_page_number": printed_p,
                                        "page_start": page_num,
                                        "page_end": page_num,
                                        "has_images": has_images,
                                        "is_scanned": False,
                                    }
                                ))
                            else:
                                missing_p.append(page_num)
                                img_info = f" ({num_page_images} image(s))" if has_images else ""
                                placeholder = f"[Page {page_num}: Scanned or Image Content{img_info} - No extractable text layer detected]"
                                pdf_docs.append(Document(
                                    page_content=placeholder,
                                    metadata={
                                        "source": filename,
                                        "filename": filename,
                                        "document_id": document_id,
                                        "conversation_id": conversation_id,
                                        "page": page_num,
                                        "page_number": page_num,
                                        "pdf_page_number": page_num,
                                        "printed_page_number": printed_p,
                                        "page_start": page_num,
                                        "page_end": page_num,
                                        "has_images": has_images,
                                        "is_scanned": True,
                                    }
                                ))

                            # Store in Page Store
                            pdf_pages.append({
                                "document_id": document_id,
                                "filename": filename,
                                "page_number": page_num,
                                "pdf_page_number": page_num,
                                "printed_page_number": printed_p,
                                "text": text,
                                "tables": page_tables,
                                "images": [{"image_index": idx + 1, "page_number": page_num} for idx in range(num_page_images)],
                                "has_text": not is_scanned,
                                "ocr_used": False,
                                "is_scanned": is_scanned,
                                "summary": text[:300].replace('\n', ' ') if text else "",
                                "created_at": now,
                            })

                        except Exception as pe:
                            print(f"[PARSE] Error extracting page {page_num} of {filename}: {pe}")
                            missing_p.append(page_num)
                            pdf_docs.append(Document(
                                page_content=f"[Page {page_num}: Error extracting page text]",
                                metadata={
                                    "source": filename,
                                    "filename": filename,
                                    "document_id": document_id,
                                    "conversation_id": conversation_id,
                                    "page": page_num,
                                    "page_number": page_num,
                                    "page_start": page_num,
                                    "page_end": page_num,
                                    "is_scanned": True,
                                }
                            ))
                            pdf_pages.append({
                                "document_id": document_id,
                                "filename": filename,
                                "page_number": page_num,
                                "pdf_page_number": page_num,
                                "printed_page_number": page_num,
                                "text": "",
                                "tables": [],
                                "images": [],
                                "has_text": False,
                                "ocr_used": False,
                                "is_scanned": True,
                                "summary": "Unreadable page",
                                "created_at": now,
                            })

                    parsed_toc = extract_table_of_contents_and_chapters(reader, pdf_pages, total_p)
                    return pdf_docs, pdf_pages, total_p, extracted_p, missing_p, parsed_toc, t_count, img_count

                documents, pages_to_store, page_count, extracted_count, missing_pages, toc_data, total_tables_count, total_images_count = await asyncio.to_thread(extract_pdf)
                log_pdf_parse(document_id, filename, page_count, extracted_count, len(toc_data))
                print(f"[PARSE] [COMPLETE] document_id={document_id} filename='{filename}' total_pages={page_count} text_pages={extracted_count} tables={total_tables_count} toc_items={len(toc_data)}")

                # ── Post-Extraction Vision OCR for Pages with Embedded Code / Images / Scanned Content ──
                ocr_tasks = []
                doc_pages_pdfium = None
                for idx, p_entry in enumerate(pages_to_store):
                    p_num = p_entry["page_number"]
                    p_text = p_entry.get("text", "")
                    p_has_images = p_entry.get("has_images") or len(p_entry.get("images", [])) > 0
                    p_is_scanned = p_entry.get("is_scanned", False)
                    clean_len = len(p_text.strip())

                    # Trigger Vision OCR smartly:
                    # 1. Truly scanned or no-text pages (clean_len < 15)
                    # 2. Slide/page with image/screenshot and extremely sparse text (clean_len < 100)
                    needs_ocr = p_is_scanned or (clean_len < 15) or (p_has_images and clean_len < 100)

                    if needs_ocr:
                        if doc_pages_pdfium is None:
                            try:
                                import pypdfium2 as pdfium
                                doc_pages_pdfium = pdfium.PdfDocument(file_path)
                            except Exception as pdfium_err:
                                print(f"[PARSE] [OCR_PDFIUM_WARN] Could not initialize pypdfium2: {pdfium_err}")
                                break

                        try:
                            p_obj = doc_pages_pdfium[p_num - 1]
                            rendered_bitmap = p_obj.render(scale=1.3)
                            pil_img = rendered_bitmap.to_pil()
                            buf = io.BytesIO()
                            pil_img.save(buf, format="JPEG", quality=80)
                            ocr_tasks.append((idx, p_num, buf.getvalue()))
                        except Exception as ren_err:
                            print(f"[PARSE] [OCR_RENDER_WARN] Page {p_num} render failed: {ren_err}")

                if doc_pages_pdfium:
                    try:
                        doc_pages_pdfium.close()
                    except Exception:
                        pass

                if ocr_tasks:
                    print(f"[PARSE] [OCR_VISION] Running Parallel Vision OCR on {len(ocr_tasks)} page(s)...")
                    sem = asyncio.Semaphore(6)

                    async def run_single_ocr(t_item):
                        t_idx, t_p_num, t_img_bytes = t_item
                        async with sem:
                            res = await self.extract_page_vision_ocr(t_img_bytes, t_p_num)
                            return t_idx, t_p_num, res

                    results = await asyncio.gather(*[run_single_ocr(t) for t in ocr_tasks], return_exceptions=True)
                    for res_item in results:
                        if isinstance(res_item, Exception) or not isinstance(res_item, tuple):
                            continue
                        idx, p_num, ocr_result = res_item
                        if ocr_result and len(ocr_result.strip()) > 10:
                            existing_text = pages_to_store[idx].get("text", "").strip()
                            combined_content = f"{existing_text}\n\n{ocr_result}" if existing_text else ocr_result
                            pages_to_store[idx]["text"] = combined_content
                            pages_to_store[idx]["has_text"] = True
                            pages_to_store[idx]["ocr_used"] = True
                            pages_to_store[idx]["is_scanned"] = False
                            pages_to_store[idx]["summary"] = combined_content[:300].replace("\n", " ")

                            # Also update Document chunk
                            if idx < len(documents):
                                documents[idx].page_content = combined_content
                                documents[idx].metadata["is_scanned"] = False
                                documents[idx].metadata["ocr_used"] = True
                                documents[idx].metadata["has_images"] = True
            else:
                loader = loaders[ext](file_path)
                docs = await asyncio.to_thread(loader.load)
                page_count = len(docs)
                for idx, doc in enumerate(docs):
                    p_num = idx + 1
                    doc.metadata["source"] = filename
                    doc.metadata["filename"] = filename
                    doc.metadata["document_id"] = document_id
                    doc.metadata["conversation_id"] = conversation_id
                    doc.metadata["page"] = p_num
                    doc.metadata["page_number"] = p_num
                    doc.metadata["page_start"] = p_num
                    doc.metadata["page_end"] = p_num
                    
                    p_tables = extract_table_structures(doc.page_content, p_num)
                    total_tables_count += len(p_tables)
                    pages_to_store.append({
                        "document_id": document_id,
                        "filename": filename,
                        "page_number": p_num,
                        "pdf_page_number": p_num,
                        "printed_page_number": p_num,
                        "text": doc.page_content,
                        "tables": p_tables,
                        "images": [],
                        "has_text": bool(doc.page_content and doc.page_content.strip()),
                        "ocr_used": False,
                        "is_scanned": False,
                        "summary": doc.page_content[:300].replace('\n', ' ') if doc.page_content else "",
                        "created_at": now,
                    })

                documents.extend([doc for doc in docs if doc.page_content and doc.page_content.strip()])
                extracted_count = len(documents)

            t_extract_end = time.perf_counter()
            print(f"[PARSE] document_id={document_id} extracted in {t_extract_end - t_extract_start:.2f}s (Pages: {page_count})")

            if not pages_to_store:
                raise ValueError("No readable pages or content found in this document")

            # ── Persist to Page Store Collection (db["pages"]) ───────────────
            await db["pages"].delete_many({"document_id": document_id})
            await db["pages"].insert_many(pages_to_store)
            print(f"[PAGE_STORE] document_id={document_id} filename='{filename}' persisted {len(pages_to_store)} page records")

            # ── Stage 2: CHUNKING (Page-by-page preservation) ────────────────
            t_chunk_start = time.perf_counter()
            await self.update_doc_status(
                db, filename, status="PROCESSING", stage="CHUNKING",
                page_count=page_count, extracted_page_count=extracted_count,
                document_id=document_id, conversation_id=conversation_id,
                toc=toc_data, tables_count=total_tables_count, images_count=total_images_count,
                status_details=f"Chunking {len(documents)} extracted pages..."
            )
            
            # Split page-by-page to guarantee zero cross-page boundary leakage
            chunks = []
            for p_doc in documents:
                p_num = p_doc.metadata.get("page_number") or p_doc.metadata.get("page") or 1
                p_splits = self.splitter.split_documents([p_doc])
                for split_doc in p_splits:
                    if split_doc.page_content and split_doc.page_content.strip():
                        split_doc.metadata.update({
                            "source": filename,
                            "filename": filename,
                            "document_id": document_id,
                            "conversation_id": conversation_id,
                            "page": p_num,
                            "page_number": p_num,
                            "pdf_page_number": p_doc.metadata.get("pdf_page_number", p_num),
                            "printed_page_number": p_doc.metadata.get("printed_page_number", p_num),
                            "total_pages": page_count,
                            "page_start": p_num,
                            "page_end": p_num,
                        })
                        chunks.append(split_doc)

            t_chunk_end = time.perf_counter()
            avg_chunk_size = sum(len(c.page_content) for c in chunks) / len(chunks) if chunks else 0
            log_chunk(document_id, len(chunks), avg_chunk_size)

            if not chunks:
                raise ValueError("No usable text chunks were produced from the document")

            for chunk_number, chunk in enumerate(chunks, start=1):
                p_num = chunk.metadata.get("page_number") or chunk.metadata.get("page") or 1
                chunk.metadata.update({
                    "chunk_number": chunk_number,
                    "chunk_index": chunk_number,
                    "chunk_id": f"{document_id}_{chunk_number}",
                    "page_number": p_num,
                    "page": p_num,
                    "page_start": chunk.metadata.get("page_start", p_num),
                    "page_end": chunk.metadata.get("page_end", p_num),
                    "total_pages": page_count,
                })

            # ── Stage 3: EMBEDDING ────────────────────────────────────────
            t_embed_start = time.perf_counter()
            await self.update_doc_status(
                db, filename, status="PROCESSING", stage="EMBEDDING",
                chunk_count=len(chunks), document_id=document_id,
                conversation_id=conversation_id, status_details=f"Generating vector embeddings for {len(chunks)} chunks..."
            )
            log_embedding(document_id, len(chunks), self.vector_dimensions or 384)
            chunks_collection = db["chunks"]
            chunk_data = [
                {
                    "chunk_id": c.metadata["chunk_id"],
                    "chunk_number": c.metadata.get("chunk_number", 1),
                    "content": c.page_content,
                    "text": c.page_content,
                    "metadata": c.metadata,
                    "page_number": c.metadata["page_number"],
                    "pdf_page_number": c.metadata.get("pdf_page_number", c.metadata["page_number"]),
                    "printed_page_number": c.metadata.get("printed_page_number", c.metadata["page_number"]),
                    "total_pages": page_count,
                    "source": filename,
                    "filename": filename,
                    "document_id": document_id,
                    "conversation_id": conversation_id
                }
                for c in chunks
            ]

            # Clear pre-existing chunks, vector records, and summaries for this document
            filename_purge_query = {
                "$or": [
                    {"document_id": document_id},
                    {"metadata.document_id": document_id}
                ]
            }
            await db["chunks"].delete_many(filename_purge_query)
            await db["vector_store"].delete_many(filename_purge_query)
            await db["summaries"].delete_many(filename_purge_query)

            if chunk_data:
                await chunks_collection.insert_many(chunk_data)

            sync_vector_collection = db_instance.get_sync_collection("vector_store")
            self.vector_db = self._get_vector_store(sync_vector_collection)

            BATCH_SIZE = 100
            for i in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[i : i + BATCH_SIZE]
                await asyncio.to_thread(self.vector_db.add_documents, batch)

            t_embed_end = time.perf_counter()
            log_vector_store(document_id, len(chunks))

            # ── Stage 4: 6-POINT AUTOMATED PRE-READY VERIFICATION ──────────
            await self.update_doc_status(
                db, filename, status="PROCESSING", stage="VERIFYING",
                document_id=document_id, conversation_id=conversation_id,
                status_details="Executing 6 automated pre-ready verification tests..."
            )
            print(f"[INDEX] [VERIFY] document_id={document_id} filename='{filename}' running 6 automated verification checks...")

            # Check 1: Document metadata check
            if not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
                raise ValueError("Verification Check 1 Failed: Physical document file missing or empty")
            if page_count <= 0:
                raise ValueError("Verification Check 1 Failed: Total page count is non-positive")

            # Check 2: Page 1 check in Page Store
            p1_record = await db["pages"].find_one({"document_id": document_id, "page_number": 1})
            if not p1_record:
                raise ValueError("Verification Check 2 Failed: Page 1 representation missing from Page Store")

            # Check 3: Random / Last Page check in Page Store
            last_p_record = await db["pages"].find_one({"document_id": document_id, "page_number": page_count})
            if not last_p_record:
                raise ValueError(f"Verification Check 3 Failed: Last page ({page_count}) representation missing from Page Store")

            # Check 4: Chunk Store parity check
            stored_chunk_count = await db["chunks"].count_documents({"document_id": document_id})
            if stored_chunk_count != len(chunks):
                raise ValueError(f"Verification Check 4 Failed: Expected {len(chunks)} chunks in MongoDB, found {stored_chunk_count}")

            # Check 5: Document ID on all chunks
            invalid_chunk = await db["chunks"].find_one({
                "document_id": document_id,
                "$or": [
                    {"metadata.document_id": {"$exists": False}},
                    {"metadata.document_id": {"$ne": document_id}}
                ]
            })
            if invalid_chunk:
                raise ValueError("Verification Check 5 Failed: Chunk metadata missing valid document_id")

            # Check 6: Valid page_number on all chunks
            invalid_p_chunk = await db["chunks"].find_one({
                "document_id": document_id,
                "$or": [
                    {"metadata.page_number": {"$exists": False}},
                    {"metadata.page_number": None},
                    {"metadata.page_number": {"$lte": 0}}
                ]
            })
            if invalid_p_chunk:
                raise ValueError("Verification Check 6 Failed: Chunk metadata missing valid positive page_number")

            # Update BM25 document cache in memory
            self.all_docs = [d for d in self.all_docs if d.metadata.get("document_id") != document_id]
            self.all_docs.extend(chunks)

            # ── Stage 5: READY ────────────────────────────────────────────
            t_pipeline_end = time.perf_counter()
            total_duration = t_pipeline_end - t_pipeline_start
            print(f"[INDEX] [READY] document_id={document_id} filename='{filename}' all 6 checks passed. Document READY (took {total_duration:.2f}s)")
            await self.update_doc_status(
                db, filename, status="READY", stage="READY",
                chunk_count=len(chunks), page_count=page_count,
                extracted_page_count=extracted_count, embedding_count=len(chunks),
                toc=toc_data, upload_status="uploaded", processing_status="ready",
                summary_status="pending", error=None,
                document_id=document_id, conversation_id=conversation_id,
                file_size=file_size, file_type=file_type, file_path=file_path,
                file_hash=file_hash, tables_count=total_tables_count, images_count=total_images_count,
                status_details="Document indexed, verified, and ready for retrieval"
            )

            # ── Stage 6: Summary generation (INDEPENDENT) ─────────────────
            print(f"[INDEX] [SUMMARY] document_id={document_id} filename='{filename}' generating summary...")
            await self.update_doc_status(db, filename, status="READY", stage="READY", summary_status="generating")
            try:
                summary = await self.generate_summary(documents)
                await db["summaries"].update_one(
                    {"$or": [{"document_id": document_id}, {"source": filename}]},
                    {"$set": {"summary": summary, "source": filename, "filename": filename, "document_id": document_id, "status": "COMPLETED"}},
                    upsert=True
                )
                await self.update_doc_status(db, filename, status="READY", stage="READY", summary_status="completed")
                print(f"[INDEX] [SUMMARY_DONE] document_id={document_id} filename='{filename}' summary saved successfully")
            except Exception as sum_err:
                print(f"[INDEX] [SUMMARY_FAIL] document_id={document_id} filename='{filename}' {sum_err} (document remains READY)")
                preview = (documents[0].page_content.strip()[:300].replace('\n', ' ') + "...") if (documents and documents[0].page_content) else "Document uploaded and indexed for Q&A."
                await db["summaries"].update_one(
                    {"$or": [{"document_id": document_id}, {"source": filename}]},
                    {"$set": {"summary": f"Summary preview: {preview}", "source": filename, "filename": filename, "document_id": document_id, "status": "FAILED", "error": str(sum_err)}},
                    upsert=True
                )
                await self.update_doc_status(db, filename, status="READY", stage="READY", summary_status="failed")

            if rebuild:
                print(f"[INDEX] [REBUILD_CHAIN] document_id={document_id} filename='{filename}' refreshing in-memory retriever...")
                await self.rebuild_chain()
            return True

        except Exception as e:
            import traceback
            traceback.print_exc()
            err_msg = str(e)[:500]
            print(f"[RAG ERROR] [{filename}] Processing failed: {err_msg}. Rolling back created records...")
            
            # Clean up all partial data created during failed upload
            try:
                filename_query = {
                    "$or": [
                        {"source": filename},
                        {"metadata.source": filename},
                        {"document_id": document_id},
                        {"metadata.document_id": document_id}
                    ]
                }
                await db["pages"].delete_many({"document_id": document_id})
                await db["chunks"].delete_many(filename_query)
                await db["vector_store"].delete_many(filename_query)
                await db["summaries"].delete_many(filename_query)
                await self.update_doc_status(
                    db, filename, status="FAILED", stage="FAILED",
                    error=err_msg, document_id=document_id, status_details=f"Failed: {err_msg}"
                )
                
                # Delete physical file on failure
                if os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                    except Exception as fe:
                        print(f"[RAG ERROR] Could not remove failed physical file {file_path}: {fe}")
            except Exception as cleanup_err:
                print(f"[RAG ERROR] Rollback failed for {filename}: {cleanup_err}")

            return False

    async def remove_file(self, filename: Optional[str] = None, document_id: Optional[str] = None) -> bool:
        """Permanently remove a file's data from disk, index, vector database, Page Store, and MongoDB collections."""
        db = await get_db()
        target = document_id or filename
        if not target:
            return False

        # Look up document record to resolve both filename, document_id, and storage_path
        doc_record = await db["documents"].find_one({
            "$or": [
                {"document_id": target},
                {"filename": target},
                {"_id": target}
            ]
        })
        actual_filename = doc_record.get("filename") if doc_record else (filename or target)
        actual_doc_id = doc_record.get("document_id") if doc_record else (document_id or target)
        storage_path = doc_record.get("storage_path") or doc_record.get("file_path") if doc_record else None

        print(f"[RAG] Removing file '{actual_filename}' (doc_id={actual_doc_id}, storage_path={storage_path})...")

        if actual_doc_id:
            purge_query = {
                "$or": [
                    {"document_id": actual_doc_id},
                    {"metadata.document_id": actual_doc_id}
                ]
            }
            doc_delete_query = {
                "$or": [
                    {"document_id": actual_doc_id},
                    {"_id": actual_doc_id}
                ]
            }
        else:
            purge_query = {
                "$or": [
                    {"source": actual_filename},
                    {"metadata.source": actual_filename},
                    {"filename": actual_filename},
                    {"metadata.filename": actual_filename}
                ]
            }
            doc_delete_query = {"filename": actual_filename}

        # 1. Remove from pages, chunks, vector_store, summaries, and documents collections
        await db["pages"].delete_many({"document_id": actual_doc_id})
        await db["chunks"].delete_many(purge_query)
        await db["vector_store"].delete_many(purge_query)
        await db["summaries"].delete_many(purge_query)
        await db["documents"].delete_many(doc_delete_query)

        # 2. Remove physical file from disk
        candidate_paths = [
            storage_path,
            os.path.join(self.upload_folder, f"{actual_doc_id}_{actual_filename}") if actual_doc_id and actual_filename else None
        ]
        if actual_filename:
            other_docs = await db["documents"].count_documents({"filename": actual_filename, "document_id": {"$ne": actual_doc_id}})
            if other_docs == 0:
                candidate_paths.append(os.path.join(self.upload_folder, actual_filename))

        for c_path in candidate_paths:
            if c_path and os.path.isfile(c_path):
                try:
                    os.remove(c_path)
                    print(f"[RAG] Deleted physical file: {c_path}")
                except Exception as e:
                    print(f"[RAG WARNING] Could not remove physical file {c_path}: {e}")

        # 3. Remove stale conversation references
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

        # 4. Update in-memory BM25 document cache
        if actual_doc_id:
            self.all_docs = [
                doc for doc in self.all_docs 
                if doc.metadata.get("document_id") != actual_doc_id
            ]
        else:
            self.all_docs = [
                doc for doc in self.all_docs 
                if doc.metadata.get("source") != actual_filename and doc.metadata.get("filename") != actual_filename
            ]

        await self.rebuild_chain()
        print(f"[RAG] File removal complete: '{actual_filename}' (doc_id={actual_doc_id})")
        return True


    async def cleanup_orphaned_documents(self) -> Dict:
        """Scan indexed documents, pages, chunks, vectors, detect missing physical files or unlinked records, and purge orphaned records."""
        try:
            await self.ensure_initialized()
            db = await get_db()
            print("[Cleanup] Starting comprehensive orphan cleanup across all 3 storage tiers...")

            docs_cursor = db["documents"].find({})
            doc_records = await docs_cursor.to_list(length=None)

            valid_disk_names = set()
            valid_doc_filenames = set()
            valid_doc_ids = set()
            orphaned_doc_records = []

            for d in doc_records:
                fn = d.get("filename")
                d_id = d.get("document_id") or str(d.get("_id"))
                s_path = get_document_storage_path(d, self.upload_folder)
                if s_path and os.path.isfile(s_path):
                    if fn:
                        valid_doc_filenames.add(fn)
                    if d_id:
                        valid_doc_ids.add(d_id)
                    valid_disk_names.add(os.path.basename(s_path))
                    if d.get("disk_filename"):
                        valid_disk_names.add(d["disk_filename"])
                    if fn:
                        valid_disk_names.add(fn)
                        if d_id:
                            valid_disk_names.add(f"{d_id}_{fn}")
                            ascii_safe = re.sub(r'[^a-zA-Z0-9_.-]', '_', os.path.basename(fn))
                            valid_disk_names.add(f"{d_id}_{ascii_safe}")
                else:
                    orphaned_doc_records.append(d)

            # 1. Purge document records that have no physical file
            docs_removed_count = 0
            for orphan_doc in orphaned_doc_records:
                fn = orphan_doc.get("filename")
                d_id = orphan_doc.get("document_id")
                print(f"[Cleanup] Purging document record without physical file: '{fn}' (id={d_id})")
                purge_q = {"$or": [{"filename": fn}, {"document_id": d_id}]}
                await db["documents"].delete_many(purge_q)
                docs_removed_count += 1

            # 2. Purge orphaned physical files on disk with no document record
            files_removed_count = 0
            now_ts = time.time()
            if os.path.exists(self.upload_folder):
                for disk_name in os.listdir(self.upload_folder):
                    if disk_name.startswith('.'):
                        continue
                    if disk_name not in valid_disk_names:
                        disk_path = os.path.join(self.upload_folder, disk_name)
                        if os.path.isfile(disk_path):
                            # Grace period: do not purge files modified/created in last 30 seconds
                            if now_ts - os.path.getmtime(disk_path) < 30:
                                continue
                            try:
                                os.remove(disk_path)
                                print(f"[Cleanup] Removed unindexed orphaned file from disk: '{disk_name}'")
                                files_removed_count += 1
                            except Exception as fe:
                                print(f"[Cleanup] Could not remove file {disk_name}: {fe}")

            # 3. Purge orphaned pages in Page Store
            pages_removed_count = 0
            if not valid_doc_ids and not valid_doc_filenames:
                p_del = await db["pages"].delete_many({})
                pages_removed_count += p_del.deleted_count
            else:
                p_del = await db["pages"].delete_many({
                    "$and": [
                        {"document_id": {"$nin": list(valid_doc_ids)}},
                        {"filename": {"$nin": list(valid_doc_filenames)}}
                    ]
                })
                pages_removed_count += p_del.deleted_count

            # 4. Purge chunks, vectors, summaries that don't belong to any valid document
            chunks_removed_count = 0
            vectors_removed_count = 0
            summaries_removed_count = 0

            unlinked_chunk_q = {
                "$and": [
                    {"document_id": {"$nin": list(valid_doc_ids)}},
                    {"source": {"$nin": list(valid_doc_filenames)}},
                    {"metadata.source": {"$nin": list(valid_doc_filenames)}}
                ]
            }
            if not valid_doc_filenames and not valid_doc_ids:
                c_del = await db["chunks"].delete_many({})
                v_del = await db["vector_store"].delete_many({})
                s_del = await db["summaries"].delete_many({})
                chunks_removed_count += c_del.deleted_count
                vectors_removed_count += v_del.deleted_count
                summaries_removed_count += s_del.deleted_count
                await db["conversations"].update_many({}, {"$set": {"selected_document_ids": [], "document_name": None}})
            else:
                c_del = await db["chunks"].delete_many(unlinked_chunk_q)
                v_del = await db["vector_store"].delete_many(unlinked_chunk_q)
                s_del = await db["summaries"].delete_many(unlinked_chunk_q)
                chunks_removed_count += c_del.deleted_count
                vectors_removed_count += v_del.deleted_count
                summaries_removed_count += s_del.deleted_count

                registered_doc_names = set(d.get("filename") for d in doc_records if d.get("filename"))
                for orphan_fn in (registered_doc_names - valid_doc_filenames):
                    await db["conversations"].update_many({}, {"$pull": {"selected_document_ids": orphan_fn}})
                    await db["conversations"].update_many({"document_name": orphan_fn}, {"$set": {"document_name": None}})

            # 5. Clean up broken conversation attachments
            broken_attachments_count = 0
            convs = await db["conversations"].find({"selected_document_ids.0": {"$exists": True}}).to_list(length=None)
            for conv in convs:
                current_ids = conv.get("selected_document_ids", [])
                valid_ids = [cid for cid in current_ids if cid in valid_doc_ids or cid in valid_doc_filenames]
                if len(valid_ids) != len(current_ids):
                    broken_count = len(current_ids) - len(valid_ids)
                    broken_attachments_count += broken_count
                    await db["conversations"].update_one(
                        {"_id": conv["_id"]},
                        {"$set": {"selected_document_ids": valid_ids}}
                    )

            # 6. Re-sync memory cache from chunks collection
            all_chunks_cursor = db["chunks"].find({})
            all_db_chunks = await all_chunks_cursor.to_list(length=None)
            self.all_docs = [
                Document(page_content=c.get("content") or c.get("text", ""), metadata=c.get("metadata", {}))
                for c in all_db_chunks
                if c.get("metadata", {}).get("source") in valid_doc_filenames or c.get("source") in valid_doc_filenames
            ]
            await self.rebuild_chain()

            # Final verified counts
            active_docs = len(valid_doc_filenames)
            active_chunks = len(self.all_docs)
            total_removed = docs_removed_count + pages_removed_count + chunks_removed_count + vectors_removed_count + summaries_removed_count + files_removed_count + broken_attachments_count

            report = {
                "success": True,
                "message": "Orphan cleanup completed successfully",
                "documents_scanned": len(doc_records),
                "orphaned_documents": docs_removed_count,
                "removed_documents": docs_removed_count,
                "removed_pages": pages_removed_count,
                "removed_chunks": chunks_removed_count,
                "removed_embeddings": vectors_removed_count,
                "broken_attachments": broken_attachments_count,
                "documents_removed": docs_removed_count,
                "chunks_removed": chunks_removed_count,
                "vectors_removed": vectors_removed_count,
                "summaries_removed": summaries_removed_count,
                "files_removed": files_removed_count,
                "records_removed": total_removed,
                "indexed_documents": active_docs,
                "indexed_chunks": active_chunks,
            }
            print(f"[Cleanup] Cleanup completed report: {report}")
            return report
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[Cleanup] ERROR: {e}")
            raise

# Singleton
rag_engine = RAGEngine()




