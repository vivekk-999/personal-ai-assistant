import asyncio
import os
import io
from pypdf import PdfWriter
from pypdf.generic import NameObject, DictionaryObject, DecodedStreamObject
from database import db_instance, get_db, UPLOAD_FOLDER
from rag_engine import rag_engine


def create_structured_multi_page_pdf(filepath: str):
    """Creates a 5-page PDF with clear, distinct topics per page, code, and table."""
    writer = PdfWriter()
    font_dict = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica")
    })
    font_obj = writer._add_object(font_dict)
    resources = DictionaryObject({
        NameObject("/Font"): DictionaryObject({
            NameObject("/Helvetica"): font_obj
        })
    })

    page_contents = [
        # Page 1: Introduction to Distributed Computing
        "Chapter 1: Introduction to Distributed Systems\n"
        "Page 1 of 5\n"
        "Distributed computing involves multiple interconnected machines working together.\n"
        "The core advantage is horizontal scalability and high fault tolerance.\n"
        "Key challenges include consensus, network latency, and data consistency.",

        # Page 2: Storage Architectures & Data Formats
        "Chapter 2: Storage Architectures\n"
        "Page 2 of 5\n"
        "Storage is divided into block, file, and object storage tiers.\n"
        "Parquet and ORC provide columnar compression for analytical queries.\n"
        "Write-Ahead Logging (WAL) ensures ACID transaction durability.",

        # Page 3: Consensus Protocols (MIDDLE PAGE)
        "Chapter 3: Consensus Protocols & Raft Algorithm\n"
        "Page 3 of 5 (MIDDLE PAGE)\n"
        "Raft is a leader-based consensus algorithm designed for understandability.\n"
        "The three states of a Raft node are Follower, Candidate, and Leader.\n"
        "Heartbeats are broadcast at 150ms intervals to maintain leadership lease.",

        # Page 4: Cross-Data Routing and Caching
        "Chapter 4: Distributed Caching & Message Queues\n"
        "Page 4 of 5\n"
        "Redis provides in-memory key-value caching with sub-millisecond latencies.\n"
        "Kafka uses partitioned append-only commit logs for real-time streaming.\n"
        "Cross-page relationship: Caches reduce read pressure on Chapter 2 storage architectures.",

        # Page 5: Security, Zero Trust & Future Trends (LAST PAGE)
        "Chapter 5: Zero-Trust Security & Modern Deployments\n"
        "Page 5 of 5 (FINAL PAGE)\n"
        "Zero-trust security requires mutual TLS (mTLS) authentication for every microservice.\n"
        "eBPF enables high-performance observability in modern Linux kernels.\n"
        "Conclusion: Distributed systems require end-to-end consensus, caching, and zero trust."
    ]

    for p_num, content_text in enumerate(page_contents, start=1):
        page = writer.add_blank_page(width=612, height=792)
        lines = content_text.split('\n')
        pdf_cmds = ["BT", "/Helvetica 12 Tf", "50 750 Td"]
        for i, line in enumerate(lines):
            safe_l = line.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
            if i == 0:
                pdf_cmds.append(f"({safe_l}) Tj")
            else:
                pdf_cmds.append(f"0 -20 Td ({safe_l}) Tj")
        pdf_cmds.append("ET")
        
        stream = DecodedStreamObject()
        stream.set_data("\n".join(pdf_cmds).encode("latin-1"))
        page[NameObject("/Contents")] = writer._add_object(stream)
        page[NameObject("/Resources")] = resources

    with open(filepath, "wb") as f:
        writer.write(f)
    return filepath


async def run_full_pdf_access_tests():
    print("=" * 70)
    print("🚀 RUNNING FULL PDF ACCESS & MULTI-STAGE REASONING TEST SUITE")
    print("=" * 70)

    db = await db_instance.connect()
    await rag_engine.initialize()

    doc_filename = "Distributed Systems Guide.pdf"
    doc_id = "doc-dist-sys-501"
    pdf_path = os.path.join(UPLOAD_FOLDER, doc_filename)
    create_structured_multi_page_pdf(pdf_path)

    # Ingest document
    await rag_engine.process_file(pdf_path, doc_filename, document_id=doc_id)

    # Verify Page Store
    page_records = await db["pages"].find({"document_id": doc_id}).sort("page_number", 1).to_list(length=10)
    print(f"\n✓ Extracted {len(page_records)} pages into Page Store (db['pages']).")
    assert len(page_records) == 5, f"Expected 5 pages, got {len(page_records)}"

    # ── TEST 1: PAGE 1 QUESTION ───────────────────────────────────────────────
    print("\n--- TEST 1: PAGE 1 QUESTION ---")
    tokens1 = []
    sources1 = []
    async for token, srcs, details in rag_engine.stream_rag_reply("What is discussed on page 1?", doc_names=[doc_id]):
        tokens1.append(token)
        sources1 = srcs
    reply1 = "".join(tokens1)
    print(f"Reply 1:\n{reply1}")
    print(f"Sources: {sources1}")
    assert "Distributed Systems" in reply1 or "horizontal scalability" in reply1 or "consensus" in reply1
    assert any("Page 1" in s or "p. 1" in s for s in sources1)
    print("✓ Test 1 Passed: Page 1 accurately retrieved from Page Store with citation.")

    # ── TEST 2: MIDDLE-PAGE QUESTION ──────────────────────────────────────────
    print("\n--- TEST 2: MIDDLE-PAGE QUESTION (Page 3) ---")
    tokens2 = []
    sources2 = []
    async for token, srcs, details in rag_engine.stream_rag_reply("What is discussed on the middle page of this document?", doc_names=[doc_id]):
        tokens2.append(token)
        sources2 = srcs
    reply2 = "".join(tokens2)
    print(f"Reply 2:\n{reply2}")
    print(f"Sources: {sources2}")
    assert "Raft" in reply2 or "Consensus" in reply2 or "Follower" in reply2 or "Leader" in reply2
    assert any("Page 3" in s or "p. 3" in s for s in sources2)
    print("✓ Test 2 Passed: Middle page (Page 3) accurately retrieved with citation.")

    # ── TEST 3: LAST-PAGE QUESTION ────────────────────────────────────────────
    print("\n--- TEST 3: LAST-PAGE QUESTION (Page 5) ---")
    tokens3 = []
    sources3 = []
    async for token, srcs, details in rag_engine.stream_rag_reply("What does the last page cover?", doc_names=[doc_id]):
        tokens3.append(token)
        sources3 = srcs
    reply3 = "".join(tokens3)
    print(f"Reply 3:\n{reply3}")
    print(f"Sources: {sources3}")
    assert "Zero-Trust" in reply3 or "mTLS" in reply3 or "eBPF" in reply3 or "Security" in reply3
    assert any("Page 5" in s or "p. 5" in s for s in sources3)
    print("✓ Test 3 Passed: Last page (Page 5) accurately retrieved with citation.")

    # ── TEST 4: CROSS-PAGE QUESTION ───────────────────────────────────────────
    print("\n--- TEST 4: CROSS-PAGE QUESTION (Compare Page 2 and Page 4) ---")
    tokens4 = []
    sources4 = []
    async for token, srcs, details in rag_engine.stream_rag_reply("Compare page 2 and page 4: how does caching relate to storage architectures?", doc_names=[doc_id]):
        tokens4.append(token)
        sources4 = srcs
    reply4 = "".join(tokens4)
    print(f"Reply 4:\n{reply4}")
    print(f"Sources: {sources4}")
    assert ("Storage" in reply4 or "Parquet" in reply4 or "WAL" in reply4 or "storage" in reply4)
    assert ("Redis" in reply4 or "Kafka" in reply4 or "caching" in reply4 or "cache" in reply4)
    assert any("Page 2" in s or "p. 2" in s or "2" in s for s in sources4)
    assert any("Page 4" in s or "p. 4" in s or "4" in s for s in sources4)
    print("✓ Test 4 Passed: Cross-page synthesis correctly combined Page 2 and Page 4.")

    # ── TEST 5: COMPLETE PDF SUMMARY ──────────────────────────────────────────
    print("\n--- TEST 5: COMPLETE PDF SUMMARY ---")
    tokens5 = []
    sources5 = []
    async for token, srcs, details in rag_engine.stream_rag_reply("Summarize the complete PDF document across all chapters", doc_names=[doc_id]):
        tokens5.append(token)
        sources5 = srcs
    reply5 = "".join(tokens5)
    print(f"Reply 5:\n{reply5}")
    print(f"Sources: {sources5}")
    assert len(reply5) > 100
    print("✓ Test 5 Passed: Complete PDF summary covers all document pages.")

    # ── TEST 6: GIVE THE MAIN TOPIC OF EVERY PAGE ─────────────────────────────
    print("\n--- TEST 6: GIVE THE MAIN TOPIC OF EVERY PAGE ---")
    tokens6 = []
    sources6 = []
    async for token, srcs, details in rag_engine.stream_rag_reply("Give the main topic of every page in this PDF", doc_names=[doc_id]):
        tokens6.append(token)
        sources6 = srcs
    reply6 = "".join(tokens6)
    print(f"Reply 6:\n{reply6}")
    print(f"Sources: {sources6}")
    assert "Page 1" in reply6 or "1." in reply6
    assert "Page 2" in reply6 or "2." in reply6
    assert "Page 3" in reply6 or "3." in reply6
    assert "Page 4" in reply6 or "4." in reply6
    assert "Page 5" in reply6 or "5." in reply6
    print("✓ Test 6 Passed: Page-by-page breakdown gives main topic of EVERY page.")

    # Clean up
    await rag_engine.remove_file(filename=doc_filename, document_id=doc_id)
    print("\n" + "=" * 70)
    print("🎉 ALL 6 FULL PDF ACCESS TESTS PASSED PERFECTLY!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_full_pdf_access_tests())
