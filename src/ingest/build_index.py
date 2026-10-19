import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import hashlib
import chromadb

from src.config import DB_PATH, COLLECTION_NAME, LECTURES_DIR, PROCESSED_LOG_PATH
from src.ingest.parse_docs import parse_lecture_docx, split_oversized_sections, FENCE, is_reading_section

PARSER_VERSION = "4"

def file_fingerprint(path):
    with open(path, "rb") as f:
        content_hash = hashlib.md5(f.read()).hexdigest()
    return f"v{PARSER_VERSION}:{content_hash}"


def load_log():
    if os.path.exists(PROCESSED_LOG_PATH):
        with open(PROCESSED_LOG_PATH, "r") as f:
            return json.load(f)
    return {}


def save_log(log):
    os.makedirs(os.path.dirname(PROCESSED_LOG_PATH), exist_ok=True)
    with open(PROCESSED_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


def build_index():
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    print("Chunks currently inside:", collection.count())

    log = load_log()

    files = sorted(
        f for f in os.listdir(LECTURES_DIR)
        if f.lower().endswith(".docx") and not f.startswith("~$")
    )

    total_chunks = 0

    for filename in files:
        path = os.path.join(LECTURES_DIR, filename)
        fingerprint = file_fingerprint(path)

        if log.get(filename) == fingerprint:
            print("SKIP (unchanged):", filename)
            continue

        print("PROCESS:", filename)

        lecture_name = os.path.splitext(filename)[0]
        sections = parse_lecture_docx(path, lecture_name)
        chunks = split_oversized_sections(sections)
        before = len(chunks)
        chunks = [c for c in chunks if not is_reading_section(c["section"])]
        skipped = before - len(chunks)

        documents = []
        metadatas = []
        ids = []

        for i, chunk in enumerate(chunks):
            documents.append(chunk["section"] + "\n" + chunk["text"])
            metadatas.append({
                "lecture": chunk["lecture"],
                "section": chunk["section"],
                "content_type": chunk["content_type"],
                "has_code": FENCE in chunk["text"],
            })
            ids.append(f"{lecture_name}::{i}")

        collection.delete(where={"lecture": lecture_name})
        collection.add(ids=ids, documents=documents, metadatas=metadatas)

        log[filename] = fingerprint
        save_log(log)

        print("    sections:", len(sections), "| stored chunks:", len(ids), "| skipped reading lists:", skipped)
        total_chunks += len(ids)

    print("TOTAL chunks added this run:", total_chunks)
    print("Chunks in database now:", collection.count())
if __name__ == "__main__":
    build_index()