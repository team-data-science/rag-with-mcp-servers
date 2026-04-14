#!/usr/bin/env python3
# VoltEdge Energy — Q&A data creator for Qdrant
#
# Reads Q&A pairs from voltedge_qa.json and ingests them into a Qdrant
# collection. Because VoltEdge is a fictional company, any correct answer
# from the RAG system proves retrieval is working — Qwen has no prior
# knowledge of these specifics.
#
# Requirements:
#   pip install qdrant-client sentence-transformers
#
# Usage:
#   python voltedge_creator.py
#   QDRANT_URL=http://localhost:6333 QDRANT_COLLECTION=voltedge-qa python voltedge_creator.py

import json
import os
import sys
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from qdrant_client.http.exceptions import ResponseHandlingException
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
QDRANT_URL        = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY", None)
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "voltedge-qa")
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
VECTOR_SIZE       = 384
BATCH_SIZE        = int(os.getenv("BATCH_SIZE", "64"))

SCRIPT_DIR = Path(__file__).parent
DEFAULT_JSON = SCRIPT_DIR / "voltedge_qa.json"
JSON_PATH = Path(os.getenv("JSON_PATH", DEFAULT_JSON))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_qa_pairs(path: Path) -> list[dict]:
    if not path.is_file():
        print(f"JSON file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print("JSON must be a list of {question, answer} objects.", file=sys.stderr)
        sys.exit(1)
    return [row for row in data if row.get("question") and row.get("answer")]


def init_client() -> QdrantClient:
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        existing = [c.name for c in client.get_collections().collections]
    except ResponseHandlingException as exc:
        print(f"Cannot connect to Qdrant at {QDRANT_URL}: {exc}", file=sys.stderr)
        sys.exit(1)

    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"Created collection '{QDRANT_COLLECTION}'.")
    else:
        print(f"Collection '{QDRANT_COLLECTION}' already exists — upserting.")
    return client


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    qa_pairs = load_qa_pairs(JSON_PATH)
    print(f"Loaded {len(qa_pairs)} Q&A pairs from {JSON_PATH}")

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    client = init_client()

    batch: list[PointStruct] = []
    total = 0

    for idx, row in enumerate(qa_pairs, start=1):
        question = row["question"]
        answer   = row["answer"]
        vector   = model.encode(question + "\n" + answer).tolist()

        batch.append(
            PointStruct(
                id=idx,
                vector=vector,
                payload={
                    "question": question,
                    "answer": answer,
                    "source": row.get("source", "voltedge-use-case"),
                },
            )
        )

        if len(batch) >= BATCH_SIZE:
            client.upsert(collection_name=QDRANT_COLLECTION, points=batch)
            total += len(batch)
            print(f"Upserted {total} points…")
            batch.clear()

    if batch:
        client.upsert(collection_name=QDRANT_COLLECTION, points=batch)
        total += len(batch)

    print(f"Done — {total} Q&A pairs ingested into '{QDRANT_COLLECTION}'.")


if __name__ == "__main__":
    main()
