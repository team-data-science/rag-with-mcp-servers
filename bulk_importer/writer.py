#!/usr/bin/env python3
# Requirements:
#   python3 -m pip install qdrant-client sentence-transformers

import os
import sys
import csv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from qdrant_client.http.exceptions import ResponseHandlingException
from sentence_transformers import SentenceTransformer

# Determine script directory to resolve relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuration
# Default to localhost; override with QDRANT_URL env var if needed
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "q-and-a")
VECTOR_SIZE = int(os.getenv("QDRANT_VECTOR_SIZE", "384"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))

# CSV path resolution
def resolve_csv_path(filename: str) -> str:
    if os.path.isabs(filename):
        return filename
    return os.path.join(SCRIPT_DIR, filename)

CSV_DEFAULT = "questions-answers.csv"
CSV_PATH = resolve_csv_path(os.getenv("CSV_PATH", CSV_DEFAULT))

# Load embedding model
_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def get_embedding(text: str) -> list:
    """Generate 384-dim embeddings using all-MiniLM-L6-v2."""
    return _model.encode(text).tolist()


def init_qdrant_client() -> QdrantClient:
    """Initialize and return a Qdrant client; check connectivity and ensure the collection exists."""
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        # Attempt to list collections to verify connection
        collections = client.get_collections().collections
    except ResponseHandlingException as e:
        print(f"Error connecting to Qdrant at {QDRANT_URL}: {e}", file=sys.stderr)
        print("Please ensure Qdrant is running and QDRANT_URL is set correctly.")
        sys.exit(1)
    # Create collection if missing
    if not any(c.name == QDRANT_COLLECTION for c in collections):
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )
        print(f"Created collection '{QDRANT_COLLECTION}' with vector size {VECTOR_SIZE}")
    else:
        print(f"Collection '{QDRANT_COLLECTION}' already exists.")
    return client


def read_qa_csv(path: str):
    """Yield tuples of (question, answer) from CSV file."""
    with open(path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            question = row.get('Questions') or row.get('Question')
            answer = row.get('Answers') or row.get('Answer')
            if question and answer:
                yield question, answer


def upsert_points(client: QdrantClient, points: list[PointStruct]):
    """Bulk upsert points into Qdrant collection."""
    client.upsert(collection_name=QDRANT_COLLECTION, points=points)


def main():
    # Check CSV
    if not os.path.isfile(CSV_PATH):
        print(f"CSV file not found: {CSV_PATH}", file=sys.stderr)
        sys.exit(1)
    # Initialize Qdrant client
    client = init_qdrant_client()

    batch: list[PointStruct] = []
    total = 0

    for idx, (question, answer) in enumerate(read_qa_csv(CSV_PATH), start=1):
        vector = get_embedding(question + "\n" + answer)
        payload = {"question": question, "answer": answer}
        batch.append(PointStruct(id=idx, vector=vector, payload=payload))

        if len(batch) >= BATCH_SIZE:
            upsert_points(client, batch)
            total += len(batch)
            print(f"Upserted {total} points so far.")
            batch.clear()

    if batch:
        upsert_points(client, batch)
        total += len(batch)
        print(f"Upserted total of {total} points.")


if __name__ == "__main__":
    main()
