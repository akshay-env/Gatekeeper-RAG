import os
import uuid
import time
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

EMBEDDING_MODEL = "models/gemini-embedding-2"
EMBEDDING_DIM = 3072      # output dimension for gemini-embedding-2
BATCH_SIZE = 100          # max texts per API call — saturates each request
SLEEP_BETWEEN_BATCHES = 15   # seconds between batches — ~4 req/min, safe for free tier
RATE_LIMIT_WAIT = 60         # seconds to wait on a 429 — full quota window reset
MAX_RETRIES = 8              # higher since 429s just need time, not a failure signal

qdrant = QdrantClient("localhost", port=6333)


def get_embeddings(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """
    Embed a batch of texts using the Gemini embedding model.
    - On 429 (rate limit): waits RATE_LIMIT_WAIT seconds — one full quota window.
    - On other errors: exponential backoff.
    """
    for attempt in range(MAX_RETRIES):
        try:
            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(task_type=task_type)
            )
            return [e.values for e in result.embeddings]

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                # Rate limit hit — wait a full minute for the quota window to reset
                wait = RATE_LIMIT_WAIT
                print(f"  [Attempt {attempt + 1}/{MAX_RETRIES}] Rate limit hit. Waiting {wait}s for quota reset...")
            else:
                # Other error — exponential backoff
                wait = 2 ** attempt
                print(f"  [Attempt {attempt + 1}/{MAX_RETRIES}] Embedding failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)

    raise RuntimeError(f"Failed to embed batch after {MAX_RETRIES} retries.")


def embed_and_store(chunks: list[dict]):
    """
    Embed all chunks and store them in Qdrant.
    Processes in batches of BATCH_SIZE with rate-limit-safe pacing.
    """
    qdrant.recreate_collection(
        collection_name="docs",
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
    )

    total = len(chunks)
    print(f"Embedding and storing {total} chunks using Gemini API ({EMBEDDING_MODEL})...")

    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [chunk["text"] for chunk in batch]

        embeddings = get_embeddings(texts, task_type="RETRIEVAL_DOCUMENT")

        points = []
        for j, embedding in enumerate(embeddings):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": batch[j]["text"],
                    "source": batch[j]["source"],
                    "breadcrumb": batch[j].get("breadcrumb", "")
                }
            ))

        qdrant.upsert(collection_name="docs", points=points)
        stored_so_far = min(i + BATCH_SIZE, total)
        print(f"  Stored {stored_so_far}/{total} chunks")

        # Rate-limit guard — sleep between batches to stay under RPM ceiling
        if stored_so_far < total:
            time.sleep(SLEEP_BETWEEN_BATCHES)

    print("Done! All chunks embedded and stored.")