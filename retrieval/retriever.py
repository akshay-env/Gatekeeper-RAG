import os
import time
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
qdrant = QdrantClient("localhost", port=6333)

EMBEDDING_MODEL = "models/gemini-embedding-2"
COLLECTION_NAME = "docs"
RATE_LIMIT_WAIT = 60
MAX_RETRIES = 8


def _embed_query(text: str) -> list[float]:
    """
    Embed a single query string using task_type=RETRIEVAL_QUERY.
    Uses RETRIEVAL_QUERY (not RETRIEVAL_DOCUMENT) — this is important:
    the model applies asymmetric encoding so queries and docs are comparable.
    """
    for attempt in range(MAX_RETRIES):
        try:
            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=[text],
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
            )
            return result.embeddings[0].values

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                wait = RATE_LIMIT_WAIT
                print(f"  [Retriever] Rate limit hit. Waiting {wait}s...")
            else:
                wait = 2 ** attempt
                print(f"  [Retriever] Embedding failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)

    raise RuntimeError("Failed to embed query after retries.")


def retrieve(query: str, top_k: int = 20) -> list[dict]:
    """
    Embed a query and search Qdrant for the top_k most similar chunks.
    Returns results as a list of dicts with text, source, breadcrumb, score.
    """
    query_vector = _embed_query(query)

    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
    )

    return [
        {
            "text": hit.payload.get("text", ""),
            "source": hit.payload.get("source", ""),
            "breadcrumb": hit.payload.get("breadcrumb", ""),
            "score": round(hit.score, 4),
        }
        for hit in hits
    ]


def multi_retrieve(queries: list[str], top_k: int = 20) -> list[dict]:
    """
    Run retrieve() for each query and merge results, deduplicating on text content.
    Used when query_rewriter generates multiple query variants (HyDE + rewrite).
    Preserves the highest score seen for each unique chunk.
    """
    seen: dict[str, dict] = {}  # text → best result dict

    for query in queries:
        results = retrieve(query, top_k=top_k)
        for r in results:
            key = r["text"]
            if key not in seen or r["score"] > seen[key]["score"]:
                seen[key] = r

    # Return merged results sorted by score descending
    merged = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
    return merged
