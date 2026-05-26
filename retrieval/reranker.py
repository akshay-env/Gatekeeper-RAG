import os
import re
import json
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

GENERATION_MODEL = "models/gemini-2.0-flash"
MAX_CHUNK_CHARS = 600   # truncate each chunk in the prompt to keep token count manageable

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_RERANK_PROMPT = """\
You are a relevance scoring system for a technical documentation RAG pipeline.

Given a user query and a numbered list of retrieved text chunks, score each chunk's \
relevance to the query on a scale of 0–10:
  10 = directly and completely answers the query
   7 = highly relevant, covers most of what's needed
   4 = partially relevant, tangentially related
   1 = barely related
   0 = completely irrelevant

Query: {query}

Retrieved chunks:
{chunks}

Return ONLY a valid JSON array of integer scores, one per chunk, in the same order.
Example for 4 chunks: [8, 2, 6, 0]
No explanation. No markdown. Just the JSON array."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rerank(query: str, chunks: list[dict], top_n: int = 5) -> list[dict]:
    """
    Re-rank retrieved chunks using Gemini as a pointwise relevance judge.

    Each chunk gets a 0–10 relevance score. Results are sorted by score
    and the top_n are returned. Falls back to original order if the LLM
    response can't be parsed.

    This is "LLM-as-judge" reranking — more precise than bi-encoder cosine
    similarity because the model reads both the query and the chunk together
    (cross-attention), rather than comparing independent embeddings.
    """
    if not chunks:
        return []

    if len(chunks) <= top_n:
        # Nothing to rerank — return as-is
        return chunks

    # Build the chunk list for the prompt (truncated to keep prompt manageable)
    chunks_text = "\n\n".join(
        f"[{i + 1}] {chunk['text'][:MAX_CHUNK_CHARS]}"
        + ("..." if len(chunk["text"]) > MAX_CHUNK_CHARS else "")
        for i, chunk in enumerate(chunks)
    )

    prompt = _RERANK_PROMPT.format(query=query, chunks=chunks_text)

    try:
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt
        )
        raw = response.text.strip()

        # Strip markdown code fences if model wraps in ```json ... ```
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

        scores = json.loads(raw)

        if not isinstance(scores, list) or len(scores) != len(chunks):
            raise ValueError(f"Score count mismatch: got {len(scores)}, expected {len(chunks)}")

        # Attach rerank scores and sort
        for i, chunk in enumerate(chunks):
            chunk["rerank_score"] = int(scores[i])

        ranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_n]

    except Exception as e:
        print(f"  [Reranker] Failed ({e}). Falling back to original retrieval order.")
        # Fall back: return top_n by original vector score
        return sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)[:top_n]
