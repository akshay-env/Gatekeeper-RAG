"""
evaluation/testset.py

Automatically generates a test set of (question, ground_truth_answer) pairs
from the indexed documents in Qdrant.

How it works:
  1. Pulls a random sample of chunks from the Qdrant collection
  2. Filters out chunks that are too short to make meaningful questions
  3. Sends each chunk to Gemini and asks it to generate a realistic Q&A pair
  4. Saves the result to evaluation/testset.json for repeated use

Run with:
  python -m evaluation.testset
"""

import os
import json
import time
import random
from pathlib import Path
from google import genai
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
qdrant = QdrantClient("localhost", port=6333)

GENERATION_MODEL = "models/gemini-2.5-flash"
COLLECTION_NAME = "docs"
TESTSET_PATH = Path(__file__).parent / "testset.json"
MIN_CHUNK_CHARS = 300       # skip chunks too short for a good question
SLEEP_BETWEEN_CALLS = 5     # seconds — stay under RPM limit

_QA_PROMPT = """\
You are creating a test dataset for evaluating a RAG (Retrieval-Augmented Generation) system.

Given the following documentation passage, generate ONE realistic question that a developer \
would ask, and the correct answer based ONLY on the passage.

Rules:
- The question must be answerable from the passage alone.
- The answer must be specific and grounded in the passage text.
- Do NOT ask "what is this passage about?" — ask a practical, technical question.
- Return ONLY valid JSON in this exact format (no markdown, no explanation):

{{
  "question": "...",
  "ground_truth": "..."
}}

Documentation passage:
{chunk}"""


def _sample_chunks(n: int) -> list[dict]:
    """Pull n random points from Qdrant and return their payloads."""
    try:
        result = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=500,          # pull a larger pool to sample from
            with_payload=True,
            with_vectors=False,
        )
        all_points = result[0]
    except UnexpectedResponse as e:
        raise RuntimeError(
            "Could not reach Qdrant or collection doesn't exist. "
            "Run `python run.py` first."
        ) from e

    # Filter to chunks long enough to generate a good question
    eligible = [
        p for p in all_points
        if len(p.payload.get("text", "")) >= MIN_CHUNK_CHARS
    ]

    if len(eligible) < n:
        print(f"  Warning: only {len(eligible)} eligible chunks found (wanted {n}).")
        n = len(eligible)

    return random.sample(eligible, n)


def _generate_qa(chunk_text: str) -> dict | None:
    """Ask Gemini to generate a Q&A pair from a single chunk. Returns None on failure."""
    prompt = _QA_PROMPT.format(chunk=chunk_text[:1500])  # cap at 1500 chars
    try:
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt
        )
        raw = response.text.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:-1])
        return json.loads(raw)
    except Exception as e:
        print(f"  Skipping chunk (generation failed: {e})")
        return None


def generate_testset(n: int = 20, save: bool = True) -> list[dict]:
    """
    Generate n test cases from the indexed documents.

    Each test case is:
      {
        "question":     str,   # auto-generated question
        "ground_truth": str,   # correct answer from the source chunk
        "source_chunk": str,   # the chunk text used to generate it
        "source":       str,   # source file path
        "breadcrumb":   str,   # section header hierarchy
      }

    Args:
        n:    Number of test cases to generate.
        save: If True, write to evaluation/testset.json.
    """
    print(f"Sampling {n} chunks from Qdrant...")
    points = _sample_chunks(n)

    testset = []
    for i, point in enumerate(points):
        payload = point.payload
        chunk_text = payload.get("text", "")
        print(f"  [{i + 1}/{len(points)}] Generating Q&A from: "
              f"{payload.get('breadcrumb', payload.get('source', '?'))[:60]}...")

        qa = _generate_qa(chunk_text)
        if qa is None:
            continue

        testset.append({
            "question":     qa.get("question", ""),
            "ground_truth": qa.get("ground_truth", ""),
            "source_chunk": chunk_text,
            "source":       payload.get("source", ""),
            "breadcrumb":   payload.get("breadcrumb", ""),
        })

        if i < len(points) - 1:
            time.sleep(SLEEP_BETWEEN_CALLS)

    print(f"\nGenerated {len(testset)} test cases.")

    if save:
        TESTSET_PATH.write_text(json.dumps(testset, indent=2), encoding="utf-8")
        print(f"Saved to {TESTSET_PATH}")

    return testset


def load_testset() -> list[dict]:
    """Load the saved testset from disk."""
    if not TESTSET_PATH.exists():
        raise FileNotFoundError(
            f"No testset found at {TESTSET_PATH}. "
            "Run `python -m evaluation.testset` to generate one."
        )
    return json.loads(TESTSET_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    generate_testset(n=20)
