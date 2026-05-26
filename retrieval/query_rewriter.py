import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

GENERATION_MODEL = "models/gemini-2.0-flash"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_REWRITE_PROMPT = """\
You are a search query optimizer for a technical documentation search engine.

Rewrite the user's question into a precise, keyword-rich search query that will \
retrieve the most relevant documentation chunks. Focus on technical terms. \
Remove filler words. Do NOT answer the question — only rewrite it.

User question: {query}

Rewritten query (one line, no explanation):"""

_HYDE_PROMPT = """\
You are a technical documentation expert.

Given the user's question, write a short passage (3–5 sentences) that would \
plausibly appear in official technical documentation and directly answer the question. \
Write in documentation style — precise, factual, and technical. \
Do NOT say "I" or answer conversationally.

User question: {query}

Hypothetical documentation passage:"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rewrite_query(query: str) -> str:
    """
    Rewrite the user's query to be more precise for vector search.
    Strips filler words and focuses on technical terminology.
    """
    prompt = _REWRITE_PROMPT.format(query=query)
    try:
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt
        )
        rewritten = response.text.strip()
        # Safety: if the model returns something empty or too long, fall back
        if not rewritten or len(rewritten) > 500:
            return query
        return rewritten
    except Exception as e:
        print(f"  [QueryRewriter] rewrite failed: {e}. Using original query.")
        return query


def generate_hyde_passage(query: str) -> str:
    """
    Generate a Hypothetical Document Embedding (HyDE) passage.

    HyDE (Gao et al. 2022) works by generating a fake-but-plausible answer,
    embedding it, and searching for real documents that are close to that
    embedding. This dramatically improves recall for precise factual questions
    because the hypothetical passage lands in the right semantic neighbourhood
    even when the raw question doesn't.
    """
    prompt = _HYDE_PROMPT.format(query=query)
    try:
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt
        )
        passage = response.text.strip()
        if not passage:
            return query
        return passage
    except Exception as e:
        print(f"  [QueryRewriter] HyDE generation failed: {e}. Using original query.")
        return query


def expand_query(query: str) -> list[str]:
    """
    Full query expansion pipeline. Returns a list of query variants to search with:
      1. The original query (always included)
      2. The rewritten/keyword-focused query
      3. The HyDE passage (fake documentation passage)

    The retriever will search with all three and merge results.
    """
    rewritten = rewrite_query(query)
    hyde = generate_hyde_passage(query)

    # Deduplicate — if rewrite returns the same as original, don't search twice
    queries = [query]
    if rewritten.lower() != query.lower():
        queries.append(rewritten)
    queries.append(hyde)

    return queries
