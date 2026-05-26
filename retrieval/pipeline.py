"""
retrieval/pipeline.py

Unified retrieval pipeline. This is the single entry point called by the
generator and API layers. Internally it runs:

  1. Query expansion  (query_rewriter.expand_query)
     ├── Original query
     ├── Rewritten/keyword-focused query
     └── HyDE passage (hypothetical documentation passage)

  2. Multi-query retrieval  (retriever.multi_retrieve)
     Searches Qdrant with all query variants, merges & deduplicates results.

  3. Reranking  (reranker.rerank)
     Gemini scores each candidate chunk 0-10, returns top_n.

The result is a ranked list of the most relevant chunks, ready to be passed
to the generator as context.
"""

from retrieval.query_rewriter import expand_query
from retrieval.retriever import multi_retrieve
from retrieval.reranker import rerank


def search(
    query: str,
    top_k: int = 20,   # candidates to retrieve per query variant
    top_n: int = 5,    # final chunks returned after reranking
    use_hyde: bool = True,
    use_rerank: bool = True,
) -> list[dict]:
    """
    Full retrieval pipeline for a user query.

    Args:
        query:      The raw user question.
        top_k:      How many candidates to pull from Qdrant per query variant.
        top_n:      How many chunks to return after reranking.
        use_hyde:   Whether to include HyDE passage in query expansion.
                    Can be disabled for speed if not needed.
        use_rerank: Whether to rerank results with Gemini.
                    Can be disabled to save API calls during testing.

    Returns:
        List of top_n chunk dicts, each with:
          - text        : chunk content
          - source      : original file path
          - breadcrumb  : header hierarchy (e.g. "Overview > Installation")
          - score       : original cosine similarity score from Qdrant
          - rerank_score: Gemini relevance score 0-10 (if reranking enabled)
    """
    # Step 1 — Query expansion
    print(f"\n[Pipeline] Query: {query!r}")

    if use_hyde:
        query_variants = expand_query(query)
    else:
        query_variants = [query]

    print(f"[Pipeline] Searching with {len(query_variants)} query variant(s)...")

    # Step 2 — Multi-query vector retrieval + deduplication
    candidates = multi_retrieve(query_variants, top_k=top_k)
    print(f"[Pipeline] Retrieved {len(candidates)} unique candidate chunks.")

    if not candidates:
        return []

    # Step 3 — Reranking
    if use_rerank:
        print(f"[Pipeline] Reranking to top {top_n}...")
        results = rerank(query, candidates, top_n=top_n)
    else:
        results = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)[:top_n]

    print(f"[Pipeline] Done. Returning {len(results)} chunks.\n")
    return results
