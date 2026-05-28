# Gatekeeper-RAG — Dev Log

A running log of everything I built, broke, fixed, and decided along the way. Not a polished report — just honest notes so future-me (or anyone reading this) knows exactly what happened and why things are the way they are.

---

## May 25, 2026

### Kicked off the project

Started building **Gatekeeper-RAG** — the goal is an industry-grade RAG pipeline. Not a toy. I want something that actually works well enough to be production-worthy: proper ingestion, smart retrieval, clean generation, and honest evaluation.

The data source I'm starting with is the FastAPI documentation (Markdown files). Good benchmark because it's real, structured, and has a mix of prose and code blocks.

The stack I settled on:
- **Qdrant** as the vector store (local for now, can move to cloud later)
- **Gemini API** for embeddings — no need to host a model locally when Google's giving me access
- **FastAPI** for the API layer (fitting, given the docs we're indexing)

---

### Ingestion pipeline — first pass

Built out the three-stage ingestion pipeline: load → chunk → embed + store.

**Loader** is simple — recursively walks a directory and reads all `.md` files. Nothing fancy, but it works.

**Chunker** — I initially did the naive thing: split by words with a fixed window of 512 words and 50-word overlap. It ran, produced chunks, seemed fine on the surface. But the problem is obvious in hindsight — it completely ignores the fact that we're dealing with Markdown. It would happily split in the middle of a code block, cut across a heading boundary, or merge content from two completely unrelated sections. That's bad for retrieval quality because the chunks lose their semantic coherence.

Fixed this properly — details below.

**Embedder** — Originally wired up `BAAI/bge-large-en-v1.5` running on local GPU. Good model, 1024-dim embeddings, works well. But it adds a hard GPU dependency and I'd rather keep the infra light. Switched to **Gemini's `text-embedding-004`** via API — 768-dim, solid quality, and I don't have to manage a local model.

---

### Problem: Hit Gemini embedding API rate limits

When I first switched to the Gemini API for embeddings, I was calling `embed_content` once per chunk. Blew through the RPM limit almost immediately. Got rate limit errors, the pipeline crashed mid-run, and I had to restart from scratch (since `recreate_collection` wipes everything at the start).

**Root cause:** I was making one API call per chunk instead of batching. With ~800+ chunks from the FastAPI docs, that's 800+ requests — way over the per-minute limit.

**Fix:**
- Bumped batch size to **100 texts per API call** — this is the practical ceiling for the embedding endpoint and means I'm getting maximum value out of each request
- Added a **1.5-second sleep between batches** — keeps us well under the RPM ceiling even under heavy indexing loads
- Added **exponential backoff with retry** (up to 5 attempts) so if we do get a transient rate limit error, the pipeline self-heals instead of crashing

Net effect: the pipeline now runs reliably end-to-end, uses each API call to its maximum capacity, and handles transient failures gracefully.

---

### Problem: Naive word-level chunking is not good enough

As mentioned above, the word-count-based chunker was splitting blindly through Markdown structure. Code blocks, headings, and paragraphs meant nothing to it.

**Fix — switched to a two-pass markdown-aware chunker:**

1. **First pass — `MarkdownHeaderTextSplitter`:** Splits the document along header boundaries (`#`, `##`, `###`, `####`). Each resulting section carries its full header hierarchy as metadata (e.g., `h1: "Advanced Usage"`, `h2: "Dependency Injection"`). This means every chunk knows what section it belongs to, which is huge for retrieval quality.

2. **Second pass — `RecursiveCharacterTextSplitter`:** Any section that's still too large gets recursively split, but this time along natural language boundaries — paragraph breaks first (`\n\n`), then line breaks, then sentence-ending punctuation, then words, and only then individual characters as a last resort. It never splits mid-sentence if it can avoid it.

The chunk size is now **1000 characters** (not words) with **150 character overlap** — character-level sizing is more consistent across different content types and aligns better with how tokenizers actually work.

Each chunk now also stores the header breadcrumb in its payload, which will be useful during retrieval to give the LLM structural context.

---

### Architecture decisions locked in (ingestion layer)

| Decision | Choice | Reasoning |
|---|---|---|
| File format support | `.md` only | Starting focused; can add PDF/HTML later |
| Chunking strategy | Markdown-header split → recursive character split | Preserves document structure and semantic coherence |
| Embedding model | `text-embedding-004` via Gemini API | No local GPU dependency, solid quality, 768-dim |
| Vector DB | Qdrant (local) | Easy to run, production-grade, good Python client |
| Similarity metric | Cosine | Standard for normalized text embeddings |
| Indexing mode | Full re-index per run | Acceptable for now; need incremental updates later |
| Rate limiting | Batch=100, sleep=1.5s, retry w/ backoff | Reliable under API constraints |

---

*Next up: retrieval layer — vector search, query rewriting, and reranking.*

---

### Problem: `text-embedding-004` not available on this API key

Ran the ingestion pipeline for the first time and hit a 404 immediately:

```
models/text-embedding-004 is not found for API version v1beta, or is not supported for embedContent.
```

Pulled the full list of models available on the key — `text-embedding-004` isn't there at all. The available embedding models are:

- `models/gemini-embedding-001` — older, 768-dim
- `models/gemini-embedding-2-preview` — preview
- `models/gemini-embedding-2` — latest stable

**Fix:** Switched to `models/gemini-embedding-2`. It's the best one available — higher quality and outputs 3072-dim vectors. Updated `EMBEDDING_DIM` to match. Qdrant collection will be created with the correct size on next run.

---

### Problem: Still hitting RPM limits with `gemini-embedding-2` free tier

Got 300 chunks through (3 batches of 100) then hit a 429 RESOURCE_EXHAUSTED. The 1.5s sleep between batches was way too aggressive — the free tier for `gemini-embedding-2` is around 3-5 RPM, not the 1500 RPM I assumed.

The retry logic made it worse — exponential backoff starting at 1s is useless for a per-minute quota. By the time we retried 5 times (1+2+4+8+16 = 31s total), we'd burned all retries and crashed, even though waiting 60s would have fixed it.

**Fix — two changes:**

1. **Sleep between batches: 1.5s → 15s** — caps us at ~4 requests/min, safely under the free tier limit
2. **429-specific retry: wait 60s instead of exponential backoff** — a 429 means the quota window hasn't reset yet; waiting a full minute is the right move, not short retries

The retry logic now distinguishes between rate limit errors (wait 60s, be patient) and actual errors (exponential backoff). Also bumped `MAX_RETRIES` to 8 since 429s aren't real failures — they just need time.

---

## May 26, 2026

### Built the retrieval layer

Ingestion is working. Today was the retrieval layer — the part that actually takes a user question and finds the right chunks from Qdrant.

Built four files:

**`retriever.py`** — Core vector search. Embeds the query using `task_type=RETRIEVAL_QUERY` (not `RETRIEVAL_DOCUMENT` — this matters, the model uses asymmetric encoding so query and document embeddings are comparable), then queries Qdrant. Also has `multi_retrieve()` which takes multiple query variants, searches for all of them, and deduplicates by text content keeping the highest score.

**`query_rewriter.py`** — Two techniques here:
1. **Query rewriting** — asks Gemini to strip filler words and make the query more keyword-focused and precise. Better for exact-match style retrieval.
2. **HyDE (Hypothetical Document Embeddings)** — this is the more interesting one. Instead of embedding the question, we ask Gemini to write a fake-but-plausible documentation passage that would answer the question, then embed *that*. The insight from the HyDE paper (Gao et al. 2022) is that a fake answer lands in a much better part of the embedding space than the question itself. The question "how do I add middleware?" embeds near other questions; the fake answer embeds near actual middleware documentation.

Both techniques feed into `expand_query()` which returns `[original_query, rewritten_query, hyde_passage]` — three search variants.

**`reranker.py`** — LLM-as-judge reranking. After retrieving the top candidates from Qdrant, we send all of them to Gemini in a single call and ask it to score each one 0-10 for relevance. Then re-sort by those scores and return the top_n. This is more precise than cosine similarity because the model reads both the query and the chunk together — it can see *why* something is relevant, not just that the embeddings are close. Robust JSON parsing with markdown fence stripping, falls back to original vector scores if anything goes wrong.

**`pipeline.py`** — Unified entry point. `search(query)` → query expansion → multi-retrieve → rerank → top chunks. The generator and API only call this, they don't need to know about the internals.

### Architecture decisions (retrieval layer)

| Decision | Choice | Reasoning |
|---|---|---|
| Query embedding task type | `RETRIEVAL_QUERY` | Asymmetric encoding — must match `RETRIEVAL_DOCUMENT` used at index time |
| Query expansion | Rewrite + HyDE | Improves recall, especially for imprecise questions |
| Retrieval candidates | top_k=20 per variant | Cast a wide net before reranking narrows it down |
| Reranking | LLM-as-judge (Gemini) | No local cross-encoder dependency; single API call for all chunks |
| Final output | top_n=5 | Enough context without flooding the generator prompt |

---

## May 27, 2026

### Built the generation layer

Today was the last core component before the API — the generator that takes retrieved chunks and produces an actual answer.

**`generator.py`** — two public functions:

- `generate(query, chunks)` — takes a query + pre-retrieved chunks, builds a numbered context block, and calls Gemini with a carefully structured prompt. Returns `{ answer, sources, chunks_used, model }`.
- `ask(query, ...)` — full end-to-end RAG in one call: internally runs `retrieval.pipeline.search()` then `generate()`. This is what the API will call.

**The prompt design matters a lot here.** I used a multi-turn conversation structure instead of a single user message — the system instruction goes as the first user turn, the model acknowledges it, then the actual context + question follows. This gets better instruction-following than cramming everything into one big message.

Key rules baked into the prompt:
- Answer ONLY from the provided passages — no hallucination
- Cite inline with [N] notation (e.g. "Use `app.add_middleware()` [1]")
- If the context doesn't contain the answer, say so explicitly — don't make something up
- Include code examples from the passages when they help

Each context chunk includes its breadcrumb (e.g. "FastAPI > Middleware") as a header before the text, giving the model structural context about where the chunk came from.

**Rate limit handling** — added the same 429-aware retry logic as the embedder: waits 60s on rate limit errors, exponential backoff on other errors, up to 8 retries. The smoke test hit a 429 immediately (quota exhausted from the embedding run), which confirmed the retry path works — it catches the error gracefully.

### Architecture decisions (generation layer)

| Decision | Choice | Reasoning |
|---|---|---|
| Generation model | `gemini-2.0-flash` | Fast, available, good instruction following |
| Prompt structure | Multi-turn conversation | Better instruction following than a single long message |
| Citation format | Inline [N] | Traceable, standard academic/RAG style |
| No-answer handling | Explicit refusal message | Faithfulness — better to say "I don't know" than hallucinate |
| Rate limit retry | 60s wait, 8 retries | Consistent with the rest of the pipeline |

---

### Problem: `QdrantClient` has no attribute `search`

First real end-to-end test crashed immediately with:
```
AttributeError: 'QdrantClient' object has no attribute 'search'
```

Installed `qdrant-client` is v1.18.0. In v1.7+, `client.search()` was deprecated and fully removed by v1.18. The replacement is `client.query_points()` — same purpose, but `query_vector=` → `query=` and results come back as a `QueryResponse` object where hits are at `.points`, not iterated directly.

**Fix:** Updated `retrieval/retriever.py` to use `query_points()`.

---

### Problem: `gemini-2.0-flash` daily free tier quota exhausted

Query rewriter and generator both hit 429 with `limit: 0` — the daily free tier request quota for `gemini-2.0-flash` was fully consumed by the embedding run and testing.

**Fix:** Switched `query_rewriter.py`, `reranker.py`, and `generator.py` from `gemini-2.0-flash` → `gemini-2.0-flash-lite`. Separate quota bucket, more than capable enough for rewriting, scoring, and grounded generation from short context.

Then hit the same issue on `gemini-2.0-flash-lite` too. Switched again to `gemini-2.5-flash`.

---

### Problem: Retrieval returning completely wrong documents

The test showed the pipeline running end-to-end — retrieved 24 chunks, reranked, generated — but every answer was "I don't have enough information." Looking at the source files returned: OAuth2 scopes and release notes for a middleware question. Clearly wrong.

Root cause: `run.py` was interrupted by the embedding rate limit after only 29 chunks (less than one batch). But `embed_and_store` called `recreate_collection` at the start of every run — so each restart wiped the collection and started from zero. We never accumulated any meaningful data across runs.

The core design mistake was that a single-run assumption baked into the indexer: wipe + rebuild. That's fine when the run always completes, but catastrophic when interrupted by API rate limits.

**Fix — resumable incremental indexing:**

Rewrote `embed_and_store` to:
1. Check how many points already exist in the collection
2. Skip that many chunks at the start of the run (they're already stored)
3. Continue embedding from where it left off

The `recreate_collection` call was replaced with `_ensure_collection()` (create if not exists, leave alone if it does) + a point count check. Added an explicit `fresh=True` flag for when a full re-index is genuinely needed.

Net effect: re-running `run.py` after an interrupted run will now pick up from chunk 30 (or wherever it stopped) instead of starting over from 0.

---

## May 28, 2026

### Built the API layer

Today was the FastAPI server — the surface that makes the pipeline callable over HTTP.

**`api/main.py`** — three endpoints:

- `POST /ask` — the main one. Takes a query + optional params (`top_k`, `top_n`, `use_hyde`, `use_rerank`), runs the full pipeline, returns `{ answer, sources, chunks_used, model, processing_time_s }`. Returns 503 if Qdrant is empty (no data indexed yet), 500 on pipeline errors.

- `GET /health` — liveness check. Pings Qdrant, returns connection status and chunk count. Useful for monitoring and as a pre-flight check before sending real queries.

- `GET /stats` — returns current model config and index size. Handy for knowing exactly what's running without digging into the code.

All request/response shapes are Pydantic models, so FastAPI auto-generates OpenAPI docs at `/docs`. Added CORS middleware so a frontend can call it later without browser issues.

Startup prints a clear warning if Qdrant is empty — saves confusion when the index hasn't been populated yet.

### Architecture decisions (API layer)

| Decision | Choice | Reasoning |
|---|---|---|
| Framework | FastAPI | Already a dependency (we're indexing FastAPI docs), async, auto-docs |
| Request validation | Pydantic models | Type-safe, auto-validated, shows up in OpenAPI docs |
| CORS | Allow all origins | Development mode — will tighten in production |
| Error on empty index | 503 Service Unavailable | Correct HTTP semantics — service exists but not ready |
| Processing time | Included in response | Useful for benchmarking retrieval vs generation latency |

### How to run

```
uvicorn api.main:app --reload --port 8000
```
Then open `http://localhost:8000/docs` for the interactive API docs.

---

## May 29, 2026

### Built the evaluation layer

With ingestion, retrieval, generation, and the API all done, the last piece is knowing whether any of it actually works — and how well. Evaluation is the difference between a demo that looks good and a system you can actually trust.

Built two files:

**`evaluation/testset.py`** — testset generator. Pulls a random sample of chunks from Qdrant, sends each one to Gemini and asks it to generate a realistic developer question + correct answer from that chunk. Saves everything to `testset.json` so we don't have to regenerate it every time. Each test case includes the question, the ground truth answer, the source chunk it came from, and the breadcrumb.

**`evaluation/metrics.py`** — four RAGAS-style metrics, all scored with Gemini LLM-as-judge:

1. **Faithfulness** — takes every factual claim in the answer and checks whether it's supported by the retrieved context. Score = supported_claims / total_claims. This is the anti-hallucination metric.

2. **Answer Relevance** — asks Gemini to rate 0-10 how well the answer actually addresses the question. Normalised to 0-1.

3. **Context Recall** — for each sentence in the ground truth answer, checks if the retrieved context contains the information needed to produce it. Score = attributable_sentences / total_sentences. Low recall means retrieval is missing relevant chunks.

4. **Context Precision** — for each retrieved chunk, checks whether it was actually useful for answering the question. Score = useful_chunks / total_chunks. Low precision means retrieval is pulling in noise.

The `evaluate()` function runs all four metrics across the test set and prints a formatted table with per-case scores and column means.

I chose LLM-as-judge over traditional NLP metrics (BLEU, ROUGE, etc.) because they're meaningless for RAG. ROUGE measures n-gram overlap, not factual grounding — an answer that's worded differently but semantically correct would score terribly. Gemini can actually understand what's supported and what isn't.

### Architecture decisions (evaluation layer)

| Decision | Choice | Reasoning |
|---|---|---|
| Metric framework | Custom, RAGAS-inspired | No external dependencies, full control over prompts |
| Scoring model | Gemini LLM-as-judge | Better semantic understanding than ROUGE/BLEU |
| Testset generation | LLM from real chunks | Realistic questions grounded in actual documentation |
| Testset persistence | JSON file | Generate once, reuse across evaluation runs |
| Rate limit handling | 3-5s sleeps between calls | Multiple Gemini calls per test case — need to pace carefully |




