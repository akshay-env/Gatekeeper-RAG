"""
api/main.py

FastAPI server exposing the Gatekeeper-RAG pipeline as HTTP endpoints.

Endpoints:
  POST /ask       — main RAG query: retrieval + generation in one call
  GET  /health    — liveness check: confirms API + Qdrant are reachable
  GET  /stats     — index statistics: chunk count, models in use

Run with:
  uvicorn api.main:app --reload --port 8000
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from generation.generator import ask, GENERATION_MODEL
from ingestion.embedder import EMBEDDING_MODEL, EMBEDDING_DIM

# ---------------------------------------------------------------------------
# Qdrant client (shared across requests)
# ---------------------------------------------------------------------------

COLLECTION_NAME = "docs"
qdrant = QdrantClient("localhost", port=6333)


def _get_chunk_count() -> int:
    try:
        info = qdrant.get_collection(COLLECTION_NAME)
        return info.points_count or 0
    except UnexpectedResponse:
        return 0


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown logic
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    count = _get_chunk_count()
    if count == 0:
        print("\n⚠️  WARNING: Qdrant collection is empty or does not exist.")
        print("   Run `python run.py` to index documents before querying.\n")
    else:
        print(f"\n✅ Gatekeeper-RAG API ready — {count} chunks indexed.\n")
    yield
    # Shutdown (nothing to clean up)
    print("Shutting down Gatekeeper-RAG API.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Gatekeeper-RAG",
    description=(
        "Industry-grade RAG pipeline over FastAPI documentation. "
        "Uses Gemini embeddings, Qdrant vector search, HyDE query expansion, "
        "LLM reranking, and grounded generation with source citations."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000,
                       description="The question to ask the RAG pipeline.")
    top_k: int = Field(default=20, ge=1, le=100,
                       description="Number of candidate chunks to retrieve per query variant.")
    top_n: int = Field(default=5, ge=1, le=20,
                       description="Final number of chunks returned after reranking.")
    use_hyde: bool = Field(default=True,
                           description="Enable HyDE query expansion (better recall, uses extra API calls).")
    use_rerank: bool = Field(default=True,
                             description="Enable LLM reranking (better precision, uses extra API calls).")


class ChunkInfo(BaseModel):
    text: str
    source: str
    breadcrumb: str
    score: float
    rerank_score: int | None = None


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: list[str]
    model: str
    processing_time_s: float
    chunks_used: list[ChunkInfo]


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    chunks_indexed: int


class StatsResponse(BaseModel):
    chunks_indexed: int
    collection: str
    embedding_model: str
    embedding_dim: int
    generation_model: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/ask", response_model=AskResponse, summary="Ask a question")
async def ask_endpoint(request: AskRequest):
    """
    Run the full RAG pipeline on a user question.

    Internally performs:
    1. Query expansion (rewrite + optional HyDE passage)
    2. Multi-query vector retrieval from Qdrant
    3. Optional LLM reranking
    4. Grounded answer generation with inline [N] citations

    Returns the answer, source files, and the chunks used as context.
    """
    if _get_chunk_count() == 0:
        raise HTTPException(
            status_code=503,
            detail="No documents indexed. Run `python run.py` first to populate the vector store."
        )

    t_start = time.perf_counter()

    try:
        result = ask(
            query=request.query,
            top_k=request.top_k,
            top_n=request.top_n,
            use_hyde=request.use_hyde,
            use_rerank=request.use_rerank,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    elapsed = round(time.perf_counter() - t_start, 2)

    return AskResponse(
        query=request.query,
        answer=result["answer"],
        sources=result["sources"],
        model=result["model"],
        processing_time_s=elapsed,
        chunks_used=[
            ChunkInfo(
                text=c.get("text", ""),
                source=c.get("source", ""),
                breadcrumb=c.get("breadcrumb", ""),
                score=c.get("score", 0.0),
                rerank_score=c.get("rerank_score"),
            )
            for c in result["chunks_used"]
        ],
    )


@app.get("/health", response_model=HealthResponse, summary="Health check")
async def health():
    """
    Confirm the API is up and Qdrant is reachable.
    Returns the number of indexed chunks.
    """
    try:
        count = _get_chunk_count()
        qdrant_status = "connected"
    except Exception:
        count = 0
        qdrant_status = "disconnected"

    return HealthResponse(
        status="ok",
        qdrant=qdrant_status,
        chunks_indexed=count,
    )


@app.get("/stats", response_model=StatsResponse, summary="Index statistics")
async def stats():
    """
    Return current index statistics and model configuration.
    """
    return StatsResponse(
        chunks_indexed=_get_chunk_count(),
        collection=COLLECTION_NAME,
        embedding_model=EMBEDDING_MODEL,
        embedding_dim=EMBEDDING_DIM,
        generation_model=GENERATION_MODEL,
    )
