import traceback
import sys

print("Starting...")

try:
    from ingestion.loader import load_docs
    print("loader imported")
    from ingestion.chunker import chunk_docs
    print("chunker imported")
    from ingestion.embedder import embed_and_store
    print("embedder imported")

    docs = load_docs('data/fastapi-docs/docs/en/docs')
    print(f"Docs loaded: {len(docs)}")

    chunks = chunk_docs(docs)
    print(f"Chunks created: {len(chunks)}")

    embed_and_store(chunks)
    print("Done!")

except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)