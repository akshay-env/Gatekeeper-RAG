from pathlib import Path

def load_docs(docs_path: str) -> list[dict]:
    docs = []
    path = Path(docs_path)
    
    for file in path.rglob("*.md"):
        try:
            text = file.read_text(encoding="utf-8", errors="ignore")
            if text.strip():  # skip empty files
                docs.append({
                    "text": text,
                    "source": str(file)
                })
        except Exception as e:
            print(f"Skipping {file}: {e}")
    
    print(f"Loaded {len(docs)} documents")
    return docs