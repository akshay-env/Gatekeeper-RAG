import re

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHUNK_SIZE = 1000       # max characters per chunk
CHUNK_OVERLAP = 150     # character overlap between consecutive chunks

# Markdown header pattern — captures level and title
HEADER_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)

# Ordered split boundaries: try paragraph first, then progressively smaller
SPLIT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_markdown_sections(text: str) -> list[dict]:
    """
    Split a markdown document into sections on header boundaries.
    Each section carries a breadcrumb built from its header hierarchy.
    Returns a list of {"text": ..., "breadcrumb": ...} dicts.
    """
    # Find all header positions
    headers = [(m.start(), len(m.group(1)), m.group(2).strip())
               for m in HEADER_RE.finditer(text)]

    if not headers:
        return [{"text": text, "breadcrumb": ""}]

    sections = []
    # Build sections between headers
    for i, (start, level, title) in enumerate(headers):
        end = headers[i + 1][0] if i + 1 < len(headers) else len(text)
        body = text[start:end].strip()
        if not body:
            continue

        # Build breadcrumb from all ancestor headers up to this one
        ancestors = [h_title for (_, h_lvl, h_title) in headers[:i + 1]
                     if h_lvl <= level]
        breadcrumb = " > ".join(ancestors)

        sections.append({"text": body, "breadcrumb": breadcrumb})

    # Any content before the first header
    preamble = text[:headers[0][0]].strip()
    if preamble:
        sections.insert(0, {"text": preamble, "breadcrumb": ""})

    return sections


def _recursive_split(text: str, separators: list[str], chunk_size: int) -> list[str]:
    """
    Recursively split text using the first separator that produces pieces
    small enough to fit within chunk_size. Falls back to the next separator
    if a piece is still too large.
    """
    if len(text) <= chunk_size:
        return [text]

    sep = separators[0] if separators else ""
    remaining_seps = separators[1:] if separators else []

    if sep == "":
        # Last resort: hard character split
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    parts = text.split(sep)
    result = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) <= chunk_size:
            result.append(part)
        else:
            result.extend(_recursive_split(part, remaining_seps, chunk_size))
    return result


def _merge_with_overlap(pieces: list[str], chunk_size: int, overlap: int) -> list[str]:
    """
    Merge small pieces into chunks up to chunk_size, with overlap between
    consecutive chunks so context isn't lost at boundaries.
    """
    chunks = []
    current = ""

    for piece in pieces:
        candidate = (current + " " + piece).strip() if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # Start new chunk with overlap from the end of the previous one
            overlap_text = current[-overlap:] if overlap and current else ""
            current = (overlap_text + " " + piece).strip() if overlap_text else piece

    if current:
        chunks.append(current)

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_docs(docs: list[dict]) -> list[dict]:
    chunks = []

    for doc in docs:
        text = doc["text"]
        source = doc["source"]

        # Pass 1: split on markdown headers — preserves document hierarchy
        sections = _split_markdown_sections(text)

        for section in sections:
            section_text = section["text"].strip()
            breadcrumb = section["breadcrumb"]

            if not section_text:
                continue

            if len(section_text) <= CHUNK_SIZE:
                # Section fits in a single chunk — use as-is
                chunks.append({
                    "text": section_text,
                    "source": source,
                    "breadcrumb": breadcrumb
                })
            else:
                # Pass 2: recursively split on natural language boundaries
                pieces = _recursive_split(section_text, SPLIT_SEPARATORS, CHUNK_SIZE)
                merged = _merge_with_overlap(pieces, CHUNK_SIZE, CHUNK_OVERLAP)

                for chunk_text in merged:
                    chunk_text = chunk_text.strip()
                    if chunk_text:
                        chunks.append({
                            "text": chunk_text,
                            "source": source,
                            "breadcrumb": breadcrumb
                        })

    print(f"Created {len(chunks)} chunks from {len(docs)} documents")
    return chunks