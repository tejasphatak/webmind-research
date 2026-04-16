"""
Semantic memory recall for Claude.

Indexes markdown memory files into a local Qdrant collection using
sentence-transformers embeddings, enabling semantic recall by query
instead of filename lookup.

Usage:
    python memory_vector.py sync                 # ingest all *.md in memory/
    python memory_vector.py recall "paper 1 scope"   # top-5 hits
    python memory_vector.py remember "..."            # add a snippet

Design notes:
- Storage: Qdrant in local-file mode (no server process required).
  Collection persists at ~/.claude/projects/.../memory/.qdrant/
- Embeddings: all-MiniLM-L6-v2 (384-dim, fast CPU inference).
  Downloaded once to ~/.cache/huggingface, reused across sessions.
- Chunking: each markdown file is split into paragraphs. Each paragraph
  is an independent record so recall can surface the specific relevant
  paragraph, not the whole file.

Requires: sentence-transformers, qdrant-client (installed in the synapse
dev venv at ~/webmind_research/.venv).

Author: Claude Opus 4.6, 2026-04-16
"""
from __future__ import annotations
import sys, os, uuid, re, hashlib
from pathlib import Path

MEMORY_DIR = Path(__file__).parent
QDRANT_DIR = MEMORY_DIR / ".qdrant"
COLLECTION = "claude_memory"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_SIZE = 384


def get_client():
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams
    QDRANT_DIR.mkdir(exist_ok=True)
    client = QdrantClient(path=str(QDRANT_DIR))
    # Ensure collection exists
    try:
        client.get_collection(COLLECTION)
    except Exception:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
    return client


def get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


def chunk_markdown(path: Path, max_words: int = 150) -> list[dict]:
    """Split a markdown file into paragraph-sized chunks.

    Skip YAML frontmatter. Each chunk gets a stable ID derived from
    (file_path, chunk_index) so re-running sync updates rather than
    duplicates."""
    text = path.read_text()
    # Strip frontmatter
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            text = parts[2]
    # Split into paragraphs, keeping heading context
    chunks = []
    current_heading = ""
    paragraph_lines: list[str] = []
    for line in text.splitlines():
        if re.match(r"^#{1,6}\s", line):
            # flush prior paragraph
            if paragraph_lines:
                body = "\n".join(paragraph_lines).strip()
                if body:
                    chunks.append({"heading": current_heading, "body": body})
                paragraph_lines = []
            current_heading = line.lstrip("# ").strip()
        elif line.strip() == "":
            if paragraph_lines:
                body = "\n".join(paragraph_lines).strip()
                if body:
                    chunks.append({"heading": current_heading, "body": body})
                paragraph_lines = []
        else:
            paragraph_lines.append(line)
    if paragraph_lines:
        body = "\n".join(paragraph_lines).strip()
        if body:
            chunks.append({"heading": current_heading, "body": body})

    # Build records with stable IDs
    records = []
    for i, c in enumerate(chunks):
        text_for_embed = (c["heading"] + "\n\n" + c["body"]) if c["heading"] else c["body"]
        # Split long chunks
        words = text_for_embed.split()
        if len(words) > max_words:
            for start in range(0, len(words), max_words):
                sub = " ".join(words[start:start + max_words])
                cid = f"{path.name}::{i}::{start}"
                records.append({
                    "id": _stable_uuid(cid),
                    "file": path.name,
                    "heading": c["heading"],
                    "text": sub,
                })
        else:
            cid = f"{path.name}::{i}"
            records.append({
                "id": _stable_uuid(cid),
                "file": path.name,
                "heading": c["heading"],
                "text": text_for_embed,
            })
    return records


def _stable_uuid(key: str) -> str:
    """Deterministic UUID5 from a key string."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def sync():
    """Ingest every *.md in MEMORY_DIR into Qdrant."""
    from qdrant_client.models import PointStruct

    client = get_client()
    model = get_model()

    all_records = []
    for path in sorted(MEMORY_DIR.glob("*.md")):
        if path.name == "MEMORY.md":
            continue  # skip the index
        all_records.extend(chunk_markdown(path))

    if not all_records:
        print("No records to index.")
        return

    texts = [r["text"] for r in all_records]
    print(f"Embedding {len(texts)} chunks with {EMBED_MODEL}...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    points = []
    for r, e in zip(all_records, embeddings):
        points.append(PointStruct(
            id=r["id"],
            vector=e.tolist(),
            payload={"file": r["file"], "heading": r["heading"], "text": r["text"]},
        ))
    client.upsert(collection_name=COLLECTION, points=points)
    print(f"Indexed {len(points)} chunks into Qdrant collection '{COLLECTION}'.")


def recall(query: str, top_k: int = 5):
    client = get_client()
    model = get_model()
    qv = model.encode(query, convert_to_numpy=True).tolist()
    hits = client.query_points(
        collection_name=COLLECTION,
        query=qv,
        limit=top_k,
    ).points
    for h in hits:
        print(f"\n[score={h.score:.3f}] {h.payload.get('file', '?')}  §{h.payload.get('heading', '')}")
        print(h.payload.get("text", "")[:400])


def remember(text: str, topic: str = "ad-hoc"):
    """Add an ad-hoc memory snippet without a markdown file."""
    from qdrant_client.models import PointStruct
    client = get_client()
    model = get_model()
    e = model.encode(text, convert_to_numpy=True).tolist()
    key = f"adhoc::{topic}::{hashlib.sha1(text.encode()).hexdigest()[:12]}"
    point = PointStruct(
        id=_stable_uuid(key),
        vector=e,
        payload={"file": f"adhoc::{topic}", "heading": topic, "text": text},
    )
    client.upsert(collection_name=COLLECTION, points=[point])
    print(f"Remembered: {key}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "sync":
        sync()
    elif cmd == "recall":
        query = " ".join(sys.argv[2:])
        if not query:
            print("recall requires a query string"); sys.exit(1)
        recall(query)
    elif cmd == "remember":
        text = " ".join(sys.argv[2:])
        if not text:
            print("remember requires text"); sys.exit(1)
        remember(text)
    else:
        print(f"Unknown command: {cmd}"); sys.exit(1)


if __name__ == "__main__":
    main()
