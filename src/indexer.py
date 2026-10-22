import os
import re
import json
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

PAGES_JSON = os.path.join("data", "processed_pages.json")
CHUNKS_JSON = os.path.join("data", "chunks.json")
CHROMA_DIR = os.path.join("data", "chroma_db")
COLLECTION_NAME = "ai_governance_docs"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def chunk_text(text: str, max_chars: int = 900, overlap: int = 150) -> list[str]:
    # Split on paragraph or sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+|\n\n+", text)
    chunks = []
    current = ""

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(current) + len(sent) + 1 <= max_chars:
            current = f"{current} {sent}".strip()
        else:
            if current:
                chunks.append(current)
            # Start new chunk with tail overlap from previous chunk
            tail = current[-overlap:] if len(current) > overlap else current
            current = f"{tail} {sent}".strip()
            # If a single sentence is huge, hard-slice it
            while len(current) > max_chars:
                chunks.append(current[:max_chars])
                current = current[max_chars - overlap:]

    if current:
        chunks.append(current)
    return chunks


def build_chunks():
    with open(PAGES_JSON, "r", encoding="utf-8") as f:
        pages = json.load(f)

    all_chunks = []
    for page in pages:
        raw_chunks = chunk_text(page["text"])
        for idx, c_text in enumerate(raw_chunks):
            chunk_id = f"{page['doc_id']}_p{page['page_number']}_c{idx}"
            enriched_text = (
                f"[Source: {page['doc_title']} ({page['doc_id']}) | Page {page['page_number']}]\n"
                f"{c_text}"
            )
            all_chunks.append({
                "chunk_id": chunk_id,
                "doc_id": page["doc_id"],
                "doc_title": page["doc_title"],
                "page_number": page["page_number"],
                "source_url": page["source_url"],
                "text": enriched_text
            })

    with open(CHUNKS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"[CHUNKED] Created {len(all_chunks)} contextual chunks from {len(pages)} pages.")
    return all_chunks


def build_vector_index(chunks: list[dict]):
    print(f"[EMBEDDING] Loading local model '{EMBED_MODEL}' (first run downloads ~130MB)...")
    ef = embedding_functions.DefaultEmbeddingFunction()
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Reset collection if re-running so we don't duplicate chunks
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        collection.add(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[{
                "doc_id": c["doc_id"],
                "doc_title": c["doc_title"],
                "page_number": c["page_number"],
                "source_url": c["source_url"]
            } for c in batch]
        )
        print(f"  -> Indexed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks into ChromaDB")

    print(f"[SUCCESS] Vector index saved to {CHROMA_DIR}")


class HybridRetriever:
    def __init__(self):
        with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.chunk_map = {c["chunk_id"]: c for c in self.chunks}

        tokenized_corpus = [re.findall(r"\w+", c["text"].lower()) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        ef = embedding_functions.DefaultEmbeddingFunction()
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)

    def retrieve(self, query: str, top_k: int = 6, max_per_doc: int = 4) -> list[dict]:
        # 1. Dense Vector Search (Top 20 candidates)
        vec_res = self.collection.query(query_texts=[query], n_results=20)
        vec_ids = vec_res["ids"][0]

        # 2. Sparse BM25 Keyword Search (Top 20 candidates)
        tokens = re.findall(r"\w+", query.lower())
        bm25_scores = self.bm25.get_scores(tokens)
        top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:20]
        bm25_ids = [self.chunks[i]["chunk_id"] for i in top_bm25_indices]

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        k_constant = 60
        for rank, cid in enumerate(vec_ids):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k_constant + rank + 1)
        for rank, cid in enumerate(bm25_ids):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k_constant + rank + 1)

        ranked_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        # 4. Document-Diverse Selection (prevents one document from crowding out others)
        results = []
        doc_counts = {}
        for cid in ranked_ids:
            chunk = self.chunk_map[cid].copy()
            doc_id = chunk["doc_id"]
            if doc_counts.get(doc_id, 0) >= max_per_doc:
                continue
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
            chunk["rrf_score"] = round(rrf_scores[cid], 5)
            results.append(chunk)
            if len(results) >= top_k:
                break

        return results


if __name__ == "__main__":
    chunks_data = build_chunks()
    build_vector_index(chunks_data)

    print("\n--- Testing Hybrid Retriever ---")
    retriever = HybridRetriever()
    test_query = "What is the maximum penalty for a personal data breach under India's DPDP Act?"
    top_chunks = retriever.retrieve(test_query, top_k=3)
    for idx, res in enumerate(top_chunks, 1):
        print(f"\nResult #{idx} (RRF Score: {res['rrf_score']}) | {res['doc_id']} Page {res['page_number']}:")
        print(res["text"][:300] + "...")
