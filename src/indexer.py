import os
import re
import math
import json
from collections import Counter
from rank_bm25 import BM25Okapi

PAGES_JSON = os.path.join("data", "processed_pages.json")
CHUNKS_JSON = os.path.join("data", "chunks.json")


def chunk_text(text: str, max_chars: int = 900, overlap: int = 150) -> list[str]:
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
            tail = current[-overlap:] if len(current) > overlap else current
            current = f"{tail} {sent}".strip()
            while len(current) > max_chars:
                chunks.append(current[:max_chars])
                current = current[max_chars - overlap:]

    if current:
        chunks.append(current)
    return chunks


def build_chunks():
    if not os.path.exists(PAGES_JSON):
        from ingest import download_pdfs, extract_pages
        download_pdfs()
        extract_pages()

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

    os.makedirs("data", exist_ok=True)
    with open(CHUNKS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"[CHUNKED] Created {len(all_chunks)} contextual chunks from {len(pages)} pages.")
    return all_chunks


def tokenize_with_stems(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    # Include 5-char prefixes to capture morphological variants (e.g. consent/consenting, penalty/penalties)
    stems = [w[:5] for w in words if len(w) > 5]
    return words + stems


class HybridRetriever:
    def __init__(self):
        if not os.path.exists(CHUNKS_JSON):
            build_chunks()
        with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.chunk_map = {c["chunk_id"]: c for c in self.chunks}

        # 1. Build BM25 Keyword Index
        tokenized_corpus = [re.findall(r"\w+", c["text"].lower()) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 2. Build Lightweight TF-IDF Vector Index (~5 MB RAM instead of 500+ MB)
        self.doc_tokens = [tokenize_with_stems(c["text"]) for c in self.chunks]
        df = Counter()
        for toks in self.doc_tokens:
            for t in set(toks):
                df[t] += 1
        n_docs = len(self.chunks)
        self.idf = {t: math.log((n_docs + 1) / (freq + 1)) + 1.0 for t, freq in df.items()}

        self.doc_vecs = []
        for toks in self.doc_tokens:
            tf = Counter(toks)
            vec = {t: (count / len(toks)) * self.idf.get(t, 1.0) for t, count in tf.items()}
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            self.doc_vecs.append({t: v / norm for t, v in vec.items()})

    def _vector_search(self, query: str, top_n: int = 20) -> list[str]:
        q_toks = tokenize_with_stems(query)
        if not q_toks:
            return [c["chunk_id"] for c in self.chunks[:top_n]]
        tf = Counter(q_toks)
        q_vec = {t: (count / len(q_toks)) * self.idf.get(t, 1.0) for t, count in tf.items()}
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
        q_vec = {t: v / q_norm for t, v in q_vec.items()}

        scores = []
        for idx, d_vec in enumerate(self.doc_vecs):
            dot = sum(q_val * d_vec.get(t, 0.0) for t, q_val in q_vec.items())
            scores.append((dot, self.chunks[idx]["chunk_id"]))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [cid for _, cid in scores[:top_n]]

    def retrieve(self, query: str, top_k: int = 6, max_per_doc: int = 4) -> list[dict]:
        # 1. Vector Cosine Search (Top 20 candidates)
        vec_ids = self._vector_search(query, top_n=20)

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

        # 4. Document-Diverse Selection
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
    build_chunks()
    retriever = HybridRetriever()
    res = retriever.retrieve("What does the DPDP Act say about consent?", top_k=3)
    for r in res:
        print(f"Matched: {r['doc_id']} Page {r['page_number']} (RRF: {r['rrf_score']})")