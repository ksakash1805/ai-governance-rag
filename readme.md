# Enterprise AI Governance & DPDP Compliance Copilot (Hybrid RAG)

An enterprise Retrieval-Augmented Generation (RAG) system that answers legal, privacy, and LLM security compliance questions strictly grounded in authoritative public regulatory frameworks, with page-level inline citations, hybrid search (ChromaDB + BM25 Reciprocal Rank Fusion), Langfuse observability, and an automated LLM-as-a-Judge evaluation suite.

## Authoritative Document Corpus (176 Content-Rich Pages)

| Document ID | Official Regulation / Standard | Scope & Use Case |
| :--- | :--- | :--- |
| `INDIA-DPDP-2023` | [India Digital Personal Data Protection Act, 2023](https://www.meity.gov.in/static/uploads/2024/06/2bf1f0e9f04e6fb4f8fef35e82c42aa5.pdf) | Data Fiduciary obligations, verifiable parental consent (Section 9), breach notifications (Section 8), and financial penalties up to Rs. 250 crore |
| `NIST-AI-100-1` | [NIST AI Risk Management Framework (AI RMF 1.0)](https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.100-1.pdf) | Foundational AI governance across the four core functions: GOVERN, MAP, MEASURE, and MANAGE |
| `NIST-AI-600-1` | [NIST RMF Generative AI Profile (AI 600-1)](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf) | Generative AI risk controls covering confabulation (hallucination), data privacy, and information integrity |
| `OWASP-LLM-2025` | [OWASP Top 10 for Large Language Model Applications (2025)](https://owasp.github.io/www-project-top-10-for-large-language-model-applications/assets/PDF/OWASP-Top-10-for-LLMs-v2025.pdf) | Application security controls for Prompt Injection (`LLM01:2025`), Sensitive Information Disclosure (`LLM02:2025`), and Vector/Embedding Weaknesses (`LLM08:2025`) |

## System Architecture Provides Zero-Cost Hybrid Retrieval and Full Observability

| Layer | Technology | Engineering Rationale |
| :--- | :--- | :--- |
| **Document Parsing** | `PyMuPDF` (`fitz`) | Extracts page-level text while preserving `doc_id`, `page_number`, and direct PDF URLs for `#page=X` deep linking |
| **Contextual Chunking** | Sentence-Aware Splitter (900 chars, 150-char overlap) | Prepends `[Source: {doc_title} ({doc_id}) \| Page {page_number}]` to every chunk so cross-framework queries retain document identity |
| **Dense Embeddings** | `sentence-transformers` (`BAAI/bge-small-en-v1.5`) | Local 384-dimensional CPU embeddings for semantic similarity without external embedding API latency or cost |
| **Hybrid Retrieval** | `ChromaDB` (Cosine) + `rank-bm25` (`BM25Okapi`) | Merges Top-20 semantic candidates and Top-20 keyword candidates via Reciprocal Rank Fusion ($k=60$) with a per-document diversity cap (`max_per_doc=4`, `top_k=6`) |
| **LLM Generation** | [Groq Cloud API](https://console.groq.com/docs/models) (`openai/gpt-oss-120b`) | Enforces strict `[DOC_ID, Page X]` inline citations at `temperature=0.1` and refuses out-of-scope questions |
| **Observability** | [Langfuse Cloud](https://langfuse.com/docs/observability/sdk/instrumentation) (`@observe` SDK) | Captures nested traces for hybrid retrieval, prompt templates, token usage, and latency |
| **Web API & Hosting** | `Flask` + `flask-cors` on [Hugging Face Spaces](https://huggingface.co/spaces) (Docker) | Serves a responsive dark-mode compliance UI and REST endpoint (`/api/chat`) inside a free 16 GB RAM container |

## Iterative Error Analysis Improved Citation Compliance by 50 Percentage Points

We evaluated the pipeline on an 8-question Golden Benchmark (`eval/run_eval.py`) spanning single-document lookups, multi-framework synthesis (`INDIA-DPDP-2023` + `OWASP-LLM-2025`), and out-of-scope adversarial refusal tests.

| Evaluation Metric | Baseline (v1) | Tuned Pipeline (v2) | Delta / Engineering Impact |
| :--- | ---: | ---: | :--- |
| **Retrieval Hit Rate** | 100.0% | 100.0% | Hybrid RRF + Diversity Cap matched ground-truth documents on 8/8 benchmark queries |
| **Citation & Guardrail Compliance** | 25.0% | 75.0% | **+50.0 percentage points** after Unicode hyphen normalization and context budget tuning |
| **LLM-as-a-Judge Faithfulness** | 56.2% | 75.0% | **+18.8 percentage points** after resolving source crowding and reasoning-token truncation |
| **Average End-to-End Latency** | 4.74s | 8.39s | Trade-off from expanding `top_k` (`5 -> 6`) and generation token budget (`900 -> 1500`) |

### Key Root-Cause Fixes Discovered via Langfuse Tracing & Evaluation

* **Eliminating Multi-Document Source Crowding:** On cross-framework queries combining India's DPDP Act and OWASP LLM Top 10, naive Top-5 retrieval filled all 5 slots with `INDIA-DPDP-2023` chunks due to dense keyword repetition. Adding a per-document diversity cap ensured balanced retrieval across both frameworks while reducing prompt tokens.
* **Balancing Diversity Against Table Contiguity:** An overly strict `max_per_doc=2` cap truncated multi-chunk penalty tables on Page 21 of the DPDP Act (`Q1_DPDP_PENALTY`). Tuning to `top_k=6, max_per_doc=4` preserved full multi-chunk tables for single-document queries while reserving 2 context slots for multi-document queries.
* **Resolving Reasoning Model Token Truncation & Unicode Drift:** Inspecting `eval/eval_results.json` revealed that `openai/gpt-oss-120b` consumed small `max_tokens` budgets on internal chain-of-thought reasoning and emitted Unicode non-breaking hyphens (`\u2011`) inside `[NIST-AI-100-1, Page X]` tags. Expanding the token budget and adding deterministic Unicode normalization restored citation verification.

## Repository Structure

```text
ai-governance-rag/
├── app.py                     # Flask REST API (/api/chat, /api/health) & web server
├── Dockerfile                 # Hugging Face Spaces container configuration (Port 7860)
├── requirements.txt           # Pinned Python dependencies
├── data/
│   ├── raw/                   # Official PDFs (DPDP 2023, NIST AI RMF, NIST GenAI, OWASP 2025)
│   ├── processed_pages.json   # Extracted page-level text and metadata (176 pages)
│   ├── chunks.json            # Contextualized overlapping chunks for BM25
│   └── chroma_db/             # Persistent ChromaDB vector store (bge-small-en-v1.5)
├── src/
│   ├── ingest.py              # Automated PDF downloader and PyMuPDF extractor
│   ├── indexer.py             # Contextual chunker & HybridRetriever (Chroma + BM25 + RRF)
│   └── rag.py                 # Groq LLM generation, citation enforcement & Langfuse tracing
├── eval/
│   ├── run_eval.py            # 8-question Golden Benchmark & LLM-as-a-Judge evaluator
│   └── eval_results.json      # Detailed per-question evaluation scorecard
└── templates/
    └── index.html             # Responsive dark-mode compliance chat UI with deep PDF links
```

## Local Quickstart

1. **Clone the repository and install dependencies:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/ai-governance-rag.git
   cd ai-governance-rag
   python -m venv venv
   source venv/bin/activate  # On Windows PowerShell: .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Configure environment variables in `.env`:**
   ```env
   GROQ_API_KEY=gsk_your_groq_key
   LANGFUSE_PUBLIC_KEY=pk-lf-your_public_key
   LANGFUSE_SECRET_KEY=sk-lf-your_secret_key
   LANGFUSE_BASE_URL=https://cloud.langfuse.com
   ```

3. **Ingest documents, build the hybrid index, and launch the Flask app:**
   ```bash
   python src/ingest.py
   python src/indexer.py
   python eval/run_eval.py
   python app.py
   ```
   Open `http://127.0.0.1:7860` in your browser.

## Embedding in Your Portfolio Website

Once deployed to Hugging Face Spaces (`https://YOUR-USERNAME-ai-governance-rag.hf.space`), paste this snippet into any section of your portfolio website (`index.html` or React component):

```html
<section style="max-width: 960px; margin: 2rem auto;">
  <h2>Live Demo: Enterprise AI Governance & DPDP Compliance Copilot</h2>
  <p>Hybrid RAG (ChromaDB + BM25 + RRF) grounded in India's DPDP Act 2023, NIST AI RMF, and OWASP LLM Top 10.</p>
  <iframe
    src="https://YOUR-USERNAME-ai-governance-rag.hf.space"
    width="100%"
    height="640"
    style="border: 1px solid #334155; border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.35);"
    allow="clipboard-write"
    loading="lazy">
  </iframe>
</section>
```
