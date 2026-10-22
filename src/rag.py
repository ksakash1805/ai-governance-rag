import os
from dotenv import load_dotenv
from groq import Groq
from langfuse import observe, get_client
from indexer import HybridRetriever

load_dotenv()

SYSTEM_PROMPT = """You are an Enterprise AI Governance & Compliance Copilot.
You answer regulatory, privacy, and LLM security questions strictly using the provided context chunks from:
- India Digital Personal Data Protection Act (INDIA-DPDP-2023)
- NIST AI Risk Management Framework 1.0 (NIST-AI-100-1)
- NIST Generative AI Profile (NIST-AI-600-1)
- OWASP Top 10 for LLM Applications 2025 (OWASP-LLM-2025)

STRICT RULES:
1. Rely ONLY on the provided context chunks. Do not assume or extrapolate from external knowledge.
2. Every single paragraph or bullet point MUST include an inline citation in the exact format [DOC_ID, Page X] using standard ASCII hyphens, for example: [INDIA-DPDP-2023, Page 21], [NIST-AI-100-1, Page 20], [NIST-AI-600-1, Page 11], or [OWASP-LLM-2025, Page 4].
3. Only if the provided context chunks are completely unrelated to the question (such as a question about unrelated countries or tax laws), reply with the exact sentence: "I cannot verify this from the indexed governance frameworks."
4. Keep answers clear, structured, and concise."""


def normalize_unicode(text: str) -> str:
    if not text:
        return ""
    replacements = {
        "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
        "\u202f": " ", "\u00a0": " ", "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


class GovernanceRAG:
    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or "paste_your" in api_key:
            raise ValueError("Missing valid GROQ_API_KEY in .env file!")
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        self.retriever = HybridRetriever()
        self.langfuse = get_client()

    @observe(name="hybrid-retrieval")
    def retrieve_context(self, query: str, top_k: int = 6) -> list[dict]:
        return self.retriever.retrieve(query, top_k=top_k, max_per_doc=4)

    def format_context(self, chunks: list[dict]) -> str:
        formatted = []
        for idx, c in enumerate(chunks, 1):
            formatted.append(
                f"--- CHUNK {idx} | Citation Tag: [{c['doc_id']}, Page {c['page_number']}] ---\n{c['text']}"
            )
        return "\n\n".join(formatted)

    @observe(name="governance-rag-pipeline")
    def answer_question(self, query: str, top_k: int = 6) -> dict:
        retrieved_chunks = self.retrieve_context(query, top_k=top_k)
        context_block = self.format_context(retrieved_chunks)

        user_prompt = (
            f"CONTEXT CHUNKS:\n{context_block}\n\n"
            f"USER QUESTION:\n{query}\n\n"
            "Answer the question using the context chunks above and include [DOC_ID, Page X] citations in every bullet:"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        with self.langfuse.start_as_current_observation(
            as_type="generation",
            name="groq-llm-generation",
            model=self.model_name,
            input=messages
        ) as gen:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=1500
            )
            raw_answer = completion.choices[0].message.content or ""
            answer_text = normalize_unicode(raw_answer)
            usage_info = {
                "input": completion.usage.prompt_tokens,
                "output": completion.usage.completion_tokens,
                "total": completion.usage.total_tokens
            }
            gen.update(output=answer_text, usage_details=usage_info)

        return {
            "query": query,
            "answer": answer_text,
            "sources": [
                {
                    "chunk_id": c["chunk_id"],
                    "doc_id": c["doc_id"],
                    "doc_title": c["doc_title"],
                    "page_number": c["page_number"],
                    "source_url": c["source_url"],
                    "rrf_score": c["rrf_score"],
                    "snippet": normalize_unicode(c["text"][:250]) + "..."
                }
                for c in retrieved_chunks
            ],
            "usage": {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens
            }
        }