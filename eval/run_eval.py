import os
import re
import sys
import json
import time

# Allow importing from src/
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from rag import GovernanceRAG

GOLDEN_DATASET = [
    {
        "id": "Q1_DPDP_PENALTY",
        "question": "What is the maximum penalty under India's DPDP Act for failing to take reasonable security safeguards to prevent a personal data breach?",
        "expected_docs": ["INDIA-DPDP-2023"],
        "ground_truth": "Under the Schedule of the DPDP Act 2023 (Section 8(5)), the penalty for breach in observing the obligation of a Data Fiduciary to take reasonable security safeguards to prevent a personal data breach may extend to two hundred and fifty crore rupees (Rs. 250 crore)."
    },
    {
        "id": "Q2_DPDP_CHILDREN",
        "question": "What are the obligations of a Data Fiduciary before processing personal data of a child under India's DPDP Act?",
        "expected_docs": ["INDIA-DPDP-2023"],
        "ground_truth": "Under Section 9 of the DPDP Act 2023, a Data Fiduciary must obtain verifiable consent of the parent or lawful guardian before processing any personal data of a child, and must not undertake processing likely to cause any detrimental effect on the well-being of a child or tracking/behavioral monitoring/targeted advertising directed at children."
    },
    {
        "id": "Q3_NIST_RMF_CORE",
        "question": "What are the four core functions of the NIST AI Risk Management Framework (AI RMF 1.0)?",
        "expected_docs": ["NIST-AI-100-1"],
        "ground_truth": "The four core functions of the NIST AI RMF 1.0 are GOVERN, MAP, MEASURE, and MANAGE."
    },
    {
        "id": "Q4_NIST_GENAI_CONFABULATION",
        "question": "How does the NIST Generative AI Profile (AI 600-1) define confabulation and what risks does it pose?",
        "expected_docs": ["NIST-AI-600-1"],
        "ground_truth": "NIST AI 600-1 defines confabulation (often called hallucination) as the production of confidently stated but erroneous or false content by generative AI systems, which risks misleading users and degrading decision-making."
    },
    {
        "id": "Q5_OWASP_PROMPT_INJECTION",
        "question": "What is Prompt Injection according to OWASP Top 10 for LLM Applications 2025 and what is the difference between direct and indirect prompt injection?",
        "expected_docs": ["OWASP-LLM-2025"],
        "ground_truth": "Prompt Injection (LLM01:2025) occurs when user inputs alter the LLM's behavior or output in unintended ways. Direct prompt injection happens when a user directly inputs malicious instructions, while indirect prompt injection occurs when an LLM processes external content (websites, documents, data) containing hidden instructions."
    },
    {
        "id": "Q6_OWASP_VECTOR_RAG",
        "question": "What are Vector and Embedding Weaknesses in OWASP LLM Top 10 2025 and how can they be mitigated in RAG systems?",
        "expected_docs": ["OWASP-LLM-2025"],
        "ground_truth": "Vector and Embedding Weaknesses (LLM08:2025) involve vulnerabilities in how embeddings are generated, stored, or retrieved in RAG systems, such as unauthorized access across tenants, data poisoning, or embedding inversion. Mitigations include permission-aware vector retrieval, input validation, and monitoring retrieval logs."
    },
    {
        "id": "Q7_CROSS_FRAMEWORK",
        "question": "If an organization faces a personal data breach in an LLM system in India, what are the breach notification duties under the DPDP Act and how does OWASP LLM Top 10 address Sensitive Information Disclosure?",
        "expected_docs": ["INDIA-DPDP-2023", "OWASP-LLM-2025"],
        "ground_truth": "Under India's DPDP Act (Section 8(6)), the Data Fiduciary must notify the Data Protection Board of India and each affected Data Principal in the event of a personal data breach. Under OWASP LLM02:2025 (Sensitive Information Disclosure), organizations should sanitize training/context data, apply input/output filtering, and enforce strict access controls."
    },
    {
        "id": "Q8_OUT_OF_SCOPE_GUARDRAIL",
        "question": "What is the standard corporate income tax rate for manufacturing companies in Brazil in 2026?",
        "expected_docs": [],
        "ground_truth": "The assistant must refuse to answer and state that it cannot verify this from the indexed governance frameworks."
    }
]


def grade_with_llm_judge(rag: GovernanceRAG, question: str, ground_truth: str, generated_answer: str) -> dict:
    judge_prompt = f"""You are a strict RAG Evaluation Judge.
Compare the GENERATED ANSWER against the GROUND TRUTH for the given QUESTION.

QUESTION: {question}
GROUND TRUTH: {ground_truth}
GENERATED ANSWER: {generated_answer}

Score the GENERATED ANSWER from 0.0 to 1.0:
- 1.0: Accurately covers the key facts in the ground truth without fabricating unsupported claims.
- 0.5: Partially accurate or misses one part of a multi-part question, but contains no hallucinations.
- 0.0: Incorrect, hallucinated, or failed to refuse an out-of-scope question.

Reply ONLY with valid JSON in this exact format:
{{"score": 1.0, "reason": "Brief 1-sentence explanation"}}"""

    completion = rag.client.chat.completions.create(
        model=rag.model_name,
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0.0,
        max_tokens=1024
    )
    raw = completion.choices[0].message.content.strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return {"score": 0.5, "reason": raw[:150]}


def run_evaluation():
    rag = GovernanceRAG()
    results = []
    print(f"Running evaluation on {len(GOLDEN_DATASET)} benchmark questions...\n")

    for idx, item in enumerate(GOLDEN_DATASET, 1):
        t0 = time.time()
        # Retry wrapper in case free-tier rate limit (429) triggers
        for attempt in range(3):
            try:
                rag_out = rag.answer_question(item["question"], top_k=6)
                break
            except Exception as e:
                print(f"  [Rate limit pause] Waiting 10s before retry ({e})...")
                time.sleep(10)

        latency_s = round(time.time() - t0, 2)
        retrieved_docs = {s["doc_id"] for s in rag_out["sources"]}

        # 1. Check Retrieval Hit Rate
        if not item["expected_docs"]:
            retrieval_hit = 1.0  # Out-of-scope query has no required doc
        else:
            hits = sum(1 for d in item["expected_docs"] if d in retrieved_docs)
            retrieval_hit = round(hits / len(item["expected_docs"]), 2)

        # 2. Check Citation / Guardrail Compliance
        answer_text = rag_out["answer"]
        if not item["expected_docs"]:
            citation_valid = 1.0 if "cannot verify" in answer_text.lower() else 0.0
        else:
            has_bracket_cite = bool(re.search(r"\[[A-Z0-9\-]+,\s*Page\s*\d+\]", answer_text))
            citation_valid = 1.0 if has_bracket_cite else 0.0

        # 3. LLM-as-a-Judge Faithfulness/Correctness Score
        judge = grade_with_llm_judge(rag, item["question"], item["ground_truth"], answer_text)
        judge_score = float(judge.get("score", 0.0))

        row = {
            "id": item["id"],
            "question": item["question"],
            "latency_sec": latency_s,
            "retrieval_hit": retrieval_hit,
            "citation_valid": citation_valid,
            "judge_score": judge_score,
            "judge_reason": judge.get("reason", ""),
            "retrieved_docs": list(retrieved_docs),
            "answer_preview": answer_text[:220].replace("\n", " ") + "..."
        }
        results.append(row)
        print(
            f"[{idx}/{len(GOLDEN_DATASET)}] {item['id']} | "
            f"Retrieval: {retrieval_hit} | Citation: {citation_valid} | "
            f"Judge: {judge_score} | Latency: {latency_s}s"
        )
        print(f"    Reason: {row['judge_reason']}")
        time.sleep(3)  # Respect Groq free-tier TPM rate limit

    rag.langfuse.flush()

    avg_retrieval = round(sum(r["retrieval_hit"] for r in results) / len(results) * 100, 1)
    avg_citation = round(sum(r["citation_valid"] for r in results) / len(results) * 100, 1)
    avg_judge = round(sum(r["judge_score"] for r in results) / len(results) * 100, 1)
    avg_latency = round(sum(r["latency_sec"] for r in results) / len(results), 2)

    summary = {
        "total_questions": len(results),
        "retrieval_hit_rate_pct": avg_retrieval,
        "citation_compliance_pct": avg_citation,
        "faithfulness_correctness_pct": avg_judge,
        "avg_latency_sec": avg_latency,
        "details": results
    }

    out_path = os.path.join("eval", "eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n================ EVALUATION SCORECARD ================")
    print(f"Questions Evaluated        : {len(results)}")
    print(f"Retrieval Hit Rate         : {avg_retrieval}%")
    print(f"Citation & Guardrail Rate  : {avg_citation}%")
    print(f"LLM-Judge Correctness      : {avg_judge}%")
    print(f"Average End-to-End Latency : {avg_latency}s")
    print(f"Saved full report to       : {out_path}")


if __name__ == "__main__":
    run_evaluation()
