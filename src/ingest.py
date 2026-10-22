import os
import re
import ssl
import json
import urllib.request
import fitz  # PyMuPDF

RAW_DIR = os.path.join("data", "raw")
OUTPUT_JSON = os.path.join("data", "processed_pages.json")

DOCUMENTS = [
    {
        "doc_id": "INDIA-DPDP-2023",
        "title": "India Digital Personal Data Protection Act (DPDP Act, 2023)",
        "filename": "INDIA.DPDP.2023.pdf",
        "url": "https://www.meity.gov.in/static/uploads/2024/06/2bf1f0e9f04e6fb4f8fef35e82c42aa5.pdf"
    },
    {
        "doc_id": "NIST-AI-100-1",
        "title": "NIST AI Risk Management Framework (AI RMF 1.0)",
        "filename": "NIST.AI.100-1.pdf",
        "url": "https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.100-1.pdf"
    },
    {
        "doc_id": "NIST-AI-600-1",
        "title": "NIST RMF Generative AI Profile (AI 600-1)",
        "filename": "NIST.AI.600-1.pdf",
        "url": "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf"
    },
    {
        "doc_id": "OWASP-LLM-2025",
        "title": "OWASP Top 10 for LLM Applications (2025)",
        "filename": "OWASP-Top-10-for-LLMs-v2025.pdf",
        "url": "https://owasp.github.io/www-project-top-10-for-large-language-model-applications/assets/PDF/OWASP-Top-10-for-LLMs-v2025.pdf"
    }
]

def download_pdfs():
    os.makedirs(RAW_DIR, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    # Unverified context fallback in case Windows SSL store flags .gov.in certificates
    ssl_ctx = ssl._create_unverified_context()
    
    for doc in DOCUMENTS:
        filepath = os.path.join(RAW_DIR, doc["filename"])
        if os.path.exists(filepath) and os.path.getsize(filepath) > 10000:
            print(f"[SKIP] Already downloaded: {doc['filename']}")
            continue
        
        print(f"[DOWNLOADING] {doc['title']}...")
        req = urllib.request.Request(doc["url"], headers=headers)
        with urllib.request.urlopen(req, timeout=60, context=ssl_ctx) as response, open(filepath, "wb") as out_file:
            out_file.write(response.read())
        print(f"[SAVED] {filepath} ({os.path.getsize(filepath) // 1024} KB)")

def clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def extract_pages():
    all_pages = []
    for doc in DOCUMENTS:
        filepath = os.path.join(RAW_DIR, doc["filename"])
        pdf = fitz.open(filepath)
        kept_pages = 0
        
        for page_index in range(len(pdf)):
            raw_text = pdf[page_index].get_text("text")
            cleaned = clean_text(raw_text)
            
            # Skip nearly blank pages or cover pages with < 100 characters
            if len(cleaned) < 100:
                continue
                
            all_pages.append({
                "doc_id": doc["doc_id"],
                "doc_title": doc["title"],
                "page_number": page_index + 1,
                "source_url": doc["url"],
                "text": cleaned
            })
            kept_pages += 1
            
        pdf.close()
        print(f"[PARSED] {doc['doc_id']}: kept {kept_pages} content-rich pages.")
        
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_pages, f, indent=2, ensure_ascii=False)
        
    print(f"\n[SUCCESS] Total extracted pages saved to {OUTPUT_JSON}: {len(all_pages)}")

if __name__ == "__main__":
    download_pdfs()
    extract_pages()