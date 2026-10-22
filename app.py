import os
import sys
import time
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Add src/ to Python path so we can import GovernanceRAG and indexer
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from rag import GovernanceRAG

app = Flask(__name__)
CORS(app)  # Allow embedding and cross-origin API calls from your portfolio website

print("[STARTUP] Initializing GovernanceRAG & Hybrid Retriever...")
rag_engine = GovernanceRAG()
print("[READY] GovernanceRAG initialized successfully!")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": rag_engine.model_name})


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Please provide a non-empty query."}), 400

    t0 = time.time()
    try:
        result = rag_engine.answer_question(query, top_k=6)
        rag_engine.langfuse.flush()
        latency = round(time.time() - t0, 2)
        result["latency_sec"] = latency
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)