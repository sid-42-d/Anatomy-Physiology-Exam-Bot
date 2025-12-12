from dotenv import load_dotenv
import os
from pathlib import Path
from flask import Flask, request, jsonify

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

# ---------- Init models & index (runs once on startup) ----------

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
assert groq_key, "GROQ_API_KEY not set in .env"

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
Settings.llm = Groq(
    model="llama-3.1-8b-instant",
    api_key=groq_key,
)

PERSIST_DIR = "storage"

if Path(PERSIST_DIR).exists():
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
else:
    documents = SimpleDirectoryReader("Data", recursive=True).load_data()
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index.storage_context.persist(persist_dir=PERSIST_DIR)

query_engine = index.as_query_engine()

# ---------- Flask app ----------

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/query", methods=["POST"])
def query():
    data = request.get_json(force=True)
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400

    resp = query_engine.query(question)
    return jsonify({"answer": str(resp)})

if __name__ == "__main__":
    # for local testing: python app.py
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
