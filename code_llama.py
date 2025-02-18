import os
import time
import json
import concurrent.futures
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS

# Model and Data Paths
MODEL_PATH = "../llama_model/llama-2-7b.Q4_K_M.gguf"
PROJECT_PATH = "repo"
EMBEDDINGS_PATH = "faiss_index"

class CodeSummarySystem:
    def __init__(self, model_path: str):
        """Initialize LlamaCpp model and FAISS retrieval"""
        self.llm = LlamaCpp(
            model_path=model_path,
            n_ctx=2048,
            max_tokens=128,
            n_gpu_layers=32,
            n_threads=8,
            n_batch=512,
            f16_kv=True,
            verbose=True
        )
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_db = None
        self.load_or_create_index()

    def load_or_create_index(self):
        """Load FAISS index if available, else create a new one"""
        global EMBEDDINGS_PATH  # Ensure it's globally accessible
        print(f"🔎 Checking if FAISS index exists at: {EMBEDDINGS_PATH}")

        if os.path.exists(EMBEDDINGS_PATH):
            print("✅ Loading FAISS index...")
            try:
                self.vector_db = FAISS.load_local(EMBEDDINGS_PATH, self.embeddings, allow_dangerous_deserialization=True)
                print("🎯 FAISS index loaded successfully!")
            except Exception as e:
                print(f"⚠️ Error loading FAISS index: {e}")
                print("🔄 Rebuilding FAISS index...")
                self.create_faiss_index()
        else:
            print("⚠️ FAISS index missing. Rebuilding embeddings...")
            self.create_faiss_index()

        # **DEBUG: Print if vector_db is assigned**
        if hasattr(self, "vector_db") and self.vector_db is not None:
            print("✅ FAISS index is now assigned to self.vector_db")
        else:
            print("❌ Critical Error: self.vector_db is still None!")
            exit(1)  # Exit if FAISS failed to load

    def create_faiss_index(self):
        """Create and store FAISS index"""
        documents = []
        for file_path in self._get_python_files(PROJECT_PATH):
            print(f"📂 Indexing file: {file_path}")
            loader = TextLoader(file_path)
            documents.extend(loader.load())

        if not documents:
            raise ValueError("❌ No Python files found in the project.")

        self.vector_db = FAISS.from_documents(documents, self.embeddings)
        self.vector_db.save_local(EMBEDDINGS_PATH)
        print("✅ FAISS index created!")

    def ask_question(self, question: str) -> dict:
        """Retrieve relevant code snippets from FAISS before invoking Llama model."""
        try:
            # Retrieve relevant documents from FAISS
            retrieved_docs = self.vector_db.similarity_search(question, k=2)  # Get top 2 matches

            if not retrieved_docs:
                return {"error": "❌ No relevant code found in FAISS database."}

            # Combine retrieved content and truncate to fit within the context window
            context = "\n\n".join([doc.page_content[:500] for doc in retrieved_docs])  # Adjust truncation as needed

            # Construct a proper prompt using retrieved context
            prompt = f"Context:\n{context}\n\nNow, summarize the function definitions: {question}"

            # Query Llama model
            response = self.llm.invoke(prompt)

            return {
                "query": question,
                "retrieved_docs": [doc.page_content[:500] for doc in retrieved_docs],  # Show first 500 chars
                "result": response
            }

        except Exception as e:
            return {"error": f"❌ Error processing request: {str(e)}"}

    def _get_python_files(self, directory: str):
        """Retrieve all Python files"""
        python_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".py"):
                    python_files.append(os.path.join(root, file))
        return python_files

# Initialize System
code_summary_system = CodeSummarySystem(MODEL_PATH)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Code Summarization API is running!"})

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    response = code_summary_system.ask_question(question)
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, threaded=True)
