import os
import argparse
import pickle
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings

app = Flask(__name__)
CORS(app)  # Enable CORS

# Initialize model
MODEL_PATH = "models/llama-2-7b-chat.gguf"
PROJECT_PATH = "temp_repo"
EMBEDDINGS_PATH = "embeddings.pkl"

class CodeQASystem:
    def __init__(self, model_path: str):
        """Initialize the LlamaCpp model with dynamic token limit."""
        self.model_path = model_path
        self.max_tokens = 256  # Start with small tokens
        self.min_tokens = 128  # Prevent too small limits
        self.max_limit = 4096  # Prevent exceeding model limit

        self.llm = LlamaCpp(
            model_path=self.model_path,
            n_ctx=4096,
            max_tokens=self.max_tokens,
            n_gpu_layers=32,
            n_batch=512,
            top_k=50,
            top_p=0.9,
            temperature=0.7,
            verbose=False
        )

        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.qa_chain = None

    def load_codebase(self, project_path: str, embeddings_path: str = "embeddings.pkl"):
        """Load FAISS index if it exists, otherwise regenerate embeddings."""
        embeddings_dir = str(Path(embeddings_path).parent)
        faiss_index_path = os.path.join(embeddings_dir, "index.faiss")

        if os.path.exists(embeddings_path) and os.path.exists(faiss_index_path):
            try:
                print("✅ Loading pre-calculated embeddings...")
                vector_db = FAISS.load_local(embeddings_dir, self.embeddings, allow_dangerous_deserialization=True)
                self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=vector_db.as_retriever(search_kwargs={"k": 1})
                )
                return
            except Exception as e:
                print(f"⚠️ FAISS index corrupted! Regenerating... ({str(e)})")


        documents = [TextLoader(file).load()[0] for file in self._get_python_files(project_path)]
        if not documents:
            raise ValueError("❌ No Python files found in the project directory.")

        vector_db = FAISS.from_documents(documents, self.embeddings)
        vector_db.save_local(embeddings_dir)

        with open(embeddings_path, "wb") as f:
            pickle.dump({"faiss_index": vector_db.index}, f)

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever(search_kwargs={"k": 1})
        )

    def ask_question(self, question: str, feedback=None) -> str:
        """Ask a question and get a response with dynamic token adjustment."""
        if not self.qa_chain:
            return "⚠️ Please load a codebase first."

        if len(question.split()) > 256:
            question = " ".join(question.split()[:256]) + "..."

        response = self.qa_chain.invoke(question)

        # Adjust token limit based on feedback
        if feedback == "correct":
            self.max_tokens = min(self.max_tokens + 256, self.max_limit)  # Increase tokens
        elif feedback == "incorrect":
            self.max_tokens = max(self.max_tokens - 128, self.min_tokens)  # Decrease tokens

        # Update Llama model with new token limit
        self.llm = LlamaCpp(
            model_path=self.model_path,
            n_ctx=4096,
            max_tokens=self.max_tokens,
            n_gpu_layers=32,
            n_batch=512,
            top_k=50,
            top_p=0.9,
            temperature=0.7,
            verbose=False
        )

        return response


    def _get_python_files(self, directory: str) -> list:
        """Retrieve all Python files from a directory."""
        python_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        return python_files

# Create a global instance
qa_system = CodeQASystem(MODEL_PATH)
qa_system.load_codebase(PROJECT_PATH, EMBEDDINGS_PATH)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "")

    print(f"🔹 Received question: {question}")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        answer = qa_system.ask_question(question)
        return jsonify({"query": question, "result": answer})
    except Exception as e:
        return jsonify({"error": f"❌ Error processing request: {str(e)}"}), 500

if __name__ == "__main__":
      app.run(host="0.0.0.0", port=5000, threaded=True)
