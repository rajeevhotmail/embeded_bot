import os
import time
import re
import concurrent.futures
import argparse
import pickle
import json
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

# Model and Data Paths
MODEL_PATH = "models/llama-2-7b-chat.gguf"
PROJECT_PATH = "repo"
EMBEDDINGS_PATH = "embeddings.pkl"


class CodeQASystem:
    def __init__(self, model_path: str):
        """Initialize the LlamaCpp model and embedding model."""
        self.llm = LlamaCpp(
       model_path=model_path,
        n_ctx=2048,  # Ensure this stays within the model's limit
        max_tokens=800,  # Explicitly limit token generation
        n_gpu_layers=32,
        n_threads=8,
        n_batch=512,
        f16_kv=True,
        verbose=True
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
                    retriever=vector_db.as_retriever(search_kwargs={"k": 5})  # Fetch 5 most relevant documents
                )
                return
            except Exception as e:
                print(f"⚠️ FAISS index corrupted! Regenerating... ({str(e)})")

        print("⚠️ FAISS index missing or invalid. Regenerating embeddings...")

        # Process and embed documents
        documents = []
        for file_path in self._get_python_files(project_path):
            print(f"📂 Indexing file: {file_path}")  # Debugging print
            loader = TextLoader(file_path)
            documents.extend(loader.load())

        if not documents:
            raise ValueError("❌ No Python files found in the project directory.")

        # Create FAISS vector store
        vector_db = FAISS.from_documents(documents, self.embeddings)
        vector_db.save_local(embeddings_dir)

        # Save embeddings metadata
        with open(embeddings_path, "wb") as f:
            pickle.dump({"faiss_index": vector_db.index}, f)

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever(search_kwargs={"k": 5})
        )

        print("✅ FAISS index and embeddings rebuilt successfully!")




    def ask_question(self, question: str) -> dict:
        """Ask a question and return a well-formatted JSON response with a strict token limit."""

        if not self.qa_chain:
            return {"error": "⚠️ Please load a codebase first."}

        start_time = time.time()

        # Limit question length to 256 words
        if len(question.split()) > 256:
            question = " ".join(question.split()[:256]) + "..."

        try:

            # Retrieve only 1 document from FAISS
            retrieval_start = time.time()
            retrieved_docs = self.qa_chain.retriever.invoke(question, search_kwargs={"k": 1})
            retrieval_time = time.time() - retrieval_start

            print(f"⏱️ FAISS Retrieval Time: {retrieval_time:.2f} seconds")
            print("🔍 Retrieved Documents:", [doc.metadata["source"] for doc in retrieved_docs])

            # If FAISS retrieval fails, return an error early
            if not retrieved_docs:
                return {
                    "error": "❌ No relevant code found. Try reindexing the FAISS database.",
                    "retrieval_time": f"{retrieval_time:.2f} seconds",
                    "total_time": f"{time.time() - start_time:.2f} seconds"
                }

            # **🛠️ Log the token count of retrieved text**
            combined_text = " ".join([doc.page_content for doc in retrieved_docs])
            token_count = len(combined_text.split())  # Rough estimation of tokens

            print(f"📏 Retrieved Document Token Count: {token_count} tokens")

            # **🛠️ Strict Token Limit: 800 Tokens (~3200 Characters)**
            max_tokens = 800  # Hard limit
            words = combined_text.split()[:max_tokens]  # Keep only the first 800 tokens
            truncated_text = " ".join(words)  # Convert back to a string
            print(f"📏 Retrieved Document Token Count: {len(combined_text.split())} tokens")
            print(f"📏 Final Input Token Count: {len(truncated_text.split())} tokens (should be ≤ 800)")
            # Construct a safe prompt
            truncated_prompt = f"Context:\n{truncated_text}\n\nBe specific: {question}"

            print(f"📏 Final Input Token Count: {len(truncated_text.split())} tokens (should be ≤ 800)")

            # Model inference with timeout
            model_start = time.time()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self.qa_chain.invoke, truncated_prompt)
                try:
                    response = future.result(timeout=15)  # Timeout in 15 seconds
                except concurrent.futures.TimeoutError:
                    return {
                        "error": "⚠️ Model took too long to respond. Try optimizing LlamaCpp settings.",
                        "retrieval_time": f"{retrieval_time:.2f} seconds",
                        "total_time": f"{time.time() - start_time:.2f} seconds"
                    }

            model_time = time.time() - model_start
            total_time = time.time() - start_time

            formatted_response = {
                "query": question,
                "answer": response,
                "retrieval_time": f"{retrieval_time:.2f} seconds",
                "model_time": f"{model_time:.2f} seconds",
                "total_time": f"{total_time:.2f} seconds"
            }

            return json.loads(json.dumps(formatted_response, indent=4))

        except Exception as e:
            return {
                "error": f"❌ Error processing request: {str(e)}",
                "retrieval_time": f"{retrieval_time:.2f} seconds",
                "total_time": f"{time.time() - start_time:.2f} seconds"
            }



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
