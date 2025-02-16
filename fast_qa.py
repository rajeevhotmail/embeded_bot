import os
import argparse
import pickle
from pathlib import Path
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings

class CodeQASystem:
    def __init__(self, model_path: str):
        """Initialize the LlamaCpp model with GPU acceleration."""
        self.llm = LlamaCpp(
            model_path=model_path,
            n_ctx=4096,
            max_tokens=256,  # Lower response size for faster inference
            n_gpu_layers=32,  # Utilize GPU for faster inference
            n_batch=512,  # Larger batch size for parallel computation
            top_k=50,  # Reduce search space for faster response
            top_p=0.9,  # Slightly reduce randomness
            temperature=0.7,  # Keep responses controlled
            verbose=False  # Disable unnecessary logging
        )
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.qa_chain = None

    def load_codebase(self, project_path: str, embeddings_path: str = "embeddings.pkl"):
        """Load FAISS index if it exists, otherwise regenerate embeddings."""
        embeddings_dir = str(Path(embeddings_path).parent)

        # Check if FAISS index and embeddings exist
        faiss_index_path = os.path.join(embeddings_dir, "index.faiss")
        if os.path.exists(embeddings_path) and os.path.exists(faiss_index_path):
            try:
                print("✅ Loading pre-calculated embeddings...")
                vector_db = FAISS.load_local(embeddings_dir, self.embeddings, allow_dangerous_deserialization=True)
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=vector_db.as_retriever(search_kwargs={"k": 2})  # Fetch only 2 most relevant documents
                )
                return
            except Exception as e:
                print(f"⚠️ FAISS index corrupted! Regenerating... ({str(e)})")

        print("⚠️ FAISS index missing or invalid. Regenerating embeddings...")

        # Process and embed documents
        documents = []
        for file_path in self._get_python_files(project_path):
            loader = TextLoader(file_path)
            documents.extend(loader.load())

        if not documents:
            raise ValueError("❌ No Python files found in the project directory.")

        # Create FAISS vector store
        vector_db = FAISS.from_documents(documents, self.embeddings)

        # Save FAISS index
        vector_db.save_local(embeddings_dir)

        # Save embeddings metadata
        with open(embeddings_path, "wb") as f:
            pickle.dump({"faiss_index": vector_db.index}, f)

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever(search_kwargs={"k": 2})  # Reduced retrieval for speed
        )

    def ask_question(self, question: str) -> str:
        """Ask a question and get a response."""
        if not self.qa_chain:
            return "⚠️ Please load a codebase first."

        # Limit question length to avoid excessive tokens
        if len(question.split()) > 256:
            question = " ".join(question.split()[:256]) + "..."

        return self.qa_chain.invoke(question)  # Using .invoke() instead of .run()

    def _get_python_files(self, directory: str) -> list:
        """Retrieve all Python files from a directory."""
        python_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        return python_files

def main():
    parser = argparse.ArgumentParser(description='Code Q&A System')
    parser.add_argument('--project-dir', help='Path to the project directory', required=True)
    parser.add_argument('--model-path', help='Path to the LlamaCpp model', required=True)

    args = parser.parse_args()

    qa_system = CodeQASystem(args.model_path)

    print("📂 Loading and analyzing codebase...")
    qa_system.load_codebase(args.project_dir)

    while True:
        question = input("\n❓ Ask a question about the code (or type 'quit' to exit): ")
        if question.lower() == 'quit':
            break

        answer = qa_system.ask_question(question)
        print("\n💡 Answer:", answer)

if __name__ == "__main__":
    main()
