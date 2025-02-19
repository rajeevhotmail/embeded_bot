import os
import json
import logging
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS

# Configure Logging
logging.basicConfig(level=logging.DEBUG)

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
        global EMBEDDINGS_PATH
        logging.debug(f"🔎 Checking if FAISS index exists at: {EMBEDDINGS_PATH}")

        if os.path.exists(EMBEDDINGS_PATH):
            logging.debug("✅ Loading FAISS index...")
            try:
                self.vector_db = FAISS.load_local(EMBEDDINGS_PATH, self.embeddings, allow_dangerous_deserialization=True)
                logging.debug("🎯 FAISS index loaded successfully!")
            except Exception as e:
                logging.error(f"⚠️ Error loading FAISS index: {e}")
                logging.debug("🔄 Rebuilding FAISS index...")
                self.create_faiss_index()
        else:
            logging.debug("⚠️ FAISS index missing. Rebuilding embeddings...")
            self.create_faiss_index()

        if hasattr(self, "vector_db") and self.vector_db is not None:
            logging.debug("✅ FAISS index is now assigned to self.vector_db")
        else:
            logging.error("❌ Critical Error: self.vector_db is still None!")
            exit(1)

    def create_faiss_index(self):
        """Create and store FAISS index"""
        documents = []
        for file_path in self._get_python_files(PROJECT_PATH):
            logging.debug(f"📂 Indexing file: {file_path}")
            loader = TextLoader(file_path)
            documents.extend(loader.load())

        if not documents:
            raise ValueError("❌ No Python files found in the project.")

        self.vector_db = FAISS.from_documents(documents, self.embeddings)
        self.vector_db.save_local(EMBEDDINGS_PATH)
        logging.debug("✅ FAISS index created!")

    def ask_question(self, question: str) -> dict:
        """Handle user queries by leveraging LLM for descriptions and AST for technical details."""
        try:
            response = ""
            retrieved_content = []

            # Define keywords to identify descriptive queries
            descriptive_keywords = ["summarize the project", "what is the project about"]

            # Check if the query is a descriptive question
            if any(keyword in question.lower() for keyword in descriptive_keywords):
                # Use LLM to provide English description of the project
                retrieved_docs = self.vector_db.similarity_search(question, k=1)
                if not retrieved_docs:
                    return {"error": "❌ No relevant code found in FAISS database."}
                retrieved_content = [doc.page_content for doc in retrieved_docs]
                context = "\n\n".join(retrieved_content)[:800]
                prompt = f"Context:\n{context}\n\nPlease summarize the project in a conversational tone, explaining what the project is about, what each file does, and the main functions in each file."
                response = self.llm.invoke(prompt)
            elif question.lower().startswith("how many python files"):
                # Use AST to count Python files
                python_files = self._get_python_files(PROJECT_PATH)
                count = len(python_files)
                response = f"There are {count} Python files in the project."
            elif question.lower().startswith("list all classes") or question.lower().startswith("what are the classes and functions"):
                # Use AST to list all classes and methods
                class_list = self.list_classes(PROJECT_PATH)
                response = "### List of Classes and Methods\n\n"
                for class_name in class_list:
                    method_list = self.list_methods(class_name, PROJECT_PATH)
                    response += f"**{class_name}**\n"
                    for method in method_list:
                        response += f"  - {method}\n"
                    response += "\n"
            elif question.lower().startswith("what is the calling stack"):
                # Use AST and LLM to summarize the calling stack
                retrieved_docs = self.vector_db.similarity_search(question, k=1)
                if not retrieved_docs:
                    return {"error": "❌ No relevant code found in FAISS database."}
                retrieved_content = [doc.page_content for doc in retrieved_docs]

                key_steps = self.extract_key_steps(retrieved_content[0])
                response = f"### Calling Stack in Normal Workflow\n\n{key_steps}"
            else:
                # Fallback: Use LLM for other types of queries
                retrieved_docs = self.vector_db.similarity_search(question, k=1)
                if not retrieved_docs:
                    return {"error": "❌ No relevant code found in FAISS database."}
                retrieved_content = [doc.page_content for doc in retrieved_docs]
                context = "\n\n".join(retrieved_content)[:800]
                prompt = f"Context:\n{context}\n\nNow, {question}"
                response = self.llm.invoke(prompt)

            formatted_response = (
                response.strip()
                + "\n\n### Retrieved Documents:\n\n"
                + "\n\n".join(retrieved_content[:1])
            )

            return {
                "query": question,
                "result": formatted_response
            }

        except Exception as e:
            logging.error(f"❌ Error processing request: {str(e)}")
            return {"error": f"❌ Error processing request: {str(e)}"}

    def extract_key_steps(self, content: str) -> str:
        """Extract key steps from the retrieved content to summarize the calling stack."""
        key_steps = []
        lines = content.split('\n')
        for line in lines:
            if "import" in line or "def " in line or "class " in line:
                key_steps.append(line.strip())
        return "\n".join(key_steps)



    def extract_key_steps(self, content: str) -> str:
        """Extract key steps from the retrieved content to summarize the calling stack."""
        key_steps = []
        lines = content.split('\n')
        for line in lines:
            if "import" in line or "def " in line or "class " in line:
                key_steps.append(line.strip())
        return "\n".join(key_steps)



    def list_classes(self, project_path: str):
        """List all classes in the project."""
        class_list = []
        for file_path in self._get_python_files(project_path):
            loader = TextLoader(file_path)
            documents = loader.load()
            for doc in documents:
                class_names = self._extract_classes(doc.page_content)
                class_list.extend(class_names)
        return class_list

    def _extract_classes(self, content: str):
        """Extract class names from the content."""
        import ast
        classes = []
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
        except Exception as e:
            logging.error(f"Error parsing content: {e}")
        return classes

    def list_methods(self, class_name: str, project_path: str):
        """List all methods in the specified class."""
        method_list = []
        for file_path in self._get_python_files(project_path):
            loader = TextLoader(file_path)
            documents = loader.load()
            for doc in documents:
                methods = self._extract_methods(doc.page_content, class_name)
                method_list.extend(methods)
        return method_list

    def _extract_methods(self, content: str, class_name: str):
        """Extract method names from the specified class in the content."""
        import ast
        methods = []
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    for subnode in node.body:
                        if isinstance(subnode, ast.FunctionDef):
                            methods.append(subnode.name)
        except Exception as e:
            logging.error(f"Error parsing content: {e}")
        return methods

    def _get_python_files(self, directory: str):
        """Retrieve all Python files"""
        python_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".py"):
                    python_files.append(os.path.join(root, file))
        return python_files

code_summary_system = CodeSummarySystem(MODEL_PATH)

@app.route("/", methods=["GET"])
def home():
    return render_template('index.html')

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.json
        question = data.get("question", "")
        if not question:
            return jsonify({"error": "No question provided"}), 400

        response = code_summary_system.ask_question(question)
        return jsonify(response)
    except Exception as e:
        logging.error(f"❌ Error processing request: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, threaded=True, debug=True)

