import os
import time
import json
import re
import concurrent.futures
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS

# Model and Data Paths
MODEL_PATH = "bigcode/starcoder"  # Use the Hugging Face model path
PROJECT_PATH = "repo"

class CodeSummarySystem:
    def __init__(self, model_path: str):
        """Initialize StarCoder model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)

    def ask_question(self, question: str) -> dict:
        """Generate a summary based on function definitions in the project"""
        try:
            print("🔍 Tokenizing the question...")  # Debugging print
            inputs = self.tokenizer(question, return_tensors="pt")
            print("🧠 Generating the response...")  # Debugging print
            outputs = self.model.generate(**inputs)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return {"query": question, "result": response}
        except Exception as e:
            return {"error": f"❌ Error processing request: {str(e)}"}

    def summarize_code(self, project_path: str):
        """Summarize and analyze all Python files in the repo."""
        summaries = {}
        for file_path in self._get_python_files(project_path):
            print(f"📂 Processing: {file_path}")  # Debugging print
            summary = self._analyze_file(file_path)
            summaries[file_path] = summary
        return summaries

    def _analyze_file(self, file_path: str):
        """Analyze a Python file and generate a summary."""
        with open(file_path, "r", encoding="utf-8") as f:
            code_content = f.read()

        # Limit Code Length to 4000 Characters (~1000 Tokens)
        truncated_code = " ".join(code_content.split()[:1000])

        # Construct the Prompt
        prompt = f"Analyze the following Python code:\n{truncated_code}\n\nSummarize the purpose, functions, and logic."

        # Run Model Inference with Timeout
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(self.ask_question, prompt)
            try:
                response = future.result(timeout=15)  # Timeout in 15 seconds
            except concurrent.futures.TimeoutError:
                return {"error": "⚠️ Model took too long to respond."}

        return {
            "file": file_path,
            "summary": response["result"] if "result" in response else response.get("error"),
            "time_taken": f"{time.time() - start_time:.2f} seconds"
        }

    def _get_python_files(self, directory: str):
        """Retrieve all Python files in a directory."""
        python_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".py"):
                    python_files.append(os.path.join(root, file))
        return python_files


# Create a Global Instance
code_summary_system = CodeSummarySystem(MODEL_PATH)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Code Summarization API is running!"})

@app.route("/summarize", methods=["GET"])
def summarize():
    print("🔹 Received Summarization Request")
    summaries = code_summary_system.summarize_code(PROJECT_PATH)
    return jsonify({"summaries": summaries})

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "")

    print(f"🔹 Received question: {question}")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        answer = code_summary_system.ask_question(question)  # Use the correct instance
        return jsonify({"query": question, "result": answer})
    except Exception as e:
        return jsonify({"error": f"❌ Error processing request: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, threaded=True)
