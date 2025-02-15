import os
import argparse
import pickle
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader

class CodeQASystem:
    def __init__(self, model_path: str):  # Accept model path in constructor
        self.llm = LlamaCpp(
            model_path=model_path,
            n_ctx=4096,
            max_tokens=4096
        )
        self.embeddings = HuggingFaceEmbeddings()
        self.qa_chain = None

    def load_codebase(self, project_path: str, embeddings_path: str = "embeddings.pkl"):
        if os.path.exists(embeddings_path):
            print("Loading pre-calculated embeddings...")
            with open(embeddings_path, "rb") as f:
                embeddings_data = pickle.load(f)
                self.embeddings = embeddings_data["embeddings"] # important to load back the embedding model
                faiss_index = embeddings_data["faiss_index"]
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=FAISS(self.embeddings, faiss_index).as_retriever()
                )
            return

        documents = []  # Initialize documents list here
        all_embeddings = []  # Initialize all_embeddings list here
        for file_path in self._get_python_files(project_path):
            loader = TextLoader(file_path)
            loaded_docs = loader.load()
            documents.extend(loaded_docs)
            for doc in loaded_docs:
                all_embeddings.append(self.embeddings.embed_query(doc.page_content))


        faiss_index = FAISS.from_embeddings(all_embeddings, self.embeddings).index

        embeddings_data = {"embeddings": self.embeddings, "faiss_index": faiss_index}
        with open(embeddings_path, "wb") as f:
            pickle.dump(embeddings_data, f)

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=FAISS(self.embeddings, faiss_index).as_retriever()
        )

    def ask_question(self, question: str) -> str:
        if not self.qa_chain:
            return "Please load a codebase first."
        return self.qa_chain.run(question)

    def _get_python_files(self, directory: str) -> list:
        python_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        return python_files

def main():
    parser = argparse.ArgumentParser(description='Code Q&A System')
    parser.add_argument('--project-dir', help='Path to the project directory', required=True) # required added
    parser.add_argument('--model-path', help='Path to the LlamaCpp model', required=True) # required added

    args = parser.parse_args()

    embeddings_path = "embeddings.pkl"
    qa_system = CodeQASystem(args.model_path) # Pass model path to constructor

    print("Loading and analyzing codebase...")
    qa_system.load_codebase(args.project_dir, embeddings_path)

    while True:
        question = input("\nAsk a question about the code (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break

        answer = qa_system.ask_question(question)
        print("\nAnswer:", answer)

if __name__ == "__main__":
    main()