import os
import sys
from typing import List

# Import loaders from modern standalone langchain packages to eliminate deprecation warnings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Fixed the import name here!
from langchain_community.vectorstores import Chroma

# Ensure project root is accessible for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Constants
KNOWLEDGE_BASE_DIR: str = "data/knowledge_base"
CHROMA_DB_DIR: str = "chroma_storage"

class KnowledgeBaseEngine:
    def __init__(self) -> None:
        """Initializes the embedding model wrapper using Google Gemini."""
        # This automatically looks for the GEMINI_API_KEY environment variable
        self.embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")

    def build_vector_store(self) -> None:
        """Loads text profiles, chunks them, and builds a local vector database."""
        if not os.path.exists(KNOWLEDGE_BASE_DIR):
            print(f"Creating empty knowledge directory at: {KNOWLEDGE_BASE_DIR}")
            os.makedirs(KNOWLEDGE_BASE_DIR)
            print("Please add text profiles (e.g., spinach.txt) to this folder and rerun.")
            return

        print(f"Loading reference articles from '{KNOWLEDGE_BASE_DIR}'...")
        loader = DirectoryLoader(KNOWLEDGE_BASE_DIR, glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()

        if not documents:
            print("No text documents found to parse.")
            return

        print(f"Successfully loaded {len(documents)} document(s). Splitting into semantic chunks...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        doc_chunks = text_splitter.split_documents(documents)
        print(f"Created {len(doc_chunks)} distinct document text chunks.")

        print(f"Generating Gemini embeddings and persisting vector index to: {CHROMA_DB_DIR}...")
        self.vector_store = Chroma.from_documents(
            documents=doc_chunks,
            embedding=self.embeddings,
            persist_directory=CHROMA_DB_DIR
        )
        print("Vector database built successfully using Gemini!")

    def query_knowledge(self, query_text: str, num_results: int = 2) -> List:
        """Performs a similarity vector search against the stored botanical text."""
        vector_db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=self.embeddings)
        results = vector_db.similarity_search(query_text, k=num_results)
        return results

if __name__ == "__main__":
    engine = KnowledgeBaseEngine()
    engine.build_vector_store()