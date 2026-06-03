import os
import sys
from typing import List
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Ensure project root is accessible for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Constants
KNOWLEDGE_BASE_DIR = "data/knowledge_base"
CHROMA_DB_DIR = "chroma_storage"

class KnowledgeBaseEngine:
    def __init__(self) -> None:
        """Initializes the embedding model wrapper using OpenAI."""
        # Note: Make sure your OPENAI_API_KEY environment variable is configured!
        self.embeddings = OpenAIEmbeddings()

    def build_vector_store(self) -> None:
        """Loads text profiles, chunks them, and builds a local vector database."""
        if not os.path.exists(KNOWLEDGE_BASE_DIR):
            print(f"Creating empty knowledge directory at: {KNOWLEDGE_BASE_DIR}")
            os.makedirs(KNOWLEDGE_BASE_DIR)
            print("Please add text profiles (e.g., spinach.txt) to this folder and rerun.")
            return

        print(f"Loading reference articles from '{KNOWLEDGE_BASE_DIR}'...")
        # Recursively load all text documents inside the directory
        loader = DirectoryLoader(KNOWLEDGE_BASE_DIR, glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()

        if not documents:
            print("No text documents found to parse.")
            return

        print(f"Successfully loaded {len(documents)} document(s). Splitting into semantic chunks...")
        
        # Split texts structurally so concepts remain cohesive
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        doc_chunks = text_splitter.split_documents(documents)
        print(f"Created {len(doc_chunks)} distinct document text chunks.")

        print(f"Generating embeddings and persisting vector index to: {CHROMA_DB_DIR}...")
        # Store vector representations locally to disk
        self.vector_store = Chroma.from_documents(
            documents=doc_chunks,
            embedding=self.embeddings,
            persist_directory=CHROMA_DB_DIR
        )
        print("Vector database built successfully!")

    def query_knowledge(self, query_text: str, num_results: int = 2) -> List:
        """Performs a similarity vector search against the stored botanical text."""
        # Re-initialize or load the collection from disk
        vector_db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=self.embeddings)
        results = vector_db.similarity_search(query_text, k=num_results)
        return results

if __name__ == "__main__":
    engine = KnowledgeBaseEngine()
    engine.build_vector_store()