import os
import sys
from typing import List
import requests

# Standalone imports to eliminate the LangChain community deprecation warning
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

# Ensure project root is accessible for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Constants
KNOWLEDGE_BASE_DIR: str = "data/knowledge_base"
CHROMA_DB_DIR: str = "chroma_storage"

class ProductionGeminiEngine:
    def __init__(self) -> None:
        """Initializes direct HTTP access to production Google AI v1 endpoints."""
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is missing!")
        
        # Strict Google AI Studio production REST endpoint URL mapping
        self.url = f"https://generativelanguage.googleapis.com/v1/models/text-embedding-004:embedContent?key={self.api_key}"
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    def get_embedding(self, text: str) -> List[float]:
        """Computes vectors via a direct REST call to the stable v1 production API."""
        # Payload structured cleanly: Model is in the URL, so we only pass content blocks here
        payload = {
            "content": {
                "parts": [{"text": text}]
            }
        }
        
        response = requests.post(self.url, json=payload)
        
        if response.status_code != 200:
            raise RuntimeError(f"Google API Error ({response.status_code}): {response.text}")
            
        response_json = response.json()
        return [float(val) for val in response_json["embedding"]["values"]]

    def build_vector_store(self) -> None:
        """Loads text files, cuts into chunks, and populates the native Chroma collection."""
        if not os.path.exists(KNOWLEDGE_BASE_DIR):
            print(f"Creating empty knowledge directory at: {KNOWLEDGE_BASE_DIR}")
            os.makedirs(KNOWLEDGE_BASE_DIR)
            return

        print(f"Loading reference articles from '{KNOWLEDGE_BASE_DIR}'...")
        loader = DirectoryLoader(KNOWLEDGE_BASE_DIR, glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()

        if not documents:
            print("No text documents found to parse.")
            return

        print(f"Successfully loaded {len(documents)} document(s). Splitting into semantic chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        doc_chunks = text_splitter.split_documents(documents)
        print(f"Created {len(doc_chunks)} distinct document text chunks.")

        print("Initializing Chroma Collection...")
        collection = self.chroma_client.get_or_create_collection(name="botanical_knowledge")

        print(f"Generating embeddings and indexing directly into: {CHROMA_DB_DIR}...")
        for idx, chunk in enumerate(doc_chunks):
            text_content = chunk.page_content
            vector = self.get_embedding(text_content)
            source_file = chunk.metadata.get("source", "unknown")

            collection.add(
                embeddings=[vector],
                documents=[text_content],
                metadatas=[{"source": source_file}],
                ids=[f"doc_chunk_{idx}"]
            )
            
        print("Vector database built successfully using direct production v1 API endpoints!")

    def query_knowledge(self, query_text: str, num_results: int = 2) -> List[str]:
        """Queries the local vector storage database for similar text segments."""
        collection = self.chroma_client.get_or_create_collection(name="botanical_knowledge")
        query_vector = self.get_embedding(query_text)
        
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=num_results
        )
        return results['documents'][0] if results['documents'] else []

if __name__ == "__main__":
    engine = ProductionGeminiEngine()
    engine.build_vector_store()