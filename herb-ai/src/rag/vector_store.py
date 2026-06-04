import os
import sys
import time
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from google import genai  # Official, modern SDK
from google.genai import errors

# Ensure project root is accessible for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Constants
KNOWLEDGE_BASE_DIR: str = "data/knowledge_base"
CHROMA_DB_DIR: str = "chroma_storage"

class StableGeminiEngine:
    def __init__(self) -> None:
        """Initializes the official Google GenAI Client and native Chroma DB."""
        # Automatically detects the GEMINI_API_KEY environment variable
        self.client = genai.Client()
        
        # This model identifier is universally stable on free developer API keys
        self.model_name = "models/embedding-001"
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    def get_embedding(self, text: str, retries: int = 3, backoff_factor: float = 2.0) -> List[float]:
        """Computes vectors using the official SDK wrapper with built-in retry logic."""
        for attempt in range(retries):
            try:
                response = self.client.models.embed_content(
                    model=self.model_name,
                    contents=text
                )
                # Parse out the raw coordinate float matrix safely
                return [float(val) for val in response.embeddings[0].values]
                
            except errors.APIError as e:
                # Handle temporary 503 overloads or 429 rate pacing flags gracefully
                if e.code in [503, 429] and attempt < retries - 1:
                    sleep_time = backoff_factor ** attempt
                    print(f"  [Warning] Google API returned {e.code}. Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                    continue
                raise e
                
        raise RuntimeError("Failed to compute embeddings after maximum retries.")

    def build_vector_store(self) -> None:
        """Loads text files manually, cuts into chunks, and populates the native Chroma collection."""
        if not os.path.exists(KNOWLEDGE_BASE_DIR):
            print(f"Creating empty knowledge directory at: {KNOWLEDGE_BASE_DIR}")
            os.makedirs(KNOWLEDGE_BASE_DIR)
            return

        print(f"Loading reference articles from '{KNOWLEDGE_BASE_DIR}'...")
        
        raw_texts: List[str] = []
        for root, _, files in os.walk(KNOWLEDGE_BASE_DIR):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        raw_texts.append(f.read())

        if not raw_texts:
            print("No text documents found to parse.")
            return

        print(f"Successfully loaded {len(raw_texts)} document(s). Splitting into semantic chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        
        doc_chunks = text_splitter.split_text("\n\n".join(raw_texts))
        print(f"Created {len(doc_chunks)} distinct document text chunks.")

        print("Initializing Chroma Collection...")
        collection = self.chroma_client.get_or_create_collection(name="botanical_knowledge")

        print(f"Generating embeddings and indexing directly into: {CHROMA_DB_DIR}...")
        for idx, text_content in enumerate(doc_chunks):
            vector = self.get_embedding(text_content)
            
            collection.add(
                embeddings=[vector],
                documents=[text_content],
                metadatas=[{"source": "knowledge_base_profile"}],
                ids=[f"doc_chunk_{idx}"]
            )
            # Safe padding delay to stay clear of concurrency caps
            time.sleep(0.5)
            
        print("Vector database built successfully using official Google GenAI SDK hooks!")

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
    engine = StableGeminiEngine()
    engine.build_vector_store()