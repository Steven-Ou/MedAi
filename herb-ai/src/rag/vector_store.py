import os
import sys
from typing import List, Any, cast

# FIX: Explicit function import resolves Pylance PrivateImportUsage issues
from google.generativeai import embed_content

from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

# Ensure project root is accessible for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Constants
KNOWLEDGE_BASE_DIR: str = "data/knowledge_base"
CHROMA_DB_DIR: str = "chroma_storage"


class ProductionGeminiEngine:
    def __init__(self) -> None:
        """Initializes the stable production Gemini API configuration and Chroma client."""
        # Note: google-generativeai implicitly reads GEMINI_API_KEY from environment variables automatically!
        if not os.environ.get("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY environment variable is missing!")

        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    def get_embedding(self, text: str) -> List[float]:
        """Computes vectors using the stable text-embedding-004 model."""
        try:
            # FIX: Call the cleanly imported function and type-cast the unknown return value
            response: Any = embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document",
            )

            if isinstance(response, dict) and "embedding" in response:
                return [float(v) for v in response["embedding"]]

            raise ValueError(
                "API response missing 'embedding' numerical payload array."
            )

        except Exception as e:
            raise RuntimeError(f"Google API stable layer failure: {e}")

    def build_vector_store(self) -> None:
        """Loads text files manually, cuts them into chunks, and populates the native Chroma collection."""
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

        print(
            f"Successfully loaded {len(raw_texts)} document(s). Splitting into semantic chunks..."
        )
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

        doc_chunks = text_splitter.split_text("\n\n".join(raw_texts))
        print(f"Created {len(doc_chunks)} distinct document text chunks.")

        print("Initializing Chroma Collection...")
        collection = self.chroma_client.get_or_create_collection(
            name="botanical_knowledge"
        )

        print(f"Generating embeddings and indexing directly into: {CHROMA_DB_DIR}...")
        for idx, text_content in enumerate(doc_chunks):
            vector = self.get_embedding(text_content)

            collection.add(
                embeddings=[vector],
                documents=[text_content],
                metadatas=[{"source": "knowledge_base_profile"}],
                ids=[f"doc_chunk_{idx}"],
            )

        print("Vector database built successfully using production API endpoints!")

    def query_knowledge(self, query_text: str, num_results: int = 2) -> List[str]:
        """Queries the local vector storage database for similar text segments."""
        collection = self.chroma_client.get_or_create_collection(
            name="botanical_knowledge"
        )

        # FIX: Call the cleanly imported function for query vectors
        query_response: Any = embed_content(
            model="models/text-embedding-004",
            content=query_text,
            task_type="retrieval_query",
        )
        query_vector = query_response["embedding"]

        results = collection.query(
            query_embeddings=[query_vector], n_results=num_results
        )
        return results["documents"][0] if results["documents"] else []


if __name__ == "__main__":
    engine = ProductionGeminiEngine()
    engine.build_vector_store()
