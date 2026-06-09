import os
import sys
from typing import List
from dotenv import load_dotenv
from google import genai
from google.genai import types

from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

# Ensure project root is accessible for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Load environmental configurations from the .env file
load_dotenv()

# Constants
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_DIR: str = os.path.abspath(os.path.join(CURRENT_DIR, "../../chroma_storage"))

KNOWLEDGE_BASE_DIR: str = os.path.abspath(os.path.join(CURRENT_DIR, "../../data/knowledge_base"))

class ProductionGeminiEngine:
    def __init__(self) -> None:
        """Initializes the modern Google GenAI Client and native Chroma client."""
        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError(
                "GEMINI_API_KEY not found. Did you set it in your .env file?"
            )

        self.client = genai.Client()
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    def get_embedding(self, text: str, is_query: bool = False) -> List[float]:
        """Computes vectors using the active mainline gemini-embedding-001 model layout."""
        try:
            # Set the appropriate task type configuration to ensure optimal vector grouping
            task_type = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"

            # Use the active mainline embedding model string identifier
            response = self.client.models.embed_content(
                model="gemini-embedding-001",
                contents=text,
                config=types.EmbedContentConfig(task_type=task_type),
            )

            # Safe parsing ensures the strict local type checkers stay completely silent
            if response.embeddings:
                values = response.embeddings[0].values
                if values:
                    return [float(v) for v in values]

            raise ValueError("API responded with an empty embedding vector structure.")

        except Exception as e:
            raise RuntimeError(f"Google GenAI SDK pipeline failure: {e}")

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
            vector = self.get_embedding(text_content, is_query=False)

            collection.add(
                embeddings=[vector],
                documents=[text_content],
                metadatas=[{"source": "knowledge_base_profile"}],
                ids=[f"doc_chunk_{idx}"],
            )

        print(
            "Vector database built successfully using official modern GenAI SDK architecture!"
        )

    def query_knowledge(self, query_text: str, num_results: int = 2) -> List[str]:
        """Queries the local vector storage database for similar text segments."""
        collection = self.chroma_client.get_or_create_collection(
            name="botanical_knowledge"
        )
        query_vector = self.get_embedding(query_text, is_query=True)

        results = collection.query(
            query_embeddings=[query_vector], n_results=num_results
        )
        return results["documents"][0] if results["documents"] else []


if __name__ == "__main__":
    engine = ProductionGeminiEngine()
    engine.build_vector_store()