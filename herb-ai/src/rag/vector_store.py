import os
import sys
from typing import List

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from google import genai

# Ensure project root is accessible for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Constants
KNOWLEDGE_BASE_DIR: str = "data/knowledge_base"
CHROMA_DB_DIR: str = "chroma_storage"


class NativeGeminiChromaEngine:
    def __init__(self) -> None:
        """Initializes direct Google GenAI SDK client and local Chroma client."""
        self.ai_client = genai.Client()
        self.model_name = "text-embedding-004"
        # Persistent local Chroma database client instance
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    def get_embedding(self, text: str) -> List[float]:
        """Directly computes vector embeddings using Google GenAI SDK."""
        response = self.ai_client.models.embed_content(
            model=self.model_name, contents=text
        )
        return [float(val) for val in response.embeddings[0].values]

    def build_vector_store(self) -> None:
        """Loads text files, cuts into chunks, and populates the native Chroma collection."""
        if not os.path.exists(KNOWLEDGE_BASE_DIR):
            print(f"Creating empty knowledge directory at: {KNOWLEDGE_BASE_DIR}")
            os.makedirs(KNOWLEDGE_BASE_DIR)
            return

        print(f"Loading reference articles from '{KNOWLEDGE_BASE_DIR}'...")
        loader = DirectoryLoader(
            KNOWLEDGE_BASE_DIR, glob="**/*.txt", loader_cls=TextLoader
        )
        documents = loader.load()

        if not documents:
            print("No text documents found to parse.")
            return

        print(
            f"Successfully loaded {len(documents)} document(s). Splitting into semantic chunks..."
        )
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        doc_chunks = text_splitter.split_documents(documents)
        print(f"Created {len(doc_chunks)} distinct document text chunks.")

        print("Initializing Chroma Collection...")
        # Get or create a clean storage collection inside Chroma
        collection = self.chroma_client.get_or_create_collection(
            name="botanical_knowledge"
        )

        print(f"Generating embeddings and indexing directly into: {CHROMA_DB_DIR}...")
        for idx, chunk in enumerate(doc_chunks):
            text_content = chunk.page_content
            # Generate the vector array natively
            vector = self.get_embedding(text_content)

            # Extract clean string metadata
            source_file = chunk.metadata.get("source", "unknown")

            # Add vectors, payloads, and structural IDs directly to database store
            collection.add(
                embeddings=[vector],
                documents=[text_content],
                metadatas=[{"source": source_file}],
                ids=[f"doc_chunk_{idx}"],
            )

        print(
            "Vector database built successfully using native Chroma & direct Google SDK hooks!"
        )

    def query_knowledge(self, query_text: str, num_results: int = 2) -> List[str]:
        """Queries the local vector storage database for similar text segments."""
        collection = self.chroma_client.get_or_create_collection(
            name="botanical_knowledge"
        )
        query_vector = self.get_embedding(query_text)

        results = collection.query(
            query_embeddings=[query_vector], n_results=num_results
        )
        # Returns the matched string components
        return results["documents"][0] if results["documents"] else []


if __name__ == "__main__":
    engine = NativeGeminiChromaEngine()
    engine.build_vector_store()
