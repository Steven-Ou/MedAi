import os
import chromadb
import httpx
from typing import List

# Ensure project root is accessible
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
CHROMA_DB_DIR = os.path.join(project_root, "chroma_storage")


class LocalVectorStoreEngine:
    def __init__(self) -> None:
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        self.collection = self.chroma_client.get_or_create_collection(
            name="botanical_knowledge"
        )

    def _get_local_embedding(self, text: str) -> List[float]:
        """Generates embedding using local Ollama nomic-embed-text to match ingest.py."""
        url = "http://localhost:11434/api/embeddings"
        payload = {"model": "nomic-embed-text", "prompt": text}
        try:
            with httpx.Client() as client:
                response = client.post(url, json=payload, timeout=30.0)
                if response.status_code == 200:
                    return response.json().get("embedding", [])
        except Exception as e:
            print(f"Local Embedding Error: {e}")
        return []

    def build_vector_store(self):
        """Rebuilds the Chroma vector store from local knowledge files."""
        text_dir = os.path.abspath(os.path.join(project_root, "../data/knowledge_base"))

        if not os.path.exists(text_dir):
            print(f"Textbook directory not found at {text_dir}, skipping vector build.")
            return

        documents = []
        metadatas = []
        ids = []

        for filename in os.listdir(text_dir):
            if filename.endswith(".txt"):
                with open(os.path.join(text_dir, filename), "r") as f:
                    content = f.read()
                    documents.append(content)
                    metadatas.append({"source": filename})
                    ids.append(filename)

        print(f"Building vector store with {len(documents)} documents...")
        for i, doc in enumerate(documents):
            embedding = self._get_local_embedding(doc)

            if not embedding:
                print(f"⚠️ Skipping document {ids[i]}: Could not generate embedding.")
                continue

            self.collection.upsert(
                ids=[ids[i]],
                embeddings=[embedding],
                documents=[doc],
                metadatas=[metadatas[i]],
            )
        print("Vector store build complete.")
