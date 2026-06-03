import os
import sys
from typing import List

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from google import genai  # Using Google's official new SDK directly

# Ensure project root is accessible for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Constants
KNOWLEDGE_BASE_DIR: str = "data/knowledge_base"
CHROMA_DB_DIR: str = "chroma_storage"

class DirectGeminiEmbeddings:
    """Custom embedding wrapper to bypass the broken LangChain-Google class."""
    def __init__(self) -> None:
        # Client automatically picks up GEMINI_API_KEY from environment variables
        self.client = genai.Client()
        # Using a model identifier format that bypasses the v1beta prefixing conflict
        self.model_name = "text-embedding-004"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds a list of string chunks using the direct Google SDK."""
        embeddings_list = []
        for text in texts:
            # We wrap the call cleanly to force native route processing
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=text
            )
            # Extract the raw float array from the response object wrapper
            vector = response.embeddings[0].values
            embeddings_list.append([float(val) for val in vector])
        return embeddings_list

    def embed_query(self, text: str) -> List[float]:
        """Embeds a single user query string."""
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=text
        )
        return [float(val) for val in response.embeddings[0].values]

class KnowledgeBaseEngine:
    def __init__(self) -> None:
        """Initializes our custom direct embedding engine."""
        self.embeddings = DirectGeminiEmbeddings()

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
        
        # Build Chroma database by passing our direct custom embedding class instance
        self.vector_store = Chroma.from_documents(
            documents=doc_chunks,
            embedding=self.embeddings,
            persist_directory=CHROMA_DB_DIR
        )
        print("Vector database built successfully using direct Gemini SDK hooks!")

    def query_knowledge(self, query_text: str, num_results: int = 2) -> List:
        """Performs a similarity vector search against the stored botanical text."""
        vector_db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=self.embeddings)
        results = vector_db.similarity_search(query_text, k=num_results)
        return results

if __name__ == "__main__":
    engine = KnowledgeBaseEngine()
    engine.build_vector_store()