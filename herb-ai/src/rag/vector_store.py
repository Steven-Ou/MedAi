import os
import sys
from typing import List

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
import google.generativeai as genai  # Switching to the classic, stable SDK engine

# Ensure project root is accessible for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Constants
KNOWLEDGE_BASE_DIR: str = "data/knowledge_base"
CHROMA_DB_DIR: str = "chroma_storage"

class LegacyGeminiChromaEngine:
    def __init__(self) -> None:
        """Initializes the stable legacy Google Generative AI configurations."""
        # Grab the key directly from environment variables
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is missing!")
        
        genai.configure(api_key=api_key)
        # The legacy engine expects 'models/text-embedding-004' directly and processes it safely
        self.model_name = "models/text-embedding-004"
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    def get_embedding(self, text: str) -> List[float]:
        """Directly computes vectors using the stable legacy generative AI SDK endpoint."""
        response = genai.embed_content(
            model=self.model_name,
            content=text,
            task_type="retrieval_document"
        )
        return [float(val) for val in response["embedding"]]

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
            
        print("Vector database built successfully using legacy Chroma & stable Google SDK paths!")

    def query_knowledge(self, query_text: str, num_results: int = 2) -> List[str]:
        """Queries the local vector storage database for similar text segments."""
        collection = self.chroma_client.get_or_create_collection(name="botanical_knowledge")
        
        # Embed query text using the appropriate retrieval task type
        response = genai.embed_content(
            model=self.model_name,
            content=query_text,
            task_type="retrieval_query"
        )
        query_vector = [float(val) for val in response["embedding"]]
        
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=num_results
        )
        return results['documents'][0] if results['documents'] else []

if __name__ == "__main__":
    engine = LegacyGeminiChromaEngine()
    engine.build_vector_store()