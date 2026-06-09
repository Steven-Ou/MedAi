# cspell:disable
import os
import sys
from typing import List
from dotenv import load_dotenv
from google import genai
from google.genai import types
import chromadb

# Ensure project root is accessible for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

load_dotenv()

# FIX: Compute absolute database storage path relative to this script file location
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_DIR: str = os.path.abspath(os.path.join(CURRENT_DIR, "../../chroma_storage"))

class BotanicalQueryEngine:
    def __init__(self) -> None:
        """Initializes the GenAI Client and connects to the active Chroma vector store."""
        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY environment variable is missing.")
            
        self.client = genai.Client()
        # Uses the pristine absolute path structure
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        self.collection = self.chroma_client.get_collection(name="botanical_knowledge")

    def _get_query_embedding(self, text: str) -> List[float]:
        """Generates a query embedding matching the database dimensionality constraints."""
        try:
            response = self.client.models.embed_content(
                model="gemini-embedding-001",
                contents=text,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
            )
            if response.embeddings and response.embeddings[0].values:
                return [float(v) for v in response.embeddings[0].values]
            raise ValueError("Empty embedding vector returned from server backend.")
        except Exception as e:
            raise RuntimeError(f"Query embedding step failed: {e}")

    def query_botanical_knowledge(self, user_query: str, n_results: int = 1) -> str:
        """Retrieves semantically close text chunks and uses Gemini to synthesize a grounded answer."""
        try:
            query_vector = self._get_query_embedding(user_query)
            
            search_results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=n_results
            )
            
            documents = search_results.get("documents")
            retrieved_context = documents[0][0] if (documents and documents[0]) else "No relevant context found."
            
            system_instruction = (
                "You are Herb-AI, an expert medical botanical knowledge agent. "
                "Answer the user's question accurately using only the provided context material. "
                "If the context doesn't contain the answer, explicitly state that you don't have enough verified data."
            )
            
            prompt = f"Scanned Plant Context:\n{retrieved_context}\n\nUser Question: {user_query}"
            
            generation_response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.1,
                )
            )
            
            return generation_response.text or "Model failed to output a response string."
            
        except Exception as e:
            return f"Query Engine failure: {e}"

if __name__ == "__main__":
    engine = BotanicalQueryEngine()
    question = "What plant handles tension headaches and clears nasal congestion?"
    print(f"Asking Herb-AI: '{question}'\n")
    answer = engine.query_botanical_knowledge(question)
    print(f"Herb-AI Response:\n{answer}")