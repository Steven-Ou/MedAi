# cspell:disable
import os
import sys
import time
import sqlite3
from typing import List, Any
from dotenv import load_dotenv
from google import genai
from google.genai import types
import chromadb

# Ensure project root is accessible for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

load_dotenv()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_DIR: str = os.path.abspath(os.path.join(CURRENT_DIR, "../../chroma_storage"))
# Locate your local SQLite telemetry file path
DB_PATH: str = os.path.abspath(os.path.join(CURRENT_DIR, "../../database/botany_telemetry.db"))

class BotanicalQueryEngine:
    def __init__(self) -> None:
        """Initializes the GenAI Client and connects to the active Chroma vector store."""
        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY environment variable is missing.")
            
        self.client = genai.Client()
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        self.collection = self.chroma_client.get_collection(name="botanical_knowledge")

    def _get_video_summary_context(self) -> str:
        """Queries the local SQL database to summarize what species the camera actually tracked."""
        if not os.path.exists(DB_PATH):
            return "No tracking telemetry recorded from active video runs yet."
        
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Query the exact unique plants seen and count how many frames they appeared in
            cursor.execute("""
                SELECT p.species_name, COUNT(t.id) 
                FROM plants p
                JOIN telemetry t ON p.id = t.plant_id
                GROUP BY p.species_name
            """)
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                return "The camera stream ran, but zero distinct plant species were verified."
                
            summary = "Real-Time Video Scan Session Analytics:\n"
            for row in rows:
                summary += f"- Detected '{row[0]}' across {row[1]} frames in this session.\n"
            return summary
        except Exception as e:
            return f"Could not read session telemetry: {e}"

    def _get_query_embedding_with_retry(self, text: str, retries: int = 3, delay: float = 2.0) -> List[float]:
        """Generates a query embedding with automatic retry logic to handle temporary 503 spikes."""
        for attempt in range(retries):
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
                if "503" in str(e) or "UNAVAILABLE" in str(e).upper():
                    if attempt < retries - 1:
                        time.sleep(delay)
                        delay *= 2
                        continue
                raise RuntimeError(f"Query embedding step failed: {e}")
        return []

    def query_botanical_knowledge(self, user_query: str, n_results: int = 1) -> str:
        """Retrieves semantically close text chunks and uses Gemini to synthesize a grounded answer."""
        try:
            # 1. Gather SQL tracking telemetry context
            video_summary = self._get_video_summary_context()
            
            # 2. Convert user question into a vector coordinate
            query_vector = self._get_query_embedding_with_retry(user_query)
            
            # 3. Search Chroma for matching profiles
            search_results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=n_results
            )
            
            documents = search_results.get("documents")
            retrieved_context = documents[0][0] if (documents and documents[0]) else "No relevant context found."
            
            system_instruction = (
                "You are Herb-AI, an expert medical botanical knowledge agent. "
                "You have access to both the textbook database context AND the actual tracking metrics from the video scan session. "
                "Answer the question accurately using these two sets of background information."
            )
            
            # 4. Inject both the Video Session Summary AND the Textbook profile information
            prompt = (
                f"{video_summary}\n"
                f"Scanned Plant Botanical Properties:\n{retrieved_context}\n\n"
                f"User Question: {user_query}"
            )
            
            gen_delay = 2.0
            for attempt in range(3):
                try:
                    generation_response = self.client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            temperature=0.2,
                        )
                    )
                    return generation_response.text or "Model failed to output a response string."
                except Exception as e:
                    if ("503" in str(e) or "UNAVAILABLE" in str(e).upper()) and attempt < 2:
                        time.sleep(gen_delay)
                        gen_delay *= 2
                        continue
                    raise e
            
            return "Failed to generate a response due to API constraints."
            
        except Exception as e:
            return f"Query Engine failure: {e}"

if __name__ == "__main__":
    engine = BotanicalQueryEngine()
    question = "What did you see in the video stream scan?"
    print(f"Asking Herb-AI: '{question}'\n")
    print(engine.query_botanical_knowledge(question))