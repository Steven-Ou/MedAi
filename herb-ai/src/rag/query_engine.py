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

class BotanicalQueryEngine:
    def __init__(self) -> None:
        """Initializes the GenAI Client and connects to the active Chroma vector store."""
        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY environment variable is missing.")
            
        self.client = genai.Client()
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        self.collection = self.chroma_client.get_collection(name="botanical_knowledge")

    def _get_database_connection(self) -> sqlite3.Connection:
        """Finds and returns a connection to the SQLite database by checking potential path paths."""
        # Check inside herb-ai/database/
        path_opts = [
            os.path.abspath(os.path.join(CURRENT_DIR, "../../database/botany_telemetry.db")),
            os.path.abspath(os.path.join(CURRENT_DIR, "../../../database/botany_telemetry.db")),
            os.path.abspath(os.path.join(os.getcwd(), "database/botany_telemetry.db")),
            os.path.abspath(os.path.join(os.getcwd(), "herb-ai/database/botany_telemetry.db"))
        ]
        
        for path in path_opts:
            if os.path.exists(path):
                # Verify it's a valid database with a plants table
                try:
                    conn = sqlite3.connect(path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='plants';")
                    if cursor.fetchone():
                        return conn
                    conn.close()
                except sqlite3.Error:
                    continue
                    
        # Fallback to default path if none exists yet
        default_path = os.path.abspath(os.path.join(CURRENT_DIR, "../../database/botany_telemetry.db"))
        os.makedirs(os.path.dirname(default_path), exist_ok=True)
        return sqlite3.connect(default_path)

    def _get_video_summary_context(self) -> str:
        """Queries the local SQL database to summarize what species the camera actually tracked."""
        try:
            conn = self._get_database_connection()
            cursor = conn.cursor()
            
            # Query the exact unique plants seen and find their frame counts
            cursor.execute("""
                SELECT p.species_name, COUNT(t.id), MAX(t.confidence_score)
                FROM plants p
                JOIN telemetry t ON p.id = t.plant_id
                GROUP BY p.species_name
            """)
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                return "The camera stream was processed, but no unique target plants have been saved to the tracking tables yet."
                
            summary = "REAL-TIME SCANNING SESSION DATA SUMMARY:\n"
            summary += "You recently ran a video scanning session. Here is what your computer vision system saw:\n"
            for row in rows:
                summary += f"- Identified '{row[0]}' in the video stream across {row[1]} separate frames, with a maximum confidence score of {row[2]:.2f}.\n"
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

    def query_botanical_knowledge(self, user_query: str, n_results: int = 2) -> str:
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
            retrieved_context = ""
            if documents and documents[0]:
                retrieved_context = "\n---\n".join([doc for doc in documents[0] if doc is not None])
            
            # Refined system prompt explaining exactly how it should answer meta-questions
            system_instruction = (
                "You are Herb-AI, an expert medical botanical knowledge agent assistant.\n"
                "You are provided with two streams of reference material:\n"
                "1. REAL-TIME SCANNING SESSION DATA SUMMARY (Shows what plants were actually seen in the video clip by the user).\n"
                "2. Scanned Plant Botanical Properties (The dictionary/textbook info about those plants).\n\n"
                "Use BOTH streams to answer the user's questions completely. If they ask what you saw, reference the video telemetry summary. "
                "If they ask general medical questions, use the botanical properties context. Always match the reality of what was seen with your knowledge base."
            )
            
            prompt = (
                f"{video_summary}\n\n"
                f"Scanned Plant Botanical Properties textbook data:\n{retrieved_context}\n\n"
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