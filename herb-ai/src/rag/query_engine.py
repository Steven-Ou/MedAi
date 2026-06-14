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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

load_dotenv()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_DIR: str = os.path.abspath(os.path.join(CURRENT_DIR, "../../chroma_storage"))

# FIX: Explicitly target the definitive database file location inside the database subfolder
DB_PATH: str = os.path.abspath(os.path.join(CURRENT_DIR, "../../database/telemetry.db"))

class BotanicalQueryEngine:
    def __init__(self) -> None:
        """Initializes the GenAI Client and connects to the active Chroma vector store."""
        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY environment variable is missing.")
            
        self.client = genai.Client()
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        self.collection = self.chroma_client.get_collection(name="botanical_knowledge")

    def _get_video_summary_context(self) -> str:
        """Queries the definitive local SQL database to summarize what species the camera actually tracked."""
        if not os.path.exists(DB_PATH):
            return "No tracking telemetry database file found on disk yet."
            
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Query unique plants seen and aggregate their logged frame count telemetry metrics
            cursor.execute("""
                SELECT p.species_name, COUNT(t.id), MAX(t.confidence_score)
                FROM plants p
                JOIN telemetry t ON p.id = t.plant_id
                GROUP p.species_name
            """)
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                return "The video scan ran, but no data entries are populated inside the tracking tables yet."
                
            summary = "REAL-TIME SCANNED VIDEO SESSION TELEMETRY SUMMARY:\n"
            summary += "You recently processed a video file with your computer vision model. Here is exactly what it tracked:\n"
            for row in rows:
                summary += f"- Identified and tracked the class '{row[0]}' in your video file across {row[1]} frames, with a maximum tracking confidence of {row[2]:.2f}.\n"
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
            video_summary = self._get_video_summary_context()
            query_vector = self._get_query_embedding_with_retry(user_query)
            
            search_results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=n_results
            )
            
            documents = search_results.get("documents")
            retrieved_context = ""
            if documents and documents[0]:
                retrieved_context = "\n---\n".join([doc for doc in documents[0] if doc is not None])
            
            system_instruction = (
                "You are Herb-AI, an expert medical botanical knowledge agent assistant.\n"
                "You are provided with two streams of reference material:\n"
                "1. REAL-TIME SCANNED VIDEO SESSION TELEMETRY SUMMARY (Shows what plants were seen in the video clip).\n"
                "2. Scanned Plant Botanical Properties textbook data (The textbook properties details about those plants).\n\n"
                "Use BOTH streams to accurately answer the user's questions completely. If they ask what you saw, directly reference your video telemetry summary data points."
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