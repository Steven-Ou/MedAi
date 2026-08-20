# herb-ai/src/rag/query_engine.py
# cspell:disable
import os
import sys
import time
import sqlite3
from typing import List, Any
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai
import httpx
import chromadb

# Ensure project root is accessible for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Import the new cache functions from your db_manager
from database.db_manager import get_cached_response, save_to_cache

load_dotenv()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_DIR: str = os.path.abspath(
    os.path.join(CURRENT_DIR, "../../../chroma_storage")
)
DB_PATH: str = os.path.abspath(
    os.path.join(CURRENT_DIR, "../../../database/telemetry.db")
)


class BotanicalQueryEngine:
    def __init__(self) -> None:
        """Initializes the GenAI Client and connects to the active Chroma vector store."""

        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../")
        )
        self.chroma_client = chromadb.PersistentClient(
            path=os.path.join(project_root, "chroma_storage")
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="botanical_knowledge"
        )

        # A simple array to hold the conversation history
        self.chat_history = []

        self.gemini_client = (
            genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            if os.getenv("GEMINI_API_KEY")
            else None
        )

    def _get_unified_session_context(self) -> str:
        """Queries local telemetry tables and recent upload directories to build a merged context window."""
        summary = "ACTIVE SESSION TELEMETRY & MULTIMODAL SNAPSHOT SUMMARY:\n"
        has_context = False

        # 1. Inspect recent unlearned photo upload snapshots on disk
        unlearned_dir = os.path.abspath(
            os.path.join(CURRENT_DIR, "../../data/clean_training/unlearned")
        )
        recent_uploads = []
        if os.path.exists(unlearned_dir):
            files = [f for f in os.listdir(unlearned_dir) if f.endswith(".jpg")]
            if files:
                # Sort by timestamp to catch the most recent file upload
                files.sort(
                    key=lambda x: os.path.getmtime(os.path.join(unlearned_dir, x)),
                    reverse=True,
                )
                for f in files[:2]:  # Grab up to the 2 latest snapshots
                    parts = f.replace(".jpg", "").split("_")
                    if len(parts) > 1:
                        clean_name = " ".join(parts[1:]).title()
                        recent_uploads.append(clean_name)
                    else:
                        recent_uploads.append(f.replace(".jpg", "").title())

        if recent_uploads:
            summary += f"[⚠️ RECENT SNAPSHOT IMAGE UPLOADS]: The user just uploaded static images to the workspace. Your vision pipeline analyzed them and identified them as: {', '.join(recent_uploads)}.\n"
            has_context = True

        # 2. Extract companion metrics from your camera's live video tracking tables
        if os.path.exists(DB_PATH):
            try:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT p.species_name, COUNT(t.id), MAX(t.confidence_score)
                    FROM plants p
                    JOIN telemetry t ON p.id = t.plant_id
                    GROUP BY p.species_name
                """)
                rows = cursor.fetchall()
                conn.close()

                if rows:
                    summary += "🎥 [VIDEO SCAN TELEMETRY RECORDS]:\n"
                    for row in rows:
                        summary += f"- Logged class '{row[0]}' across {row[1]} moving video frames (Max Confidence: {row[2]:.2f}).\n"
                    has_context = True
            except Exception:
                pass

        if not has_context:
            return "No recent video telemetry scans or photograph uploads have been captured in this current interface session."

        return summary

    model = SentenceTransformer("all-mpnet-base-v2")

    def _get_query_embedding_with_retry(self, text: str) -> List[float]:
        """Generates a query embedding using local SentenceTransformer."""
        try:
            return self.model.encode(text).tolist()
        except Exception as e:
            print(f"Embedding error: {e}")
            return []

    def query_botanical_knowledge(self, user_query: str, n_results: int = 6) -> str:
        """Retrieves textbook reference vectors and synthesizes an answer using local Ollama."""
        try:
            self.chat_history.append(f"User: {user_query}")

            cached_answer = get_cached_response(user_query)
            if cached_answer:
                print("⚡ Cache hit! Returning saved answer instantly.")
                self.chat_history.append(f"Herb-AI: {cached_answer}")
                return cached_answer

            print("🔄 Cache miss. Proceeding with vector search...")
            session_context = self._get_unified_session_context()
            query_vector = self._get_query_embedding_with_retry(user_query)

            search_results = self.collection.query(
                query_embeddings=[query_vector], n_results=n_results
            )

            documents = search_results.get("documents")
            retrieved_context = (
                "\n---\n".join([doc for doc in documents[0] if doc is not None])
                if documents and documents[0]
                else "No relevant textbook data found."
            )

            history_str = "\n".join(self.chat_history[-4:])

            prompt = (
                f"You are Herb-AI, an expert medical botanical vision agent.\n"
                "CRITICAL RULES:\n"
                "1. You ARE a multimodal vision agent. Use the 'Session Context' below to know exactly what plants you just identified in the user's video or image.\n"
                "2. Answer the User's question using ONLY the Textbook Context provided below.\n"
                "3. If the textbook context is empty or doesn't contain the answer, say: 'I do not have enough specific clinical data in my knowledge base to answer that yet.'\n\n"
                f"--- SESSION CONTEXT ---\n{session_context}\n\n"
                f"--- TEXTBOOK CONTEXT ---\n{retrieved_context}\n\n"
                f"--- CONVERSATION HISTORY ---\n{history_str}\n\n"
                f"User Question: {user_query}\n"
                f"Answer clearly and focus on clinical benefits."
            )

            # NEW: Cascade to Gemini first to avoid the 5-minute HTTP timeout
            if self.gemini_client:
                try:
                    response = self.gemini_client.models.generate_content(
                        model="gemini-3.5-flash", contents=prompt
                    )
                    answer = response.text.strip()
                    self.chat_history.append(f"Herb-AI: {answer}")
                    save_to_cache(user_query, answer)
                    return answer
                except Exception as e:
                    print(
                        f"⚠️ Gemini RAG generation failed: {e}. Falling back to Ollama..."
                    )

            # FALLBACK: Existing Ollama implementation
            ollama_url = "http://localhost:11434/api/generate"
            payload = {
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3},
            }

            with httpx.Client() as client:
                response = client.post(ollama_url, json=payload, timeout=300.0)
                if response.status_code == 200:
                    answer = response.json().get("response", "").strip()
                    self.chat_history.append(f"Herb-AI: {answer}")
                    save_to_cache(user_query, answer)
                    return answer
                else:
                    return f"Ollama Error: Status {response.status_code}"

        except Exception as e:
            return f"Query Engine failure: {e}"


if __name__ == "__main__":
    engine = BotanicalQueryEngine()
    question = "What was the name of the herb photo I just uploaded?"
    print(engine.query_botanical_knowledge(question))
