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
from openai import OpenAI
import httpx
import chromadb
import json

# Ensure project root is accessible for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Import the new cache functions from your db_manager
from database.db_manager import get_cached_response, save_to_cache
from frontend.src.rag.know_gen import AutoKnowledgeGenerator
from frontend.src.rag.vector_store import LocalVectorStoreEngine

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

        # A simple array to hold the conversation history
        self.chat_history = []

        self.gemini_client = (
            genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            if os.getenv("GEMINI_API_KEY")
            else None
        )
        self.openai_client = (
            OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            if os.getenv("OPENAI_API_KEY")
            else None
        )
        self.groq_client = (
            OpenAI(
                api_key=os.getenv("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1",
            )
            if os.getenv("GROQ_API_KEY")
            else None
        )

    def get_collection(self):
        return self.chroma_client.get_or_create_collection(name="botanical_knowledge")

    def _get_unified_session_context(self, session_id: str = "default_session") -> str:
        """Queries local telemetry tables to build a merged context window strictly for this user's session."""
        summary = "ACTIVE SESSION TELEMETRY & MULTIMODAL SNAPSHOT SUMMARY:\n"
        has_context = False

        try:
            from database.db_manager import get_conn

            conn = get_conn()
            cursor = conn.cursor()
            cursor.execute(
                """
                    SELECT p.species_name, COUNT(t.id), MAX(t.confidence_score)
                    FROM plants p
                    JOIN telemetry t ON p.id = t.plant_id
                    WHERE t.session_id = %s
                    GROUP BY p.species_name
                    ORDER BY MAX(t.id) DESC
                """,
                (session_id,),
            )
            rows = cursor.fetchall()
            conn.close()

            if rows:
                summary += (
                    "🎥 [SESSION TELEMETRY RECORDS] (Ordered newest to oldest):\n"
                )
                for i, row in enumerate(rows):
                    # Flag the newest item for the LLM
                    label = "[MOST RECENTLY SCANNED] " if i == 0 else ""
                    summary += f"- {label}Logged class '{row[0]}' across {row[1]} moving video frames (Max Confidence: {row[2]:.2f}).\n"
                has_context = True
        except Exception:
            pass

        if not has_context:
            return "No recent video telemetry scans or photograph uploads have been captured in this current interface session."

        return summary

    model = SentenceTransformer("all-mpnet-base-v2")

    def _ensure_knowledge_exists(self, user_query: str, session_id: str):
        """Checks if the user is asking about a detected plant and generates its profile JIT."""
        try:
            from database.db_manager import get_conn

            conn = get_conn()
            cursor = conn.cursor()
            # Grab all plants seen in this session, ordered by highest confidence
            cursor.execute(
                "SELECT p.species_name FROM plants p JOIN telemetry t ON p.id = t.plant_id WHERE t.session_id = %s GROUP BY p.species_name ORDER BY MAX(t.confidence_score) DESC",
                (session_id,),
            )
            rows = cursor.fetchall()
            conn.close()

            rebuild_needed = False
            kg = AutoKnowledgeGenerator()

            plants_to_generate = []

            # 1. Check if specific plants were explicitly asked about
            for row in rows:
                if row[0].lower() in user_query.lower():
                    plants_to_generate.append(row[0])

            # 2. Fallback: If no specific plant is mentioned, generate the top 2 most confident plants
            if not plants_to_generate and rows:
                plants_to_generate = [row[0] for row in rows[:2]]

            for plant_name in plants_to_generate:
                if kg.generate_profile_if_new(plant_name):
                    print(f"📝 Just-In-Time knowledge synced for: {plant_name}")
                    rebuild_needed = True

            if rebuild_needed:
                LocalVectorStoreEngine().build_vector_store()

        except Exception as e:
            print(f"⚠️ JIT Generation failed: {e}")

    def _get_query_embedding_with_retry(self, text: str) -> List[float]:
        """Generates a query embedding using local SentenceTransformer."""
        try:
            return self.model.encode(text).tolist()
        except Exception as e:
            print(f"Embedding error: {e}")
            return []

    def query_botanical_knowledge(
        self, user_query: str, session_id: str = "default_session", n_results: int = 6
    ) -> str:
        """Retrieves textbook reference vectors and synthesizes an answer using local Ollama."""
        try:
            self._ensure_knowledge_exists(user_query, session_id)

            self.chat_history.append(f"User: {user_query}")

            print("🔄 Cache miss. Proceeding with vector search...")
            session_context = self._get_unified_session_context(session_id=session_id)
            query_vector = self._get_query_embedding_with_retry(user_query)

            collection = self.get_collection()
            search_results = collection.query(
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
                "You are Herb-AI, an expert medical botanical vision agent. Answer the user's question conversationally.\n"
                "CRITICAL RULES:\n"
                "1. Answer the User Question using ONLY the information provided in the [Clinical Data] and [Vision Telemetry] blocks below.\n"
                "2. DO NOT echo or repeat the raw context block headers. Synthesize the answer naturally.\n"
                "3. If the user uses pronouns like 'it' or 'this', they are referring to the [MOST RECENTLY SCANNED] plant in the Vision Telemetry.\n"
                "4. Format your response beautifully using Markdown. Use tables, bolding, and bullet points where appropriate.\n"
                "5. If you do not have enough specific clinical data, state: 'I do not have enough textbook data on that specific topic.'\n\n"
                f"[Vision Telemetry]\n{session_context}\n\n"
                f"[Clinical Data]\n{retrieved_context}\n\n"
                f"[Conversation History]\n{history_str}\n\n"
                f"User Question: {user_query}\n"
                "Herb-AI Answer:"
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

            if self.groq_client or self.openai_client:
                client_to_use = self.groq_client or self.openai_client
                model_to_use = (
                    "llama-3.1-8b-instant" if self.groq_client else "gpt-4o-mini"
                )
                try:
                    response = client_to_use.chat.completions.create(
                        model=model_to_use,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                    )
                    answer = response.choices[0].message.content.strip()
                    self.chat_history.append(f"Herb-AI: {answer}")
                    save_to_cache(user_query, answer)
                    return answer
                except Exception as e:
                    print(
                        f"⚠️ OpenAI/Groq generation failed: {e}. Falling back to Ollama..."
                    )

            # FALLBACK: Existing Ollama implementation
            ollama_url = "http://localhost:11434/api/generate"
            payload = {
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_ctx": 1024, "num_thread": 4},
            }

            with httpx.Client() as client:
                response = client.post(ollama_url, json=payload, timeout=900.0)
                if response.status_code == 200:
                    answer = response.json().get("response", "").strip()
                    self.chat_history.append(f"Herb-AI: {answer}")
                    save_to_cache(user_query, answer)
                    return answer
                else:
                    return f"Ollama Error: Status {response.status_code}"

        except Exception as e:
            return f"Query Engine failure: {e}"

    def stream_botanical_knowledge(
        self, user_query: str, session_id: str = "default_session", n_results: int = 6
    ):
        """Streams the response chunk-by-chunk for the frontend typing effect."""

        self._ensure_knowledge_exists(user_query, session_id)

        self.chat_history.append(f"User: {user_query}")

        session_context = self._get_unified_session_context(session_id=session_id)
        query_vector = self._get_query_embedding_with_retry(user_query)

        collection = self.get_collection()
        search_results = collection.query(
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
            "You are Herb-AI, an expert medical botanical vision agent. Answer the user's question conversationally.\n"
            "CRITICAL RULES:\n"
            "1. Answer the User Question using ONLY the information provided in the [Clinical Data] and [Vision Telemetry] blocks below.\n"
            "2. DO NOT echo or repeat the raw context block headers. Synthesize the answer naturally.\n"
            "3. If the user uses pronouns like 'it' or 'this', they are referring to the [MOST RECENTLY SCANNED] plant in the Vision Telemetry.\n"
            "4. Format your response beautifully using Markdown. Use tables, bolding, and bullet points where appropriate.\n"
            "5. If you do not have enough specific clinical data, state: 'I do not have enough textbook data on that specific topic.'\n\n"
            f"[Vision Telemetry]\n{session_context}\n\n"
            f"[Clinical Data]\n{retrieved_context}\n\n"
            f"[Conversation History]\n{history_str}\n\n"
            f"User Question: {user_query}\n"
            "Herb-AI Answer:"
        )

        full_answer = ""

        # 1. Gemini Streaming
        if self.gemini_client:
            try:
                response = self.gemini_client.models.generate_content_stream(
                    model="gemini-3.5-flash", contents=prompt
                )
                for chunk in response:
                    if chunk.text:
                        full_answer += chunk.text
                        yield chunk.text
                self.chat_history.append(f"Herb-AI: {full_answer}")
                save_to_cache(user_query, full_answer)
                return
            except Exception as e:
                print(f"⚠️ Gemini stream failed: {e}. Falling back to Ollama...")

        if self.groq_client or self.openai_client:
            client_to_use = self.groq_client or self.openai_client
            model_to_use = (
                "llama-3.3-70b-versatile" if self.groq_client else "gpt-4o-mini"
            )

            try:
                response_stream = client_to_use.chat.completions.create(
                    model=model_to_use,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                    temperature=0.3,
                )
                for chunk in response_stream:
                    if chunk.choices[0].delta.content:
                        text_chunk = chunk.choices[0].delta.content
                        full_answer += text_chunk
                        yield text_chunk

                self.chat_history.append(f"Herb-AI: {full_answer}")
                save_to_cache(user_query, full_answer)
                return
            except Exception as e:
                print(f"⚠️ OpenAI/Groq stream failed: {e}. Falling back to Ollama...")

        # 2. Ollama Fallback Streaming
        import json

        ollama_url = "http://localhost:11434/api/generate"
        payload = {
            "model": "llama3.2",
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": 0.3},
        }

        with httpx.Client() as client:
            with client.stream(
                "POST", ollama_url, json=payload, timeout=300.0
            ) as response:
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        chunk_text = data.get("response", "")
                        full_answer += chunk_text
                        yield chunk_text

        self.chat_history.append(f"Herb-AI: {full_answer}")
        save_to_cache(user_query, full_answer)


if __name__ == "__main__":
    engine = BotanicalQueryEngine()
    question = "What was the name of the herb photo I just uploaded?"
    print(engine.query_botanical_knowledge(question))
