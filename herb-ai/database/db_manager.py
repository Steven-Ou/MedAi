import os
import psycopg2
from typing import Tuple, Optional
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.getenv("SUPABASE_DB_URL")


def get_conn():
    if not DB_URL:
        raise ValueError(
            "🚨 SUPABASE_DB_URL is missing! Check your .env file or Cloud Secrets."
        )
    return psycopg2.connect(DB_URL)


def init_db() -> None:
    conn = get_conn()
    cursor = conn.cursor()

    # Supabase Plants Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS plants (
            id SERIAL PRIMARY KEY,
            species_name TEXT NOT NULL UNIQUE
        );
    """)

    # Supabase LLM Cache Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ai_cache (
            query_text TEXT PRIMARY KEY,
            ai_response TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # Create the baseline table if it is a fresh database
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS telemetry (
            id SERIAL PRIMARY KEY,
            plant_id INTEGER NOT NULL REFERENCES plants(id),
            frame_number INTEGER NOT NULL,
            bbox_xmin REAL NOT NULL,
            ymin REAL NOT NULL,
            bbox_xmax REAL NOT NULL,
            ymax REAL NOT NULL,
            confidence_score REAL NOT NULL,
            evidence_image_url TEXT
        );
    """)

    # 🔥 BULLETPROOF FIX: Force Postgres to add the column to the existing table
    cursor.execute("""
        ALTER TABLE telemetry 
        ADD COLUMN IF NOT EXISTS session_id TEXT DEFAULT 'default_session';
    """)

    conn.commit()
    conn.close()
    print("✅ Supabase PostgreSQL successfully initialized!")


def add_new_plant(species_name: str) -> int:
    conn = get_conn()
    cursor = conn.cursor()

    # Postgres uses ON CONFLICT DO NOTHING instead of INSERT OR IGNORE
    cursor.execute(
        """
        INSERT INTO plants (species_name) 
        VALUES (%s) 
        ON CONFLICT (species_name) DO NOTHING 
        RETURNING id;
    """,
        (species_name,),
    )

    result = cursor.fetchone()
    if result:
        plant_id = result[0]
    else:
        cursor.execute(
            "SELECT id FROM plants WHERE species_name = %s;", (species_name,)
        )
        plant_id = cursor.fetchone()[0]

    conn.commit()
    conn.close()
    return plant_id


def clear_session_telemetry(session_id: str) -> None:
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM telemetry WHERE session_id = %s;", (session_id,))
    conn.commit()
    conn.close()


def insert_telemetry(
    session_id: str,
    plant_id: int,
    frame_number: int,
    bbox: Tuple[float, float, float, float],
    confidence_score: float,
    evidence_url: str = None,
) -> None:
    conn = get_conn()
    cursor = conn.cursor()
    xmin, ymin, xmax, ymax = bbox

    cursor.execute(
        """
        INSERT INTO telemetry (session_id, plant_id, frame_number, bbox_xmin, ymin, bbox_xmax, ymax, confidence_score, evidence_image_url)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
    """,
        (
            session_id,
            plant_id,
            frame_number,
            xmin,
            ymin,
            xmax,
            ymax,
            confidence_score,
            evidence_url,
        ),
    )
    conn.commit()
    conn.close()


def get_cached_response(query_text: str) -> Optional[str]:
    try:
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ai_response FROM ai_cache WHERE query_text = %s;", (query_text,)
        )
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None
    except Exception:
        return None


def save_to_cache(query_text: str, ai_response: str) -> None:
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO ai_cache (query_text, ai_response) 
        VALUES (%s, %s)
        ON CONFLICT (query_text) DO UPDATE SET ai_response = EXCLUDED.ai_response;
    """,
        (query_text, ai_response),
    )
    conn.commit()
    conn.close()
