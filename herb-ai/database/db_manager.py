import sqlite3
import os
from contextlib import contextmanager
from typing import Generator, Tuple, Optional

DB_PATH: str = os.path.join(os.path.dirname(__file__), "telemetry.db")

@contextmanager
def get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    """Context manager to handle safe database connections and closures."""
    conn: sqlite3.Connection = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
    finally:
        conn.close()

def init_db() -> None:
    """Initializes the relational database schema for video tracking telemetry."""
    with get_db_connection() as conn:
        cursor: sqlite3.Cursor = conn.cursor()
        
        # Table 1: Unique tracked plant entities
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tracked_plants (
                plant_id INTEGER PRIMARY KEY AUTOINCREMENT,
                species_name TEXT NOT NULL,
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        ''')

        # Table 2: Per-frame visual telemetry
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS video_telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plant_id INTEGER,
                frame_number INTEGER NOT NULL,
                bbox_xmin REAL NOT NULL,
                bbox_ymin REAL NOT NULL,
                bbox_xmax REAL NOT NULL,
                bbox_ymax REAL NOT NULL,
                confidence_score REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (plant_id) REFERENCES tracked_plants(plant_id) ON DELETE CASCADE
            );
        ''')
        conn.commit()
    print(f"Database successfully initialized at: {DB_PATH}")

def add_new_plant(species_name: str) -> int:
    """Inserts a newly detected unique plant entity and returns its ID."""
    with get_db_connection() as conn:
        cursor: sqlite3.Cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO tracked_plants (species_name) 
            VALUES (?);
        ''', (species_name,))
        conn.commit()
        last_id: Optional[int] = cursor.lastrowid
        return last_id if last_id is not None else 0

def update_plant_last_seen(plant_id: int) -> None:
    """Updates the last_seen timestamp when a tracking ID is active in a frame."""
    with get_db_connection() as conn:
        cursor: sqlite3.Cursor = conn.cursor()
        cursor.execute('''
            UPDATE tracked_plants 
            SET last_seen = CURRENT_TIMESTAMP 
            WHERE plant_id = ?;
        ''', (plant_id,))
        conn.commit()

def insert_telemetry(plant_id: int, frame_number: int, bbox: Tuple[float, float, float, float], confidence_score: float) -> None:
    """
    Logs spatial coordinates for a plant track at a specific video frame.
    bbox expected format: (xmin, ymin, xmax, ymax)
    """
    xmin, ymin, xmax, ymax = bbox
    with get_db_connection() as conn:
        cursor: sqlite3.Cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO video_telemetry (plant_id, frame_number, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, confidence_score)
            VALUES (?, ?, ?, ?, ?, ?, ?);
        ''', (plant_id, frame_number, float(xmin), float(ymin), float(xmax), float(ymax), confidence_score))
        conn.commit()
    
    # Keeping timestamps fresh as frames process
    update_plant_last_seen(plant_id)

if __name__ == "__main__":
    init_db()