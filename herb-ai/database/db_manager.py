# cspell:disable
import os
import sqlite3
from typing import Tuple, Any

# FIX: Force DB_PATH to resolve as an absolute path relative to this file's location
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH: str = os.path.abspath(os.path.join(CURRENT_DIR, "telemetry.db"))


def init_db() -> None:
    """Initializes the SQLite database and ensures the schema layout is strictly structured."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Structure the target plants tracking directory index
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS plants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            species_name TEXT NOT NULL UNIQUE
        );
    """)

    # Structure spatial boundary logs and confidence thresholds
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS telemetry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plant_id INTEGER NOT NULL,
            frame_number INTEGER NOT NULL,
            xmin REAL NOT NULL,
            ymin REAL NOT NULL,
            xmax REAL NOT NULL,
            ymax REAL NOT NULL,
            confidence_score REAL NOT NULL,
            FOREIGN KEY (plant_id) REFERENCES plants(id)
        );
    """)

    conn.commit()
    conn.close()
    print(f"Database successfully initialized at: {DB_PATH}")


def add_new_plant(species_name: str) -> int:
    """Inserts a new plant species entry if it doesn't exist and returns its primary key."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM plants WHERE species_name = ?;", (species_name,))
    row = cursor.fetchone()

    if row:
        plant_id = int(row[0])
    else:
        cursor.execute("INSERT INTO plants (species_name) VALUES (?);", (species_name,))
        conn.commit()
        plant_id = int(cursor.lastrowid)

    conn.close()
    return plant_id


def insert_telemetry(
    plant_id: int,
    frame_number: int,
    bbox: Tuple[float, float, float, float],
    confidence_score: float,
) -> None:
    """Logs raw frame tracking metrics and spatial coordinates directly into SQLite tables."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    xmin, ymin, xmax, ymax = bbox
    cursor.execute(
        """
        INSERT INTO telemetry (plant_id, frame_number, xmin, ymin, xmax, ymax, confidence_score)
        VALUES (?, ?, ?, ?, ?, ?, ?);
    """,
        (plant_id, frame_number, xmin, ymin, xmax, ymax, confidence_score),
    )

    conn.commit()
    conn.close()
