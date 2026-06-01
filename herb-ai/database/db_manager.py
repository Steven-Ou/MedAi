import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "telemetry.db")

def init_db():
    """Initializes the relational database schema for video tracking telemetry."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Enable foreign key support in SQLite
    cursor.execute("PRAGMA foreign_keys = ON;")

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
    conn.close()
    print(f"Database successfully initialized at: {DB_PATH}")

if __name__ == "__main__":
    init_db()