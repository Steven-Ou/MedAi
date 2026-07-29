import os
import sys
import sqlite3
import traceback
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure project root is accessible for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "herb-ai")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from database.db_manager import init_db, DB_PATH
from frontend.src.rag.query_engine import BotanicalQueryEngine
from frontend.src.vision.detector import BotanicalDetector
from frontend.src.vision.tracker import BotanicalTracker

app = FastAPI(title="Herb-AI Medical Botanical API Hub")
CURRENT_SESSION_PLANT = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Database Schema on API boot
init_db()

@app.get("/")
def read_root():
    return {"status": "Herb-AI Backend is running"}

class QueryRequest(BaseModel):
    query_text: str


def background_video_scan():
    """Runs the tracking pipeline cleanly via the BotanicalTracker object."""
    video_path = os.path.join(project_root, "data/processed/sample_garden_walk.mp4")
    model_path = os.path.join(project_root, "best.pt")
    
    if not os.path.exists(model_path) or not os.path.exists(video_path):
        print("Missing weights or video file for scanning.")
        return

    # Clear tables for fresh session
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM telemetry;")
    cursor.execute("DELETE FROM plants;")
    conn.commit()
    conn.close()

    # Execute the modular tracker
    tracker = BotanicalTracker(model_path=model_path)
    tracker.process_video(video_path, show_live_feed=False)

@app.post("/api/scan")
def trigger_scan(background_tasks: BackgroundTasks):
    background_tasks.add_task(background_video_scan)
    return {
        "status": "processing",
        "message": "Video scanning pipeline kicked off successfully.",
    }
    
    
# --- TELEMETRY ENDPOINT (Fixes the 404s) ---
@app.get("/api/telemetry")
def get_telemetry():
    if not os.path.exists(DB_PATH):
        return {"data": []}
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

    return {
        "data": [
            {
                "species": row[0],
                "framesTracked": row[1],
                "maxConfidence": round(row[2], 2),
            }
            for row in rows
        ]
    }

# --- DETECT ENDPOINT (Fixes the 422 and matches frontend requests) ---
@app.post("/api/detect")
async def upload_image_inference(file: UploadFile = File(...)):
    global CURRENT_SESSION_PLANT
    
    contents = await file.read()
    model_path = os.path.join(project_root, "best.pt")
    
    detector = BotanicalDetector(model_path=model_path)
    result = detector.analyze_image(contents)
    
    # Store for RAG context
    CURRENT_SESSION_PLANT = result.get("predicted_class")
    
    return {"status": "success", "results": result}

# --- RAG QUERY ENDPOINT ---
@app.post("/api/query")
def query_agent(payload: QueryRequest):
    print(f"🤖 Herb-AI: Starting analysis pipeline for question: {payload.query_text}")
    try:
        query_engine = BotanicalQueryEngine()
        augmented_query = payload.query_text
        
        if CURRENT_SESSION_PLANT and CURRENT_SESSION_PLANT != "Unidentified Anomaly":
            augmented_query = f"Context: We are discussing the plant '{CURRENT_SESSION_PLANT}'. User Question: {payload.query_text}"

        answer = query_engine.query_botanical_knowledge(augmented_query)
        return {"response": answer}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))