import os
import sys
import sqlite3
import traceback
import shutil
import tempfile
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
    query_text: str = None
    question: str = None


# --- TELEMETRY ENDPOINT ---
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


# --- IMAGE DETECTION (Supports both /api/detect and /api/upload-image) ---
async def handle_image_upload(file: UploadFile):
    global CURRENT_SESSION_PLANT
    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400, detail="No image file provided. Please select a file."
        )

    model_path = os.path.join(project_root, "best.pt")

    detector = BotanicalDetector(model_path=model_path)
    result = detector.analyze_image(contents)

    CURRENT_SESSION_PLANT = result.get("predicted_class")
    return {
        "status": "success",
        "predicted_class": result.get("predicted_class"),
        "confidence": result.get("confidence"),
        "results": result,
    }


@app.post("/api/detect")
async def detect_alias(file: UploadFile = File(...)):
    return await handle_image_upload(file)


@app.post("/api/upload-image")
async def upload_image_alias(file: UploadFile = File(...)):
    return await handle_image_upload(file)


# --- CHAT & RAG QUERY (Supports both /api/query and /api/chat) ---
def process_query_text(text: str):
    print(f"🤖 Herb-AI: Starting analysis pipeline for question: {text}")
    try:
        query_engine = BotanicalQueryEngine()
        augmented_query = text
        
        # Override the LLM's natural refusal by explicitly telling it what it "saw"
        if CURRENT_SESSION_PLANT and CURRENT_SESSION_PLANT != "Unidentified Anomaly":
            augmented_query = (
                f"System Context: You are Herb-AI's advanced vision agent. "
                f"You just successfully analyzed the user's uploaded media (image/video) "
                f"and visually detected the plant '{CURRENT_SESSION_PLANT}'. "
                f"Use this context to directly answer their question. "
                f"User Question: {text}"
            )

        answer = query_engine.query_botanical_knowledge(augmented_query)
        return {"response": answer, "answer": answer}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query")
def query_alias(payload: QueryRequest):
    q = payload.query_text or payload.question
    return process_query_text(q)


@app.post("/api/chat")
def chat_alias(payload: QueryRequest):
    q = payload.query_text or payload.question
    return process_query_text(q)


# --- VIDEO SCANNING BACKGROUND TASK ---
def background_video_scan(video_path: str):
    """Runs the tracking pipeline cleanly via the BotanicalTracker object on any dynamic video."""
    model_path = os.path.join(project_root, "best.pt")
    
    if not os.path.exists(model_path) or not os.path.exists(video_path):
        print(f"Missing weights or video file for scanning. Path: {video_path}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM telemetry;")
    cursor.execute("DELETE FROM plants;")
    conn.commit()
    conn.close()

    tracker = BotanicalTracker(model_path=model_path)
    tracker.process_video(video_path, show_live_feed=False)

@app.post("/api/scan")
async def trigger_scan(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    
    with open(temp_video.name, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    background_tasks.add_task(background_video_scan, temp_video.name)
    
    return {
        "status": "processing",
        "message": f"Video scanning pipeline kicked off successfully for {file.filename}.",
    }