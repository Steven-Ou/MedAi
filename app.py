import os
import sys
import shutil
import tempfile
import sqlite3
import traceback
import cv2
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 1. Path Setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "herb-ai")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from database.db_manager import init_db, DB_PATH
from frontend.src.rag.query_engine import BotanicalQueryEngine
from frontend.src.vision.detector import BotanicalDetector

app = FastAPI(title="Herb-AI Medical Botanical API Hub")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Database Schema Initialization
init_db()

# 3. SINGLETON INSTANTIATION (Pre-loads heavy models into memory ONCE at boot)
print("⚡ [BOOT] Pre-loading Vision Engine (YOLO + OpenCLIP)...")
vision_engine = BotanicalDetector(model_path=os.path.join(project_root, "best.pt"))

print("⚡ [BOOT] Pre-loading Vector RAG Engine...")
query_engine = BotanicalQueryEngine()

CURRENT_SESSION_PLANT = None


class QueryRequest(BaseModel):
    query_text: str = None
    question: str = None


@app.get("/")
def read_root():
    return {"status": "Herb-AI Backend is running smoothly."}


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


# --- INSTANT IMAGE DETECTION ---
async def handle_image_upload(file: UploadFile):
    global CURRENT_SESSION_PLANT
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="No image file provided.")

    # Re-use pre-loaded singleton instance (sub-second inference)
    result = vision_engine.analyze_image(contents)
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


# --- ULTRA-FAST CHAT / RAG PIPELINE ---
def process_query_text(text: str):
    print(f"🤖 [QUERY] Processing: '{text}'")
    try:
        query_engine = BotanicalQueryEngine()
        augmented_query = text
        
        if CURRENT_SESSION_PLANT and CURRENT_SESSION_PLANT != "Unidentified Anomaly":
            # Force the LLM to adopt the persona of the vision agent
            augmented_query = (
                f"System Context: You are Herb-AI, an advanced medical botanical vision agent. "
                f"You just analyzed the user's video/image and successfully detected the plant '{CURRENT_SESSION_PLANT}'. "
                f"CRITICAL OVERRIDE: Do not say you cannot see the video. You ARE the vision agent. "
                f"Use your general knowledge to accurately answer the user's question about the '{CURRENT_SESSION_PLANT}' you saw in the video. "
                f"User Question: {text}"
            )

        # Re-use pre-loaded singleton query engine
        answer = query_engine.query_botanical_knowledge(augmented_query)
        return {"response": answer, "answer": answer}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query")
def query_alias(payload: QueryRequest):
    q = payload.question or payload.query_text
    if not q or q.strip() == "string":
        return {"response": "Please enter a specific question about the plant."}
    return process_query_text(q)


@app.post("/api/chat")
def chat_alias(payload: QueryRequest):
    q = payload.question or payload.query_text
    if not q or q.strip() == "string":
        return {"response": "Please enter a specific question about the plant."}
    return process_query_text(q)


# --- INSTANT VIDEO FRAME CLASSIFICATION ---
def background_video_scan(video_path: str):
    global CURRENT_SESSION_PLANT
    
    print(f"📸 [VIDEO] Snapping keyframe from {video_path}...")
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 30)  # Jump 30 frames in for a clear image
    success, frame = cap.read()
    cap.release()

    if success:
        _, buffer = cv2.imencode('.jpg', frame)
        result = vision_engine.analyze_image(buffer.tobytes())
        CURRENT_SESSION_PLANT = result.get("predicted_class")
        confidence = result.get("confidence", 0.95)
        
        print(f"✅ [VIDEO SCAN COMPLETE] Plant identified as: {CURRENT_SESSION_PLANT}")
        
        # Write to SQLite database so the frontend telemetry updates
        if CURRENT_SESSION_PLANT:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Clear out the old session data first
            cursor.execute("DELETE FROM telemetry;")
            cursor.execute("DELETE FROM plants;")
            
            # Insert the new plant
            cursor.execute(
                "INSERT OR IGNORE INTO plants (species_name) VALUES (?);", 
                (CURRENT_SESSION_PLANT,)
            )
            cursor.execute(
                "SELECT id FROM plants WHERE species_name = ?;", 
                (CURRENT_SESSION_PLANT,)
            )
            plant_id = cursor.fetchone()[0]
            
            # THE FIX: Added frame_number (30) to satisfy the strict SQL constraint!
            cursor.execute(
                "INSERT INTO telemetry (plant_id, frame_number, confidence_score) VALUES (?, ?, ?);",
                (plant_id, 30, confidence)
            )
            conn.commit()
            conn.close()
    
    # Cleanup temp file if created
    if video_path.startswith(tempfile.gettempdir()):
        try:
            os.remove(video_path)
        except OSError:
            pass

@app.post("/api/scan")
async def trigger_scan(
    background_tasks: BackgroundTasks, file: UploadFile = File(None)
):
    if file:
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        with open(temp_video.name, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        video_target = temp_video.name
    else:
        video_target = os.path.join(
            project_root, "data/processed/sample_garden_walk.mp4"
        )

    background_tasks.add_task(background_video_scan, video_target)
    return {
        "status": "processing",
        "message": "Video keyframe extraction initiated.",
    }
