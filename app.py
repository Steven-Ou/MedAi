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
from ultralytics import YOLO

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
    model_path = os.path.join(project_root, "best.pt")
    
    # Load the model directly 
    model = YOLO(model_path)
    
    print(f"🎬 [VIDEO TRACKING] Initiating full-court frame-by-frame analysis...")
    
    # 1. Setup OpenCV Video Writer to actively draw and save the green boxes
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    
    # We will save the tracked video to a temporary output path first
    output_path = video_path.replace(".mp4", "_tracked.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    # 2. Run YOLO's native tracker (persist=True keeps IDs locked onto the object)
    results = model.track(source=video_path, stream=True, persist=True, conf=0.5)
    
    # 3. Prepare the Database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM telemetry;")
    cursor.execute("DELETE FROM plants;")
    
    detected_plants = {}
    frame_number = 0
    
    # 4. The Analytics Loop
    for r in results:
        frame_number += 1
        
        # r.plot() generates the image frame WITH the green bounding boxes and tracking IDs!
        annotated_frame = r.plot()
        out.write(annotated_frame)
        
        # If YOLO successfully locked onto an object and assigned an ID in this frame
        if r.boxes is not None and r.boxes.id is not None:
            for box, track_id, cls, conf in zip(r.boxes.xyxy, r.boxes.id, r.boxes.cls, r.boxes.conf):
                plant_name = model.names[int(cls)]
                
                # Keep a running tally of what we see the most
                detected_plants[plant_name] = detected_plants.get(plant_name, 0) + 1
                
                # Insert Telemetry
                cursor.execute("INSERT OR IGNORE INTO plants (species_name) VALUES (?);", (plant_name,))
                cursor.execute("SELECT id FROM plants WHERE species_name = ?;", (plant_name,))
                plant_id = cursor.fetchone()[0]
                
                cursor.execute(
                    "INSERT INTO telemetry (plant_id, frame_number, confidence_score) VALUES (?, ?, ?);",
                    (plant_id, frame_number, float(conf))
                )
    
    # 5. Cleanup and Save
    out.release()
    cap.release()
    conn.commit()
    conn.close()
    
    # Replace the original video with the new one so your frontend displays the boxes
    os.replace(output_path, video_path)
    
    # 6. Update the RAG Agent Context
    if detected_plants:
        # Set the RAG context to the plant that appeared in the highest volume of frames
        CURRENT_SESSION_PLANT = max(detected_plants, key=detected_plants.get)
        print(f"✅ [TRACKING COMPLETE] {frame_number} frames analyzed. Agent context locked to: {CURRENT_SESSION_PLANT}")
    else:
        print("⚠️ [TRACKING COMPLETE] No stable objects tracked.")

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
