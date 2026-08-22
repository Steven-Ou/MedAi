import os
import sys
import shutil
import tempfile
import sqlite3
import traceback
import base64
import cv2
import aiofiles
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "herb-ai")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from database.db_manager import init_db, DB_PATH
from frontend.src.rag.query_engine import BotanicalQueryEngine
from frontend.src.vision.detector import BotanicalDetector
from frontend.src.rag.know_gen import AutoKnowledgeGenerator
from frontend.src.rag.vector_store import LocalVectorStoreEngine
from fastapi.responses import StreamingResponse

app = FastAPI(title="Herb-AI Medical Botanical API Hub")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

IS_SCANNING = False

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


@app.get("/api/scan-status")
def get_scan_status():
    return {"is_scanning": IS_SCANNING}


# --- TELEMETRY ENDPOINT ---
@app.get("/api/telemetry")
def get_telemetry():
    if not os.path.exists(DB_PATH):
        return {"data": []}
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            p.species_name, 
            COUNT(t.id), 
            MAX(t.confidence_score), 
            (SELECT evidence_image FROM telemetry WHERE plant_id = p.id ORDER BY confidence_score DESC LIMIT 1)
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
                "evidenceImage": row[3],
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

    result = vision_engine.analyze_image(contents)
    CURRENT_SESSION_PLANT = result.get("predicted_class")

    # FIX: Clear old DB and insert the new static image telemetry
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM telemetry;")
        cursor.execute("DELETE FROM plants;")
        conn.commit()

        if CURRENT_SESSION_PLANT:
            evidence_base64 = base64.b64encode(contents).decode("utf-8")
            cursor.execute(
                "INSERT OR IGNORE INTO plants (species_name) VALUES (?);",
                (CURRENT_SESSION_PLANT,),
            )
            cursor.execute(
                "SELECT id FROM plants WHERE species_name = ?;",
                (CURRENT_SESSION_PLANT,),
            )
            plant_row = cursor.fetchone()

            if plant_row:
                plant_id = plant_row[0]
                conf = result.get("confidence", 0.0)
                cursor.execute(
                    "INSERT INTO telemetry (plant_id, frame_number, xmin, ymin, xmax, ymax, confidence_score, evidence_image) VALUES (?, 1, 0, 0, 0, 0, ?, ?);",
                    (plant_id, float(conf), evidence_base64),
                )
        conn.commit()
        conn.close()

    return {
        "status": "success",
        "predicted_class": CURRENT_SESSION_PLANT,
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
        answer = query_engine.query_botanical_knowledge(text)
        return {"response": answer, "answer": answer}

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query")
def query_alias(payload: QueryRequest):
    q = payload.question or payload.query_text
    if not q or q.strip() == "string":
        return {"response": "Please enter a specific question about the plant."}
    return process_query_text(q)

@app.post("/api/query/stream")
def query_stream_alias(payload: QueryRequest):
    q = payload.question or payload.query_text
    if not q or q.strip() == "string":
        raise HTTPException(status_code=400, detail="Invalid query")
    
    return StreamingResponse(
        query_engine.stream_botanical_knowledge(q),
        media_type="application/octet-stream"
    )

@app.post("/api/chat")
def chat_alias(payload: QueryRequest):
    q = payload.question or payload.query_text
    if not q or q.strip() == "string":
        return {"response": "Please enter a specific question about the plant."}
    return process_query_text(q)


# --- INSTANT VIDEO FRAME CLASSIFICATION ---
def background_video_scan(video_path: str):
    global CURRENT_SESSION_PLANT, IS_SCANNING
    IS_SCANNING = True
    model_path = os.path.join(project_root, "best.pt")

    try:
        model = YOLO(model_path)
        print(f"🎬 [VIDEO SCAN] Initiating frame-by-frame analysis...")

        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

        output_path = video_path.replace(".mp4", "_tracked.mp4")
        out = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

        results = model.predict(
            source=video_path, stream=True, conf=0.45, imgsz=640, vid_stride=5
        )

        conn = sqlite3.connect(DB_PATH, timeout=15.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        cursor = conn.cursor()

        cursor.execute("DELETE FROM telemetry;")
        cursor.execute("DELETE FROM plants;")
        conn.commit()

        detected_plants = {}
        frame_number = 0

        for r in results:
            frame_number += 1

            annotated_frame = r.plot()
            out.write(annotated_frame)

            if r.probs is not None:
                top5_indices = r.probs.top5
                top5_confs = r.probs.top5conf.tolist()

                evidence_base64 = None

                for idx, conf in zip(top5_indices, top5_confs):
                    if conf >= 0.01:
                        plant_name = model.names[idx]

                        detected_plants[plant_name] = (
                            detected_plants.get(plant_name, 0) + 1
                        )

                        if not evidence_base64:
                            _, buffer = cv2.imencode(".jpg", annotated_frame)
                            evidence_base64 = base64.b64encode(buffer).decode("utf-8")

                        cursor.execute(
                            "INSERT OR IGNORE INTO plants (species_name) VALUES (?);",
                            (plant_name,),
                        )
                        cursor.execute(
                            "SELECT id FROM plants WHERE species_name = ?;",
                            (plant_name,),
                        )
                        plant_id = cursor.fetchone()[0]

                        cursor.execute(
                            "INSERT INTO telemetry (plant_id, frame_number, xmin, ymin, xmax, ymax, confidence_score, evidence_image) VALUES (?, ?, 0, 0, 0, 0, ?, ?);",
                            (plant_id, frame_number, float(conf), evidence_base64),
                        )
                        conn.commit()

        out.release()
        cap.release()
        conn.commit()
        conn.close()

        if detected_plants:
            CURRENT_SESSION_PLANT = ", ".join(detected_plants.keys())
            print(
                f"✅ [SCAN COMPLETE] {frame_number} frames analyzed. Context locked to: {CURRENT_SESSION_PLANT}"
            )
            knowledge_gen = AutoKnowledgeGenerator()
            vector_engine = LocalVectorStoreEngine()
            rebuild_needed = False

            for plant_name in detected_plants.keys():
                if knowledge_gen.generate_profile_if_new(plant_name):
                    print(f"📝 Syncing local vector knowledge for: {plant_name}")
                    rebuild_needed = True

            if rebuild_needed:
                print("Updating Chroma vector store with new video discoveries...")
                vector_engine.build_vector_store()
        else:
            print("⚠️ [SCAN COMPLETE] No classes detected even with 1% threshold.")

    finally:
        IS_SCANNING = False
        if os.path.exists(video_path):
            os.remove(video_path)
        if "output_path" in locals() and os.path.exists(output_path):
            os.remove(output_path)


@app.post("/api/scan")
async def trigger_scan(background_tasks: BackgroundTasks, file: UploadFile):
    global IS_SCANNING

    # 2. Prevent concurrent heavy ML scans
    if IS_SCANNING:
        raise HTTPException(
            status_code=429, detail="A video scan is already in progress. Please wait."
        )

    if not file or not file.filename:
        raise HTTPException(
            status_code=400, detail="No video file provided for scanning."
        )

    # Lock the scanning state immediately
    IS_SCANNING = True

    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    video_target = temp_video.name

    try:
        # 3. Unblock the event loop using async file writing
        async with aiofiles.open(video_target, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):  # Read in 1MB chunks
                await buffer.write(chunk)

    except Exception as e:
        # If upload fails, unlock and clean up
        IS_SCANNING = False
        if os.path.exists(video_target):
            os.remove(video_target)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

    # Add the background task
    background_tasks.add_task(background_video_scan, video_target)

    return {"message": "Video scan started successfully."}
