# app.py
import os
import sys
import shutil
import tempfile
import traceback
import cv2
import uuid
import chromadb
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from ultralytics import YOLO
from huggingface_hub import HfApi

# Setup paths before local imports to satisfy Python module resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "herb-ai")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Ruff Rule E402 Bypass: We must manipulate sys.path before these imports
from database.db_manager import (
    init_db,
    add_new_plant,
    insert_telemetry,
    get_conn,
    clear_session_telemetry,
)  # noqa: E402
from frontend.src.rag.query_engine import BotanicalQueryEngine  # noqa: E402
from frontend.src.vision.detector import BotanicalDetector  # noqa: E402
from frontend.src.rag.know_gen import AutoKnowledgeGenerator  # noqa: E402
from frontend.src.rag.vector_store import LocalVectorStoreEngine  # noqa: E402

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

# 3. SINGLETON INSTANTIATION
print("⚡ [BOOT] Pre-loading Vision Engine (YOLO + OpenCLIP)...")
vision_engine = BotanicalDetector(model_path=os.path.join(project_root, "best.pt"))

print("⚡ [BOOT] Pre-loading Vector RAG Engine...")
query_engine = BotanicalQueryEngine()


class QueryRequest(BaseModel):
    query_text: str = None
    question: str = None
    session_id: str = "default_session"


@app.get("/")
def read_root():
    return {"status": "Herb-AI Backend is running smoothly."}


ACTIVE_SCANS = set()


@app.get("/api/scan-status")
def get_scan_status(session_id: str = None):
    if not session_id:
        return {"is_scanning": False}
    return {"is_scanning": session_id in ACTIVE_SCANS}


@app.get("/api/telemetry")
def get_telemetry(session_id: str):
    try:
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT p.species_name, COUNT(t.id), MAX(t.confidence_score), 
                   (SELECT evidence_image_url FROM telemetry WHERE plant_id = p.id AND session_id = %s ORDER BY confidence_score DESC LIMIT 1)
            FROM plants p
            JOIN telemetry t ON p.id = t.plant_id
            WHERE t.session_id = %s
            GROUP BY p.species_name, p.id
        """,
            (session_id, session_id),
        )
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
    except Exception as e:
        print(f"Telemetry Fetch Error: {e}")
        return {"data": []}


def upload_to_huggingface(image_bytes: bytes, predicted_class: str) -> str:
    hf_token = os.getenv("HF_TOKEN")
    repo_id = "steveo223/herb-ai-vault"
    file_name = f"{predicted_class.replace(' ', '_')}/{uuid.uuid4().hex}.jpg"

    api = HfApi()
    try:
        api.upload_file(
            path_or_fileobj=image_bytes,
            path_in_repo=file_name,
            repo_id=repo_id,
            repo_type="dataset",
            token=hf_token,
        )
        return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{file_name}"
    except Exception as e:
        print(f"HF Upload Error: {e}")
        return None


async def handle_image_upload(file: UploadFile, session_id: str = "default_session"):
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="No image file provided.")

    result = await asyncio.to_thread(vision_engine.analyze_image, contents)
    predicted_plant = result.get("predicted_class")

    if predicted_plant:
        print(f"☁️ Uploading {predicted_plant} to Hugging Face Vault...")
        evidence_url = upload_to_huggingface(contents, predicted_plant)

        plant_id = add_new_plant(predicted_plant)
        conf = result.get("confidence", 0.0)

        insert_telemetry(
            session_id=session_id,
            plant_id=plant_id,
            frame_number=1,
            bbox=(0, 0, 0, 0),
            confidence_score=float(conf),
            evidence_url=evidence_url,
        )

    return {
        "status": "success",
        "predicted_class": predicted_plant,
        "confidence": result.get("confidence"),
        "results": result,
    }


@app.post("/api/detect")
async def detect_alias(
    file: UploadFile = File(...), session_id: str = Form("default_session")
):
    return await handle_image_upload(file, session_id)


@app.post("/api/upload-image")
async def upload_image_alias(
    file: UploadFile = File(...), session_id: str = Form("default_session")
):
    return await handle_image_upload(file, session_id)


@app.post("/api/predict")
async def predict_alias(
    file: UploadFile = File(...), session_id: str = Form("default_session")
):
    return await handle_image_upload(file, session_id)


def process_query_text(text: str):
    print(f"🤖 [QUERY] Processing: '{text}'")
    try:
        answer = query_engine.query_botanical_knowledge(text)
        return {"response": answer, "answer": answer}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query")
def query_alias(payload: QueryRequest):
    q = payload.question or payload.query_text
    if not q or q.strip() == "string":
        return {"response": "Please enter a specific question about the plant."}
    answer = query_engine.query_botanical_knowledge(q, session_id=payload.session_id)
    return {"response": answer, "answer": answer}


@app.post("/api/query/stream")
def query_stream_alias(payload: QueryRequest):
    q = payload.question or payload.query_text
    if not q or q.strip() == "string":
        raise HTTPException(status_code=400, detail="Invalid query")

    return StreamingResponse(
        query_engine.stream_botanical_knowledge(q, session_id=payload.session_id),
        media_type="application/octet-stream",
    )


@app.post("/api/chat")
def chat_alias(payload: QueryRequest):
    q = payload.question or payload.query_text
    if not q or q.strip() == "string":
        return {"response": "Please enter a specific question about the plant."}
    return process_query_text(q)


def background_video_scan(video_path: str, session_id: str):
    ACTIVE_SCANS.add(session_id)
    model_path = os.path.join(project_root, "best.pt")

    try:
        clear_session_telemetry(session_id)
        print(f"🧹 Cleared old telemetry data for session {session_id}.")

        chroma_client = chromadb.PersistentClient(
            path=os.path.join(project_root, "chroma_storage")
        )
        try:
            visual_collection = chroma_client.get_collection(name="visual_memory")
            visual_collection.delete(where={"session_id": session_id})
            print(f"🧹 Cleared ChromaDB RAG Context for session {session_id}.")
        except Exception as e:
            print(f"⚠️ Could not clear ChromaDB context: {e}")

    except Exception as e:
        print(f"Failed to clear telemetry: {e}")

    try:
        model = YOLO(model_path)
        print("🎬 [VIDEO SCAN] Initiating frame-by-frame analysis...")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ [ERROR] Could not open video file.")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

        output_path = video_path.replace(".mp4", "_tracked.mp4")
        out = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

        results = model.predict(
            source=video_path,
            stream=True,
            conf=0.05,
            imgsz=320,
            vid_stride=30,
        )

        detected_plants = {}
        frame_number = 0

        for r in results:
            frame_number += 1
            annotated_frame = r.plot()
            out.write(annotated_frame)

            detected_names = []

            if r.probs is not None:
                top5_indices = r.probs.top5
                top5_confs = r.probs.top5conf.tolist()
                for idx, conf in zip(top5_indices, top5_confs):
                    if conf >= 0.01:
                        detected_names.append((model.names[idx], float(conf)))
            elif r.boxes is not None:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf >= 0.01:
                        detected_names.append((model.names[int(box.cls[0])], conf))

            evidence_url = None
            for plant_name, conf in detected_names:
                detected_plants[plant_name] = detected_plants.get(plant_name, 0) + 1

                if not evidence_url:
                    _, buffer = cv2.imencode(".jpg", annotated_frame)
                    evidence_url = upload_to_huggingface(buffer.tobytes(), plant_name)

                plant_id = add_new_plant(plant_name)
                insert_telemetry(
                    session_id=session_id,
                    plant_id=plant_id,
                    frame_number=frame_number,
                    bbox=(0, 0, 0, 0),
                    confidence_score=float(conf),
                    evidence_url=evidence_url,
                )

        out.release()
        cap.release()

        if detected_plants:
            print(
                f"✅ [SCAN COMPLETE] {frame_number} frames analyzed. Saving context for ALL detected plants for session {session_id}"
            )

            knowledge_gen = AutoKnowledgeGenerator()
            vector_engine = LocalVectorStoreEngine()

            rebuild_needed = False
            for plant_name in detected_plants.keys():
                # This now loops through every single plant it spotted
                if knowledge_gen.generate_profile_if_new(plant_name):
                    print(f"📝 Syncing local vector knowledge for: {plant_name}")
                    rebuild_needed = True

            # Only rebuild the vector store once after all new profiles are generated
            if rebuild_needed:
                vector_engine.build_vector_store()

        else:
            print("⚠️ [SCAN COMPLETE] No classes detected even with 1% threshold.")

    finally:
        ACTIVE_SCANS.discard(session_id)
        if os.path.exists(video_path):
            os.remove(video_path)
        if "output_path" in locals() and os.path.exists(output_path):
            os.remove(output_path)


@app.post("/api/scan")
async def trigger_scan(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session_id: str = Form("default_session"),
):
    if session_id in ACTIVE_SCANS:
        raise HTTPException(
            status_code=429,
            detail="Your video scan is already in progress. Please wait.",
        )

    if not file or not file.filename:
        raise HTTPException(
            status_code=400, detail="No video file provided for scanning."
        )

    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    video_target = temp_video.name

    try:
        with open(video_target, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        if os.path.exists(video_target):
            os.remove(video_target)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

    background_tasks.add_task(background_video_scan, video_target, session_id)
    return {"message": "Video scan started successfully."}
