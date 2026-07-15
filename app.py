# app.py
import os
import sys
import sqlite3
import io
import shutil
import time
import random
from datetime import datetime
from PIL import Image
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import traceback
from ultralytics import YOLO
from dotenv import load_dotenv
from google import genai
import cv2
import base64
import httpx

# Ensure project root is accessible for imports
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    gemini_client = genai.Client(api_key=api_key)
else:
    gemini_client = None
    print("⚠️ Warning: GEMINI_API_KEY not found in .env file. Cloud fallback will fail.")


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "herb-ai")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from database.db_manager import init_db, DB_PATH, add_new_plant, insert_telemetry
from frontend.src.rag.query_engine import BotanicalQueryEngine
from frontend.src.rag.know_gen import AutoKnowledgeGenerator
from frontend.src.rag.vector_store import LocalVectorStoreEngine

app = FastAPI(title="Herb-AI Medical Botanical API Hub")
CURRENT_SESSION_PLANT = None
# Enable CORS so your React/Next.js frontend can communicate with it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Database Schema on API boot
init_db()


class QueryRequest(BaseModel):
    question: str


# Background task to process video frames without locking up the UI HTTP request
def background_video_scan():
    video_path = os.path.join(project_root, "data/processed/sample_garden_walk.mp4")
    model_path = os.path.join(project_root, "best.pt")
    
    if not os.path.exists(model_path) or not os.path.exists(video_path):
        return

    # Clear tables for fresh session
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM telemetry;")
    cursor.execute("DELETE FROM plants;")
    conn.commit()
    conn.close()

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    knowledge_gen = AutoKnowledgeGenerator()
    new_plant_discovered = False
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 5 == 0:
            results = model(frame, verbose=False)
            if results and results[0].probs:
                top_idx = results[0].probs.top1
                class_name = results[0].names[top_idx]
                confidence = float(results[0].probs.top1conf)

                if confidence < 0.35:
                    class_name = "Non-Botanical Anomaly"
                    add_new_plant(class_name)
                else:
                    add_new_plant(class_name)

                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id FROM plants WHERE species_name = ?;", (class_name,)
                )
                plant_id = cursor.fetchone()[0]
                conn.close()

                insert_telemetry(
                    plant_id, frame_count, (0.0, 0.0, 0.0, 0.0), confidence
                )

                if class_name != "Non-Botanical Anomaly":
                    was_generated = knowledge_gen.generate_profile_if_new(class_name)
                    if was_generated:
                        new_plant_discovered = True
    cap.release()

    if new_plant_discovered:
        vector_engine = LocalVectorStoreEngine()
        vector_engine.build_vector_store()


@app.post("/api/scan")
def trigger_scan(background_tasks: BackgroundTasks):
    """Triggers the computer vision video processing pipeline in the background."""
    background_tasks.add_task(background_video_scan)
    return {
        "status": "processing",
        "message": "Video scanning pipeline kicked off successfully.",
    }


@app.get("/api/telemetry")
def get_telemetry():
    """Fetches currently tracked plants and frame configurations directly from SQLite."""
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


def smart_backoff(attempt):
    """Wait longer after each failed attempt to avoid slamming the API."""
    wait_time = (2**attempt) + random.random()
    print(f"Quota hit/High demand. Backing off for {wait_time:.2f} seconds...")
    time.sleep(wait_time)


@app.post("/api/upload-image")
def upload_image_inference(file: UploadFile = File(...)):
    """Processes static image uploads using a Hybrid Cloud-to-Edge fallback pipeline."""
    global CURRENT_SESSION_PLANT
    model_path = os.path.join(project_root, "best.pt")
    model = YOLO(model_path)    

    contents = file.file.read()
    image = Image.open(io.BytesIO(contents))

    # 1. Local YOLO Inference (Fastest Edge)
    temp_target = "/tmp/temp_inference_target.jpg"
    image.save(temp_target)

    results = model(temp_target, verbose=False)
    top_idx = results[0].probs.top1
    predicted_class = results[0].names[top_idx]
    confidence = float(results[0].probs.top1conf)

    # 2. Hybrid LLM Vision Pivot
    if confidence < 0.70:
        print("🔍 YOLO confidence low. Attempting Cloud Vision (Gemini 1.5 Flash)...")

        try:
            # --- PRIMARY: CLOUD INFERENCE (GEMINI) ---
            if not gemini_client:
                raise ValueError("Gemini Client not initialized.")

            prompt = "Identify this plant. Reply only with the common name. Do not include punctuation or extra text."

            # A cascade of Gemini models (Ordered from fastest/newest to most robust)
            gemini_models = [
                "gemini-2.5-flash",
                "gemini-2.0-flash",
                "gemini-1.5-flash",
                "gemini-1.5-pro",
            ]

            response = None
            last_cloud_error = None
            successful_model = None

            # Loop through the models until one succeeds
            for model_name in gemini_models:
                try:
                    # Optional: uncomment the line below if you want to see the cascade in the terminal
                    # print(f"☁️ Attempting Cloud Vision with {model_name}...")
                    response = gemini_client.models.generate_content(
                        model=model_name, contents=[prompt, image]
                    )
                    successful_model = model_name
                    break  # Success! Break out of the loop
                except Exception as e:
                    print(f"⚠️ {model_name} failed: {e}. Trying next model...")
                    last_cloud_error = e
                    continue

            # If the loop finished and we never got a response, force the Ollama fallback
            if not response:
                raise ValueError(
                    f"All Gemini models failed. Last error: {last_cloud_error}"
                )

            discovered_name = response.text.strip()

            if "Unidentified" in discovered_name or not discovered_name:
                predicted_class = "Unidentified Anomaly"
                confidence = 0.0
            else:
                print(f"☁️ Cloud Vision Success ({successful_model}): {discovered_name}")
                predicted_class = discovered_name
                confidence = 0.95  # Assign very high confidence to Cloud LLM

        except Exception as cloud_error:
            # --- SECONDARY: EDGE INFERENCE (OLLAMA FALLBACK) ---
            print(
                f"☁️ Cloud Vision Exhausted ({cloud_error}). Auto-failing over to Edge Vision (Ollama)..."
            )

            # Compress image to save RAM before sending to local model
            buffered = io.BytesIO()
            if image.mode != "RGB":
                image_converted = image.convert("RGB")
            else:
                image_converted = image

            image_converted.thumbnail((512, 512))  # Downsize for local VRAM safety
            image_converted.save(buffered, format="JPEG", quality=85)
            base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

            ollama_url = "http://localhost:11434/api/generate"
            payload = {
                "model": "moondream",  # Lightweight vision fallback
                "prompt": "Identify this plant. Reply only with the common name.",
                "images": [base64_image],
                "stream": False,
                "options": {"temperature": 0.0, "num_ctx": 1024},
            }

            try:
                with httpx.Client() as client:
                    response = client.post(ollama_url, json=payload, timeout=120.0)
                    if response.status_code == 200:
                        discovered_name = response.json().get("response", "").strip()
                        if "Unidentified" in discovered_name or not discovered_name:
                            predicted_class = "Unidentified Anomaly"
                            confidence = 0.0
                        else:
                            print(f"🔌 Edge Vision Success: {discovered_name}")
                            predicted_class = discovered_name
                            confidence = 0.85
            except Exception as e:
                print(f"❌ Both Cloud and Edge vision inferences failed: {e}")
                predicted_class = f"Uncertain: {predicted_class} (Total Offline Mode)"

        # 3. Handle Auto-Knowledge Generation (Shared Logic for Both Cloud & Edge)
        if confidence > 0.70 and predicted_class != "Unidentified Anomaly":
            CURRENT_SESSION_PLANT = predicted_class
            knowledge_gen = AutoKnowledgeGenerator()
            if knowledge_gen.generate_profile_if_new(predicted_class):
                print(f"📝 Syncing local vector knowledge for: {predicted_class}")
                engine = LocalVectorStoreEngine()
                engine.build_vector_store()

    if os.path.exists(temp_target):
        os.remove(temp_target)

    print(f"✅ Final Identification: {predicted_class} ({confidence})")
    return {"predicted_class": predicted_class, "confidence": round(confidence, 2)}


@app.post("/api/chat")
def query_agent(payload: QueryRequest):
    print(f"🤖 Herb-AI: Starting analysis pipeline for question: {payload.question}")
    try:
        # RAG Pipeline
        print("📚 Herb-AI: Retrieving botanical knowledge from RAG...")
        query_engine = BotanicalQueryEngine()

        augmented_query = payload.question
        if CURRENT_SESSION_PLANT and CURRENT_SESSION_PLANT != "Unidentified Anomaly":
            augmented_query = f"Context: We are discussing the plant '{CURRENT_SESSION_PLANT}'. User Question: {payload.question}"

        answer = query_engine.query_botanical_knowledge(augmented_query)

        print("✨ Herb-AI: Generation complete.")
        return {"answer": answer}

    except Exception as e:
        print("❌ Error in pipeline:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
