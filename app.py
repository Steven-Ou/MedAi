import os
import sys
import shutil
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

# 1. Inject the hyphenated folder into Python's path so imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "herb-ai")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 2. Import your actual AI engines using the corrected path
from frontend.src.vision.detector import BotanicalDetector
from frontend.src.rag.query_engine import BotanicalQueryEngine

app = FastAPI(title="Herb-AI Vision API")

# 3. Initialize the heavy models ONCE at startup to prevent memory crashes
print("Loading Botanical Detector...")
vision_engine = BotanicalDetector()

print("Loading Vector Storage Engine...")
rag_engine = BotanicalQueryEngine()


class QueryRequest(BaseModel):
    query_text: str


@app.get("/")
def health_check():
    return {"status": "Online", "message": "Herb-AI is actively listening."}


@app.post("/api/detect")
async def run_detection(file: UploadFile = File(...)):
    # Read the file directly into memory as bytes (Matches detector.py requirement)
    image_bytes = await file.read()

    # Pass the bytes directly to your actual method name
    inference_result = vision_engine.analyze_image(image_bytes)

    return {"status": "success", "results": inference_result}


@app.post("/api/query")
def run_rag_query(request: QueryRequest):
    # Pass the JSON string to your actual query method
    answer = rag_engine.query_botanical_knowledge(request.query_text)

    return {"response": answer}


# FIX: Import your database manager schema setup tools to guarantee tables exist
# (Replace 'init_db' with whatever table setup function is named inside your db_manager.py, e.g., create_tables)
from database.db_manager import init_db  # noqa: E402



