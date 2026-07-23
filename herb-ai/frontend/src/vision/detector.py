import os
import io
import base64
import httpx
import sys
from PIL import Image
from ultralytics import YOLO
from google import genai
from dotenv import load_dotenv

# Ensure database/rag imports work cleanly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from frontend.src.rag.know_gen import AutoKnowledgeGenerator
from frontend.src.rag.vector_store import LocalVectorStoreEngine


class BotanicalDetector:
    def __init__(self, model_path: str = "herb-ai/best.pt"):
        """Initializes the YOLO vision model and Cloud AI fallback clients."""
        print(f"Loading Static Image Vision model: {model_path}...")
        self.model = YOLO(model_path)

        load_dotenv()
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.gemini_client = (
            genai.Client(api_key=self.api_key) if self.api_key else None
        )

    def analyze_image(self, image_bytes: bytes) -> dict:
        """Processes a single static image through Edge YOLO, Cloud Gemini, and Edge Moondream fallbacks."""
        image = Image.open(io.BytesIO(image_bytes))

        # 1. Local YOLO Inference (Fastest Edge)
        temp_target = "/tmp/temp_inference_target.jpg"
        image.save(temp_target)

        results = self.model(temp_target, verbose=False)
        top_idx = results[0].probs.top1
        predicted_class = results[0].names[top_idx]
        confidence = float(results[0].probs.top1conf)

        # 2. Hybrid LLM Vision Pivot
        if confidence < 0.70:
            print("🔍 YOLO confidence low. Attempting Cloud Vision...")
            try:
                if not self.gemini_client:
                    raise ValueError("Gemini Client not initialized.")

                prompt = "Identify this plant. Reply only with the common name. Do not include punctuation or extra text."
                # Using 1.5-flash as the primary fast fallback based on your cascade
                response = self.gemini_client.models.generate_content(
                    model="gemini-1.5-flash-latest", contents=[prompt, image]
                )

                discovered_name = response.text.strip()
                if "Unidentified" in discovered_name or not discovered_name:
                    predicted_class = "Unidentified Anomaly"
                    confidence = 0.0
                else:
                    print(f"☁️ Cloud Vision Success: {discovered_name}")
                    predicted_class = discovered_name
                    confidence = 0.95

            except Exception as cloud_error:
                print(
                    f"☁️ Cloud Vision Exhausted ({cloud_error}). Auto-failing over to Edge Vision (Ollama)..."
                )

                # Compress image to save RAM before sending to local model
                buffered = io.BytesIO()
                img_converted = image.convert("RGB") if image.mode != "RGB" else image
                img_converted.thumbnail((512, 512))
                img_converted.save(buffered, format="JPEG", quality=85)
                base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

                try:
                    with httpx.Client() as client:
                        resp = client.post(
                            "http://localhost:11434/api/generate",
                            json={
                                "model": "moondream",
                                "prompt": "Identify this plant. Reply only with the common name.",
                                "images": [base64_image],
                                "stream": False,
                            },
                            timeout=120.0,
                        )

                        if resp.status_code == 200:
                            discovered_name = resp.json().get("response", "").strip()
                            if "Unidentified" in discovered_name or not discovered_name:
                                predicted_class = "Unidentified Anomaly"
                                confidence = 0.0
                            else:
                                print(f"🔌 Edge Vision Success: {discovered_name}")
                                predicted_class = discovered_name
                                confidence = 0.85
                except Exception as e:
                    print(f"❌ Both Cloud and Edge vision inferences failed: {e}")
                    predicted_class = (
                        f"Uncertain: {predicted_class} (Total Offline Mode)"
                    )

        # 3. Handle Auto-Knowledge Generation
        if confidence > 0.70 and predicted_class != "Unidentified Anomaly":
            knowledge_gen = AutoKnowledgeGenerator()
            if knowledge_gen.generate_profile_if_new(predicted_class):
                print(f"📝 Syncing local vector knowledge for: {predicted_class}")
                engine = LocalVectorStoreEngine()
                engine.build_vector_store()

        if os.path.exists(temp_target):
            os.remove(temp_target)

        return {"predicted_class": predicted_class, "confidence": round(confidence, 2)}
