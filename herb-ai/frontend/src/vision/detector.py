import os
import io
import base64
import httpx
import sys
import torch
import open_clip
from PIL import Image
from ultralytics import YOLO
from google import genai
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from frontend.src.rag.know_gen import AutoKnowledgeGenerator
from frontend.src.rag.vector_store import LocalVectorStoreEngine


class BotanicalDetector:
    def __init__(self, model_path: str = "herb-ai/best.pt"):
        print(f"Loading Edge YOLO model: {model_path}...")
        self.model = YOLO(model_path)

        load_dotenv()
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.gemini_client = (
            genai.Client(api_key=self.api_key) if self.api_key else None
        )

        # Initialize OpenCLIP for zero-cost visual memory
        print("🧠 Loading OpenCLIP visual encoder...")
        self.clip_model, _, self.clip_preprocess = (
            open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k"
            )
        )
        self.clip_model.eval()

    def _get_image_vector(self, pil_image: Image.Image) -> list:
        """Encodes an image into a 512-dimensional vector."""
        image_tensor = self.clip_preprocess(pil_image).unsqueeze(0)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().tolist()[0]

    def analyze_image(self, image_bytes: bytes) -> dict:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image.thumbnail((640, 640))
        engine = LocalVectorStoreEngine()

        # Add a dedicated client for visual memory isolation
        import chromadb

        chroma_client = chromadb.PersistentClient(
            path=os.path.join(project_root, "chroma_storage")
        )
        visual_collection = chroma_client.get_or_create_collection(name="visual_memory")

        # STEP 1: CHECK CHROMADB VISUAL MEMORY (Free, 0ms API Cost)
        image_vector = self._get_image_vector(image)
        try:
            # CHANGE: Query the visual collection, not engine.collection
            memory_query = visual_collection.query(
                query_embeddings=[image_vector], n_results=1
            )
            # If distance is under threshold, we already "learned" this image
            if memory_query["distances"] and memory_query["distances"][0]:
                dist = memory_query["distances"][0][0]
                if dist < 0.25:  # Cosine distance match
                    learned_name = memory_query["metadatas"][0][0]["plant_name"]
                    print(
                        f"⚡ Visual Memory Recall: {learned_name} (Distance: {dist:.3f})"
                    )
                    return {"predicted_class": learned_name, "confidence": 0.98}
        except Exception as mem_err:
            print(f"⚠️ Vector memory lookup bypassed: {mem_err}")

        # STEP 2: LOCAL YOLO INFERENCE
        temp_target = "/tmp/temp_inference_target.jpg"
        image.save(temp_target)

        results = self.model(temp_target, verbose=False)
        top_idx = results[0].probs.top1
        predicted_class = results[0].names[top_idx]
        confidence = float(results[0].probs.top1conf)

        # STEP 3: HYBRID LLM FALLBACK (When YOLO is uncertain)
        cloud_success = False
        if confidence < 0.70:
            print("🔍 YOLO confidence low. Attempting Cloud Vision...")
            try:
                if not self.gemini_client:
                    raise ValueError("Gemini Client not initialized.")

                prompt = "Identify this plant. Reply only with the common name. Do not include punctuation or extra text."

                active_gemini_models = [
                    "gemini-3.6-flash",
                    "gemini-3.5-flash",
                    "gemini-3.1-flash-lite",
                    "gemini-2.5-flash",
                ]
                for gemini_model in active_gemini_models:
                    try:
                        print(f"☁️ Attempting Cloud Vision with {gemini_model}...")
                        response = self.gemini_client.models.generate_content(
                            model=gemini_model, contents=[prompt, image]
                        )
                        discovered_name = response.text.strip()
                        if "Unidentified" in discovered_name or not discovered_name:
                            predicted_class = "Unidentified Anomaly"
                            confidence = 0.0
                        else:
                            print(
                                f"☁️ Cloud Vision Success: {discovered_name} via {gemini_model}"
                            )
                            predicted_class = discovered_name
                            confidence = 0.95
                            cloud_success = True
                        break
                    except Exception as model_error:
                        print(f"⚠️ API rejection for {gemini_model}: {model_error}")
                        continue

                if not cloud_success:
                    raise RuntimeError("All available Gemini fallbacks exhausted.")

            except Exception as cloud_error:
                print(
                    f"☁️ Cloud Vision Exhausted ({cloud_error}). Auto-failing over to Edge Vision (Ollama)..."
                )
                buffered = io.BytesIO()
                image.thumbnail((512, 512))
                image.save(buffered, format="JPEG", quality=85)
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
                            if not discovered_name or "Unidentified" in discovered_name:
                                predicted_class = "Unidentified Anomaly"
                                confidence = 0.0
                            else:
                                print(f"🔌 Edge Vision Success: {discovered_name}")
                                predicted_class = discovered_name
                                confidence = 0.85
                except Exception as e:
                    print(f"❌ Inferences failed: {e}")
                    predicted_class = "Unidentified Anomaly"
                    confidence = 0.0

        # STEP 4: SAVE VISUAL EMBEDDING & KNOWLEDGE TO CHROMADB
        if confidence > 0.70 and predicted_class != "Unidentified Anomaly":
            try:
                # CHANGE: Add to visual_collection, not engine.collection
                visual_collection.add(
                    embeddings=[image_vector],
                    metadatas=[{"plant_name": predicted_class}],
                    ids=[f"img_{predicted_class}_{os.urandom(4).hex()}"],
                )
                print(
                    f"💾 Visual embedding stored in ChromaDB for future instant recall: {predicted_class}"
                )
            except Exception as save_err:
                print(f"⚠️ Failed to store vector memory: {save_err}")

            # Generate RAG text profile
            knowledge_gen = AutoKnowledgeGenerator()
            if knowledge_gen.generate_profile_if_new(predicted_class):
                print(f"📝 Syncing local vector knowledge for: {predicted_class}")
                engine.build_vector_store()

        if os.path.exists(temp_target):
            os.remove(temp_target)

        return {"predicted_class": predicted_class, "confidence": round(confidence, 2)}
