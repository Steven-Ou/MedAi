# herb-ai/src/rag/know_gen.py
import os
import sys
import requests
import google.generativeai as genai
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

load_dotenv()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

KNOWLEDGE_BASE_DIR = os.path.abspath(
    os.path.join(PROJECT_ROOT, "../data/knowledge_base")
)
os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)


class AutoKnowledgeGenerator:
    def __init__(self):
        self.kb_dir = KNOWLEDGE_BASE_DIR
        self.token = os.getenv("HF_TOKEN")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")

        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)

        self.cloud_models = [
            "meta-llama/Llama-3.2-1B-Instruct",
            "microsoft/Phi-3-mini-4k-instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
        ]

    def generate_profile_if_new(self, plant_name: str) -> bool:
        """Generates a text profile for discovered herbs cascading through models."""
        os.makedirs(self.kb_dir, exist_ok=True)
        file_filename = f"{plant_name.lower().replace(' ', '_')}.txt"
        target_path = os.path.join(self.kb_dir, file_filename)

        if os.path.exists(target_path):
            print(
                f"[{plant_name}] Profile already exists at '{target_path}'. Skipping."
            )
            return False

        print(f"New plant discovered: '{plant_name}'! Generating profile...")

        prompt = (
            f"Write a brief textbook clinical overview for the medicinal substance: {plant_name}.\n"
            f"CRITICAL: Treat this strictly as a botanical medicine, natural remedy, or Traditional Chinese Medicine (TCM) substance.\n"
            f"Focus EXCLUSIVELY on its medicinal effects, active chemical compounds, and traditional biological properties across global herbalism (e.g., TCM's 'clearing heat', Western herbalism's 'anti-inflammatory', or Ayurvedic 'adaptogens').\n"
            f"DO NOT describe it as a medical condition or disease. DO NOT list surgical treatments.\n"
            f"Keep it concise, accurate, and professional."
        )

        # 1. PRIMARY TIER: Gemini API (<1 sec)
        if self.gemini_api_key:
            try:
                print("⚡ Attempting ultra-fast generation with Gemini Flash...")
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(prompt)
                response_text = response.text.strip()

                if response_text:
                    with open(target_path, "w", encoding="utf-8") as f:
                        f.write(response_text)
                    print(f"✅ Successfully saved profile via Gemini: {target_path}")
                    return True
            except Exception as e:
                print(f"⚠️ Gemini generation failed: {e}. Falling back to HF...")

        # 2. SECOND TIER: Hugging Face Inference API
        messages = [{"role": "user", "content": prompt}]
        client = InferenceClient(token=self.token)

        for model_name in self.cloud_models:
            try:
                print(f"☁️ Attempting HF generation with: {model_name}...")
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=512,
                )

                response_text = completion.choices[0].message.content.strip()

                if response_text:
                    with open(target_path, "w", encoding="utf-8") as f:
                        f.write(response_text)
                    print(f"✅ Successfully saved profile via HF: {target_path}")
                    return True

            except Exception as e:
                print(f"⚠️ {model_name} failed: {e}. Trying next...")

        # 3. FALLBACK TIER: Local Ollama CPU
        print("🚨 Remote APIs failed. Falling back to local Ollama...")
        return self.run_local_ollama_fallback(prompt, target_path)

    def run_local_ollama_fallback(self, prompt: str, target_path: str) -> bool:
        try:
            payload = {
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 512, "temperature": 0.3},
            }
            response = requests.post(
                "http://localhost:11434/api/generate", json=payload
            )
            response.raise_for_status()

            response_text = response.json().get("response", "").strip()
            if response_text:
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(response_text)
                print(f"✅ Saved profile via local Ollama: {target_path}")
                return True
        except Exception as e:
            print(f"❌ Local generation failed: {e}")

        return False
