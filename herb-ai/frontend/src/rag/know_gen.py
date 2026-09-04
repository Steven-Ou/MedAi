# herb-ai/src/rag/know_gen.py
import os
import sys
import requests
from google import genai
from huggingface_hub import InferenceClient
from openai import OpenAI
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
        self.gemini_client = (
            genai.Client(api_key=self.gemini_api_key) if self.gemini_api_key else None
        )

        self.openai_client = (
            OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            if os.getenv("OPENAI_API_KEY")
            else None
        )
        self.groq_client = (
            OpenAI(
                api_key=os.getenv("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1",
            )
            if os.getenv("GROQ_API_KEY")
            else None
        )

        self.cloud_models = [
            "meta-llama/Llama-3.2-1B-Instruct",
            "microsoft/Phi-3-mini-4k-instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
        ]

    def generate_profile_if_new(self, plant_name: str) -> bool:
        """Generates a text profile for discovered herbs cascading through models."""

        if "unidentified" in plant_name.lower() or "anomaly" in plant_name.lower():
            print(f"🚫 Skipping profile generation for invalid edge case: {plant_name}")
            return False

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
            f"You are a master clinical botanist. Write a highly structured, accurate textbook profile for the medicinal substance: {plant_name}.\n"
            f"CRITICAL RULES:\n"
            f"1. Treat this STRICTLY as a botanical medicine or Traditional Chinese Medicine (TCM) substance.\n"
            f"2. Focus EXCLUSIVELY on its medicinal effects, active chemical compounds, and traditional biological properties.\n"
            f"3. DO NOT describe it as a medical condition. DO NOT hallucinate properties it does not have. If you do not know it, state 'Insufficient clinical data'.\n"
            f"4. Format the output with clear, structured data points so it can be parsed into tables. Use Markdown.\n"
            f"Keep it concise, rigorous, and professional. Avoid fluff."
        )

        # 1. PRIMARY TIER: Gemini API Cascade
        if self.gemini_client:
            active_gemini_models = [
                "gemini-3.6-flash",
                "gemini-3.5-flash",
                "gemini-3.1-flash-lite",
                "gemini-2.5-flash",
            ]

            for gemini_model in active_gemini_models:
                try:
                    print(f"⚡ Attempting generation with {gemini_model}...")
                    response = self.gemini_client.models.generate_content(
                        model=gemini_model, contents=prompt
                    )
                    response_text = response.text.strip()

                    if response_text:
                        with open(target_path, "w", encoding="utf-8") as f:
                            f.write(response_text)
                        print(
                            f"✅ Successfully saved profile via Gemini: {target_path}"
                        )
                        return True
                except Exception as e:
                    print(f"⚠️ {gemini_model} failed: {e}. Trying next...")

        if self.groq_client or self.openai_client:
            client_to_use = self.groq_client or self.openai_client
            model_to_use = "llama-3.3-70b-versatile" if self.groq_client else "gpt-4o-mini"

            try:
                print(f"☁️ Attempting generation with {model_to_use}...")
                response = client_to_use.chat.completions.create(
                    model=model_to_use,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1024,
                    temperature=0.3,
                )
                response_text = response.choices[0].message.content.strip()

                if response_text:
                    with open(target_path, "w", encoding="utf-8") as f:
                        f.write(response_text)
                    print(
                        f"✅ Successfully saved profile via OpenAI/Groq: {target_path}"
                    )
                    return True
            except Exception as e:
                print(f"⚠️ OpenAI/Groq failed: {e}. Trying HF next...")

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
