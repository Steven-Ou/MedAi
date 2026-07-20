# herb-ai/src/rag/know_gen.py
# cspell:disable
import os
import sys
import requests
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Ensure project root is accessible for imports
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

        # List models in order of preference (Waterfall fallback)
        self.cloud_models = [
            "meta-llama/Llama-3.2-1B-Instruct",  # Tries this first (Gated)
            "microsoft/Phi-3-mini-4k-instruct",  # Fallback 1 (Ungated)
            "Qwen/Qwen2.5-1.5B-Instruct",  # Fallback 2 (Ungated)
        ]

    def generate_profile_if_new(self, plant_name: str) -> bool:
        """Generates a text profile for discovered herbs cascading through models."""
        os.makedirs(self.kb_dir, exist_ok=True)
        file_filename = f"{plant_name.lower().replace(' ', '_')}.txt"
        target_path = os.path.join(self.kb_dir, file_filename)

        if os.path.exists(target_path):
            print(
                f"[{plant_name}] Profile already exists at '{target_path}'. Skipping generation."
            )
            return False

        print(f"New plant discovered: '{plant_name}'! Attempting generation...")

        prompt = (
            f"Write a brief textbook clinical overview for the herb: {plant_name}.\n"
            f"Include sections for:\n"
            f"- Common and botanical names\n"
            f"- Primary medicinal properties (anti-inflammatory, immunomodulator, etc.)\n"
            f"- Active chemical compounds\n"
            f"- Primary clinical health benefits.\n"
            f"Keep it concise, accurate, and professional."
        )

        messages = [{"role": "user", "content": prompt}]
        client = InferenceClient(token=self.token)

        # Waterfall through the Hugging Face models
        for model_name in self.cloud_models:
            try:
                print(f"☁️ Attempting remote generation with: {model_name}...")
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=512,  # Strictly fixes the 18-token cutoff!
                )

                response_text = completion.choices[0].message.content.strip()

                if response_text:
                    with open(target_path, "w", encoding="utf-8") as f:
                        f.write(response_text)
                    print(
                        f"✅ Successfully saved new knowledge base file to: {target_path}"
                    )
                    return True
                else:
                    print(
                        f"⚠️ {model_name} returned an empty response. Trying next model..."
                    )

            except Exception as e:
                print(f"⚠️ {model_name} failed (Error: {e}). Trying next model...")

        # If we loop through all cloud models and they ALL fail, drop to local
        print("🚨 All cloud API models failed. Booting local Ollama instance...")
        return self.run_local_ollama_fallback(prompt, target_path)

    def run_local_ollama_fallback(self, prompt: str, target_path: str) -> bool:
        """Fallback to local Llama 3.2 3B instance if remote API fails."""
        try:
            payload = {
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 512, "temperature": 0.3},
            }

            # Make request to local Ollama server
            response = requests.post(
                "http://localhost:11434/api/generate", json=payload
            )
            response.raise_for_status()

            response_text = response.json().get("response", "").strip()

            if response_text:
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(response_text)
                print(
                    f"✅ Successfully saved new knowledge base file via Local Ollama to: {target_path}"
                )
                return True
            else:
                print("❌ Local Ollama returned an empty response.")

        except requests.exceptions.RequestException as e:
            print(
                f"❌ Local generation completely failed. Is Ollama running? Error: {e}"
            )

        return False


if __name__ == "__main__":
    gen = AutoKnowledgeGenerator()
    # Test it out
    gen.generate_profile_if_new("Peppermint")
