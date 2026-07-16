# herb-ai/src/rag/know_gen.py
# cspell:disable
import os
import sys
import httpx
from dotenv import load_dotenv

# Ensure project root is accessible for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

load_dotenv()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
KNOWLEDGE_BASE_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "../data/knowledge_base"))
os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)


class AutoKnowledgeGenerator:
    def __init__(self, model_name="llama3.2"):
        # You can change 'llama3' to a smaller model here if it's lagging
        self.model_name = model_name
        self.kb_dir = KNOWLEDGE_BASE_DIR
        self.ollama_url = "http://localhost:11434/api/generate"

    def generate_profile_if_new(self, plant_name: str) -> bool:
        """Generates a text profile for discovered herbs using local Ollama."""
        os.makedirs(self.kb_dir, exist_ok=True)
        file_filename = f"{plant_name.lower().replace(' ', '_')}.txt"
        target_path = os.path.join(self.kb_dir, file_filename)

        if os.path.exists(target_path):
            print(
                f"[{plant_name}] Profile already exists at '{target_path}'. Skipping generation."
            )
            return False

        print(
            f"New plant discovered: '{plant_name}'! Generating medical and botanical background using Ollama ({self.model_name})..."
        )

        prompt = (
            f"Write a brief textbook clinical overview for the herb: {plant_name}.\n"
            f"Include sections for:\n"
            f"- Common and botanical names\n"
            f"- Primary medicinal properties (anti-inflammatory, immunomodulator, etc.)\n"
            f"- Active chemical compounds\n"
            f"- Primary clinical health benefits.\n"
            f"Keep it concise, accurate, and professional."
        )

        payload = {"model": self.model_name, "prompt": prompt, "stream": False}

        try:
            # We set a high timeout (120s) because local generation can take a minute on slower hardware
            with httpx.Client() as client:
                response = client.post(self.ollama_url, json=payload, timeout=300.0)

                if response.status_code == 200:
                    response_text = response.json().get("response", "").strip()

                    if response_text:
                        with open(target_path, "w", encoding="utf-8") as f:
                            f.write(response_text)
                        print(
                            f"Successfully saved new knowledge base file to: {target_path}"
                        )
                        return True
                else:
                    print(f"❌ Ollama API Error: Status {response.status_code}")

        except Exception as e:
            print(f"❌ Local generation failure via Ollama: {e}")

        return False


if __name__ == "__main__":
    gen = AutoKnowledgeGenerator()
    # Test it out
    gen.generate_profile_if_new("Peppermint")
