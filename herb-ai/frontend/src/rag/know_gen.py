# herb-ai/src/rag/know_gen.py
# cspell:disable
import os
from huggingface_hub import InferenceClient
import sys
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
    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct"):
        # We swapped the model name to the Hugging Face repo ID
        self.model_name = model_name
        self.kb_dir = KNOWLEDGE_BASE_DIR
        
        # Initialize the serverless Hugging Face client
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=os.getenv("HF_TOKEN")
        )

    def generate_profile_if_new(self, plant_name: str) -> bool:
        """Generates a text profile for discovered herbs using Hugging Face."""
        os.makedirs(self.kb_dir, exist_ok=True)
        file_filename = f"{plant_name.lower().replace(' ', '_')}.txt"
        target_path = os.path.join(self.kb_dir, file_filename)

        if os.path.exists(target_path):
            print(
                f"[{plant_name}] Profile already exists at '{target_path}'. Skipping generation."
            )
            return False

        print(
            f"New plant discovered: '{plant_name}'! Generating medical and botanical background using Hugging Face ({self.model_name})..."
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

        # Format the prompt for the Chat API
        messages = [{"role": "user", "content": prompt}]

        try:
            # Replaced the local httpx/Ollama call with the Hugging Face call
            completion = self.client.chat.completions.create(
                model=self.model_name, 
                messages=messages, 
                max_tokens=512 # This strictly fixes the 18-token cutoff!
            )
            
            response_text = completion.choices[0].message.content.strip()

            if response_text:
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(response_text)
                print(
                    f"Successfully saved new knowledge base file to: {target_path}"
                )
                return True
            else:
                print(f"❌ Hugging Face API Error: Empty response received.")

        except Exception as e:
            print(f"❌ Remote generation failure via Hugging Face API: {e}")

        return False


if __name__ == "__main__":
    gen = AutoKnowledgeGenerator()
    # Test it out
    gen.generate_profile_if_new("Peppermint")