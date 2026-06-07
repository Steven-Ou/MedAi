# cspell:disable
import os
import sys
from google import genai
from dotenv import load_dotenv

# Ensure project root is accessible for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

load_dotenv()

KNOWLEDGE_BASE_DIR: str = "data/knowledge_base"

class AutoKnowledgeGenerator:
    def __init__(self) -> None:
        """Initializes the GenAI client to generate plant profiles automatically."""
        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY not found in your environment properties.")
        self.client = genai.Client()

    def generate_profile_if_new(self, plant_name: str) -> bool:
        """Generates a detailed medical text profile if the plant hasn't been scanned before."""
        if not os.path.exists(KNOWLEDGE_BASE_DIR):
            os.makedirs(KNOWLEDGE_BASE_DIR)

        # Normalize the filename to avoid file path mapping issues
        safe_filename = plant_name.lower().replace(" ", "_")
        file_path = os.path.join(KNOWLEDGE_BASE_DIR, f"{safe_filename}.txt")

        # Skip generation if we already have this plant on record
        if os.path.exists(file_path):
            print(f"[{plant_name}] Profile already exists at '{file_path}'. Skipping generation.")
            return False

        print(f"New plant discovered: '{plant_name}'! Generating medical and botanical background...")

        prompt = (
            f"Write a brief, comprehensive medical and botanical profile for: {plant_name}.\n"
            f"Format it exactly like this template structure:\n"
            f"{plant_name} (Scientific Name) is a ...\n"
            f"Medicinal Properties: [List core clinical or therapeutic properties]\n"
            f"Active Compounds: [List primary chemical active elements]\n"
            f"Health Benefits: [List major use cases or physiological advantages]\n"
            f"Keep it concise, clear, and around 100-150 words."
        )

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            
            profile_text = response.text
            if not profile_text:
                raise ValueError("Model returned an empty response string.")

            # Save the text file directly into your knowledge base folder
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(profile_text.strip())

            print(f"Successfully saved new knowledge base file to: {file_path}")
            return True

        except Exception as e:
            print(f"Failed to generate automated profile properties: {e}")
            return False

if __name__ == "__main__":
    # Quick standalone testing check
    gen = AutoKnowledgeGenerator()
    gen.generate_profile_if_new("Peppermint")