import os
import sys

# Ensure imports route correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from herb_ai.frontend.src.vision.detector import BotanicalDetector

# 1. Grab ANY random plant folder path from your validation directory
test_image_path = "/Users/steve/CS/MedAi/herb-ai/data/clean_training/val/Tomato/img_6480_20190919_171532.jpg"

if os.path.exists(test_image_path):
    print("Loading BotanicalDetector and initializing models...")
    detector = BotanicalDetector()
    
    # Read the image as bytes to simulate your actual frontend payload
    with open(test_image_path, "rb") as image_file:
        image_bytes = image_file.read()
    
    print("\n" + "="*50)
    print(f"TARGET IMAGE: {os.path.basename(test_image_path)}")
    print("RUNNING FULL MULTIMODAL CASCADE TEST...")
    print("="*50)
    
    # This will trigger YOLO -> Gemini -> Moondream
    final_result = detector.analyze_image(image_bytes)
    
    print("\n" + "="*50)
    print("PIPELINE OUTPUT:")
    print(f"PREDICTED SPECIES: {final_result.get('predicted_class')}")
    print(f"CONFIDENCE SCORE: {final_result.get('confidence')}")
    print("="*50)
else:
    print(f"Please check the path to your validation image folder: {test_image_path}")