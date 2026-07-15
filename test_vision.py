import os
from ultralytics import YOLO

model_path = "/Users/steve/CS/MedAi/herb-ai/research/herb_runs/botany_classification-2/weights/best.pt"
model = YOLO(model_path)

# 1. Grab ANY random plant folder path from your validation directory
# Example: Let's pick a clean Tomato or Tulsi image path from your data stack
test_image_path = "/Users/steve/CS/MedAi/herb-ai/data/clean_training/val/Tomato/img_6480_20190919_171532.jpg"

if os.path.exists(test_image_path):
    results = model(test_image_path)
    top_idx = results[0].probs.top1
    predicted_class = results[0].names[top_idx]
    confidence = float(results[0].probs.top1conf)
    
    print("\n" + "="*40)
    print(f"TARGET IMAGE: {os.path.basename(test_image_path)}")
    print(f"PREDICTED SPECIES: {predicted_class}")
    print(f"CONFIDENCE SCORE: {confidence:.2f}")
    print("="*40)
else:
    print(f"Please check the path to your validation image folder: {test_image_path}")