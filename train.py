import os
import glob
import shutil
from ultralytics import YOLO

project_root = "/Users/steve/CS/MedAi/herb-ai"
raw_data_dir = os.path.join(project_root, "data/raw")
clean_base_dir = os.path.join(project_root, "data/clean_training")

print("=" * 60)
print("          HERB-AI CLASSIFICATION TREE COMPILER          ")
print("=" * 60)

# 1. Gather all raw image binaries
supported_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPEG", "*.PNG"]
found_raw_images = []
for ext in supported_extensions:
    found_raw_images.extend(glob.glob(os.path.join(raw_data_dir, "**", ext), recursive=True))

print(f"Located {len(found_raw_images)} source images.")

# 2. Parse paths and copy them into split classification folders
for idx, raw_path in enumerate(found_raw_images):
    parts = raw_path.split(os.sep)
    
    # Extract clean category folder name by avoiding the ".zip" suffix string
    category = "unknown_plant"
    for part in reversed(parts):
        if "dataset" in part.lower() or part == parts[-1]:
            continue
        category = part.replace(".zip", "").replace(" ", "_")
        break

    # Determine train vs validation split (90% train, 10% val)
    split_type = "train" if (idx % 10 != 0) else "val"
    target_dir = os.path.join(clean_base_dir, split_type, category)
    os.makedirs(target_dir, exist_ok=True)

    dest_path = os.path.join(target_dir, f"img_{idx}_{os.path.basename(raw_path)}")
    if not os.path.exists(dest_path):
        shutil.copy(raw_path, dest_path)

print(f"Dataset successfully compiled inside: {clean_base_dir}")

# 3. Fire up the YOLO classification training sequence loop!
print("\n🚀 Launching YOLO Classification Pipeline...")
model = YOLO("yolov8n-cls.pt")  # Loading native classification layers
results = model.train(
    data=clean_base_dir,       # Points to the root directory containing train/ and val/ folders
    epochs=30,                  # Quick run to generate weights file
    imgsz=224,                 # Standard optimal image classification size
    project=os.path.join(project_root, "research/herb_runs"),
    name="botany_classification",
    device="cpu"               # Switch to 'mps' if your M3 Mac terminal environment is native ARM
)