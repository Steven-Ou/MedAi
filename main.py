# herb-ai/main.py
# cspell:disable
import os
import sys

# Add current root directory to the python path references
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.vision.detector import BotanicalTracker
from src.rag.query_engine import BotanicalQueryEngine

def start_herb_ai() -> None:
    print("=" * 50)
    print("         HERB-AI INTEGRATED AGENT SYSTEMS         ")
    print("==================================================\n")

    # 1. Ask the user if they want to run a live scan session first
    run_scan = input("Do you want to run the computer vision scanning pipeline? (y/n): ")
    
    if run_scan.strip().lower() == 'y':
        video_path = "data/processed/sample_garden_walk.mp4"
        if os.path.exists(video_path):
            print(f"\nInitializing real-time plant tracking on: {video_path}")
            # Automatically uses yolov8n.pt until your custom weights are exported
            tracker = BotanicalTracker(model_path="yolov8n.pt")
            tracker.process_video(video_path, show_live_feed=True)
        else:
            print(f"\nClip missing at '{video_path}'. Skipping video tracking layer.")

    # 2. Boot up the RAG Clinical Consulting Engine
    print("\nInitializing Vector Storage Search Engines...")
    query_engine = BotanicalQueryEngine()
    
    print("\n" + "=" * 45)
    print("  HERB-AI INTERACTIVE TERMINAL ONLINE  ")
    print("=" * 45)
    print("Type your medical/botanical questions below.")
    print("Type 'exit' to cleanly close the application hub session.\n")

    while True:
        user_query = input("Herb-AI User Question > ")
        if user_query.strip().lower() == "exit":
            print("\nShutting down Herb-AI master processes safely. Goodbye!")
            break
            
        if not user_query.strip():
            continue

        print("\nSearching context indices and generating medical verification text properties...")
        response = query_engine.query_botanical_knowledge(user_query)
        print(f"\nHerb-AI Answer:\n{response}")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    start_herb_ai()