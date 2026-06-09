# cspell:disable
import os
import sys

# 1. Dynamically identify the absolute inner project directory path layout
current_dir: str = os.path.dirname(os.path.abspath(__file__))
project_root: str = os.path.join(current_dir, "herb-ai")

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 2. Tell Ruff that these imports must follow path updates by appending noqa
from src.vision.detector import BotanicalTracker  # noqa: E402
from src.rag.query_engine import BotanicalQueryEngine  # noqa: E402

def start_herb_ai() -> None:
    print("=" * 50)
    print("         HERB-AI INTEGRATED AGENT SYSTEMS         ")
    print("==================================================\n")

    # Ask the user if they want to run a live scan session first
    run_scan: str = input("Do you want to run the computer vision scanning pipeline? (y/n): ")
    
    if run_scan.strip().lower() == 'y':
        video_path: str = os.path.join(project_root, "data/processed/sample_garden_walk.mp4")
        model_path: str = os.path.join(project_root, "yolov8n.pt")
        
        if os.path.exists(video_path):
            print(f"\nInitializing real-time plant tracking on: {video_path}")
            tracker: BotanicalTracker = BotanicalTracker(model_path=model_path)
            tracker.process_video(video_path, show_live_feed=True)
        else:
            print(f"\nClip missing at '{video_path}'. Skipping video tracking layer.")

    # Boot up the RAG Clinical Consulting Engine
    print("\nInitializing Vector Storage Search Engines...")
    
    # FIX: Overwrite the active current working directory state context so that Chroma 
    # internal engines naturally load the correct persistent storage directories
    original_cwd: str = os.getcwd()
    try:
        os.chdir(project_root)
        query_engine: BotanicalQueryEngine = BotanicalQueryEngine()
    finally:
        os.chdir(original_cwd)
    
    print("\n" + "=" * 45)
    print("  HERB-AI INTERACTIVE TERMINAL ONLINE  ")
    print("=" * 45)
    print("Type your medical/botanical questions below.")
    print("Type 'exit' to cleanly close the application hub session.\n")

    while True:
        user_query: str = input("Herb-AI User Question > ")
        if user_query.strip().lower() == "exit":
            print("\nShutting down Herb-AI master processes safely. Goodbye!")
            break
            
        if not user_query.strip():
            continue

        print("\nSearching context indices and generating medical verification text properties...")
        
        # Ensure runtime query paths map back smoothly
        try:
            os.chdir(project_root)
            response: str = query_engine.query_botanical_knowledge(user_query)
        finally:
            os.chdir(original_cwd)
            
        print(f"\nHerb-AI Answer:\n{response}")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    start_herb_ai()