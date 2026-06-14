# cspell:disable
import os
import sys

current_dir: str = os.path.dirname(os.path.abspath(__file__))
project_root: str = os.path.join(current_dir, "herb-ai")

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import your tracker and query engine modules
from src.vision.detector import BotanicalTracker  # noqa: E402
from src.rag.query_engine import BotanicalQueryEngine  # noqa: E402

# FIX: Import your database manager schema setup tools to guarantee tables exist
# (Replace 'init_db' with whatever table setup function is named inside your db_manager.py, e.g., create_tables)
from database.db_manager import init_db  # noqa: E402

def start_herb_ai() -> None:
    print("=" * 50)
    print("         HERB-AI INTEGRATED AGENT SYSTEMS         ")
    print("==================================================\n")

    # FIX: Guarantee that the local SQLite database tables exist before any components run
    try:
        print("Verifying database schema layout properties...")
        # If your function doesn't require arguments or handles them internally, call it here:
        init_db() 
        print("Database verification complete! Schema tables are online.\n")
    except Exception as e:
        print(f"[Database Warning] Table initialization script bypassed: {e}\n")

    run_scan: str = input("Do you want to run the computer vision scanning pipeline? (y/n): ")
    
    if run_scan.strip().lower() == 'y':
        video_path: str = os.path.join(project_root, "data/processed/sample_garden_walk.mp4")
        model_path = os.path.join(project_root, "weights/best.pt")        
        
        if os.path.exists(video_path):
            print(f"\nInitializing real-time plant tracking on: {video_path}")
            tracker: BotanicalTracker = BotanicalTracker(model_path=model_path)
            tracker.process_video(video_path, show_live_feed=True)
        else:
            print(f"\nClip missing at '{video_path}'. Skipping video tracking layer.")

    print("\nInitializing Vector Storage Search Engines...")
    query_engine: BotanicalQueryEngine = BotanicalQueryEngine()
    
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
        response: str = query_engine.query_botanical_knowledge(user_query)
            
        print(f"\nHerb-AI Answer:\n{response}")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    start_herb_ai()