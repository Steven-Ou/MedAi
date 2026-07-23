# cspell:disable
import os
import glob
import sys

current_dir: str = os.path.dirname(os.path.abspath(__file__))
project_root: str = os.path.join(current_dir, "herb-ai")

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import your tracker and query engine modules
from frontend.src.vision.detector import BotanicalDetector  # noqa: E402
from frontend.src.rag.query_engine import BotanicalQueryEngine  # noqa: E402

# FIX: Import your database manager schema setup tools to guarantee tables exist
# (Replace 'init_db' with whatever table setup function is named inside your db_manager.py, e.g., create_tables)
from database.db_manager import init_db  # noqa: E402


def start_herb_ai() -> None:
    print("=" * 50)
    print("         HERB-AI INTEGRATED AGENT SYSTEMS         ")
    print("==================================================\n")

    # Guarantee that local SQL tables exist
    try:
        print("Verifying database schema layout properties...")
        init_db()
        print("Database verification complete! Schema tables are online.\n")
    except Exception as e:
        print(f"[Database Warning] Table initialization script bypassed: {e}\n")

    # --- ADDED: FORCE CHROMA COHERENCY CHECK ON BOOT ---
    try:
        from frontend.src.rag.vector_store import LocalVectorStoreEngine
        import chromadb
        
        vector_engine = LocalVectorStoreEngine()
        # Test if the collection exists, if not, force a full clean build
        try:
            vector_engine.chroma_client.get_collection(name="botanical_knowledge")
            print("Vector storage engines online and verified.")
        except (chromadb.errors.NotFoundError, Exception):
            print("[Vector Storage Warning] Collection missing. Compiling reference vectors...")
            vector_engine.build_vector_store()
            print("Vector storage rebuild complete.\n")
    except Exception as e:
        print(f"[Vector Warning] Vector checking pipeline bypassed: {e}\n")
    # ---------------------------------------------------

    run_scan: str = input(
        "Do you want to run the computer vision scanning pipeline? (y/n): "
    )

    if run_scan.strip().lower() == "y":
        video_path: str = os.path.join(project_root, "data/processed/sample_garden_walk.mp4")
        model_path = "/Users/steve/CS/MedAi/herb-ai/research/herb_runs/botany_classification-2/weights/best.pt"

        if os.path.exists(model_path) and os.path.exists(video_path):
            print(f"\nInitializing classification parsing pipeline on: {video_path}")
            print(f"Loading custom fine-tuned weights: {model_path}")

            from ultralytics import YOLO
            import cv2
            import sqlite3  # Import sqlite3 to clear the tables cleanly
            from database.db_manager import DB_PATH, add_new_plant, insert_telemetry

            # --- ADDED: WIPE CORES ON NEW VIDEO SCAN RUNS ---
            print("Resetting telemetry history tables for fresh session execution...")
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM telemetry;")
            cursor.execute("DELETE FROM plants;")
            conn.commit()
            conn.close()
            # ------------------------------------------------

            from frontend.src.rag.know_gen import AutoKnowledgeGenerator
            from frontend.src.rag.vector_store import LocalVectorStoreEngine
            
            knowledge_gen = AutoKnowledgeGenerator()
            new_plant_discovered = False
            # -------------------------------------------------------

            model = YOLO(model_path)
            cap = cv2.VideoCapture(video_path)

            print("Processing video frames sequentially...")
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % 5 == 0:
                    results = model(frame, verbose=False)
                    if results and results[0].probs:
                        top_idx = results[0].probs.top1
                        class_name = results[0].names[top_idx]
                        confidence = float(results[0].probs.top1conf)

                        if confidence > 0.35:
                            print(f"[Frame {frame_count}] Identified: {class_name} ({confidence:.2f}) -> Logging...")
                            
                            plant_id = add_new_plant(class_name)
                            dummy_bbox = (0.0, 0.0, 0.0, 0.0)

                            insert_telemetry(
                                plant_id=plant_id,
                                frame_number=frame_count,
                                bbox=dummy_bbox,
                                confidence_score=confidence,
                            )
                            
                            # --- AUTOMATIC DISCOVERY AND PROFILE GENERATION TRIGGER ---
                            # If it's a new plant, generate its medical profile text file
                            was_generated = knowledge_gen.generate_profile_if_new(class_name)
                            if was_generated:
                                new_plant_discovered = True
                            # ----------------------------------------------------------

            cap.release()
            print(f"Video pipeline finished. Total processed frames: {frame_count}")
            
            # --- REBUILD THE VECTOR DATABASE IF NEW SPECIES WERE FOUND ---
            if new_plant_discovered:
                print("\nNew botanical profiles generated! Syncing vector store indices...")
                vector_engine = LocalVectorStoreEngine()
                vector_engine.build_vector_store()
            # -------------------------------------------------------------

        elif not os.path.exists(model_path):
            print(f"\nWeights file missing at '{model_path}'.")
        else:
            print(f"\nClip missing at '{video_path}'.")

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

        print(
            "\nSearching context indices and generating medical verification text properties..."
        )
        response: str = query_engine.query_botanical_knowledge(user_query)

        print(f"\nHerb-AI Answer:\n{response}")
        print("-" * 50 + "\n")


if __name__ == "__main__":
    start_herb_ai()
