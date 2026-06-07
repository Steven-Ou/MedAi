# cspell:disable
import cv2
import os
import sys
from ultralytics import YOLO

# Ensure database and RAG modules can be imported cleanly depending on how executed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from database.db_manager import insert_telemetry, add_new_plant

# FIXED: Hooking up your RAG and Knowledge Generator pipeline dependencies
from src.rag.knowledge_generator import AutoKnowledgeGenerator
from src.rag.vector_store import ProductionGeminiEngine

class BotanicalTracker:
    def __init__(self, model_path="yolov8n.pt"):
        """
        Initializes the YOLO vision model and hooks up the autonomous RAG engines.
        """
        print(f"Loading Computer Vision model: {model_path}...")
        self.model = YOLO(model_path)
        
        # Keeps track of mapping between YOLO internal track IDs and our SQL plant_ids
        self.track_to_db_map = {}
        
        # FIXED: Initialize the automated RAG pipeline sub-engines
        self.knowledge_gen = AutoKnowledgeGenerator()
        self.vector_engine = ProductionGeminiEngine()

    def process_video(self, video_path, show_live_feed=True):
        """
        Processes a video file frame-by-frame, runs tracking inference,
        saves spatial telemetry, and triggers automated AI profile synthesis.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file at {video_path}")
            return

        frame_number = 0
        print(f"Starting video ingestion pipeline for: {video_path}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break # Video has ended or frame drop occurred

            frame_number += 1

            # Run tracking inference on the current frame
            results = self.model.track(frame, persist=True, verbose=False)

            # Check if boxes were detected with valid tracking IDs
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()  # [xmin, ymin, xmax, ymax]
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

                # Fetch class name map from model metadata
                names = self.model.names

                # Flag to check if we need to sync the Chroma vector store at the end of the frame processing
                rebuild_vector_store = False

                for bbox, track_id, conf, cls_id in zip(boxes, track_ids, confidences, class_ids):
                    species_name = names[cls_id]

                    # If this is a brand new track ID unseen by the system, run discovery lifecycle
                    if track_id not in self.track_to_db_map:
                        # Step A: Log the plant entity to your SQL telemetry database
                        db_plant_id = add_new_plant(species_name)
                        self.track_to_db_map[track_id] = db_plant_id
                        print(f"[NEW ENTITY] Detected {species_name} - Logged to Database with Plant ID: {db_plant_id}")
                        
                        # Step B: FIXED - Check if text profile exists, write it via Gemini if it's brand new
                        was_generated = self.knowledge_gen.generate_profile_if_new(species_name)
                        if was_generated:
                            rebuild_vector_store = True
                    
                    # Retrieve the assigned persistent database primary key
                    assigned_plant_id = self.track_to_db_map[track_id]

                    # Insert spatial metrics and boundary coordinates for this specific frame
                    insert_telemetry(
                        plant_id=assigned_plant_id,
                        frame_number=frame_number,
                        bbox=bbox,
                        confidence_score=float(conf)
                    )

                    # Optional UI: Draw bounding boxes and labels on screen
                    if show_live_feed:
                        xmin, ymin, xmax, ymax = map(int, bbox)
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        label = f"ID {assigned_plant_id}: {species_name} ({conf:.2f})"
                        cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # FIXED: If new plant files were generated during this frame, update the Chroma indices immediately
                if rebuild_vector_store:
                    print("New knowledge base profiles detected. Updating Chroma vector index storage...")
                    self.vector_engine.build_vector_store()
                    print("Vector database sync complete!")

            # Display window frame if active
            if show_live_feed:
                cv2.imshow("Herb-AI Live Vision Tracking Stream", frame)
                # Press 'q' on keyboard to cancel video stream manual override
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Processing terminated by user command.")
                    break

        cap.release()
        cv2.destroyAllWindows()
        print(f"Video pipeline finished. Total processed frames: {frame_number}")

if __name__ == "__main__":
    SAMPLE_VIDEO = "data/processed/sample_garden_walk.mp4" 
    
    if os.path.exists(SAMPLE_VIDEO):
        tracker = BotanicalTracker()
        tracker.process_video(SAMPLE_VIDEO, show_live_feed=True)
    else:
        print(f"Please place a valid test video file at '{SAMPLE_VIDEO}' to run local testing.")