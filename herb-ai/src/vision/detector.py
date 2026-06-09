# cspell:disable
import cv2
import os
import sys
from typing import Dict, Any, Tuple
import numpy as np
from ultralytics import YOLO

# Ensure database module can be imported cleanly depending on how main.py is executed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from database.db_manager import insert_telemetry, add_new_plant
from src.rag.know_gen import AutoKnowledgeGenerator
from src.rag.vector_store import ProductionGeminiEngine

class BotanicalTracker:
    def __init__(self, model_path: str = "weights/best.pt") -> None:
        """Initializes the YOLO vision model and hooks up the autonomous RAG engines."""
        print(f"Loading Computer Vision model: {model_path}...")
        self.model = YOLO(model_path)
        
        # Explicitly type hint the tracking map dictionary to prevent type inference failures
        self.track_to_db_map: Dict[int, int] = {}
        
        self.knowledge_gen = AutoKnowledgeGenerator()
        self.vector_engine = ProductionGeminiEngine()

    def process_video(self, video_path: str, show_live_feed: bool = True) -> None:
        """
        Processes a video file frame-by-frame, runs tracking inference,
        and saves spatial telemetry metadata into the SQL database.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file at {video_path}")
            return

        frame_number: int = 0
        print(f"Starting video ingestion pipeline for: {video_path}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Video has ended or frame drop occurred

            frame_number += 1

            # FIX: Cast model to Any before accessing .track to fully mute reportUnknownMemberType warnings
            model_any: Any = self.model
            results_list: list[Any] = model_any.track(frame, persist=True, verbose=False)

            # Check if boxes were detected with valid tracking IDs
            if results_list and results_list[0].boxes is not None and results_list[0].boxes.id is not None:
                boxes_obj = results_list[0].boxes
                
                # Sift out tensor primitives into NumPy matrices
                boxes: np.ndarray[Any, Any] = boxes_obj.xyxy.cpu().numpy()
                track_ids: np.ndarray[Any, Any] = boxes_obj.id.cpu().numpy().astype(int)
                confidences: np.ndarray[Any, Any] = boxes_obj.conf.cpu().numpy()
                class_ids: np.ndarray[Any, Any] = boxes_obj.cls.cpu().numpy().astype(int)

                names: Dict[int, str] = self.model.names
                rebuild_vector_store: bool = False

                for b, t_id, c, cls_id in zip(boxes, track_ids, confidences, class_ids):
                    track_id: int = int(t_id)
                    conf: float = float(c)
                    species_name: str = names[int(cls_id)]

                    # If this is a brand new track ID unseen by the system, log it to SQL and RAG
                    if track_id not in self.track_to_db_map:
                        db_plant_id: int = add_new_plant(species_name)
                        self.track_to_db_map[track_id] = db_plant_id
                        print(f"[NEW ENTITY] Detected {species_name} - Logged to Database with Plant ID: {db_plant_id}")
                        
                        was_generated: bool = self.knowledge_gen.generate_profile_if_new(species_name)
                        if was_generated:
                            rebuild_vector_store = True
                    
                    # Retrieve the assigned persistent database primary key
                    assigned_plant_id: int = self.track_to_db_map[track_id]

                    # Safely parse array components into standard Python Float Tuple to pass types cleanly
                    bbox_tuple: Tuple[float, float, float, float] = (
                        float(b[0]), float(b[1]), float(b[2]), float(b[3])
                    )

                    # Insert spatial metrics and boundary coordinates for this specific frame
                    insert_telemetry(
                        plant_id=assigned_plant_id,
                        frame_number=frame_number,
                        bbox=bbox_tuple,
                        confidence_score=conf
                    )

                    # Optional UI: Draw bounding boxes and labels on screen
                    if show_live_feed:
                        xmin, ymin, xmax, ymax = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        label: str = f"ID {assigned_plant_id}: {species_name} ({conf:.2f})"
                        cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
    tracker = BotanicalTracker(model_path="weights/best.pt")
    
    SAMPLE_VIDEO: str = "data/processed/sample_garden_walk.mp4"
    if os.path.exists(SAMPLE_VIDEO):
        tracker.process_video(SAMPLE_VIDEO, show_live_feed=True)
    else:
        print(f"Please place a valid test video file at '{SAMPLE_VIDEO}' to run local testing.")