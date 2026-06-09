# cspell:disable
import cv2
import os
import sys
from typing import Dict, Any, Tuple, List, cast
import numpy as np
from ultralytics import YOLO

# Ensure database module can be imported cleanly depending on how main.py is executed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from database.db_manager import insert_telemetry, add_new_plant
from src.rag.know_gen import AutoKnowledgeGenerator
from src.rag.vector_store import ProductionGeminiEngine

class BotanicalTracker:
    def __init__(self, model_path: str = "yolov8n.pt") -> None:
        """Initializes the YOLO vision model and links up the automated RAG engines."""
        print(f"Loading Computer Vision model: {model_path}...")
        self.model = YOLO(model_path)
        
        # FIX: Type hint the tracker dictionary map to silence member resolution alerts
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

            # FIX: Cast self.model to Any right here to completely hide un-typed internal library signatures from Pylance
            model_any: Any = self.model
            results: List[Any] = model_any.track(frame, persist=True, verbose=False)

            # Check if boxes were detected with valid tracking IDs
            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes_obj = results[0].boxes
                
                # Explicitly cast items to clear up the untyped multi-layer duck typing layers
                boxes: np.ndarray[Any, Any] = cast(np.ndarray[Any, Any], boxes_obj.xyxy.numpy())
                track_ids: np.ndarray[Any, Any] = cast(np.ndarray[Any, Any], boxes_obj.id.numpy().astype(int))
                confidences: np.ndarray[Any, Any] = cast(np.ndarray[Any, Any], boxes_obj.conf.numpy())
                class_ids: np.ndarray[Any, Any] = cast(np.ndarray[Any, Any], boxes_obj.cls.numpy().astype(int))

                # Fetch class name map from model metadata safely (removed redundant cast)
                names: Dict[int, str] = self.model.names
                rebuild_vector_store: bool = False

                for b, t_id, c, cls_id in zip(boxes, track_ids, confidences, class_ids):
                    bbox: np.ndarray[Any, Any] = cast(np.ndarray[Any, Any], b)
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

                    # FIX: Explicitly convert the bounding box to a pure Tuple[float, float, float, float]
                    # This satisfies the strict tuple schema contract required by insert_telemetry
                    bbox_tuple: Tuple[float, float, float, float] = (
                        float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
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
                        xmin, ymin, xmax, ymax = map(int, bbox)
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
    SAMPLE_VIDEO: str = "data/processed/sample_garden_walk.mp4" 
    
    # We default back to yolov8n.pt for safety so the program runs fine locally.
    # When your training completes in data.ipynb, change this path parameter to "weights/best.pt"
    if os.path.exists(SAMPLE_VIDEO):
        tracker = BotanicalTracker(model_path="yolov8n.pt")
        tracker.process_video(SAMPLE_VIDEO, show_live_feed=True)
    else:
        print(f"Please place a valid test video file at '{SAMPLE_VIDEO}' to run local testing.")