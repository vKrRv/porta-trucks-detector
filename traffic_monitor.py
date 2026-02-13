import cv2
import datetime
import os
import json
from ultralytics import YOLO
from collections import defaultdict

class TrafficMonitor:
    def __init__(self, model_path, output_folder="heavy_truck_crops"):
        print(f"🔄 Loading Traffic Model: {model_path} ...")
        self.model = YOLO(model_path)
        
        # Configuration
        self.output_folder = output_folder
        self.line_position = 0.5
        self.heavy_size_threshold = 40000
        self.conf_threshold = 0.3
        self.target_classes = [2, 3, 5, 7]  # Car, Bike, Bus, Truck
        
        # State
        self.track_history = defaultdict(lambda: [])
        self.crossed_ids = set()
        self.counts = {"Car": 0, "Motorcycle": 0, "Bus": 0, "Truck": 0, "Heavy Truck": 0}
        self.current_status = "NORMAL"

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def get_vehicle_type(self, cls_id, box):
        """Determine specific vehicle type based on class and size."""
        x1, y1, x2, y2 = map(int, box)
        area = (x2 - x1) * (y2 - y1)
        
        if cls_id == 7: # Truck
            return ("Heavy Truck", (0, 0, 255)) if area > self.heavy_size_threshold else ("Truck", (0, 255, 255))
        elif cls_id == 5: return "Bus", (0, 0, 255)
        elif cls_id == 2: return "Car", (0, 255, 0)
        elif cls_id == 3: return "Motorcycle", (0, 255, 0)
        return "Unknown", (255, 255, 255)

    def update_status(self):
        hour = datetime.datetime.now().hour
        total = sum(self.counts.values())
        heavy = self.counts["Heavy Truck"] + self.counts["Bus"]
        
        if 8 <= hour < 14:
            if total > 45 or heavy > 5: self.current_status = "CONGESTED"
            elif total > 30 or heavy > 2: self.current_status = "MODERATE"
            else: self.current_status = "NORMAL"
        else:
            self.current_status = "MODERATE" if total > 35 else "NORMAL"

    def process_frame(self, frame):
        height, width = frame.shape[:2]
        line_y = int(height * self.line_position)
        truck_event = None 
        
        # Track
        results = self.model.track(frame, persist=True, conf=self.conf_threshold, 
                                   classes=self.target_classes, verbose=False)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            cls_ids = results[0].boxes.cls.int().cpu().tolist()
            
            for box, track_id, cls_id in zip(boxes, track_ids, cls_ids):
                label, color = self.get_vehicle_type(cls_id, box)
                x1, y1, x2, y2 = map(int, box)
                
                # Update Track
                track = self.track_history[track_id]
                track.append(y2)
                if len(track) > 30: track.pop(0)
                
                # Check Crossing
                if len(track) >= 2 and track[-2] < line_y and track[-1] > line_y:
                    if track_id not in self.crossed_ids:
                        self.crossed_ids.add(track_id)
                        self.counts[label] += 1
                        self.update_status()
                        
                        # --- RESTORED PRINT STATEMENTS ---
                        print(f"✅ COUNTED: #{track_id} as {label}")
                        # ---------------------------------

                        if label in ["Truck", "Heavy Truck"]:
                            truck_crop = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
                            truck_event = (truck_crop, track_id)

                # Draw Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"#{track_id} {label}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw UI
        cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 255), 2)
        self._draw_sidebar(frame)
        
        return frame, truck_event

    def _draw_sidebar(self, frame):
        # Draw Black Background
        cv2.rectangle(frame, (10, 10), (280, 250), (0, 0, 0), -1)
        # Header
        cv2.putText(frame, "TRAFFIC MONITOR", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"STATUS: {self.current_status}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        y = 100
        for k, v in self.counts.items():
            color = (0,0,255) if k in ["Heavy Truck", "Bus"] else (0,255,0)
            cv2.putText(frame, f"{k}: {v}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 30