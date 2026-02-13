import os
# --- SUPPRESS LINUX QT WARNINGS ---
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_LOGGING_RULES"] = "qt.qpa.*=false"
# ----------------------------------

import cv2
import time
from ultralytics import YOLO
import requests
from datetime import datetime, timezone

# --- IMPORTS ---
from traffic_monitor import TrafficMonitor
from license_plate_reader import read_license_plate, initialize_reader

# --- BACKEND INTEGRATION ---
BACKEND_URL = "https://smart-logistics-web.vercel.app/api/traffic"
CAMERA_ID = "CAM_01_KING_ABDULAZIZ"

def classify_traffic_status(vehicle_count: int) -> str:
    """Classify traffic based on vehicle count"""
    if vehicle_count < 100:
        return "NORMAL"
    elif vehicle_count <= 150:
        return "MODERATE"
    else:
        return "CONGESTED"

def send_traffic_update(camera_id: str, vehicle_count: int, truck_count: int, recommendation: str = ""):
    """Send traffic data to Team 1's backend"""

    # Format timestamp as YYYY-MM-DDTHH:MM:SSZ (no microseconds)
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')

    payload = {
        "camera_id": camera_id,
        "timestamp": timestamp,
        "status": classify_traffic_status(vehicle_count),
        "vehicle_count": vehicle_count,
        "truck_count": truck_count,
        "recommendation": recommendation
    }

    try:
        response = requests.post(BACKEND_URL, json=payload, timeout=5)
        response.raise_for_status()

        result = response.json()
        print(f"✅ Traffic update sent: {result['success']}")
        print(f"   Permits affected: {result.get('permits_affected', 0)}")
        print(f"   Permits protected: {result.get('permits_protected', 0)}")

        return result

    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to send traffic update: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Response: {e.response.text}")
            print(f"   Payload sent: {payload}")
        return None

class PortaPipeline:
    def __init__(self, traffic_model="yolo11x.pt", plate_model="plate-detector.pt"):
        self.monitor = TrafficMonitor(traffic_model)

        print("🔄 Loading Plate Model...")
        self.plate_model = YOLO(plate_model)

        print("🔄 Initializing OCR...")
        initialize_reader(use_gpu=True)
        print("✅ Pipeline Ready.\n")

        # Backend integration
        self.last_update_time = 0
        self.update_interval = 1800  # Send updates every 30 minutes (1800 seconds)

    def detect_and_read_plate(self, truck_crop):
        results = self.plate_model(truck_crop, verbose=False, conf=0.25)
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_img = truck_crop[y1:y2, x1:x2]
                
                temp_path = f"temp_plate_{time.time()}.jpg"
                cv2.imwrite(temp_path, plate_img)
                
                ocr_result, _ = read_license_plate(temp_path)
                
                if os.path.exists(temp_path): os.remove(temp_path)
                
                if "error" not in ocr_result:
                    return ocr_result['data']['english']['full']
        return None

    def run(self, video_source, duration=60):
        if not os.path.exists(video_source):
            print(f"❌ ERROR: File not found at {video_source}")
            return

        cap = cv2.VideoCapture(video_source)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        out = cv2.VideoWriter("final_output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        print(f"🚀 Running on {video_source}")
        start_time = time.time()

        while cap.isOpened() and (time.time() - start_time < duration):
            success, frame = cap.read()
            if not success: break

            processed_frame, truck_event = self.monitor.process_frame(frame)

            if truck_event:
                truck_crop, track_id = truck_event
                print(f"🚛 Analyzing Truck #{track_id}...")

                plate_text = self.detect_and_read_plate(truck_crop)

                if plate_text:
                    print(f"✨ FOUND PLATE: {plate_text}")
                    cv2.putText(processed_frame, f"PLATE: {plate_text}", (20, 300),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

            # Send traffic updates every 30 seconds
            current_time = time.time()
            if current_time - self.last_update_time >= self.update_interval:
                # Calculate total vehicle count and truck count
                total_vehicles = sum(self.monitor.counts.values())
                truck_count = self.monitor.counts.get("Truck", 0) + self.monitor.counts.get("Heavy Truck", 0)

                recommendation = f"Detected {total_vehicles} vehicles on King Abdulaziz Road"
                if self.monitor.current_status == "CONGESTED":
                    recommendation += " - High congestion detected"
                elif self.monitor.current_status == "MODERATE":
                    recommendation += " - Moderate traffic flow"

                send_traffic_update(
                    camera_id=CAMERA_ID,
                    vehicle_count=total_vehicles,
                    truck_count=truck_count,
                    recommendation=recommendation
                )

                self.last_update_time = current_time

            out.write(processed_frame)

            try:
                cv2.imshow("PortaPipeline", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            except cv2.error:
                pass  # Skip display if GUI not available

        cap.release()
        out.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass  # Skip if GUI not available
        print("Done.")

if __name__ == "__main__":
    pipeline = PortaPipeline(traffic_model='yolo11x.pt', plate_model='plate-detector.pt')

    # NOTE: Increase duration=300 if your video is longer than 60 seconds!
    pipeline.run(video_source='test.mp4', duration=60)