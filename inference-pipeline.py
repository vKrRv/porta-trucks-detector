import os
# --- SUPPRESS LINUX QT WARNINGS ---
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_LOGGING_RULES"] = "qt.qpa.*=false"
# ----------------------------------

import cv2
import time
from ultralytics import YOLO

# --- IMPORTS ---
from traffic_monitor import TrafficMonitor
from license_plate_reader import read_license_plate, initialize_reader

class PortaPipeline:
    def __init__(self, traffic_model="yolo11x.pt", plate_model="plate-detector.pt"):
        self.monitor = TrafficMonitor(traffic_model)
        
        print("🔄 Loading Plate Model...")
        self.plate_model = YOLO(plate_model)
        
        print("🔄 Initializing OCR...")
        initialize_reader(use_gpu=True)
        print("✅ Pipeline Ready.\n")

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

            out.write(processed_frame)
            cv2.imshow("PortaPipeline", processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("Done.")

if __name__ == "__main__":
    pipeline = PortaPipeline(traffic_model='yolo11x.pt', plate_model='plate-detector.pt')
    
    # NOTE: Increase duration=300 if your video is longer than 60 seconds!
    pipeline.run(video_source='/home/ali/Desktop/test_video.mp4', duration=60)