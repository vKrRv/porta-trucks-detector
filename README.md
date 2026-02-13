# Porta Trucks Detector

Computer vision pipeline for traffic monitoring and license plate detection from road CCTV footage. Integrates with backend API for traffic management and permit systems.

## Project Structure

```
porta-trucks-detector/
├── inference-pipeline.py       # Main execution script
├── traffic_monitor.py          # Vehicle detection and tracking
├── license_plate_reader.py     # OCR for license plates
├── plate-detector.pt           # YOLO plate detection model
├── yolo11x.pt                  # YOLO11 traffic model
├── requirements.txt            # Python dependencies
├── heavy_truck_crops/          # Output: truck image crops
├── notebooks/                  # Development notebooks
└── archive/                    # Archived models and data
```

## System Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- Internet connection (for backend API)

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

All required packages:
- ultralytics (YOLO models)
- opencv-python (video processing)
- easyocr (OCR engine)
- requests (HTTP client)
- torch, torchvision (deep learning)
- numpy (numerical operations)
- lapx (linear assignment)

### 2. Required Models

Place the following model files in the root directory:
- `yolo11x.pt` - YOLO11 traffic detection model
- `plate-detector.pt` - License plate detection model

## Usage

### Basic Execution

```bash
python inference-pipeline.py
```

### Configuration

Edit `inference-pipeline.py` to modify:

**Video Source** (Line 165):
```python
pipeline.run(video_source='test.mp4', duration=60)
```

**Update Interval** (Line 73):
```python
self.update_interval = 30  # seconds between backend updates
```

**Camera ID** (Line 19):
```python
CAMERA_ID = "CAM_01_KING_ABDULAZIZ"
```

## Backend Integration

### API Endpoint

```
POST https://smart-logistics-web.vercel.app/api/traffic
```

### Payload Format

```json
{
  "camera_id": "CAM_01_KING_ABDULAZIZ",
  "timestamp": "2026-02-13T15:23:45Z",
  "status": "NORMAL",
  "vehicle_count": 42,
  "truck_count": 8,
  "recommendation": "Detected 42 vehicles on King Abdulaziz Road"
}
```

### Traffic Classification

- **NORMAL**: vehicle_count < 100
- **MODERATE**: 100 <= vehicle_count <= 150
- **CONGESTED**: vehicle_count > 150

### Response Format

```json
{
  "success": true,
  "permits_affected": 3,
  "permits_protected": 1
}
```

## Components

### TrafficMonitor (`traffic_monitor.py`)

- Vehicle detection and tracking using YOLO11
- Counts vehicles by type: Car, Motorcycle, Bus, Truck, Heavy Truck
- Crossing line detection for accurate counting
- Real-time traffic status classification

### LicensePlateReader (`license_plate_reader.py`)

- YOLO-based plate detection
- EasyOCR for text extraction
- Supports English and Arabic characters
- GPU acceleration support

### PortaPipeline (`inference-pipeline.py`)

- Integrates traffic monitoring and plate reading
- Sends periodic updates to backend API
- Outputs annotated video: `final_output.mp4`
- Handles OpenCV display errors gracefully

## Output

**Console:**
- Real-time vehicle counts
- License plate detections
- Backend API responses

**Files:**
- `final_output.mp4` - Annotated video with detections
- `heavy_truck_crops/` - Cropped images of detected trucks

## Error Handling

- Network failures: Logs error, continues processing
- Missing video file: Exits with error message
- OpenCV display errors: Runs headless, saves video output
- API timeouts: 5-second timeout, non-blocking

## Testing

### Test Backend Connection

```bash
curl -X POST https://smart-logistics-web.vercel.app/api/traffic \
  -H "Content-Type: application/json" \
  -d '{"camera_id":"CAM_01_KING_ABDULAZIZ","timestamp":"2026-02-13T15:00:00Z","status":"NORMAL","vehicle_count":50,"truck_count":5,"recommendation":"Test"}'
```

### Reduce Update Interval

For testing, set `update_interval = 5` to see backend updates every 5 seconds.

## Known Limitations

- Requires `lap` package which needs C++ build tools on Windows
- OpenCV GUI support may not work on headless systems
- EasyOCR model download required on first run (~500MB)
- Video processing speed depends on GPU availability

## Development

### Notebooks

Development notebooks are in `notebooks/`:
- `traffic-monitor.ipynb` - Traffic detection development
- `license-plate-detector.ipynb` - Plate detection development
- `ocr_license_plate_reader.ipynb` - OCR testing
- `truck-detector.ipynb` - Truck classification experiments

### Archive

Deprecated files in `archive/`:
- `truck-detector.pt` - Legacy truck detection model
- `sample_data/` - Test datasets

## License

Internal project for traffic management system integration.
