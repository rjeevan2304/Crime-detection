# Real-Time Crime & Anomaly Detection from Video Surveillance

Automated real-time criminal activity detection and alerting pipeline using Deep Learning (CNN + Temporal Attention LSTM / BiLSTM) with PyTorch and OpenCV.

---

## 📌 Documentation & Reports
- [Project Brief (PDF)](https://github.com/user-attachments/files/21161407/Brief.pdf)
- [Detailed Technical Report (PDF)](https://github.com/user-attachments/files/21161414/Report.in.detail.pdf)

---

## 🚀 Key Optimizations over Baseline

| Feature | Baseline Notebook | Optimized Pipeline |
| :--- | :--- | :--- |
| **Feature Extraction** | Frame-by-frame on CPU (~20–150s per video) | **Batched GPU processing (AMP FP16)** (~15–30x faster) |
| **Backbone Support** | ResNet-50 only (`pretrained=True` deprecated) | **ResNet-50, MobileNet-V3-Large, EfficientNet-B0** |
| **Temporal Modeling** | Unidirectional LSTM using only final hidden state `h_n[-1]` | **Bidirectional LSTM + Temporal Attention Pooling** |
| **Keyframe Selection** | Saves the last frame of the video | **Saves frame with maximum temporal attention** |
| **Inference Mode** | Offline video batch analysis only | **Live CCTV / RTSP / Webcam sliding window detector** |
| **Alerting** | Hardcoded credentials, synchronous blocking SMS | **Non-blocking asynchronous alerts with debounce cooldown** |
| **Architecture** | Monolithic Jupyter Notebook | **Clean, modular Python package + CLI scripts** |

---

## 📂 Project Structure

```
Crime-detection/
├── config/
│   └── settings.yaml          # Hyperparameters and pipeline configuration
├── src/
│   ├── __init__.py
│   ├── models.py              # OptimizedCrimeClassifier, Attention & Legacy models
│   ├── feature_extractor.py   # Batched CNN feature extractor (ResNet / MobileNet)
│   ├── dataset.py             # PyTorch Dataset & padded collation loader
│   ├── train.py               # Training loop with validation & checkpointing
│   ├── alert_service.py       # Asynchronous Twilio SMS alerting with cooldown
│   └── realtime_detector.py   # Live sliding-window CCTV / stream detector
├── .env.example               # Environment variable template for API keys
├── .gitignore
├── requirements.txt           # Project dependencies
├── code.ipynb                 # Interactive notebook with clean demonstrations
├── crime_lstm.pth             # Pretrained baseline model weights
└── README.md
```

---

## 🛠️ Installation & Setup

### 1. Clone & create virtual environment
```bash
git clone https://github.com/rjeevan2304/Crime-detection.git
cd Crime-detection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and configure your Twilio credentials:
```bash
cp .env.example .env
```
Fill in `.env`:
```ini
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_PHONE_NUMBER=+1234567890
ALERT_RECIPIENT_PHONE=+1987654321
```
*(If Twilio credentials are not set, alerts will be safely simulated in console logs without throwing errors).*

---

## 💻 Usage

### 1. Fast Batched Feature Extraction
Extract 2048-dimensional features across your dataset videos using GPU batching:
```python
from src.feature_extractor import FastFeatureExtractor

extractor = FastFeatureExtractor(backbone="resnet50")
# Process an entire folder of videos in parallel batches
extractor.process_directory(
    input_dir="robbery",
    output_dir="processed_frames/robbery",
    frame_skip=5,
    max_frames=100,
    batch_size=32
)
```

### 2. Train the Optimized Classifier
```bash
python -m src.train \
  --data-dir processed_frames \
  --epochs 30 \
  --batch-size 16 \
  --lr 0.0003 \
  --model-type optimized \
  --save-path checkpoints/crime_model_best.pth
```

### 3. Real-Time CCTV Stream Inference
Run real-time inference on a webcam, video file, or RTSP network camera stream:
```bash
# Webcam stream
python -m src.realtime_detector --source 0 --checkpoint crime_lstm.pth

# Video file
python -m src.realtime_detector --source test_video.mp4 --threshold 0.75

# Headless server (no GUI window)
python -m src.realtime_detector --source rtsp://camera-ip:554/live --no-display
```

---

## 🏷️ Anomaly Classes
- `0`: Abuse
- `1`: Assault
- `2`: Vandalism
- `3`: Robbery
- `4`: Burglary

