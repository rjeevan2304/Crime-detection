import os
import argparse
from collections import deque
from datetime import datetime
from typing import Optional, Dict

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .models import OptimizedCrimeClassifier, LegacyCrimeClassifierLSTM


class RealTimeCrimeDetector:
    """
    Real-time video anomaly detection pipeline with sliding window buffer,
    peak attention frame extraction, and async alerting.
    """

    def __init__(
        self,
        checkpoint_path: str,
        backbone: str = "resnet50",
        window_size: int = 32,
        frame_interval: int = 3,
        threshold: float = 0.75,
        save_dir: str = "detected_frames",
        cooldown_seconds: int = 60,
        device: Optional[torch.device] = None
    ):
        self.device = device or get_default_device()
        self.window_size = window_size
        self.frame_interval = frame_interval
        self.threshold = threshold
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        print(f"Initializing FastFeatureExtractor ({backbone}) on {self.device}...")
        self.feature_extractor = FastFeatureExtractor(backbone=backbone, device=self.device)

        # Invert label map: id -> class name
        self.idx_to_label: Dict[int, str] = {v: k for k, v in DEFAULT_LABEL_MAP.items()}

        # Load Classifier
        self.is_legacy = False
        self.classifier = self._load_model(checkpoint_path)
        self.alert_manager = AlertManager(cooldown_seconds=cooldown_seconds)

    def _load_model(self, checkpoint_path: str) -> torch.nn.Module:
        input_dim = self.feature_extractor.feature_dim
        num_classes = len(self.idx_to_label)

        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint '{checkpoint_path}' not found! Instantiating untrained model.")
            model = OptimizedCrimeClassifier(input_dim=input_dim, hidden_dim=256, num_classes=num_classes)
            return model.to(self.device).eval()

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Check if saved with full metadata or raw state_dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            model_type = checkpoint.get("model_type", "optimized")
            if "label_map" in checkpoint:
                self.idx_to_label = {v: k for k, v in checkpoint["label_map"].items()}
        else:
            state_dict = checkpoint
            # Inspect keys: if no attention layer, it's legacy
            model_type = "legacy" if any("fc.weight" in k for k in state_dict.keys()) and not any("attention" in k for k in state_dict.keys()) else "optimized"

        if model_type == "legacy":
            print("Loading LegacyCrimeClassifierLSTM architecture...")
            self.is_legacy = True
            model = LegacyCrimeClassifierLSTM(input_dim=input_dim, hidden_dim=256, num_classes=num_classes)
        else:
            print("Loading OptimizedCrimeClassifier architecture...")
            model = OptimizedCrimeClassifier(input_dim=input_dim, hidden_dim=256, num_classes=num_classes)

        model.load_state_dict(state_dict)
        return model.to(self.device).eval()

    def process_stream(
        self,
        source: str = "0",
        display: bool = True
    ):
        """
        Runs live anomaly detection on webcam, video file, or RTSP camera stream.
        """
        # Parse numeric camera index or string path
        video_src = int(source) if source.isdigit() else source
        cap = cv2.VideoCapture(video_src)

        if not cap.isOpened():
            print(f"Failed to open video source: {source}")
            return

        feature_buffer = deque(maxlen=self.window_size)
        frame_buffer = deque(maxlen=self.window_size)
        frame_idx = 0

        current_class = "normal"
        current_conf = 0.0

        print(f"Streaming from '{source}'... Press 'q' in video window to exit.")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if frame_idx % self.frame_interval != 0:
                    continue

                # Extract features for single frame
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                feats = self.feature_extractor.extract_from_frames([rgb], batch_size=1)
                feats_tensor = torch.as_tensor(feats, dtype=torch.float32)

                feature_buffer.append(feats_tensor)
                frame_buffer.append(frame)

                # Predict once the temporal buffer has enough context
                if len(feature_buffer) >= min(16, self.window_size):
                    seq_tensor = torch.cat(list(feature_buffer), dim=0).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        out = self.classifier(seq_tensor)
                        if isinstance(out, tuple):
                            logits, attn_weights = out
                        else:
                            logits, attn_weights = out, None

                        probs = F.softmax(logits, dim=1).squeeze(0)
                        top_prob, top_idx = torch.max(probs, dim=0)
                        current_conf = top_prob.item()
                        current_class = self.idx_to_label.get(top_idx.item(), f"class_{top_idx.item()}")

                    if current_conf >= self.threshold:
                        # Extract the most critical frame using attention weights
                        if attn_weights is not None:
                            peak_idx = torch.argmax(attn_weights.squeeze()).item()
                            peak_idx = min(peak_idx, len(frame_buffer) - 1)
                            peak_frame = frame_buffer[peak_idx]
                        else:
                            peak_frame = frame

                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        snap_name = f"{current_class}_{timestamp}.jpg"
                        snap_path = os.path.join(self.save_dir, snap_name)
                        cv2.imwrite(snap_path, peak_frame)

                        self.alert_manager.trigger_alert(current_class, current_conf, snap_path)

                # Render live status overlay
                if display:
                    color = (0, 0, 255) if current_conf >= self.threshold else (0, 255, 0)
                    status_text = f"{current_class.upper()} ({current_conf * 100:.1f}%)"
                    cv2.putText(
                        frame,
                        status_text,
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        color,
                        2,
                        cv2.LINE_AA
                    )
                    cv2.imshow("Crime Detection Stream", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
            self.alert_manager.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Real-Time Crime Detection")
    parser.add_argument("--source", type=str, default="0", help="Video file path, RTSP stream URL, or webcam ID")
    parser.add_argument("--checkpoint", type=str, default="crime_lstm.pth", help="Path to model weights")
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "mobilenet_v3", "efficientnet_b0"])
    parser.add_argument("--threshold", type=float, default=0.70, help="Confidence threshold for alerting")
    parser.add_argument("--window-size", type=int, default=32, help="Temporal sliding window size")
    parser.add_argument("--frame-interval", type=int, default=3, help="Sample every Nth frame")
    parser.add_argument("--no-display", action="store_true", help="Disable GUI display window (headless mode)")
    args = parser.parse_args()

    detector = RealTimeCrimeDetector(
        checkpoint_path=args.checkpoint,
        backbone=args.backbone,
        window_size=args.window_size,
        frame_interval=args.frame_interval,
        threshold=args.threshold
    )
    detector.process_stream(source=args.source, display=not args.no_display)


if __name__ == "__main__":
    main()
