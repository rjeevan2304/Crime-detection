import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from typing import List, Tuple, Optional
from tqdm import tqdm


def get_default_device() -> torch.device:
    """Select the fastest available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FastFeatureExtractor:
    """
    High-throughput batched CNN feature extractor.
    Eliminates frame-by-frame overhead by batching tensor transformations
    and using GPU mixed-precision inference.
    """

    SUPPORTED_BACKBONES = {
        "resnet50": 2048,
        "mobilenet_v3": 960,
        "efficientnet_b0": 1280,
    }

    def __init__(
        self,
        backbone: str = "resnet50",
        device: Optional[torch.device] = None
    ):
        self.device = device or get_default_device()
        self.backbone = backbone.lower()

        if self.backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT
            base = models.resnet50(weights=weights)
            # Remove final classification FC layer
            self.model = nn.Sequential(*list(base.children())[:-1])
            self.pool = nn.Flatten(1)
            self.feature_dim = 2048
            self.transform = weights.transforms()
        elif self.backbone == "mobilenet_v3":
            weights = models.MobileNet_V3_Large_Weights.DEFAULT
            base = models.mobilenet_v3_large(weights=weights)
            self.model = base.features
            self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(1))
            self.feature_dim = 960
            self.transform = weights.transforms()
        elif self.backbone == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT
            base = models.efficientnet_b0(weights=weights)
            self.model = base.features
            self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(1))
            self.feature_dim = 1280
            self.transform = weights.transforms()
        else:
            raise ValueError(
                f"Unsupported backbone: {backbone}. "
                f"Choose from {list(self.SUPPORTED_BACKBONES.keys())}"
            )

        self.model.to(self.device).eval()

    @torch.no_grad()
    def extract_from_frames(
        self,
        frames: List[np.ndarray],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Extract features from a list of RGB numpy frames (H, W, 3).
        Processes in batches for maximum throughput.
        """
        if not frames:
            return np.empty((0, self.feature_dim), dtype=np.float32)

        features_batches = []
        is_cuda = self.device.type == "cuda"

        for i in range(0, len(frames), batch_size):
            chunk = frames[i : i + batch_size]
            # Convert numpy uint8 RGB (B, H, W, C) -> PyTorch (B, C, H, W)
            tensor_list = [
                self.transform(torch.from_numpy(f).permute(2, 0, 1))
                for f in chunk
            ]
            batch_tensor = torch.stack(tensor_list).to(self.device)

            if is_cuda:
                with torch.amp.autocast(device_type="cuda"):
                    out = self.model(batch_tensor)
                    feats = self.pool(out)
            else:
                out = self.model(batch_tensor)
                feats = self.pool(out)

            features_batches.append(feats.cpu().float().numpy())

        return np.concatenate(features_batches, axis=0)

    def extract_from_video(
        self,
        video_path: str,
        frame_skip: int = 5,
        max_frames: int = 100,
        batch_size: int = 32,
        return_last_frame: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Extract features directly from a video file path.
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        frames_rgb = []
        last_bgr = None
        frame_count = 0

        while cap.isOpened() and len(frames_rgb) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_rgb.append(rgb)
                last_bgr = frame

            frame_count += 1

        cap.release()

        feats = self.extract_from_frames(frames_rgb, batch_size=batch_size)
        return (feats, last_bgr) if return_last_frame else (feats, None)

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        frame_skip: int = 5,
        max_frames: int = 100,
        batch_size: int = 32
    ) -> int:
        """
        Batch-extract features for all video files in input_dir and save as .npy.
        """
        os.makedirs(output_dir, exist_ok=True)
        valid_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        video_files = [
            f for f in sorted(os.listdir(input_dir))
            if os.path.splitext(f)[1].lower() in valid_exts
        ]

        processed = 0
        for fname in tqdm(video_files, desc=f"Extracting {os.path.basename(input_dir)}"):
            vpath = os.path.join(input_dir, fname)
            out_name = os.path.splitext(fname)[0] + ".npy"
            out_path = os.path.join(output_dir, out_name)

            if os.path.exists(out_path):
                continue

            try:
                feats, _ = self.extract_from_video(
                    vpath,
                    frame_skip=frame_skip,
                    max_frames=max_frames,
                    batch_size=batch_size
                )
                np.save(out_path, feats)
                processed += 1
            except Exception as e:
                print(f"Error processing {fname}: {e}")

        return processed
