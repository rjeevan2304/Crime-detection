"""Crime Detection Package - High-performance video anomaly classification."""

from .models import OptimizedCrimeClassifier, LegacyCrimeClassifierLSTM, TemporalAttention
from .feature_extractor import FastFeatureExtractor
from .dataset import CrimeDataset, pad_collate_fn
from .alert_service import AlertManager

__all__ = [
    "OptimizedCrimeClassifier",
    "LegacyCrimeClassifierLSTM",
    "TemporalAttention",
    "FastFeatureExtractor",
    "CrimeDataset",
    "pad_collate_fn",
    "AlertManager",
]
