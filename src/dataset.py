import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Tuple, Optional

DEFAULT_LABEL_MAP = {
    "abuse": 0,
    "assault": 1,
    "vandalism": 2,
    "robbery": 3,
    "burglary": 4
}


class CrimeDataset(Dataset):
    """
    Loads pre-extracted feature .npy files from class directories.
    Features are expected to have shape (seq_len, feature_dim).
    """

    def __init__(
        self,
        data_dir: str,
        label_map: Optional[Dict[str, int]] = None,
        max_seq_len: Optional[int] = None
    ):
        self.data_dir = data_dir
        self.label_map = label_map or DEFAULT_LABEL_MAP
        self.max_seq_len = max_seq_len
        self.samples: List[Tuple[str, int]] = []

        if not os.path.exists(data_dir):
            return

        for class_name, label in self.label_map.items():
            class_path = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            npy_files = sorted(glob.glob(os.path.join(class_path, "*.npy")))
            for fpath in npy_files:
                self.samples.append((fpath, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path, label = self.samples[idx]
        feats = np.load(file_path)

        # Handle empty features edge case
        if feats.ndim == 1:
            feats = feats.reshape(1, -1)
        if len(feats) == 0:
            feats = np.zeros((1, 2048), dtype=np.float32)

        if self.max_seq_len is not None and len(feats) > self.max_seq_len:
            # Subsample or truncate
            feats = feats[: self.max_seq_len]

        # Use torch.from_numpy / torch.as_tensor for zero-copy efficiency
        feats_tensor = torch.as_tensor(feats, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return feats_tensor, label_tensor


def pad_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pads variable-length feature sequences into a uniform batch tensor.

    Returns:
        padded_seqs: (batch_size, max_seq_len, feature_dim)
        labels: (batch_size,)
        lengths: (batch_size,) original lengths before padding
    """
    sequences = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch], dim=0)
    lengths = torch.tensor([seq.shape[0] for seq in sequences], dtype=torch.long)

    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    return padded_seqs, labels, lengths
