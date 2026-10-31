import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from typing import Tuple

from .dataset import CrimeDataset, pad_collate_fn, DEFAULT_LABEL_MAP
from .models import OptimizedCrimeClassifier, LegacyCrimeClassifierLSTM


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for padded_seqs, labels, lengths in dataloader:
        padded_seqs = padded_seqs.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        outputs = model(padded_seqs, lengths)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs

        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping to prevent exploding gradients in RNNs
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, preds = torch.max(logits, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = (correct / max(total, 1)) * 100.0
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for padded_seqs, labels, lengths in dataloader:
        padded_seqs = padded_seqs.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        outputs = model(padded_seqs, lengths)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs

        loss = criterion(logits, labels)
        running_loss += loss.item() * labels.size(0)

        _, preds = torch.max(logits, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = (correct / max(total, 1)) * 100.0
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description="Train Crime Detection Model")
    parser.add_argument("--data-dir", type=str, default="processed_frames", help="Path to processed .npy frames")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension of LSTM")
    parser.add_argument("--input-dim", type=int, default=2048, help="Feature dimension (2048 for ResNet50, 960 for MobileNetV3)")
    parser.add_argument("--save-path", type=str, default="checkpoints/crime_model_best.pth", help="Checkpoint save destination")
    parser.add_argument("--model-type", type=str, choices=["optimized", "legacy"], default="optimized")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation set fraction")
    args = parser.parse_args()

    set_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"
    )
    print(f"Using device: {device}")

    full_dataset = CrimeDataset(args.data_dir, label_map=DEFAULT_LABEL_MAP)
    total_samples = len(full_dataset)
    if total_samples == 0:
        print(f"Warning: No .npy files found in {args.data_dir}. Please run feature extraction first.")
        return

    val_size = int(total_samples * args.val_split)
    train_size = total_samples - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate_fn
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate_fn
    )

    print(f"Dataset split: {train_size} train samples, {val_size} val samples.")

    if args.model_type == "optimized":
        model = OptimizedCrimeClassifier(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            num_classes=len(DEFAULT_LABEL_MAP),
            dropout=0.3
        )
    else:
        model = LegacyCrimeClassifierLSTM(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            num_classes=len(DEFAULT_LABEL_MAP)
        )

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        print(
            f"Epoch [{epoch + 1:02d}/{args.epochs:02d}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_acc": val_acc,
                    "model_type": args.model_type,
                    "input_dim": args.input_dim,
                    "hidden_dim": args.hidden_dim,
                    "label_map": DEFAULT_LABEL_MAP,
                },
                args.save_path
            )
            print(f"--> Saved new best checkpoint to {args.save_path} (Val Acc: {val_acc:.2f}%)")


if __name__ == "__main__":
    main()
