import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TemporalAttention(nn.Module):
    """Computes soft attention weights across frames to highlight anomaly timestamps."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(
        self,
        rnn_outputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            rnn_outputs: Tensor of shape (batch_size, seq_len, hidden_dim)
            mask: Optional boolean tensor (batch_size, seq_len), True for valid frames
        Returns:
            context: (batch_size, hidden_dim)
            weights: (batch_size, seq_len, 1)
        """
        scores = self.attn(rnn_outputs)  # (batch_size, seq_len, 1)
        if mask is not None:
            # Mask out padded positions
            scores = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))

        weights = F.softmax(scores, dim=1)  # (batch_size, seq_len, 1)
        # Avoid NaN if whole sequence was masked
        weights = torch.nan_to_num(weights, nan=0.0)
        context = torch.sum(weights * rnn_outputs, dim=1)  # (batch_size, hidden_dim)
        return context, weights


class OptimizedCrimeClassifier(nn.Module):
    """
    Modern Video Anomaly Classifier with LayerNorm, Bidirectional LSTM,
    Temporal Attention Pooling, and Dropout Regularization.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 256,
        num_classes: int = 5,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        self.layer_norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attention = TemporalAttention(hidden_dim * num_directions)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * num_directions, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, seq_len, input_dim)
            lengths: Optional 1D tensor with sequence lengths
        Returns:
            logits: (batch_size, num_classes)
            attention_weights: (batch_size, seq_len, 1)
        """
        x = self.layer_norm(x)

        mask = None
        if lengths is not None:
            max_len = x.size(1)
            lengths_cpu = lengths.cpu()
            mask = torch.arange(max_len, device=x.device)[None, :] < lengths[:, None]
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            out, _ = self.lstm(x)

        context, attn_weights = self.attention(out, mask=mask)
        logits = self.fc(context)
        return logits, attn_weights


class LegacyCrimeClassifierLSTM(nn.Module):
    """
    Exact architecture matching original 'crime_lstm.pth' weights
    to ensure full backward compatibility.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 256,
        num_classes: int = 5,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, (h_n, _) = self.lstm(packed)
        else:
            _, (h_n, _) = self.lstm(x)

        out = self.fc(h_n[-1])
        return out
