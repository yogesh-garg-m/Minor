"""
pipeline/model.py
Single source of truth for the AnomalyDetector architecture.
Must match stage3_train.py exactly — do NOT change layer params without
also retraining the model.

Architecture:
    Input : (B, T, 2048)   ← X3D-M features
    TCN   : (B, T, 2048) → (B, T, 512)
    Scorer: (B, T, 512)  → (B, T, 1)   sigmoid scores ∈ [0, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Constants (match stage3_train.py) ─────────────────────────────────────────
FEATURE_DIM = 2048
TCN_HID     = 512
MIL_HID1    = 256
MIL_HID2    = 128


class MultiScaleTCN(nn.Module):
    """
    Multi-scale dilated temporal convolutional network.

    Stage 1 : Linear dim reduction  2048 → 512  (Conv1d 1×1)
    Stage 2 : Three parallel dilated branches, each 512 → 256
              - short  : dilation=1,  receptive field ≈ 3  segments
              - medium : dilation=4,  receptive field ≈ 9  segments
              - long   : dilation=16, receptive field ≈ 33 segments
    Stage 3 : Fusion  768 → 512  (Conv1d 1×1 + Dropout + ReLU)
    Stage 4 : Refinement with residual  (Conv1d 3×1, dilation=1)

    Dilations MUST match training — changing them post-training silently
    corrupts scores (same weight shapes, different computation).
    """

    def __init__(self, in_dim: int = FEATURE_DIM, hid: int = TCN_HID):
        super().__init__()

        # Stage 1 — stored as Sequential but forward only calls [0] (Conv1d).
        # LayerNorm at [1] and ReLU at [2] are applied manually after transpose.
        # This matches the exact state_dict key layout from training.
        self.reduce = nn.Sequential(
            nn.Conv1d(in_dim, hid, kernel_size=1),   # reduce.0
            nn.LayerNorm(hid),                        # reduce.1  (applied after transpose)
            nn.ReLU(inplace=True),                    # reduce.2  (dead — never called)
        )
        self.ln1 = nn.LayerNorm(hid)

        branch_out = hid // 2  # 256

        # Stage 2 — dilations from training: 1 / 4 / 16
        self.branch_short  = nn.Conv1d(hid, branch_out, kernel_size=3,
                                       dilation=1,  padding=1)
        self.branch_medium = nn.Conv1d(hid, branch_out, kernel_size=3,
                                       dilation=4,  padding=4)
        self.branch_long   = nn.Conv1d(hid, branch_out, kernel_size=3,
                                       dilation=16, padding=16)

        # Stage 3
        self.fusion = nn.Sequential(
            nn.Conv1d(branch_out * 3, hid, kernel_size=1),
            nn.Dropout(p=0.3),
            nn.ReLU(inplace=True),
        )
        self.ln2 = nn.LayerNorm(hid)

        # Stage 4
        self.refine = nn.Conv1d(hid, hid, kernel_size=3, dilation=1, padding=1)
        self.ln3    = nn.LayerNorm(hid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T, C)
        x = x.transpose(1, 2)           # (B, C, T)
        x = self.reduce[0](x)           # Conv1d  → (B, 512, T)
        x = x.transpose(1, 2)           # (B, T, 512)
        x = self.ln1(x)
        x = F.relu(x, inplace=True)
        x = x.transpose(1, 2)           # (B, 512, T)

        a = F.relu(self.branch_short(x),  inplace=True)   # (B, 256, T)
        b = F.relu(self.branch_medium(x), inplace=True)
        c = F.relu(self.branch_long(x),   inplace=True)

        fused   = torch.cat([a, b, c], dim=1)   # (B, 768, T)
        x_fused = self.fusion(fused)             # (B, 512, T)
        x_fused = x_fused.transpose(1, 2)
        x_fused = self.ln2(x_fused)
        x_fused = x_fused.transpose(1, 2)

        refined = self.refine(x_fused)           # (B, 512, T)
        x_out   = x_fused + refined              # residual
        x_out   = x_out.transpose(1, 2)          # (B, T, 512)
        x_out   = self.ln3(x_out)
        return x_out                             # (B, T, 512)


class MILScorer(nn.Module):
    """
    MIL anomaly scorer.
    Input : (B, T, 512)
    Output: (B, T, 1)  — sigmoid scores ∈ [0, 1]
    """

    def __init__(self, in_dim: int = TCN_HID):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, MIL_HID1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(MIL_HID1, MIL_HID2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(MIL_HID2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)   # (B, T, 1)


class AnomalyDetector(nn.Module):
    """
    Full pipeline: features → per-segment anomaly scores.
    Input : (B, T, 2048)
    Output: (B, T, 1)
    """

    def __init__(self):
        super().__init__()
        self.tcn    = MultiScaleTCN(FEATURE_DIM, TCN_HID)
        self.scorer = MILScorer(TCN_HID)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats  = self.tcn(x)        # (B, T, 512)
        scores = self.scorer(feats)  # (B, T, 1)
        return scores


def load_detector(checkpoint_path: str, device: torch.device) -> AnomalyDetector:
    """
    Instantiate AnomalyDetector and load weights from checkpoint.
    Raises FileNotFoundError if checkpoint is missing.
    """
    from pathlib import Path
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = AnomalyDetector().to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model
