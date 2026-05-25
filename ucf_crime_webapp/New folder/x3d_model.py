# x3d_model.py - Drop-in replacement for i3d_model.py
# Loads X3D-M from PyTorchVideo (facebookresearch/pytorchvideo),
# hooks the global-pool layer to extract 2048-D features.
#
# model(batch) signature:
#   Input : (B, C, 16, 224, 224)  float32, ImageNet mean/std normalised
#   Output: (B, 2048)             float32

import torch
import torch.nn as nn


def _ensure_pytorchvideo():
    try:
        import pytorchvideo  # noqa: F401
    except ImportError:
        import subprocess, sys
        print("[x3d_model] pytorchvideo not found - installing...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "pytorchvideo", "fvcore", "iopath", "--quiet",
        ])
        print("[x3d_model] Installation complete.")

_ensure_pytorchvideo()


class X3DFeatureExtractor(nn.Module):
    """
    Wraps X3D-M so that forward() returns (B, 2048) feature vectors
    instead of class logits.

    Architecture:
        blocks[0]   : stem
        blocks[1-4] : residual stages
        blocks[5]   : head (pool + dropout + linear)
                      we replace the linear with Identity to get 2048-D output
    """

    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device

        print("[x3d_model] Loading X3D-M weights from torch.hub...")
        model = torch.hub.load(
            "facebookresearch/pytorchvideo",
            "x3d_m",
            pretrained=True,
        )

        # Remove the final classification linear from the head
        # so the model outputs the 2048-D pooled vector instead of logits.
        head = model.blocks[5]
        if hasattr(head, "proj"):
            head.proj = nn.Identity()
        elif hasattr(head, "output_pool"):
            proj_seq = list(head.output_pool.children())
            head.output_pool = nn.Sequential(*proj_seq[:-1]) if len(proj_seq) > 1 else nn.Identity()
        else:
            # Fallback: find and replace any Linear in the head
            for name, child in head.named_children():
                if isinstance(child, nn.Linear):
                    setattr(head, name, nn.Identity())
                    break

        # Remove Softmax activation -- without this, features collapse to ~1/2048
        if hasattr(head, "activation"):
            head.activation = nn.Identity()

        self.backbone = model.to(device)
        print("[x3d_model] X3D-M ready. Output dim: 2048")

    def forward(self, x):
        """
        Args:
            x : (B, C, T, H, W) float32, ImageNet-normalised, T=16, H=W=224
        Returns:
            feats : (B, 2048) float32
        """
        feats = self.backbone(x)

        # Squeeze any leftover spatial/temporal dims, e.g. (B, 2048, 1, 1, 1)
        if feats.dim() > 2:
            feats = feats.flatten(1)

        return feats  # (B, 2048)


def load_x3d_pretrained(device='cuda'):
    """
    Factory function used by stage2_extract_features.py.

    Usage:
        from x3d_model import load_x3d_pretrained
        model = load_x3d_pretrained(device='cuda')
        model.eval()
        # model(batch) -> (B, 2048)

    Weights are downloaded automatically from torch.hub (~130 MB,
    cached in ~/.cache/torch/hub/).
    """
    return X3DFeatureExtractor(device=device)