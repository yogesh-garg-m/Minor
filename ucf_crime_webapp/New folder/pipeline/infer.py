"""
pipeline/infer.py
Stage 3 inference for the webapp: score a single .npy feature file.

Key fixes over the original stage3_inference.py:
  - Dilations match training: branch_medium=4, branch_long=16
    (original had 2 and 4 — same weight shapes so no crash, but wrong computation)
  - Model is defined in pipeline/model.py (single source of truth)
  - No argparse / file-scanning — takes a single path, returns a dict
"""

import numpy as np
import torch
import logging
from pathlib import Path
from scipy.ndimage import median_filter

from pipeline.model import AnomalyDetector, load_detector

logger = logging.getLogger(__name__)

# Segment → time constants (must match feature extraction)
WINDOW_FRAMES = 16
STRIDE_FRAMES = 8
FPS           = 30


def _segment_to_time(seg_idx: int) -> float:
    """Convert segment index to start time in seconds."""
    return (seg_idx * STRIDE_FRAMES) / FPS


def _merge_events(anomalous_indices: np.ndarray) -> list[dict]:
    """
    Merge consecutive anomalous segment indices into time-stamped events.

    Args:
        anomalous_indices: 1-D array of segment indices flagged as anomalous.

    Returns:
        List of dicts: [{start_sec, end_sec, start_hms, end_hms}, ...]
    """
    if len(anomalous_indices) == 0:
        return []

    events = []
    s = e = anomalous_indices[0]

    for idx in anomalous_indices[1:]:
        if idx == e + 1:
            e = idx
        else:
            events.append(_make_event(s, e))
            s = e = idx

    events.append(_make_event(s, e))
    return events


def _make_event(seg_start: int, seg_end: int) -> dict:
    t_start = _segment_to_time(seg_start)
    t_end   = _segment_to_time(seg_end) + WINDOW_FRAMES / FPS
    return {
        "start_sec" : round(t_start, 2),
        "end_sec"   : round(t_end,   2),
        "start_hms" : _sec_to_hms(t_start),
        "end_hms"   : _sec_to_hms(t_end),
    }


def _sec_to_hms(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mm for display."""
    h  = int(seconds // 3600)
    m  = int((seconds % 3600) // 60)
    s  = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def score_npy(
    npy_path: str | Path,
    model: AnomalyDetector,
    device: torch.device,
    score_threshold: float = 0.5,
    smooth_window: int = 7,
) -> dict:
    """
    Run the anomaly detector on a single .npy feature file.

    Args:
        npy_path        : Path to (N, 2048) numpy feature array.
        model           : Loaded AnomalyDetector in eval mode.
        device          : torch.device.
        score_threshold : Segments with smoothed score ≥ this are anomalous.
        smooth_window   : Median filter window size (in segments).

    Returns:
        {
          "n_segments"    : int,
          "duration_sec"  : float,
          "is_anomalous"  : bool,
          "max_score"     : float,
          "mean_score"    : float,
          "events"        : [ {start_sec, end_sec, start_hms, end_hms}, ... ],
          "scores"        : [float, ...]   ← per-segment raw scores for plotting
          "smooth_scores" : [float, ...]   ← smoothed scores for plotting
        }
    """
    npy_path = Path(npy_path)
    arr = np.load(npy_path).astype(np.float32)   # (N, 2048)

    # Single video → add batch dim: (1, N, 2048)
    x = torch.from_numpy(arr).unsqueeze(0).to(device)

    with torch.no_grad():
        raw = model(x)          # (1, N, 1)
    
    scores = raw.squeeze().cpu().numpy()   # (N,)
    if scores.ndim == 0:
        scores = scores.reshape(1)

    # Median filter smoothing
    win    = min(smooth_window, len(scores))
    smooth = median_filter(scores, size=win)

    anomalous_idx = np.where(smooth >= score_threshold)[0]
    events        = _merge_events(anomalous_idx)

    n_segments   = len(scores)
    duration_sec = _segment_to_time(n_segments - 1) + WINDOW_FRAMES / FPS

    result = {
        "n_segments"    : n_segments,
        "duration_sec"  : round(duration_sec, 2),
        "is_anomalous"  : len(events) > 0,
        "max_score"     : round(float(scores.max()),  4),
        "mean_score"    : round(float(scores.mean()), 4),
        "events"        : events,
        "scores"        : [round(float(s), 4) for s in scores.tolist()],
        "smooth_scores" : [round(float(s), 4) for s in smooth.tolist()],
    }

    logger.info(
        f"  Scored {n_segments} segments | "
        f"max={result['max_score']:.3f} | "
        f"anomalous={result['is_anomalous']} | "
        f"{len(events)} event(s)"
    )
    return result
