"""
pipeline/extract_features.py
Stage 2 for the webapp: extract X3D-M features from a single preprocessed video.

Mirrors stage2_extract_features.py exactly:
  - Same ImageNet normalisation (mean=0.45, std=0.225 on [0,1] pixels)
  - Same segment parameters: window=16 frames, 50% overlap (stride=8)
  - Same chunked loading to handle arbitrarily long videos
  - Same unfold trick for zero-copy segment creation
  - Output: numpy array (N, 2048) saved as .npy

Two device paths:
  CPU path — chunk_size=5000, batch_size=4   (RAM-safe for typical server)
  GPU path — chunk_size=30000, batch_size=32  (matches original H100 settings)
"""

import gc
import cv2
import numpy as np
import torch
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ImageNet normalisation constants — X3D-M was trained with these
_IMAGENET_MEAN = np.array([0.45, 0.45, 0.45], dtype=np.float32)
_IMAGENET_STD  = np.array([0.225, 0.225, 0.225], dtype=np.float32)

# Segment parameters — must match Stage 2 training extraction exactly
SEGMENT_LENGTH = 16   # frames per clip (X3D-M input length)
OVERLAP        = 0.5
STRIDE         = int(SEGMENT_LENGTH * (1 - OVERLAP))  # = 8


def _count_frames(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def _read_frames_chunk(video_path: Path, start_frame: int, num_frames: int) -> np.ndarray | None:
    """
    Read a contiguous chunk of frames from a video.
    Returns uint8 array (T, 224, 224, 3) or None on failure.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    return np.array(frames, dtype=np.uint8) if frames else None


def _normalize(frames: np.ndarray) -> np.ndarray:
    """
    uint8 (T, H, W, 3) → float32 (T, H, W, 3), ImageNet-normalised.
    Matches stage2_extract_features.py normalise_frames() exactly.
    """
    f = frames.astype(np.float32) / 255.0
    return (f - _IMAGENET_MEAN) / _IMAGENET_STD


def _extract_chunk_features(
    frames_norm: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> np.ndarray | None:
    """
    Build 16-frame overlapping segments via torch.unfold (zero-copy view),
    run X3D-M in mini-batches, return (N, 2048) float32.

    Mirrors create_segments_and_extract() from stage2 exactly.
    """
    if len(frames_norm) < SEGMENT_LENGTH:
        return None

    # (T, H, W, C) → (C, T, H, W) — shares numpy memory
    tensor_frames = torch.from_numpy(frames_norm).permute(3, 0, 1, 2)

    # unfold along T dim: (C, T, H, W) → (C, N, H, W, seg_len)
    segments = tensor_frames.unfold(1, SEGMENT_LENGTH, STRIDE)

    # (C, N, H, W, seg_len) → (N, C, seg_len, H, W)
    segments = segments.permute(1, 0, 4, 2, 3).contiguous()
    del tensor_frames

    num_segments = len(segments)
    all_features = []

    for i in range(0, num_segments, batch_size):
        batch = segments[i : i + batch_size].float().to(device)

        with torch.no_grad():
            feats = model(batch)   # (B, 2048)

        all_features.append(feats.cpu().numpy())

        del batch, feats
        if device.type == "cuda":
            torch.cuda.empty_cache()

    del segments
    return np.concatenate(all_features, axis=0)   # (N, 2048)


def extract_features(
    video_path: str | Path,
    output_npy_path: str | Path,
    x3d_model: torch.nn.Module,
    device: torch.device,
    chunk_size: int,
    batch_size: int,
    progress_callback=None,
) -> tuple[dict | None, str | None]:
    """
    Extract X3D-M features from a single preprocessed video.

    Args:
        video_path       : Path to preprocessed .mp4 (224×224, 30fps).
        output_npy_path  : Where to save the (N, 2048) feature array.
        x3d_model        : Loaded X3D-M model (already on device, eval mode).
        device           : torch.device('cpu') or torch.device('cuda').
        chunk_size       : Frames to load per chunk (5000 CPU / 30000 GPU).
        batch_size       : Segments per forward pass (4 CPU / 32 GPU).
        progress_callback: Optional callable(chunk_idx, num_chunks) for UI updates.

    Returns:
        (metadata_dict, None)   on success
        (None, error_string)    on failure
    """
    video_path      = Path(video_path)
    output_npy_path = Path(output_npy_path)
    output_npy_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        total_frames = _count_frames(video_path)
        if total_frames < SEGMENT_LENGTH:
            return None, f"Video too short: {total_frames} frames (need ≥ {SEGMENT_LENGTH})"

        logger.info(
            f"Feature extraction: {video_path.name} | "
            f"{total_frames} frames ({total_frames/30/60:.1f} min) | "
            f"device={device.type} | chunk={chunk_size} | batch={batch_size}"
        )

        all_features = []

        if total_frames <= chunk_size:
            # ── Full video loading (short video or large chunk_size on GPU) ──
            logger.info("  Strategy: full video")
            frames = _read_frames_chunk(video_path, 0, total_frames)
            if frames is None:
                return None, "Failed to read video frames"

            frames_norm = _normalize(frames)
            del frames

            feats = _extract_chunk_features(frames_norm, x3d_model, device, batch_size)
            del frames_norm

            if feats is None:
                return None, "Chunk too short for segment extraction"
            all_features.append(feats)

        else:
            # ── Chunked loading (long video / CPU memory constraint) ──────
            num_chunks = (total_frames + chunk_size - 1) // chunk_size
            logger.info(f"  Strategy: chunked ({num_chunks} chunks)")

            for chunk_idx in range(num_chunks):
                start = chunk_idx * chunk_size

                # 8-frame overlap between chunks to avoid boundary artefacts
                # (same logic as stage2_extract_features.py)
                if chunk_idx > 0:
                    start -= STRIDE   # = 8 frames

                frames_to_read = min(chunk_size + STRIDE, total_frames - start)

                logger.info(
                    f"    Chunk {chunk_idx+1}/{num_chunks}: "
                    f"frames {start}–{start+frames_to_read-1}"
                )

                if progress_callback:
                    progress_callback(chunk_idx, num_chunks)

                frames = _read_frames_chunk(video_path, start, frames_to_read)
                if frames is None:
                    logger.warning(f"    Chunk {chunk_idx+1}: failed to read, skipping")
                    continue

                frames_norm = _normalize(frames)
                del frames

                feats = _extract_chunk_features(frames_norm, x3d_model, device, batch_size)
                del frames_norm

                if feats is None:
                    logger.warning(f"    Chunk {chunk_idx+1}: too short, skipping")
                    continue

                # Drop the first segment from non-first chunks (boundary overlap)
                if chunk_idx > 0 and len(feats) > 1:
                    feats = feats[1:]

                all_features.append(feats)
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        if not all_features:
            return None, "No features extracted from any chunk"

        final = np.concatenate(all_features, axis=0)   # (N, 2048)
        np.save(output_npy_path, final)

        n_segments = len(final)
        logger.info(f"  ✓ Saved {n_segments} segments → {output_npy_path.name}")

        del final, all_features
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return {
            "total_frames" : total_frames,
            "n_segments"   : n_segments,
            "duration_sec" : total_frames / 30.0,
        }, None

    except Exception as e:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        logger.exception(f"Feature extraction failed: {e}")
        return None, str(e)
