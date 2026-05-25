"""
pipeline/preprocess.py
Stage 1 for the webapp: preprocess a single video with FFmpeg.

Mirrors stage1_preprocess.py exactly:
  - Resize to 224×224
  - Standardize to 30 fps
  - Remove audio
  - Codec: libx264, crf=23, preset=medium

No DataFrames, no CSV, no metadata — just one video in, one video out.
"""

import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def preprocess_video(
    input_path: str | Path,
    output_path: str | Path,
    target_fps: int = 30,
    width: int = 224,
    height: int = 224,
    timeout_sec: int = 1800,
) -> tuple[bool, str | None]:
    """
    Preprocess a single video using FFmpeg.

    Args:
        input_path   : Path to the raw input video.
        output_path  : Destination path for the preprocessed video.
        target_fps   : Output frame rate (default 30, matches training data).
        width/height : Resize dimensions (default 224×224, matches X3D-M input).
        timeout_sec  : Kill FFmpeg if it exceeds this many seconds (default 30 min).

    Returns:
        (True,  None)          on success
        (False, error_message) on failure
    """
    input_path  = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        return False, f"Input file not found: {input_path}"

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-i",  str(input_path),
        "-vf", f"fps={target_fps},scale={width}:{height}",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-an",          # strip audio
        "-y",           # overwrite without prompt
        str(output_path),
    ]

    logger.info(f"FFmpeg: {input_path.name} → {output_path.name}")
    logger.debug(f"CMD: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            check=False,
        )

        if result.returncode == 0:
            size_mb = output_path.stat().st_size / (1024 ** 2)
            logger.info(f"  ✓ Preprocessed: {size_mb:.1f} MB")
            return True, None
        else:
            err = result.stderr.decode("utf-8", errors="ignore")
            # Truncate to first 300 chars — FFmpeg stderr is very verbose
            short_err = err[:300].strip()
            logger.error(f"  ✗ FFmpeg failed (rc={result.returncode}): {short_err}")
            return False, f"FFmpeg error (rc={result.returncode}): {short_err}"

    except subprocess.TimeoutExpired:
        msg = f"FFmpeg timed out after {timeout_sec}s"
        logger.error(f"  ✗ {msg}")
        return False, msg

    except FileNotFoundError:
        msg = "FFmpeg not found. Install it with: sudo apt install ffmpeg"
        logger.error(f"  ✗ {msg}")
        return False, msg

    except Exception as e:
        logger.error(f"  ✗ Unexpected error: {e}")
        return False, str(e)
