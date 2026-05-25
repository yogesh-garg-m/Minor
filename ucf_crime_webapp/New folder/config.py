"""
config.py — Central configuration for the UCF-Crime inference webapp.
Edit paths and device settings here; nothing else needs to change.
"""

from pathlib import Path

# ── Project root (your HPC path) ─────────────────────────────────────────────
BASE_DIR = Path("")
# /Data3/cs_23103166/ucf_crime_project
# ── Model checkpoint ──────────────────────────────────────────────────────────
CHECKPOINT_PATH = BASE_DIR / "outputs" / "stage3" / "checkpoints" / "best_model.pt"

# ── Device configuration ──────────────────────────────────────────────────────
# Set USE_GPU = True if a CUDA-capable GPU is available on this machine.
# The app will fall back to CPU automatically if CUDA is not found even when True.
USE_GPU = False          # ← flip to True when GPU is available

# ── X3D-M feature extraction ──────────────────────────────────────────────────
# Keep chunk_size low on CPU to avoid RAM exhaustion.
# On GPU (H100) you can push this back to 30000.
CHUNK_SIZE_CPU = 5000    # frames per chunk on CPU
CHUNK_SIZE_GPU = 30000   # frames per chunk on GPU

BATCH_SIZE_CPU = 4       # X3D-M segments per forward pass on CPU
BATCH_SIZE_GPU = 32      # safe limit on GPU (cuDNN 32-bit index)

# ── Preprocessing ─────────────────────────────────────────────────────────────
TARGET_FPS    = 30
TARGET_WIDTH  = 224
TARGET_HEIGHT = 224

# ── Inference ─────────────────────────────────────────────────────────────────
SCORE_THRESHOLD = 0.5    # segments above this are flagged anomalous
SMOOTH_WINDOW   = 7      # median filter size (segments)

# ── Webapp temp storage ───────────────────────────────────────────────────────
WEBAPP_DIR      = Path(__file__).parent
TEMP_DIR        = WEBAPP_DIR / "temp"
UPLOAD_DIR      = TEMP_DIR / "uploads"
PREPROCESSED_DIR = TEMP_DIR / "preprocessed"
FEATURES_DIR    = TEMP_DIR / "features"
RESULTS_DIR     = TEMP_DIR / "results"

# Max upload size (bytes) — 2 GB
MAX_CONTENT_LENGTH = 2 * 1024 * 1024 * 1024

# ── Segment parameters (must match Stage 2 exactly) ───────────────────────────
WINDOW_FRAMES = 16
STRIDE_FRAMES = 8    # 50 % overlap
FPS           = 30
