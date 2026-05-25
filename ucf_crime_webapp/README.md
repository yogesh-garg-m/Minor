# UCF-Crime Anomaly Detection Webapp

End-to-end inference: raw video → anomaly timestamps.

## Project Structure

```
ucf_crime_webapp/
├── app.py                      # Flask app, models loaded at startup
├── config.py                   # All paths + device settings (edit this)
├── requirements.txt
├── pipeline/
│   ├── model.py                # AnomalyDetector definition (single source of truth)
│   ├── preprocess.py           # Stage 1: FFmpeg 224×224 30fps
│   ├── extract_features.py     # Stage 2: X3D-M chunked feature extraction
│   └── infer.py                # Stage 3: TCN+MIL scoring → timestamps
├── templates/
│   └── index.html              # UI: upload → progress → timeline + events table
└── temp/                       # Created at runtime, persists for the session
    ├── uploads/
    ├── preprocessed/
    ├── features/
    └── results/
```

## Setup

```bash
# 1. Copy your project's x3d_model.py into this directory
cp /Data3/cs_23103166/ucf_crime_project/x3d_model.py .

# 2. Install dependencies
pip install -r requirements.txt

# 3. Edit config.py if needed
#    CHECKPOINT_PATH and USE_GPU are the two fields you'll touch most
```

## Running

### CPU mode (current default)
```bash
python app.py
```
Open http://localhost:5000


### GPU mode (when CUDA is available)
In `config.py`:
```python
USE_GPU = True
```
Then:
```bash
python app.py
```
The app detects CUDA at startup and automatically switches to:
- `chunk_size = 30000` (vs 5000 on CPU)
- `batch_size = 32`    (vs 4 on CPU)

These match the original H100 training settings.

## CPU vs GPU — what changes

| Setting         | CPU                  | GPU (H100)            |
|----------------|----------------------|-----------------------|
| `USE_GPU`       | `False`              | `True`                |
| `chunk_size`    | 5000 frames          | 30000 frames          |
| `batch_size`    | 4 segments           | 32 segments           |
| Feature time    | ~2–10 min/video      | ~10–30 sec/video      |
| Memory          | ~2–4 GB RAM          | ~4–8 GB VRAM          |

The pipeline logic, normalisation, segment parameters, and model weights
are identical between CPU and GPU paths. Only throughput differs.

## API

### POST /analyze
Upload a video file (multipart `video` field).

Returns:
```json
{
  "status":        "ok",
  "job_id":        "abc123...",
  "filename":      "example.mp4",
  "duration_sec":  120.5,
  "n_segments":    450,
  "is_anomalous":  true,
  "max_score":     0.923,
  "mean_score":    0.312,
  "events": [
    { "start_sec": 34.1, "end_sec": 41.6, "start_hms": "00:00:34.10", "end_hms": "00:00:41.60" }
  ],
  "scores":        [...],
  "smooth_scores": [...],
  "timings": {
    "preprocess_sec": 8.2,
    "features_sec":   180.3,
    "infer_sec":      0.4,
    "total_sec":      189.1
  }
}
```

### GET /result/<job_id>
Retrieve a previously computed result.

### GET /health
Check model load status and device.

## Architecture notes

### Dilation fix
`stage3_inference.py` (original) had wrong dilations:
- `branch_medium`: dilation=2 (should be 4)
- `branch_long`:   dilation=4 (should be 16)

`pipeline/model.py` corrects this to match training exactly.
The checkpoint loads without error either way (same weight shapes),
but scores are wrong with the original inference dilations.

### Single-video batch handling
The model expects `(B, T, 2048)`. For a single video we `unsqueeze(0)`
to get `(1, T, 2048)`. No batch handling is needed beyond this since
inference is always one video at a time.

### Threading
A `threading.Lock` in `app.py` ensures only one video is processed at
a time. Subsequent requests block and queue behind it. This prevents
GPU OOM and CPU RAM exhaustion from concurrent heavy workloads.
