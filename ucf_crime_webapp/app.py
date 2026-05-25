"""
app.py — UCF-Crime Anomaly Detection Webapp

Synchronous pipeline:
    Upload video → Stage 1 (FFmpeg preprocess) → Stage 2 (X3D-M features)
    → Stage 3 (AnomalyDetector inference) → JSON results → UI renders timeline

Models are loaded ONCE at startup and reused across requests.
A threading.Lock ensures only one video is processed at a time on the GPU/CPU.
"""

import uuid
import json
import logging
import threading
from pathlib import Path
from datetime import datetime

import torch
from flask import Flask, request, jsonify, render_template, send_file

import config
from pipeline.preprocess      import preprocess_video
from pipeline.extract_features import extract_features
from pipeline.infer            import score_npy
from pipeline.model            import load_detector

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("app")

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "ucf-crime-webapp-secret"
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

# Create temp dirs
for d in [config.UPLOAD_DIR, config.PREPROCESSED_DIR,
          config.FEATURES_DIR, config.RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Device selection ──────────────────────────────────────────────────────────
# CPU path: USE_GPU=False in config.py
# GPU path: USE_GPU=True  in config.py
#           Falls back to CPU automatically if CUDA is not available.
if config.USE_GPU and torch.cuda.is_available():
    DEVICE     = torch.device("cuda")
    CHUNK_SIZE = config.CHUNK_SIZE_GPU
    BATCH_SIZE = config.BATCH_SIZE_GPU
    logger.info(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
else:
    DEVICE     = torch.device("cpu")
    CHUNK_SIZE = config.CHUNK_SIZE_CPU
    BATCH_SIZE = config.BATCH_SIZE_CPU
    if config.USE_GPU and not torch.cuda.is_available():
        logger.warning("USE_GPU=True but CUDA not available — falling back to CPU")
    logger.info(f"Device: CPU | chunk={CHUNK_SIZE} | batch={BATCH_SIZE}")

# ── Load models once at startup ───────────────────────────────────────────────
logger.info("Loading X3D-M feature extractor…")
try:
    from x3d_model import load_x3d_pretrained
    X3D_MODEL = load_x3d_pretrained(device=str(DEVICE))
    X3D_MODEL.eval()
    logger.info("X3D-M loaded ✓")
except Exception as e:
    logger.error(f"Failed to load X3D-M: {e}")
    X3D_MODEL = None

logger.info(f"Loading AnomalyDetector from {config.CHECKPOINT_PATH}…")
try:
    ANOMALY_MODEL = load_detector(config.CHECKPOINT_PATH, DEVICE)
    logger.info("AnomalyDetector loaded ✓")
except Exception as e:
    logger.error(f"Failed to load AnomalyDetector: {e}")
    ANOMALY_MODEL = None

# Serialise access — one video at a time
_PIPELINE_LOCK = threading.Lock()
_JOBS = {}
_JOBS_LOCK = threading.Lock()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.after_request
def add_cors_headers(response):
    """Allow the React app to call this API and read generated videos."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.route("/api/jobs", methods=["POST"])
def create_job():
    """Start an async analysis job so React can poll stage progress."""
    if "video" not in request.files:
        return jsonify({"status": "error", "message": "No video file in request"}), 400

    video_file = request.files["video"]
    if video_file.filename == "":
        return jsonify({"status": "error", "message": "Empty filename"}), 400

    if X3D_MODEL is None or ANOMALY_MODEL is None:
        return jsonify({"status": "error",
                        "message": "Models not loaded — check server logs"}), 503

    job_id = uuid.uuid4().hex
    suffix = Path(video_file.filename).suffix.lower() or ".mp4"
    upload_path = config.UPLOAD_DIR / f"{job_id}{suffix}"
    preprocessed_path = config.PREPROCESSED_DIR / f"{job_id}.mp4"
    features_path = config.FEATURES_DIR / f"{job_id}.npy"
    result_path = config.RESULTS_DIR / f"{job_id}.json"

    video_file.save(str(upload_path))
    logger.info(f"[{job_id[:8]}] Saved async upload: {upload_path.name} "
                f"({upload_path.stat().st_size / 1e6:.1f} MB)")

    now = _now()
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "status": "queued",
            "job_id": job_id,
            "filename": video_file.filename,
            "created_at": now,
            "stages": _fresh_stages(upload_done_at=now),
            "result": None,
            "message": None,
        }

    worker = threading.Thread(
        target=_run_async_job,
        args=(job_id, video_file.filename, upload_path, preprocessed_path, features_path, result_path),
        daemon=True,
    )
    worker.start()

    return jsonify({"status": "queued", "job_id": job_id})


@app.route("/api/jobs/<job_id>")
def get_job(job_id: str):
    """Return async job progress, stage timings, and final result when complete."""
    if not job_id.isalnum():
        return jsonify({"status": "error", "message": "Invalid job id"}), 400

    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return jsonify({"status": "error", "message": "Job not found"}), 404
        snapshot = json.loads(json.dumps(job))

    now = _now()
    for stage in snapshot["stages"]:
        if stage["status"] == "running" and stage.get("started_at"):
            stage["elapsed_sec"] = round(now - stage["started_at"], 1)

    return jsonify(snapshot)


@app.route("/api/analyze", methods=["POST"])
@app.route("/analyze", methods=["POST"])
def analyze():
    """
    POST /analyze
    Multipart form field: video (file)

    Returns JSON:
    {
      "status"       : "ok" | "error",
      "message"      : str,            # only on error
      "job_id"       : str,
      "filename"     : str,
      "duration_sec" : float,
      "n_segments"   : int,
      "is_anomalous" : bool,
      "max_score"    : float,
      "mean_score"   : float,
      "events"       : [ {start_sec, end_sec, start_hms, end_hms}, ... ],
      "scores"       : [float, ...],
      "smooth_scores": [float, ...],
      "timings"      : {preprocess_sec, features_sec, infer_sec, total_sec}
    }
    """
    if "video" not in request.files:
        return jsonify({"status": "error", "message": "No video file in request"}), 400

    video_file = request.files["video"]
    if video_file.filename == "":
        return jsonify({"status": "error", "message": "Empty filename"}), 400

    if X3D_MODEL is None or ANOMALY_MODEL is None:
        return jsonify({"status": "error",
                        "message": "Models not loaded — check server logs"}), 503

    # Unique job ID so concurrent uploads don't collide on filenames
    job_id = uuid.uuid4().hex
    suffix = Path(video_file.filename).suffix.lower() or ".mp4"

    upload_path       = config.UPLOAD_DIR       / f"{job_id}{suffix}"
    preprocessed_path = config.PREPROCESSED_DIR / f"{job_id}.mp4"
    features_path     = config.FEATURES_DIR     / f"{job_id}.npy"
    result_path       = config.RESULTS_DIR      / f"{job_id}.json"

    # Save the upload
    video_file.save(str(upload_path))
    logger.info(f"[{job_id[:8]}] Saved upload: {upload_path.name} "
                f"({upload_path.stat().st_size / 1e6:.1f} MB)")

    acquired = _PIPELINE_LOCK.acquire(blocking=False)
    if not acquired:
        # Another request is already processing — queue it (simple blocking wait)
        logger.info(f"[{job_id[:8]}] Waiting for pipeline lock…")
        _PIPELINE_LOCK.acquire(blocking=True)

    try:
        timings = {}
        t0 = _now()

        # ── Stage 1: Preprocess ───────────────────────────────────────────
        logger.info(f"[{job_id[:8]}] Stage 1: preprocess")
        ok, err = preprocess_video(
            input_path=upload_path,
            output_path=preprocessed_path,
            target_fps=config.TARGET_FPS,
            width=config.TARGET_WIDTH,
            height=config.TARGET_HEIGHT,
        )
        timings["preprocess_sec"] = round(_now() - t0, 2)

        if not ok:
            return jsonify({"status": "error",
                            "message": f"Preprocessing failed: {err}"}), 500

        # ── Stage 2: Feature extraction ───────────────────────────────────
        logger.info(f"[{job_id[:8]}] Stage 2: extract features")
        t1 = _now()
        meta, err = extract_features(
            video_path=preprocessed_path,
            output_npy_path=features_path,
            x3d_model=X3D_MODEL,
            device=DEVICE,
            chunk_size=CHUNK_SIZE,
            batch_size=BATCH_SIZE,
        )
        timings["features_sec"] = round(_now() - t1, 2)

        if meta is None:
            return jsonify({"status": "error",
                            "message": f"Feature extraction failed: {err}"}), 500

        # ── Stage 3: Inference ────────────────────────────────────────────
        logger.info(f"[{job_id[:8]}] Stage 3: inference")
        t2 = _now()
        result = score_npy(
            npy_path=features_path,
            model=ANOMALY_MODEL,
            device=DEVICE,
            score_threshold=config.SCORE_THRESHOLD,
            smooth_window=config.SMOOTH_WINDOW,
        )
        timings["infer_sec"] = round(_now() - t2, 2)
        timings["total_sec"] = round(_now() - t0, 2)

        response = {
            "status"        : "ok",
            "job_id"        : job_id,
            "filename"      : video_file.filename,
            "duration_sec"  : meta["duration_sec"],
            "n_segments"    : result["n_segments"],
            "is_anomalous"  : result["is_anomalous"],
            "max_score"     : result["max_score"],
            "mean_score"    : result["mean_score"],
            "events"        : result["events"],
            "scores"        : result["scores"],
            "smooth_scores" : result["smooth_scores"],
            "timings"       : timings,
            "preprocessed_video_url": f"/video/{job_id}",
        }

        # Persist result JSON for the session
        with open(result_path, "w") as f:
            json.dump(response, f, indent=2)

        logger.info(
            f"[{job_id[:8]}] Done — {timings['total_sec']}s | "
            f"anomalous={result['is_anomalous']} | "
            f"{len(result['events'])} event(s)"
        )
        return jsonify(response)

    except Exception as e:
        logger.exception(f"[{job_id[:8]}] Pipeline error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

    finally:
        _PIPELINE_LOCK.release()


@app.route("/result/<job_id>")
def get_result(job_id: str):
    """Retrieve a previously computed result by job_id."""
    result_path = config.RESULTS_DIR / f"{job_id}.json"
    if not result_path.exists():
        return jsonify({"status": "error", "message": "Result not found"}), 404
    with open(result_path) as f:
        return jsonify(json.load(f))


@app.route("/video/<job_id>")
def get_preprocessed_video(job_id: str):
    """Serve the standardized 30 fps video generated by the Flask pipeline."""
    if not job_id.isalnum():
        return jsonify({"status": "error", "message": "Invalid job id"}), 400

    video_path = config.PREPROCESSED_DIR / f"{job_id}.mp4"
    if not video_path.exists():
        return jsonify({"status": "error", "message": "Video not found"}), 404

    return send_file(video_path, mimetype="video/mp4", conditional=True)


@app.route("/health")
def health():
    return jsonify({
        "status"        : "ok",
        "device"        : str(DEVICE),
        "x3d_loaded"    : X3D_MODEL is not None,
        "model_loaded"  : ANOMALY_MODEL is not None,
        "checkpoint"    : str(config.CHECKPOINT_PATH),
    })


def _fresh_stages(upload_done_at=None):
    return [
        {
            "id": "upload",
            "label": "UPLOAD",
            "description": "Receiving video file",
            "status": "done" if upload_done_at else "pending",
            "started_at": upload_done_at,
            "ended_at": upload_done_at,
            "duration_sec": 0.0 if upload_done_at else None,
        },
        {
            "id": "preprocess",
            "label": "PREPROCESS",
            "description": "FFmpeg · 224x224 · 30 fps · strip audio",
            "status": "pending",
            "started_at": None,
            "ended_at": None,
            "duration_sec": None,
        },
        {
            "id": "features",
            "label": "FEATURE EXTRACTION",
            "description": "X3D-M · 16-frame windows · 50% overlap · (N, 2048)",
            "status": "pending",
            "started_at": None,
            "ended_at": None,
            "duration_sec": None,
        },
        {
            "id": "infer",
            "label": "INFERENCE",
            "description": "MultiScaleTCN + MIL scorer · median smoothing",
            "status": "pending",
            "started_at": None,
            "ended_at": None,
            "duration_sec": None,
        },
    ]


def _set_job_status(job_id, *, status=None, message=None, result=None):
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        if status is not None:
            job["status"] = status
        if message is not None:
            job["message"] = message
        if result is not None:
            job["result"] = result


def _set_job_stage(job_id, stage_id, status, duration_sec=None):
    now = _now()
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        for stage in job["stages"]:
            if stage["id"] != stage_id:
                continue
            stage["status"] = status
            if status == "running":
                stage["started_at"] = now
                stage["ended_at"] = None
                stage["duration_sec"] = None
            elif status in {"done", "error"}:
                stage["ended_at"] = now
                if duration_sec is not None:
                    stage["duration_sec"] = round(duration_sec, 2)
                elif stage.get("started_at"):
                    stage["duration_sec"] = round(now - stage["started_at"], 2)
            return


def _run_async_job(job_id, filename, upload_path, preprocessed_path, features_path, result_path):
    acquired = _PIPELINE_LOCK.acquire(blocking=False)
    if not acquired:
        logger.info(f"[{job_id[:8]}] Async job waiting for pipeline lock…")
        _set_job_status(job_id, status="queued", message="Waiting for the model pipeline")
        _PIPELINE_LOCK.acquire(blocking=True)

    timings = {}
    t0 = _now()
    _set_job_status(job_id, status="running", message=None)

    try:
        logger.info(f"[{job_id[:8]}] Async Stage 1: preprocess")
        _set_job_stage(job_id, "preprocess", "running")
        s0 = _now()
        ok, err = preprocess_video(
            input_path=upload_path,
            output_path=preprocessed_path,
            target_fps=config.TARGET_FPS,
            width=config.TARGET_WIDTH,
            height=config.TARGET_HEIGHT,
        )
        timings["preprocess_sec"] = round(_now() - s0, 2)
        _set_job_stage(job_id, "preprocess", "done" if ok else "error", timings["preprocess_sec"])
        if not ok:
            raise RuntimeError(f"Preprocessing failed: {err}")

        logger.info(f"[{job_id[:8]}] Async Stage 2: extract features")
        _set_job_stage(job_id, "features", "running")
        s1 = _now()
        meta, err = extract_features(
            video_path=preprocessed_path,
            output_npy_path=features_path,
            x3d_model=X3D_MODEL,
            device=DEVICE,
            chunk_size=CHUNK_SIZE,
            batch_size=BATCH_SIZE,
        )
        timings["features_sec"] = round(_now() - s1, 2)
        _set_job_stage(job_id, "features", "done" if meta is not None else "error", timings["features_sec"])
        if meta is None:
            raise RuntimeError(f"Feature extraction failed: {err}")

        logger.info(f"[{job_id[:8]}] Async Stage 3: inference")
        _set_job_stage(job_id, "infer", "running")
        s2 = _now()
        result = score_npy(
            npy_path=features_path,
            model=ANOMALY_MODEL,
            device=DEVICE,
            score_threshold=config.SCORE_THRESHOLD,
            smooth_window=config.SMOOTH_WINDOW,
        )
        timings["infer_sec"] = round(_now() - s2, 2)
        timings["total_sec"] = round(_now() - t0, 2)
        _set_job_stage(job_id, "infer", "done", timings["infer_sec"])

        response = {
            "status": "ok",
            "job_id": job_id,
            "filename": filename,
            "duration_sec": meta["duration_sec"],
            "n_segments": result["n_segments"],
            "is_anomalous": result["is_anomalous"],
            "max_score": result["max_score"],
            "mean_score": result["mean_score"],
            "events": result["events"],
            "scores": result["scores"],
            "smooth_scores": result["smooth_scores"],
            "timings": timings,
            "preprocessed_video_url": f"/video/{job_id}",
        }

        with open(result_path, "w") as f:
            json.dump(response, f, indent=2)

        _set_job_status(job_id, status="complete", result=response)
        logger.info(f"[{job_id[:8]}] Async done — {timings['total_sec']}s")

    except Exception as e:
        logger.exception(f"[{job_id[:8]}] Async pipeline error: {e}")
        _set_job_status(job_id, status="error", message=str(e))

    finally:
        _PIPELINE_LOCK.release()


def _now() -> float:
    import time
    return time.time()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
