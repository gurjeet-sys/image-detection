import io
import logging
import math
import os
import threading
import time

import psutil
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
MODEL_PATH = os.environ.get("YOLO_MODEL", os.path.join(BASE_DIR, "yolo11s.pt"))
MODEL_NAME = os.path.basename(MODEL_PATH)
INFERENCE_CONFIDENCE = 0.05
FULL_IMAGE_SIZE = 1280
MAX_DETECTIONS = 300
TILE_SIZE = 1024
TILE_OVERLAP = 0.2
NMS_IOU_THRESHOLD = 0.4

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES
app.logger.setLevel(logging.INFO)

_model = None
_model_lock = threading.Lock()
_network_lock = threading.Lock()
_previous_network_sample = None


def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from ultralytics import YOLO

                app.logger.info("Loading object detection model: %s", MODEL_NAME)
                _model = YOLO(MODEL_PATH)
                app.logger.info("Object detection model loaded")
    return _model


def _tile_positions(length, tile_size=TILE_SIZE, overlap=TILE_OVERLAP):
    if length <= tile_size:
        return [0]
    stride = max(1, int(tile_size * (1 - overlap)))
    positions = list(range(0, length - tile_size + 1, stride))
    last_position = length - tile_size
    if positions[-1] != last_position:
        positions.append(last_position)
    return positions


def _detections_from_result(result, image_width, image_height, offset=(0, 0), source_id="full"):
    detections = []
    if result.boxes is None:
        return detections

    offset_x, offset_y = offset
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        score = float(box.conf[0].item())
        x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
        if not math.isfinite(score) or not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            continue
        left = min(max(float(x1) + offset_x, 0.0), float(image_width))
        top = min(max(float(y1) + offset_y, 0.0), float(image_height))
        right = min(max(float(x2) + offset_x, 0.0), float(image_width))
        bottom = min(max(float(y2) + offset_y, 0.0), float(image_height))
        if right <= left or bottom <= top:
            continue
        detections.append({
            "class": str(result.names[class_id]),
            "score": min(max(score, 0.0), 1.0),
            "bbox": [left, top, right - left, bottom - top],
            "_source": source_id,
        })
    return detections


def _suppress_duplicate_detections(detections):
    ranked = sorted(detections, key=lambda item: item["score"], reverse=True)
    kept = []
    for candidate in ranked:
        x, y, width, height = candidate["bbox"]
        candidate_area = width * height
        duplicate = False
        for accepted in kept:
            if candidate["class"] != accepted["class"]:
                continue
            accepted_x, accepted_y, accepted_width, accepted_height = accepted["bbox"]
            intersection_width = max(0.0, min(x + width, accepted_x + accepted_width) - max(x, accepted_x))
            intersection_height = max(0.0, min(y + height, accepted_y + accepted_height) - max(y, accepted_y))
            intersection = intersection_width * intersection_height
            union = candidate_area + accepted_width * accepted_height - intersection
            smaller_area = min(candidate_area, accepted_width * accepted_height)
            if union > 0 and (
                intersection / union > NMS_IOU_THRESHOLD
                or (smaller_area > 0 and intersection / smaller_area >= 0.8)
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME})


@app.get("/api/system-status")
def system_status():
    global _previous_network_sample

    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        root_path = os.path.splitdrive(BASE_DIR)[0] + os.sep or os.sep
        storage = psutil.disk_usage(root_path)
        network = psutil.net_io_counters()
        if network is None:
            raise RuntimeError("Network counters are unavailable.")
        sampled_at = time.monotonic()

        with _network_lock:
            if _previous_network_sample is None:
                received_per_second = None
                sent_per_second = None
            else:
                previous_time, previous_network = _previous_network_sample
                elapsed = sampled_at - previous_time
                if elapsed <= 0:
                    received_per_second = None
                    sent_per_second = None
                else:
                    received_per_second = max(0, network.bytes_recv - previous_network.bytes_recv) / elapsed
                    sent_per_second = max(0, network.bytes_sent - previous_network.bytes_sent) / elapsed
            _previous_network_sample = (sampled_at, network)

        return jsonify({
            "cpu_percent": cpu_percent,
            "memory": {
                "percent": memory.percent,
                "used_bytes": memory.used,
                "total_bytes": memory.total,
            },
            "storage": {
                "percent": storage.percent,
                "used_bytes": storage.used,
                "total_bytes": storage.total,
                "free_bytes": storage.free,
            },
            "network": {
                "received_bytes_per_second": received_per_second,
                "sent_bytes_per_second": sent_per_second,
            },
        })
    except (OSError, psutil.Error, RuntimeError):
        app.logger.exception("Could not read system status")
        return jsonify({"error": "System metrics are currently unavailable."}), 503


@app.post("/api/detect")
def detect():
    uploaded = request.files.get("image")
    if uploaded is None:
        return jsonify({"error": "Choose an image to detect."}), 400
    if uploaded.mimetype not in ALLOWED_MIME_TYPES:
        return jsonify({"error": "Use a JPG, PNG, or WEBP image."}), 400

    try:
        image_bytes = uploaded.read(MAX_UPLOAD_BYTES + 1)
        if len(image_bytes) > MAX_UPLOAD_BYTES:
            return jsonify({"error": "The image must be 15 MB or smaller."}), 413
        with Image.open(io.BytesIO(image_bytes)) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                return jsonify({"error": "The image dimensions are too large to process."}), 400
            source.verify()
        with Image.open(io.BytesIO(image_bytes)) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                return jsonify({"error": "The image dimensions are too large to process."}), 400
            source.load()
            oriented = ImageOps.exif_transpose(source)
            if "A" in oriented.getbands() or "transparency" in oriented.info:
                rgba = oriented.convert("RGBA")
                image = Image.new("RGB", rgba.size, (255, 255, 255))
                image.paste(rgba, mask=rgba.getchannel("A"))
            else:
                image = oriented.convert("RGB")
            width, height = image.size
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        return jsonify({"error": "The uploaded file is not a valid supported image."}), 400

    try:
        model = get_model()
        results = model.predict(
            source=image,
            conf=INFERENCE_CONFIDENCE,
            max_det=MAX_DETECTIONS,
            imgsz=FULL_IMAGE_SIZE,
            device="cpu",
            verbose=False,
        )
        detections = _detections_from_result(results[0], width, height)

        if width > TILE_SIZE or height > TILE_SIZE:
            for top in _tile_positions(height):
                for left in _tile_positions(width):
                    tile = image.crop((
                        left,
                        top,
                        min(left + TILE_SIZE, width),
                        min(top + TILE_SIZE, height),
                    ))
                    tile_result = model.predict(
                        source=tile,
                        conf=INFERENCE_CONFIDENCE,
                        max_det=MAX_DETECTIONS,
                        imgsz=TILE_SIZE,
                        device="cpu",
                        verbose=False,
                    )[0]
                    detections.extend(_detections_from_result(
                        tile_result,
                        width,
                        height,
                        (left, top),
                        source_id=(left, top),
                    ))

        detections = _suppress_duplicate_detections(detections)
        return jsonify({
            "width": width,
            "height": height,
            "predictions": [
                {key: value for key, value in detection.items() if key != "_source"}
                for detection in detections
            ],
        })
    except Exception:
        app.logger.exception("Object detection failed")
        return jsonify({"error": "The server could not analyze this image. Please try again."}), 500


@app.errorhandler(RequestEntityTooLarge)
def handle_too_large(_error):
    return jsonify({"error": "The image must be 15 MB or smaller."}), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)
