# Excerpt from tracking_api/routes.py

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from tracking_api.config import COCO_IMAGE_SIZE, COCO_MODEL_NAME, GUN_IMAGE_SIZE, GUN_MODEL_NAME, STATIC_DIR
from tracking_api.schemas import serialize_detections, serialize_tracks
from tracking_api.services.detector import Detector
from tracking_api.services.tracking import TrackingSession
from tracking_api.utils.images import decode_jpeg


router = APIRouter()
logger = logging.getLogger(__name__)


def get_detector(request_or_websocket: Request | WebSocket) -> Detector:
    detector = getattr(request_or_websocket.app.state, "detector", None)
    if detector is None:
        raise RuntimeError("Detector is not initialized.")
    return detector


@router.get("/")


async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})


@router.get("/health")


async def health(request: Request) -> dict[str, Any]:
    detector = get_detector(request)
    return {
        "ok": True,
        "device": detector.device,
        "torch": {
            "version": detector.torch_version,
            "requested_device": detector.requested_device,
            "cuda_available": detector.cuda_available,
            "cuda_device_count": detector.cuda_device_count,
            "cpu_count": detector.cpu_count,
            "torch_threads": detector.torch_threads,
        },
        "models": {
            "coco": COCO_MODEL_NAME,
            "gun": GUN_MODEL_NAME,
        },
        "image_sizes": {
            "coco": COCO_IMAGE_SIZE,
            "gun": GUN_IMAGE_SIZE,
        },
    }


@router.websocket("/ws/track")


async def track_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    detector = get_detector(websocket)
    session = TrackingSession()
    frame_number = 0

    while True:
        try:
            jpeg_bytes = await websocket.receive_bytes()
        except WebSocketDisconnect:
            return

        started_at = time.perf_counter()
        try:
            decode_started = time.perf_counter()
            frame = decode_jpeg(jpeg_bytes)
            decode_ms = (time.perf_counter() - decode_started) * 1000

            detections, timings = await detector.detect_with_timings(frame)

            track_started = time.perf_counter()
            tracks = session.update(detections, frame)
            track_ms = (time.perf_counter() - track_started) * 1000
        except Exception:
            logger.exception("Failed to process frame %d; skipping it.", frame_number + 1)
            try:
                await websocket.send_json({"frame": frame_number, "error": "Failed to process frame."})
            except (WebSocketDisconnect, RuntimeError):
                return
            continue

        elapsed_ms = (time.perf_counter() - started_at) * 1000
        frame_number += 1

        timings = {
            **timings,
            "decode_ms": round(decode_ms, 2),
            "track_ms": round(track_ms, 2),
            "server_total_ms": round(elapsed_ms, 2),
            "frame_bytes": len(jpeg_bytes),
        }

        try:
            await websocket.send_json(
                {
                    "frame": frame_number,
                    "width": int(frame.shape[1]),
                    "height": int(frame.shape[0]),
                    "elapsed_ms": round(elapsed_ms, 2),
                    "timings": timings,
                    "fps": round(1000 / elapsed_ms, 2) if elapsed_ms > 0 else 0,
                    "detections": serialize_detections(detections),
                    "tracks": serialize_tracks(tracks),
                    "counts": {
                        "detections": {
                            "person": sum(1 for detection in detections if detection.label == "person"),
                            "gun": sum(1 for detection in detections if detection.label == "gun"),
                        },
                        "tracks": {
                            "person": sum(1 for track in tracks if track.label == "person"),
                            "gun": sum(1 for track in tracks if track.label == "gun"),
                        },
                    },
                }
            )
        except (WebSocketDisconnect, RuntimeError):
            return
