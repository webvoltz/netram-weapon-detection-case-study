# Excerpt from tracking_api/services/tracking.py

from __future__ import annotations

from typing import Any

import numpy as np
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker, STrack
from ultralytics.utils import YAML, IterableSimpleNamespace

from tracking_api.config import (
    BYTETRACK_CONFIG_PATH,
    BYTETRACK_GUN_CONFIG_PATH,
    GUN_CLASS_ID,
    PERSON_TRACK_CLASS_ID,
)
from tracking_api.schemas import Detection, Track


class PersonSTrack(STrack):
    """STrack with its own ID counter so person IDs never share a sequence with gun IDs."""

    _count = 0

    @staticmethod
    def next_id() -> int:
        PersonSTrack._count += 1
        return PersonSTrack._count

    @staticmethod
    def reset_id() -> None:
        PersonSTrack._count = 0


class GunSTrack(STrack):
    """STrack with its own ID counter so gun IDs never share a sequence with person IDs."""

    _count = 0

    @staticmethod
    def next_id() -> int:
        GunSTrack._count += 1
        return GunSTrack._count

    @staticmethod
    def reset_id() -> None:
        GunSTrack._count = 0


def create_tracker(track_class: type, config_path: Any) -> Any:
    args = IterableSimpleNamespace(**YAML.load(config_path))
    tracker = BYTETracker(args)
    tracker.track_class = track_class
    return tracker


class TrackingSession:
    def __init__(self) -> None:
        PersonSTrack.reset_id()
        GunSTrack.reset_id()
        self.person_tracker = create_tracker(PersonSTrack, BYTETRACK_CONFIG_PATH)
        self.gun_tracker = create_tracker(GunSTrack, BYTETRACK_GUN_CONFIG_PATH)

    def update(self, detections: list[Detection], frame: Any) -> list[Track]:
        person_tracks = self._update_label(
            self.person_tracker, detections, "person", PERSON_TRACK_CLASS_ID, frame
        )
        gun_tracks = self._update_label(self.gun_tracker, detections, "gun", GUN_CLASS_ID, frame)
        return [*person_tracks, *gun_tracks]

    @staticmethod
    def _update_label(
        tracker: Any, detections: list[Detection], label: str, class_id: int, frame: Any
    ) -> list[Track]:
        height, width = frame.shape[:2]
        label_detections = [detection for detection in detections if detection.label == label]
        rows = np.asarray(
            [
                [
                    detection.x1,
                    detection.y1,
                    detection.x2,
                    detection.y2,
                    detection.confidence,
                    class_id,
                ]
                for detection in label_detections
            ],
            dtype=np.float32,
        )
        if rows.size == 0:
            rows = np.empty((0, 6), dtype=np.float32)

        boxes = Boxes(rows, (height, width))
        tracked_rows = tracker.update(boxes, img=frame)
        return TrackingSession._tracks_from_rows(tracked_rows, label, label_detections)

    @staticmethod
    def _tracks_from_rows(rows: Any, label: str, detections: list[Detection]) -> list[Track]:
        tracks: list[Track] = []
        for row in rows:
            x1, y1, x2, y2, track_id, confidence, _class_id, detection_index = row.tolist()
            int_detection_index = int(detection_index)
            display_label = label
            if 0 <= int_detection_index < len(detections):
                display_label = detections[int_detection_index].source_label

            int_track_id = int(track_id)
            tracks.append(
                Track(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    confidence=float(confidence),
                    label=label,
                    display_label=display_label,
                    track_id=int_track_id,
                    track_key=f"{label}-{int_track_id}",
                )
            )
        return tracks
