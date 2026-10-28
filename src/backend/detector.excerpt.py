# Excerpt from tracking_api/services/detector.py ("# ..." marks omitted code)


class Detector:
    def __init__(self) -> None:
        # ...
        self.coco_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yolo-coco")
        self.gun_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yolo-gun")
        # ...

    async def detect_with_timings(self, frame: Any) -> tuple[list[Detection], dict[str, float]]:
        loop = asyncio.get_running_loop()
        started_at = time.perf_counter()

        # Person and full-frame gun passes run in parallel
        coco_future = loop.run_in_executor(self.coco_executor, self.coco_model.predict, frame)
        gun_baseline_future = loop.run_in_executor(self.gun_executor, self.gun_model.predict, frame)
        parallel_started = time.perf_counter()
        (coco_result, person_stats), (gun_baseline_result, gun_stats) = await asyncio.gather(
            coco_future, gun_baseline_future
        )
        parallel_wall_ms = (time.perf_counter() - parallel_started) * 1000

        extract_started = time.perf_counter()
        person_detections = self._extract_detections(coco_result, "coco")
        baseline_gun_detections = self._extract_detections(gun_baseline_result, "gun")
        extract_ms = (time.perf_counter() - extract_started) * 1000

        # Second gun pass on person crops, only when someone is in frame
        refined_gun_detections, refined_regions, crop_stats = await self._detect_guns_from_person_crops(
            frame, person_detections
        )
        merge_started = time.perf_counter()

        # Inside a re-examined crop, the crop result wins
        surviving_baseline = [
            detection
            for detection in baseline_gun_detections
            if not self._point_in_any_region(
                (detection.x1 + detection.x2) / 2, (detection.y1 + detection.y2) / 2, refined_regions
            )
        ]

        gun_detections = self._dedup_gun_detections([*surviving_baseline, *refined_gun_detections])
        merge_ms = (time.perf_counter() - merge_started) * 1000

        detections = [*person_detections, *gun_detections]
        # ...
        return detections, timings

    async def _detect_guns_from_person_crops(
        self, frame: Any, person_detections: list[Detection]
    ) -> tuple[list[Detection], list[tuple[float, float, float, float]], dict[str, float]]:
        """Higher-resolution gun pass on a crop around each of the top persons."""
        if not person_detections:
            return [], [], {"total_ms": 0.0, "images": 0}

        loop = asyncio.get_running_loop()
        capped_persons = sorted(person_detections, key=lambda detection: detection.confidence, reverse=True)[
            :GUN_MAX_PERSON_CROPS
        ]

        frame_height, frame_width = frame.shape[:2]
        crops: list[Any] = []
        origins: list[tuple[float, float]] = []
        regions: list[tuple[float, float, float, float]] = []
        for person in capped_persons:
            x1, y1, x2, y2 = self._expand_box(
                person.x1, person.y1, person.x2, person.y2, frame_width, frame_height
            )
            if x2 <= x1 or y2 <= y1:
                continue
            crops.append(frame[y1:y2, x1:x2])
            origins.append((float(x1), float(y1)))
            regions.append((float(x1), float(y1), float(x2), float(y2)))

        if not crops:
            return [], [], {"total_ms": 0.0, "images": 0}

        gun_results, crop_stats = await loop.run_in_executor(
            self.gun_executor, self.gun_model.predict_batch, crops, GUN_CROP_IMAGE_SIZE
        )

        gun_detections: list[Detection] = []
        for result, (offset_x, offset_y) in zip(gun_results, origins):
            gun_detections.extend(self._extract_detections(result, "gun", offset=(offset_x, offset_y)))
        return gun_detections, regions, crop_stats

    @staticmethod
    def _point_in_any_region(x: float, y: float, regions: list[tuple[float, float, float, float]]) -> bool:
        return any(x1 <= x <= x2 and y1 <= y <= y2 for x1, y1, x2, y2 in regions)

    @staticmethod
    def _expand_box(
        x1: float, y1: float, x2: float, y2: float, frame_width: int, frame_height: int
    ) -> tuple[int, int, int, int]:
        width = x2 - x1
        height = y2 - y1
        pad_x = max(width * GUN_CROP_MARGIN_RATIO, GUN_CROP_MIN_PADDING_PX)
        pad_y = max(height * GUN_CROP_MARGIN_RATIO, GUN_CROP_MIN_PADDING_PX)
        new_x1 = max(0, int(x1 - pad_x))
        new_y1 = max(0, int(y1 - pad_y))
        new_x2 = min(frame_width, int(x2 + pad_x))
        new_y2 = min(frame_height, int(y2 + pad_y))
        return new_x1, new_y1, new_x2, new_y2

    @staticmethod
    def _dedup_gun_detections(detections: list[Detection]) -> list[Detection]:
        """Greedy dedup: keep the higher-confidence box of any overlapping pair."""
        if len(detections) < 2:
            return detections

        ordered = sorted(detections, key=lambda detection: detection.confidence, reverse=True)
        kept: list[Detection] = []
        for candidate in ordered:
            if not any(
                Detector._iou(candidate, other) > GUN_CROP_DEDUP_IOU_THRESHOLD for other in kept
            ):
                kept.append(candidate)
        return kept

    @staticmethod
    def _iou(a: Detection, b: Detection) -> float:
        inter_x1 = max(a.x1, b.x1)
        inter_y1 = max(a.y1, b.y1)
        inter_x2 = min(a.x2, b.x2)
        inter_y2 = min(a.y2, b.y2)
        inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
        if inter_area <= 0.0:
            return 0.0
        area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
        area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
        union_area = area_a + area_b - inter_area
        return inter_area / union_area if union_area > 0 else 0.0
