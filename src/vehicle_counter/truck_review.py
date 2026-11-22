"""Optional crop review of crossings that YOLO's track vote labels as trucks."""

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from vehicle_counter.contracts import TrackObservation
from vehicle_counter.exporting import file_sha256
from vehicle_counter.run_analysis import write_rows

LIGHT_VEHICLE_LABELS = frozenset(
    {
        "ambulance",
        "beach wagon",
        "cab",
        "jeep",
        "limousine",
        "minivan",
        "police van",
        "racer",
        "sports car",
    }
)
LIGHT_VEHICLE_THRESHOLD = 0.40
WEIGHTS_SHA256 = "5c1a416349c4cf298f2a6a5e2600ed0ee55e604713578f5e74e6bc8bcaef7997"


def light_vehicle_decision(
    top_label: str, light_score: float, threshold: float, *, minivan_only: bool = False
) -> bool:
    """Lower thresholds relax only minivan evidence, not ambiguous truck lookalikes."""
    return (
        top_label in LIGHT_VEHICLE_LABELS
        and (not minivan_only or top_label == "minivan")
        and light_score >= threshold
        and (light_score >= LIGHT_VEHICLE_THRESHOLD or top_label == "minivan")
    )


class TruckCropReviewer:
    """Use the fixed MobileNetV3-Large ImageNet weights with no implicit download."""

    def __init__(
        self, weights: str, *, light_threshold: float = 0.40, minivan_only: bool = False
    ) -> None:
        import torch
        from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

        path = Path(weights).expanduser().resolve()
        if file_sha256(path) != WEIGHTS_SHA256:
            raise ValueError("Checksum bobot MobileNetV3-Large berbeda dari profil klasifikasi.")
        specification = MobileNet_V3_Large_Weights.IMAGENET1K_V2
        self._threshold = light_threshold
        self._minivan_only = minivan_only
        self._labels = specification.meta["categories"]
        self.rows: list[dict] = []
        self.corrected_track_ids: set[int] = set()
        self._transforms = specification.transforms()
        self._light_indices = tuple(
            i
            for i, label in enumerate(specification.meta["categories"])
            if label in LIGHT_VEHICLE_LABELS
        )
        self._model = mobilenet_v3_large(weights=None)
        self._model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self._model.eval()
        self.metadata = {
            "model": "mobilenet_v3_large_imagenet1k_v2",
            "weights_sha256": WEIGHTS_SHA256,
            "weights_url": specification.url,
            "light_labels": sorted(LIGHT_VEHICLE_LABELS),
            "light_threshold": light_threshold,
            "minivan_only": minivan_only,
            "relaxed_top_labels": ["minivan"],
            "general_light_threshold_floor": LIGHT_VEHICLE_THRESHOLD,
        }

    def __call__(self, frame: NDArray[np.uint8], observation: TrackObservation) -> bool:
        return self._review(frame, observation)

    def review_passenger_conflict(
        self, frame: NDArray[np.uint8], observation: TrackObservation
    ) -> bool:
        """Allow SUV evidence when recent votes would overturn an existing car consensus."""
        return self._review(frame, observation, passenger_conflict=True)

    def _review(
        self,
        frame: NDArray[np.uint8],
        observation: TrackObservation,
        *,
        passenger_conflict: bool = False,
    ) -> bool:
        import torch
        from PIL import Image

        x1, y1, x2, y2 = map(int, observation.bbox_xyxy)
        crop = frame[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
        if min(crop.shape[:2]) < 24:
            self._record(observation, "", None, False, "crop_too_small")
            return False
        rgb = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        # Depthwise convolutions need the portable kernel on older VM CPUs.
        with torch.inference_mode(), torch.backends.mkldnn.flags(enabled=False):
            probabilities = self._model(self._transforms(rgb).unsqueeze(0)).softmax(1)[0]
        top_index = int(probabilities.argmax())
        top_label = self._labels[top_index]
        light_score = float(probabilities[list(self._light_indices)].sum())
        correct = light_vehicle_decision(
            top_label, light_score, self._threshold, minivan_only=self._minivan_only
        )
        if passenger_conflict and top_label == "jeep":
            correct |= float(probabilities[top_index]) >= 0.20 and light_score >= self._threshold
        reason = "passenger_class_conflict" if passenger_conflict else "classified"
        self._record(observation, top_label, light_score, correct, reason)
        if correct:
            self.corrected_track_ids.add(observation.track_id)
        return correct

    def _record(self, observation, top_label, light_score, correct, reason) -> None:
        self.rows.append(
            {
                "frame_index": observation.frame_index,
                "timestamp_seconds": observation.timestamp_seconds,
                "track_id": observation.track_id,
                "voted_class": "truck",
                "detector_class": observation.class_name,
                "top_crop_label": top_label,
                "light_score": light_score,
                "light_threshold": self._threshold,
                "minivan_only": self._minivan_only,
                "corrected_to_car": correct,
                "final_class": "car" if correct else "truck",
                "reason": reason,
            }
        )

    def export(self, output_dir: Path) -> dict[str, str]:
        path = output_dir / "truck_review.csv"
        write_rows(
            path,
            self.rows,
            [
                "frame_index",
                "timestamp_seconds",
                "track_id",
                "voted_class",
                "detector_class",
                "top_crop_label",
                "light_score",
                "light_threshold",
                "minivan_only",
                "corrected_to_car",
                "final_class",
                "reason",
            ],
        )
        return {"truck_review_csv": str(path)}
