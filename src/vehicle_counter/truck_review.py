"""Optional crop review of crossings that YOLO's track vote labels as trucks."""

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from vehicle_counter.contracts import TrackObservation
from vehicle_counter.exporting import file_sha256

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


class TruckCropReviewer:
    """Use the fixed MobileNetV3-Large ImageNet weights with no implicit download."""

    def __init__(self, weights: str) -> None:
        import torch
        from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

        path = Path(weights).expanduser().resolve()
        if file_sha256(path) != WEIGHTS_SHA256:
            raise ValueError("Checksum bobot MobileNetV3-Large berbeda dari profil klasifikasi.")
        specification = MobileNet_V3_Large_Weights.IMAGENET1K_V2
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
            "light_threshold": LIGHT_VEHICLE_THRESHOLD,
        }

    def __call__(self, frame: NDArray[np.uint8], observation: TrackObservation) -> bool:
        import torch
        from PIL import Image

        x1, y1, x2, y2 = map(int, observation.bbox_xyxy)
        crop = frame[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
        if min(crop.shape[:2]) < 24:
            return False
        rgb = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        with torch.inference_mode():
            probabilities = self._model(self._transforms(rgb).unsqueeze(0)).softmax(1)[0]
        return (
            int(probabilities.argmax()) in self._light_indices
            and float(probabilities[list(self._light_indices)].sum()) >= LIGHT_VEHICLE_THRESHOLD
        )
