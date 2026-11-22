"""Conservative, optional filtering of conflicting vehicle boxes before tracking."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

DUPLICATE_BOX_IOU = 0.90
CONFLICT_CLASSES = frozenset({"car", "bus", "truck"})


def distinct_vehicle_indices(boxes: NDArray, scores: NDArray, classes: Sequence[str]) -> list[int]:
    """Keep the higher score only for near-identical, conflicting car/bus/truck boxes.

    Same-class detections and motorcycles are left to the detector/tracker. Apply
    this after selecting vehicle classes, so a person score cannot discard a bike.
    Return indices in the original order for stable association tie-breaking.
    """
    kept: list[int] = []
    for index in np.argsort(-scores, kind="stable"):
        name = classes[index]
        duplicate = False
        if name in CONFLICT_CLASSES:
            box = boxes[index]
            for previous in kept:
                if classes[previous] == name or classes[previous] not in CONFLICT_CLASSES:
                    continue
                other = boxes[previous]
                size = np.maximum(
                    0, np.minimum(box[2:], other[2:]) - np.maximum(box[:2], other[:2])
                )
                intersection = float(size.prod())
                union = float(np.prod(box[2:] - box[:2]) + np.prod(other[2:] - other[:2]))
                union -= intersection
                if union > 0 and intersection / union >= DUPLICATE_BOX_IOU:
                    duplicate = True
                    break
        if not duplicate:
            kept.append(int(index))
    return sorted(kept)
