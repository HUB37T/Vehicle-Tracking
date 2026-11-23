"""Image-coordinate bridge for manual polygons and dataset bounding boxes."""

import base64
from pathlib import Path

from vehicle_counter.polygons import validated_polygon


def decode_polygon(value):
    if not isinstance(value, dict) or set(value) not in ({"polygon"}, {"polygon", "edit_id"}):
        raise ValueError("Data zona tidak valid.")
    return validated_polygon(value["polygon"])


def render_shapes(image_bgr, *, mode, key, polygon=(), boxes=(), overlays=()):
    import cv2
    import streamlit.components.v1 as components

    if mode not in ("polygon", "boxes"):
        raise ValueError("Mode editor tidak dikenal.")
    scale = min(1, 1000 / max(image_bgr.shape[:2]))
    image = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, data = cv2.imencode(".jpg", image)
    if not ok:
        raise ValueError("Preview tidak dapat dibuat.")
    editor = components.declare_component(
        "vehicle_shapes", path=str(Path(__file__).parent / "editor" / "shapes")
    )
    return editor(
        image="data:image/jpeg;base64," + base64.b64encode(data).decode("ascii"),
        mode=mode,
        polygon=polygon,
        boxes=boxes,
        overlays=overlays,
        key=key,
        default=None,
    )
