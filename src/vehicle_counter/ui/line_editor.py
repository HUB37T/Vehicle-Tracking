"""Validated bridge between the line editor and normalized domain coordinates."""

import base64
from pathlib import Path

from vehicle_counter.contracts import CongestionConfig, CountingLineConfig


def decode_editor(value: dict, original: CountingLineConfig):
    if not isinstance(value, dict) or set(value) not in (
        {"p1", "p2", "roi"},
        {"p1", "p2", "roi", "edit_id"},
    ):
        raise ValueError("Data editor garis tidak lengkap.")
    if "edit_id" in value and (
        not isinstance(value["edit_id"], str) or not 0 < len(value["edit_id"]) <= 64
    ):
        raise ValueError("Identitas edit tidak valid.")
    line = CountingLineConfig(
        line_id=original.line_id,
        p1=value["p1"],
        p2=value["p2"],
        epsilon_height_fraction=original.epsilon_height_fraction,
    )
    area = CongestionConfig(roi=value["roi"])
    return line, area.roi


def render_editor(
    image_bgr,
    line,
    roi,
    *,
    key: str,
    edit_roi: bool = False,
    edit_line: bool = True,
    show_roi: bool = False,
    saved_edit_id: str | None = None,
):
    import cv2
    import streamlit.components.v1 as components

    ok, data = cv2.imencode(".jpg", image_bgr)
    if not ok:
        raise ValueError("Preview tidak dapat dikodekan.")
    editor = components.declare_component(
        "vehicle_line_editor", path=str(Path(__file__).parent / "editor")
    )
    return editor(
        image="data:image/jpeg;base64," + base64.b64encode(data).decode("ascii"),
        image_height=1000 * image_bgr.shape[0] / image_bgr.shape[1],
        p1=line.p1,
        p2=line.p2,
        roi=roi,
        edit_roi=edit_roi,
        show_roi=show_roi,
        edit_line=edit_line,
        saved_edit_id=saved_edit_id,
        default=None,
        key=key,
    )
