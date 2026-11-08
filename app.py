"""Streamlit front end for the local vehicle counting pipeline."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from math import hypot
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import pandas as pd
import streamlit as st

from vehicle_counter.config import load_config
from vehicle_counter.contracts import DIRECTIONS, PREPROCESS_MODES, VEHICLE_CLASSES, RunResult
from vehicle_counter.geometry import PixelCountingLine
from vehicle_counter.pipeline import process_video
from vehicle_counter.video import VideoReader

ROOT = Path(__file__).resolve().parent
CLASSES = dict(zip(VEHICLE_CLASSES, ("Mobil", "Motor", "Bus", "Truk"), strict=True))
MODES = {
    "none": "Tanpa prapemrosesan",
    "clahe": "CLAHE",
    "gamma": "Gamma correction",
    "gaussian": "Gaussian smoothing",
    "unsharp": "Unsharp masking",
}
DEMO_EXPERIMENTS = {
    "holdout-istanbul-stabilized.mp4": (
        "istanbul-recovery.yaml",
        "[Ahmet Akpolat / Pexels](https://www.pexels.com/video/traffic-on-highway-14359967/)"
        " · Distabilkan dan diperkecil.",
    ),
    "holdout-hvsd-rain.mp4": (
        "hvsd-rain-recovery.yaml",
        "[Hadi Sadoghi Yazdi / HVSD](https://data.mendeley.com/datasets/zdgmdg7p25/1)"
        " · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · Dikonversi ke 30 FPS.",
    ),
}


@contextmanager
def _source_file(local: Path | None, data: bytes | None) -> Iterator[Path]:
    if local is not None:
        yield local
        return
    with TemporaryDirectory(prefix="vehicle-counter-") as temp:
        path = Path(temp) / "upload.mp4"
        path.write_bytes(data or b"")
        yield path


@st.cache_data(max_entries=6, show_spinner=False)
def _preview(local: Path | None, data: bytes | None, modified_ns: int | None):
    # modified_ns is part of the cache key for locally replaced files.
    with _source_file(local, data) as source, VideoReader(source) as reader:
        metadata = reader.metadata
        _, frame = next(iter(reader))
        scale = min(1, 960 / metadata.width, 540 / metadata.height)
        size = (max(2, round(metadata.width * scale)), max(2, round(metadata.height * scale)))
        return metadata, cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def _line_preview(frame, config):
    image = frame.copy()
    h, w = image.shape[:2]
    line = PixelCountingLine.from_config(config, width=w, height=h)
    p1, p2 = tuple(map(round, line.p1)), tuple(map(round, line.p2))
    yellow = (0, 225, 255)
    cv2.line(image, p1, p2, yellow, 2, cv2.LINE_AA)
    cv2.circle(image, p1, 5, yellow, -1)
    cv2.circle(image, p2, 5, yellow, -1)
    dx, dy = line.p2[0] - line.p1[0], line.p2[1] - line.p1[1]
    length = hypot(dx, dy)
    middle_x, middle_y = (line.p1[0] + line.p2[0]) / 2, (line.p1[1] + line.p2[1]) / 2
    for sign, label in ((-1, "A"), (1, "B")):
        x = min(max(1, round(middle_x - sign * 24 * dy / length)), max(1, w - 18))
        y = min(max(1, round(middle_y + sign * 24 * dx / length)), max(1, h - 3))
        for color, thickness in (((0, 0, 0), 4), (yellow, 2)):
            cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, thickness)
    return image


def _show_result(result: RunResult, source_name: str, mode: str, profile: str) -> None:
    st.header("Hasil analisis")
    st.caption(f"{source_name} · {profile} · {MODES[mode]} · Run {result.run_id}")
    if result.status != "completed":
        st.error("Analisis berhenti sebelum selesai. Periksa detail proses.")
        for warning in result.warnings:
            st.write(warning)
        return

    directions = [
        sum(counts[direction] for counts in result.summary.by_class_direction.values())
        for direction in DIRECTIONS
    ]
    for column, label, value in zip(
        st.columns(3),
        ("Total crossing", "A → B", "B → A"),
        (result.summary.total, *directions),
        strict=True,
    ):
        column.metric(label, value)
    table, video, evidence, exports = st.tabs(("Rekap", "Video", "Bukti materi", "Unduhan"))
    with table:
        rows = [
            {"Kendaraan": CLASSES[name], "A → B": counts["A_TO_B"], "B → A": counts["B_TO_A"]}
            for name, counts in result.summary.by_class_direction.items()
        ]
        frame = pd.DataFrame(rows)
        st.dataframe(frame, hide_index=True)
        st.bar_chart(frame, x="Kendaraan", y=["A → B", "B → A"], stack=False)
    with video:
        st.video(Path(result.artifact_paths["annotated_video"]))
        st.caption("Video beranotasi disimpan tanpa audio sumber.")
    with evidence:
        if path := result.artifact_paths.get("process_evidence"):
            st.image(path, caption="Ruang warna, CLAHE, YOLO, dan ByteTrack", width="stretch")
        else:
            st.info("Run ini dibuat sebelum panel bukti proses tersedia.")
    with exports:
        for key, label, mime in (
            ("events_csv", "Unduh daftar crossing", "text/csv"),
            ("counts_csv", "Unduh rekap hitungan", "text/csv"),
            ("annotated_video", "Unduh video beranotasi", "video/mp4"),
            ("process_evidence", "Unduh panel bukti", "image/png"),
            ("run_json", "Unduh metadata run", "application/json"),
        ):
            if filename := result.artifact_paths.get(key):
                path = Path(filename)
                st.download_button(label, path.read_bytes(), path.name, mime, on_click="ignore")
        if result.warnings:
            with st.expander("Catatan run"):
                for warning in result.warnings:
                    st.write(warning)


def main() -> None:
    st.set_page_config(page_title="Penghitung Kendaraan", layout="wide")
    st.title("Penghitung Kendaraan")
    st.caption("Praktikum pengolahan citra · YOLO11n dan ByteTrack")
    config = load_config(ROOT / "configs" / "default.yaml")
    local_videos = {p.name: p for p in sorted((ROOT / "data" / "videos").glob("*.mp4"))}
    setup, preview = st.columns([1, 2])
    with setup:
        st.subheader("1. Pilih video")
        source_name = st.selectbox("Sumber video", ["Unggah MP4", *local_videos])
        local = local_videos.get(source_name)
        uploaded = st.file_uploader("Unggah video MP4", type="mp4") if local is None else None
        if uploaded is not None:
            source_name = uploaded.name
        data = uploaded.getvalue() if uploaded is not None else None
        st.subheader("2. Atur analisis")
        profile = "Baseline"
        use_evaluation_settings = False
        if local is not None and local.name in DEMO_EXPERIMENTS:
            profile_file, credit = DEMO_EXPERIMENTS[local.name]
            st.caption(credit)
            profile = st.radio("Profil analisis", ("Baseline", "Pemulihan crossing", "Kustom"))
            use_evaluation_settings = profile != "Kustom"
            if use_evaluation_settings:
                recovery = load_config(ROOT / "configs" / "experiments" / profile_file)
                if profile == "Pemulihan crossing":
                    config = recovery
                else:
                    config = replace(
                        config, counting_line=recovery.counting_line, preprocess=recovery.preprocess
                    )

        if use_evaluation_settings:
            mode, line = config.preprocess.mode, config.counting_line
            st.caption(f"{MODES[mode]} · Garis dikunci sesuai evaluasi klip ini.")
        else:
            mode = st.selectbox("Prapemrosesan", PREPROCESS_MODES, format_func=MODES.__getitem__)
            st.caption("P1 dan P2 memakai koordinat 0–1 dari sudut kiri atas.")
            x1 = st.slider("P1 · X", 0.0, 1.0, config.counting_line.p1[0], 0.01)
            y1 = st.slider("P1 · Y", 0.0, 1.0, config.counting_line.p1[1], 0.01)
            x2 = st.slider("P2 · X", 0.0, 1.0, config.counting_line.p2[0], 0.01)
            y2 = st.slider("P2 · Y", 0.0, 1.0, config.counting_line.p2[1], 0.01)
            line = None
            try:
                line = replace(config.counting_line, p1=(x1, y1), p2=(x2, y2))
            except ValueError as error:
                st.error(str(error))

    ready = False
    with preview:
        st.subheader("Preview garis")
        if local is None and data is None:
            st.info("Pilih rekaman dari kamera diam untuk menempatkan garis di kedua lajur.")
        else:
            try:
                metadata, image = _preview(local, data, local.stat().st_mtime_ns if local else None)
                st.caption(
                    f"{source_name} · {metadata.width} × {metadata.height} · "
                    f"{metadata.source_fps:g} fps"
                )
                st.image(
                    _line_preview(image, line) if line else image, channels="BGR", width="stretch"
                )
                ready = line is not None
            except (ValueError, OSError, cv2.error) as error:
                st.error(f"Video tidak dapat dibaca. {error}")

    if setup.button("Mulai analisis", type="primary", disabled=not ready):
        progress = st.progress(0, text="Menyiapkan model dan video")

        def update(step):
            fraction = step.processed_frames / step.total_frames if step.total_frames else 0
            total = f" / {step.total_frames}" if step.total_frames else ""
            progress.progress(
                min(0.99, fraction), text=f"Memproses frame {step.processed_frames}{total}"
            )

        st.session_state.pop("result", None)
        try:
            with _source_file(local, data) as source:
                result = process_video(
                    source,
                    replace(
                        config, counting_line=line, preprocess=replace(config.preprocess, mode=mode)
                    ),
                    ROOT / "outputs" / "ui",
                    update,
                )
            st.session_state.result = (result, source_name, mode, profile)
            if result.status == "completed":
                progress.progress(1.0, text="Analisis selesai")
            else:
                progress.empty()
        except (ValueError, OSError, RuntimeError, cv2.error) as error:
            progress.empty()
            st.error(f"Analisis gagal. {error}")
    if "result" in st.session_state:
        _show_result(*st.session_state.result)


if __name__ == "__main__":
    main()
