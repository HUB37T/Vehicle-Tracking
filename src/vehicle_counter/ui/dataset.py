"""Manual box annotation on original source frames, separate from crossing review."""

from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import streamlit as st

from vehicle_counter.exporting import file_sha256
from vehicle_counter.training_labels import (
    export_dataset,
    latest_labels,
    save_labels,
    validate_boxes,
)
from vehicle_counter.ui.shape_editor import render_shapes


@st.cache_data(max_entries=8, show_spinner=False)
def _checksum(path, modified, size):
    return file_sha256(Path(path))


def _read_frame(path, index):
    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok or abs(capture.get(cv2.CAP_PROP_POS_FRAMES) - (index + 1)) > 0.1:
            raise ValueError("Frame sumber yang diminta belum dapat dibaca dengan tepat.")
        return frame
    finally:
        capture.release()


def show_dataset_labels(result, manifest, root):
    with st.expander("Beri kotak manual untuk calon dataset YOLO"):
        if not st.checkbox(
            "Buka editor kotak manual", value=False, key=f"open-labels-{result.run_id}"
        ):
            return
        st.caption(
            "Gambar yang disimpan adalah frame sumber tanpa overlay. Periksa "
            "seluruh kendaraan, kelas, dan kotaknya. Catatan crossing dan "
            "prediksi model tidak otomatis menjadi label pelatihan."
        )
        source = Path(manifest["source"]["path"])
        uploaded = None
        if not source.is_file():
            st.info(
                "Video sumber run lama tidak tersedia. Unggah kembali video asli "
                "dengan checksum yang sama."
            )
            uploaded = st.file_uploader(
                "Video asli untuk pelabelan", type="mp4", key=f"label-source-{result.run_id}"
            )
            if uploaded is None:
                return
        with TemporaryDirectory(prefix="vehicle-label-") as temporary:
            if uploaded:
                source = Path(temporary) / "source.mp4"
                source.write_bytes(uploaded.getvalue())
            try:
                stat = source.stat()
                if (
                    _checksum(str(source), stat.st_mtime_ns, stat.st_size)
                    != manifest["source"]["sha256"]
                ):
                    st.error("Checksum video berbeda dari sumber run. Label tidak disimpan.")
                    return
                index = st.number_input(
                    "Frame sumber yang diberi kotak",
                    0,
                    manifest["processed_frames"] - 1,
                    0,
                    key=f"label-frame-{result.run_id}",
                )
                frame = _read_frame(source, int(index))
                initial = next(
                    (
                        r["boxes"]
                        for _, r in latest_labels(root)
                        if r["source_sha256"] == manifest["source"]["sha256"]
                        and r["frame_index"] == index
                    ),
                    [],
                )
                value = render_shapes(
                    frame, mode="boxes", key=f"boxes-{result.run_id}-{index}", boxes=initial
                )
                boxes = validate_boxes(value["boxes"] if value is not None else initial)
                confirmed = st.checkbox(
                    "Seluruh kotak dan kelas sudah saya periksa; tanpa kotak berarti frame negatif",
                    key=f"label-confirm-{result.run_id}-{index}",
                )
                if st.button(
                    "Simpan label frame",
                    disabled=not confirmed,
                    key=f"save-label-{result.run_id}-{index}",
                ):
                    save_labels(
                        root,
                        source_sha256=manifest["source"]["sha256"],
                        frame_index=int(index),
                        frame_bgr=frame,
                        boxes=boxes,
                        confirmed=confirmed,
                        source_label=manifest["video_metadata"]["source_name"],
                    )
                    st.success("Label manual tersimpan sebagai revisi baru.")
                st.caption(
                    f"{len(boxes)} kotak pada frame ini. "
                    "Kelas dataset: 0 mobil, 1 motor, 2 bus, 3 truk."
                )
            except (ValueError, OSError, TypeError, KeyError, cv2.error) as error:
                st.error(f"Label belum bisa diproses. {error}")
        if st.button("Siapkan ZIP label YOLO", key=f"prepare-labels-{result.run_id}"):
            try:
                st.session_state[f"dataset-zip-{result.run_id}"] = export_dataset(root)
            except (ValueError, OSError) as error:
                st.error(str(error))
        if archive := st.session_state.get(f"dataset-zip-{result.run_id}"):
            st.download_button(
                "Unduh calon dataset YOLO",
                archive,
                "manual-yolo-labels.zip",
                "application/zip",
                on_click="ignore",
                key=f"labels-download-{result.run_id}",
            )
        st.caption(
            "Pisahkan train/validation/test menurut video sumber. Jangan "
            "memasukkan klip evaluasi ke pelatihan lalu menganggap evaluasinya"
            " tetap independen."
        )
