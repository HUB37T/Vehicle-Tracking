"""Streamlit front end for the local vehicle counting pipeline."""

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from importlib.util import find_spec
from math import hypot
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import cv2
import pandas as pd
import streamlit as st

from vehicle_counter.analytics_dashboard import (
    configure_zones,
    show_heatmap,
    show_image_analytics,
    show_quality,
    show_traffic_zones,
)
from vehicle_counter.config import load_config
from vehicle_counter.contracts import (
    DIRECTIONS,
    PREPROCESS_MODES,
    VEHICLE_CLASSES,
    CountSummary,
    RunResult,
)
from vehicle_counter.dashboard import (
    show_comparison,
    show_diagnostics,
    show_review_form,
    show_timeline,
    show_traffic,
)
from vehicle_counter.dataset_dashboard import show_dataset_labels
from vehicle_counter.geometry import PixelCountingLine
from vehicle_counter.history import list_runs, load_run
from vehicle_counter.job_dashboard import show_jobs
from vehicle_counter.jobs import ACTIVE, JobQueue
from vehicle_counter.line_editor import decode_editor, render_editor
from vehicle_counter.quality import inspect_quality
from vehicle_counter.tracking import load_tracker_settings
from vehicle_counter.ui_layout import QUEUE, RESULTS, SETUP, open_view, show_header, workspace_tabs
from vehicle_counter.video import VideoReader

ROOT = Path(__file__).resolve().parent
PUBLIC_DEMO = os.environ.get("VEHICLE_COUNTER_PUBLIC_DEMO") == "1"
CLASSES = dict(zip(VEHICLE_CLASSES, ("Mobil", "Motor", "Bus", "Truk"), strict=True))
MODES = {
    "none": "Tanpa prapemrosesan",
    "clahe": "CLAHE",
    "gamma": "Gamma correction",
    "gaussian": "Gaussian smoothing",
    "unsharp": "Unsharp masking",
    "median": "Median (pengurangan noise)",
    "bilateral_clahe": "Bilateral + CLAHE (eksperimen)",
}
DEMO_EXPERIMENTS = {
    "vehicles-3-bytetrack-test.mp4": (
        "vehicles-3-coursework.yaml",
        "Klip antrean lokal tim untuk pengembangan dan pemeriksaan indikator macet.",
    ),
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


@st.cache_data(max_entries=4, show_spinner=False)
def _quality_preview(local, data, modified_ns, config):
    with _source_file(local, data) as source:
        report, panel = inspect_quality(source, config)
        ok, encoded = cv2.imencode(".png", panel)
        if not ok:
            raise ValueError("Panel kualitas tidak dapat dibuat.")
        return report, encoded.tobytes()


def _line_preview(frame, config, roi=None):
    image = frame.copy()
    h, w = image.shape[:2]
    line = PixelCountingLine.from_config(config, width=w, height=h)
    p1, p2 = tuple(map(round, line.p1)), tuple(map(round, line.p2))
    yellow = (0, 225, 255)
    if roi is not None:
        left, top, right, bottom = roi
        cv2.rectangle(
            image,
            (round(left * w), round(top * h)),
            (round(right * w), round(bottom * h)),
            (255, 190, 30),
            2,
        )
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


@st.cache_data(max_entries=24, show_spinner=False)
def _audit_frame(filename: str, frame_index: int):
    capture = cv2.VideoCapture(filename)
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            raise ValueError("Frame crossing tidak dapat dibaca dari video hasil.")
        return frame
    finally:
        capture.release()


def _show_result(result: RunResult, source_name: str, mode: str, profile: str) -> None:
    st.header("Hasil analisis")
    st.caption(f"{source_name} · {profile} · {MODES[mode]} · Run {result.run_id}")
    if result.status != "completed":
        if result.status == "cancelled":
            st.warning(
                "Analisis dibatalkan. Hitungan berikut masih parsial dan belum "
                "mencakup seluruh rentang video."
            )
        else:
            st.error("Analisis berhenti sebelum selesai. Periksa detail proses.")
        for warning in result.warnings:
            st.write(warning)
        for key in ("events_csv", "counts_csv", "run_json"):
            if path := result.artifact_paths.get(key):
                st.download_button(
                    f"Unduh hasil parsial {Path(path).name}",
                    Path(path).read_bytes(),
                    Path(path).name,
                    key=f"partial-{result.run_id}-{key}",
                    on_click="ignore",
                )
        return

    manifest = json.loads(Path(result.artifact_paths["run_json"]).read_text(encoding="utf-8"))
    used = manifest["config"]
    fps = manifest.get("processing_fps")
    st.caption(
        f"Model {Path(used['model_weights']).name} · masukan {used['imgsz']} px · "
        f"confidence {used['detector_conf']:g}"
        + (f" · pemrosesan {fps:.1f} frame/detik" if fps else "")
    )
    if manifest.get("analysis_scope", {}).get("kind") == "preview":
        st.info(
            f"Hasil uji singkat pada {manifest['analyzed_duration_seconds']:.2f} detik awal video."
        )

    audit_rows = None
    if path := result.artifact_paths.get("crossing_audit"):
        audit_rows = pd.read_csv(path)
        audit_rows["review_reason"] = audit_rows["review_reason"].astype("string").fillna("")
        flagged = audit_rows["review_reason"].ne("").sum()
        if flagged:
            st.warning(
                f"{flagged} event perlu ditinjau. Buka Audit crossing untuk memeriksa frame; "
                "tanda ini belum membuktikan event salah."
            )

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
        column.metric(label, value, border=True)
    table, video, traffic, heatmap, evidence, audit, exports = st.tabs(
        (
            "Rekap",
            "Video",
            "Lalu lintas",
            "Heatmap",
            "Bukti materi",
            "Audit crossing",
            "Unduhan",
        )
    )
    with table:
        rows = [
            {"Kendaraan": CLASSES[name], "A → B": counts["A_TO_B"], "B → A": counts["B_TO_A"]}
            for name, counts in result.summary.by_class_direction.items()
        ]
        frame = pd.DataFrame(rows)
        counts_table, counts_chart = st.columns([1, 1.5])
        with counts_table:
            st.subheader("Hitungan per kelas", anchor=False)
            st.dataframe(frame, hide_index=True)
        counts_chart.bar_chart(frame, x="Kendaraan", y=["A → B", "B → A"], stack=False)
        with st.container(border=True):
            show_timeline(result)
    with video:
        st.video(Path(result.artifact_paths["annotated_video"]))
        st.caption("Video beranotasi disimpan tanpa audio sumber.")
    with evidence:
        show_image_analytics(result, manifest)
        with st.expander("Diagnostik deteksi dan tracking"):
            show_diagnostics(result, manifest, _audit_frame)
    with heatmap:
        show_heatmap(result)
    with audit:
        st.caption(
            "Hitungan berasal dari event tracker. Kotak yang bertumpuk perlu diperiksa pada "
            "video; event tanpa tanda juga belum membuktikan identitas kendaraan benar."
        )
        selected_row = None
        if audit_rows is not None:
            st.dataframe(audit_rows, hide_index=True)
            if not audit_rows.empty:
                selected = st.selectbox(
                    "Periksa event",
                    range(len(audit_rows)),
                    format_func=lambda i: (
                        f"Frame {audit_rows.iloc[i]['frame_index']} · "
                        f"ID {audit_rows.iloc[i]['track_id']}"
                    ),
                )
                row = audit_rows.iloc[selected]
                selected_row = row
                try:
                    st.image(
                        _audit_frame(
                            result.artifact_paths["annotated_video"], int(row.frame_index)
                        ),
                        caption=f"Frame crossing ID {row.track_id} · {row.timestamp_seconds:.3f} s",
                        channels="BGR",
                        width="stretch",
                    )
                except (ValueError, cv2.error) as error:
                    st.error(str(error))
        else:
            st.info("Run ini dibuat sebelum ekspor audit crossing tersedia.")
        show_review_form(result, manifest, selected_row, _audit_frame)
        label_root = ROOT / "outputs" / "dataset-labels"
        if PUBLIC_DEMO:
            label_root /= st.session_state.job_owner
        show_dataset_labels(result, manifest, label_root)
    with traffic:
        show_traffic_zones(result, manifest, _audit_frame)
        show_traffic(result, manifest, _audit_frame)
    with exports:
        for key, label, mime in (
            ("events_csv", "Unduh daftar crossing", "text/csv"),
            ("counts_csv", "Unduh rekap hitungan", "text/csv"),
            ("crossing_audit", "Unduh audit crossing", "text/csv"),
            ("timeline_csv", "Unduh grafik waktu CSV", "text/csv"),
            ("traffic_csv", "Unduh status lalu lintas", "text/csv"),
            ("traffic_episodes_csv", "Unduh waktu indikasi macet", "text/csv"),
            ("traffic_zones_csv", "Unduh status per zona", "text/csv"),
            ("traffic_zone_episodes_csv", "Unduh kejadian macet per zona", "text/csv"),
            ("heatmap_png", "Unduh heatmap", "image/png"),
            ("heatmap_npz", "Unduh sel waktu heatmap", "application/octet-stream"),
            ("heatmap_json", "Unduh metadata heatmap", "application/json"),
            ("quality_json", "Unduh pengukuran kualitas", "application/json"),
            ("quality_png", "Unduh panel histogram, FFT, dan ORB", "image/png"),
            ("diagnostics_json", "Unduh diagnostik", "application/json"),
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


def _show_history(queue, owned_jobs) -> None:
    with st.container(border=True):
        st.subheader("Riwayat analisis", anchor=False)
        if PUBLIC_DEMO:
            st.caption("Riwayat demo menampilkan pekerjaan dari sesi browser ini.")
        if st.checkbox("Tampilkan riwayat tersimpan", value=False):
            if PUBLIC_DEMO:
                history = [
                    run for job in owned_jobs for run in list_runs(queue.root / job["job_id"])
                ]
            else:
                history = list_runs(ROOT / "outputs")
            if history:
                query = st.text_input("Cari nama video atau ID run")
                history = [
                    r for r in history if query.lower() in (r["source"] + r["run_id"]).lower()
                ]
                if history:
                    selected = st.selectbox(
                        "Hasil tersimpan",
                        range(len(history)),
                        format_func=lambda i: (
                            f"{history[i]['source']} · {history[i]['model']} · "
                            f"{history[i]['scope']} · {history[i]['total']} crossing · "
                            f"{history[i]['status']} · {history[i]['run_id']}"
                        ),
                    )
                    if st.button("Buka hasil tersimpan"):
                        try:
                            restored, info = load_run(Path(history[selected]["path"]))
                            st.session_state.result = (
                                restored,
                                info["video_metadata"]["source_name"],
                                info["config"]["preprocess"]["mode"],
                                "Riwayat",
                            )
                            open_view(RESULTS)
                        except (ValueError, OSError, KeyError) as error:
                            st.error(f"Hasil belum bisa dibuka. {error}")
                else:
                    st.info("Tidak ada hasil yang cocok dengan pencarian.")
            else:
                st.info("Belum ada analisis tersimpan.")


def _show_setup(queue, owner, has_active) -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    local_videos = {p.name: p for p in sorted((ROOT / "data" / "videos").glob("*.mp4"))}
    with st.container(key="vehicle_setup"):
        settings_column, preview_column = st.columns([1, 1.9], gap="large")
    setup = settings_column.container(border=True)
    preview = preview_column.container(border=True)
    actions = preview_column.container(border=True)
    tools = preview_column.container()
    with setup:
        st.subheader("Sumber dan pengaturan", anchor=False)
        source_name = st.selectbox("Sumber video", ["Unggah MP4", *local_videos])
        local = local_videos.get(source_name)
        uploaded = st.file_uploader("Unggah video MP4", type="mp4") if local is None else None
        if uploaded is not None:
            source_name = uploaded.name
        data = uploaded.getvalue() if uploaded is not None else None
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

        source_version = (
            local.stat().st_mtime_ns if local else (uploaded.file_id if uploaded else "none")
        )
        editor_key = f"editor-{source_name}-{source_version}-{profile}"
        stored_points = st.session_state.get(editor_key + "-points")
        initial_line = config.counting_line
        initial_congestion = config.congestion
        if stored_points:
            saved_line, saved_roi = decode_editor(stored_points, initial_line)
            if not use_evaluation_settings:
                initial_line = saved_line
            initial_congestion = replace(initial_congestion, roi=saved_roi)
        traffic_enabled = st.checkbox("Aktifkan indikator macet", value=config.congestion.enabled)
        heatmap_enabled = st.checkbox(
            "Buat heatmap keberadaan kendaraan", value=config.heatmap_enabled
        )
        congestion = replace(initial_congestion, enabled=traffic_enabled)
        numeric_roi = False
        if traffic_enabled:
            st.caption(
                "Untuk kamera diam. Pilih area satu lajur/arah pada preview "
                "agar status lebih mudah ditafsirkan."
            )
            with st.expander("Pengaturan indikator macet"):
                count = st.number_input(
                    "Minimal kendaraan dalam area", 2, 50, congestion.min_vehicles
                )
                occupancy = st.slider(
                    "Minimal area tertutup kendaraan", 0.01, 1.0, congestion.min_occupancy, 0.01
                )
                speed = st.number_input(
                    "Ambang gerak lambat (diagonal kotak/detik)",
                    0.01,
                    5.0,
                    congestion.slow_speed,
                    0.01,
                )
                hold = st.number_input(
                    "Durasi kondisi lambat sebelum status macet (detik)",
                    1.0,
                    30.0,
                    congestion.hold_seconds,
                    0.5,
                )
                congestion = replace(
                    congestion,
                    min_vehicles=int(count),
                    min_occupancy=occupancy,
                    slow_speed=speed,
                    hold_seconds=hold,
                )
                numeric_roi = st.checkbox("Atur area dengan angka", value=False)
                if numeric_roi:
                    values = [
                        st.slider(label, 0.0, 1.0, float(value), 0.01)
                        for label, value in zip(
                            ("Area X kiri", "Area Y atas", "Area X kanan", "Area Y bawah"),
                            congestion.roi,
                            strict=True,
                        )
                    ]
                    try:
                        congestion = replace(congestion, roi=tuple(values))
                    except ValueError as error:
                        st.error(str(error))
                        congestion = None
        with st.expander("Model dan pelacakan"):
            tracker_name = st.selectbox("Tracker", ("ByteTrack", "DeepSORT (eksperimen)"))
            deepsort_selected = tracker_name.startswith("DeepSORT")
            deepsort_available = find_spec("deep_sort_realtime") is not None
            deepsort_ready = deepsort_available
            if deepsort_selected:
                st.caption(
                    "DeepSORT memakai fitur tampilan MobileNetV2 pretrained ImageNet pada CPU. "
                    "Belum dilatih khusus identitas kendaraan dan dapat lebih lambat."
                )
                if not deepsort_available:
                    st.error("DeepSORT belum dipasang. Ikuti bagian DeepSORT pada README.")
                else:
                    try:
                        _, settings = load_tracker_settings(str(ROOT / "configs" / "deepsort.yaml"))
                        if settings["tracker_type"] != "deepsort":
                            raise ValueError("Konfigurasi bukan DeepSORT.")
                    except (ValueError, OSError):
                        deepsort_ready = False
                        st.error(
                            "Konfigurasi DeepSORT belum bisa dipakai oleh proses aplikasi ini. "
                            "Mulai ulang aplikasi setelah memperbarui kode; jika tetap gagal, "
                            "periksa configs/deepsort.yaml."
                        )
            model_choice = st.selectbox(
                "Model deteksi", ("Sesuai profil", "YOLO11n — ringan", "YOLO11s — pembanding")
            )
            if model_choice != "Sesuai profil":
                variant = "yolo11s" if model_choice.startswith("YOLO11s") else "yolo11n"
                config = replace(config, model_weights=str(ROOT / "models" / f"{variant}.pt"))
            size = st.selectbox("Ukuran masukan YOLO", ("Sesuai profil", 640, 960, 1280))
            conf = st.number_input(
                "Confidence minimum detektor", 0.01, 1.0, config.detector_conf, 0.01
            )
            config = replace(
                config, imgsz=config.imgsz if size == "Sesuai profil" else size, detector_conf=conf
            )
            st.caption(
                "YOLO11s dan masukan lebih besar membutuhkan waktu lebih lama. "
                "Periksa crossing terhadap anotasi sebelum memilihnya. Confidence tinggi "
                "dapat membuang deteksi lemah yang dibutuhkan ByteTrack."
            )
            model_path = Path(config.model_weights)
            if model_path.parent == Path("."):
                model_path = ROOT / "models" / model_path
            model_ready = model_path.is_file()
            if not model_ready:
                st.warning(
                    f"Bobot {model_path.name} belum tersedia. Siapkan melalui prepare-model "
                    f"--variant {model_path.stem}, lalu mulai ulang aplikasi."
                )
            mode = config.preprocess.mode
            if use_evaluation_settings:
                st.caption(f"Prapemrosesan profil: {MODES[mode]}")
            else:
                mode = st.selectbox(
                    "Prapemrosesan", PREPROCESS_MODES, format_func=MODES.__getitem__
                )
            quality_enabled = st.checkbox(
                "Sertakan pemeriksaan kualitas video", value=config.quality_check
            )

    ready = False
    zones = ()
    with preview:
        st.subheader("Pratinjau dan garis hitung", anchor=False)
        if use_evaluation_settings:
            line = config.counting_line
            editor_mode = "Geser di gambar" if traffic_enabled else "Koordinat angka"
            st.caption("Profil ini memakai garis dari evaluasi klip.")
        else:
            editor_mode = st.radio(
                "Pengaturan garis", ("Geser di gambar", "Koordinat angka"), horizontal=True
            )
            line = initial_line
            if editor_mode == "Koordinat angka":
                st.caption("P1 dan P2 memakai koordinat 0–1 dari sudut kiri atas.")
                x1 = st.slider("P1 · X", 0.0, 1.0, initial_line.p1[0], 0.01)
                y1 = st.slider("P1 · Y", 0.0, 1.0, initial_line.p1[1], 0.01)
                x2 = st.slider("P2 · X", 0.0, 1.0, initial_line.p2[0], 0.01)
                y2 = st.slider("P2 · Y", 0.0, 1.0, initial_line.p2[1], 0.01)
                try:
                    line = replace(line, p1=(x1, y1), p2=(x2, y2))
                    if congestion is not None:
                        st.session_state[editor_key + "-points"] = {
                            "p1": list(line.p1),
                            "p2": list(line.p2),
                            "roi": list(congestion.roi),
                        }
                except ValueError as error:
                    line = None
                    st.error(str(error))

        if local is None and data is None:
            with st.container(key="vehicle_empty_preview"):
                st.markdown("### Mulai dengan video MP4")
                st.write("Pilih video yang tersedia atau unggah rekaman hingga 100 MB.")
                st.caption("Setelah video terbaca, atur garis hitung pada pratinjau.")
        else:
            try:
                metadata, image = _preview(local, data, local.stat().st_mtime_ns if local else None)
                st.caption(
                    f"{source_name} · {metadata.width} × {metadata.height} · "
                    f"{metadata.source_fps:g} fps"
                )
                if editor_mode == "Geser di gambar" and line is not None and congestion is not None:
                    key = editor_key
                    result = render_editor(
                        image,
                        line,
                        congestion.roi,
                        key=key,
                        edit_roi=traffic_enabled and not numeric_roi,
                        show_roi=traffic_enabled,
                        edit_line=not use_evaluation_settings,
                    )
                    if result is not None and result != st.session_state.get(key + "-last-value"):
                        candidate_line, area = decode_editor(result, line)
                        if not use_evaluation_settings:
                            line = candidate_line
                        if not numeric_roi:
                            congestion = replace(congestion, roi=area)
                        st.session_state[key + "-last-value"] = result
                    st.session_state[key + "-points"] = {
                        "p1": list(line.p1),
                        "p2": list(line.p2),
                        "roi": list(congestion.roi),
                    }
                    st.caption(
                        f"P1 ({line.p1[0]:.2f}, {line.p1[1]:.2f})   "
                        f"P2 ({line.p2[0]:.2f}, {line.p2[1]:.2f})"
                    )
                else:
                    st.image(
                        _line_preview(
                            image, line, congestion.roi if traffic_enabled and congestion else None
                        )
                        if line
                        else image,
                        channels="BGR",
                        width="stretch",
                    )
                with tools:
                    if traffic_enabled and congestion is not None:
                        zones = configure_zones(
                            image, replace(config, congestion=congestion), editor_key
                        )
                    ready = line is not None and congestion is not None and zones is not None
                    if st.button("Periksa kualitas video sebelum analisis"):
                        with st.spinner("Memeriksa kualitas citra dan fitur latar"):
                            report, panel = _quality_preview(
                                local,
                                data,
                                source_version,
                                replace(config, traffic_zones=zones or ()),
                            )
                        show_quality(report, panel)
            except (ValueError, OSError, cv2.error) as error:
                st.error(f"Video tidak dapat dibaca. {error}")

    actions.subheader("Jalankan analisis", anchor=False)
    time_control, run_control = actions.columns([1, 1])
    short_seconds = time_control.select_slider(
        "Durasi uji singkat (detik)", options=[3, 5, 10], value=5
    )
    disabled = (
        not ready or not model_ready or has_active or (deepsort_selected and not deepsort_ready)
    )
    short = run_control.button("Uji singkat", type="primary", disabled=disabled, width="stretch")
    full = run_control.button(
        "Mulai analisis",
        disabled=disabled,
        width="stretch",
    )
    if has_active:
        actions.caption("Sesi ini punya pekerjaan aktif. Pantau progres di tab Antrean.")
    elif not model_ready or (deepsort_selected and not deepsort_ready):
        actions.caption("Buka Model dan pelacakan untuk memeriksa kesiapan model dan tracker.")
    elif not ready:
        actions.caption("Siapkan video dan garis hitung untuk mulai.")
    actual_config = (
        replace(
            config,
            counting_line=line,
            preprocess=replace(config.preprocess, mode=mode),
            congestion=congestion,
            traffic_zones=zones or (),
            heatmap_enabled=heatmap_enabled,
            quality_check=quality_enabled,
            tracker_config=str(ROOT / "configs" / "deepsort.yaml")
            if deepsort_selected
            else config.tracker_config,
        )
        if ready
        else None
    )
    if short or full:
        try:
            with _source_file(local, data) as source:
                job_id = queue.enqueue(
                    source,
                    actual_config,
                    owner=owner,
                    source_label=source_name,
                    max_seconds=short_seconds if short else None,
                )
            st.session_state.pending_job = job_id
            queue.ensure_worker()
            open_view(QUEUE)
        except (ValueError, OSError, RuntimeError, cv2.error) as error:
            actions.error(f"Tidak dapat memasukkan video ke antrean. {error}")
    with st.expander("Perbandingan metode"):
        compare_kind = st.radio(
            "Yang dibandingkan",
            (
                "Tanpa vs CLAHE",
                "ByteTrack vs DeepSORT",
                "Keduanya (4 run)",
                "YOLO11n vs YOLO11s",
                "CLAHE vs prapemrosesan terpilih",
            ),
        )
        full_comparison = st.checkbox("Bandingkan seluruh video", value=False)
        compare_models = compare_kind == "YOLO11n vs YOLO11s"
        compare_filter = compare_kind == "CLAHE vs prapemrosesan terpilih"
        compare_requires_deep = compare_kind in ("ByteTrack vs DeepSORT", "Keduanya (4 run)")
        model_weights = tuple(str(ROOT / "models" / f"{v}.pt") for v in ("yolo11n", "yolo11s"))
        comparison_disabled = (
            not ready
            or has_active
            or (compare_filter and mode == "clahe")
            or (
                not deepsort_ready
                if compare_requires_deep
                else (deepsort_selected and not deepsort_ready)
            )
            or (
                not all(Path(w).is_file() for w in model_weights)
                if compare_models
                else not model_ready
            )
        )
        if compare_models and not all(Path(w).is_file() for w in model_weights):
            st.info("Siapkan kedua bobot YOLO11n dan YOLO11s untuk membandingkan model.")
        if compare_filter and mode == "clahe":
            st.info("Pilih profil Kustom dan prapemrosesan pembanding yang berbeda dari CLAHE.")
        if st.button(
            "Bandingkan metode",
            disabled=comparison_disabled,
        ):
            try:
                if compare_requires_deep:
                    load_tracker_settings(str(ROOT / "configs" / "deepsort.yaml"))
                modes = (
                    ("none", "clahe")
                    if compare_kind in ("Tanpa vs CLAHE", "Keduanya (4 run)")
                    else (mode,)
                )
                if compare_filter:
                    modes = ("clahe", mode)
                trackers = (
                    ("bytetrack.yaml", str(ROOT / "configs" / "deepsort.yaml"))
                    if compare_requires_deep
                    else (actual_config.tracker_config,)
                )
                with _source_file(local, data) as source:
                    job_id = queue.enqueue(
                        source,
                        actual_config,
                        owner=owner,
                        source_label=source_name,
                        max_seconds=None if full_comparison else short_seconds,
                        comparison={
                            "modes": modes,
                            "trackers": trackers,
                            "weights": model_weights if compare_models else None,
                        },
                    )
                st.session_state.pending_job = job_id
                queue.ensure_worker()
                open_view(QUEUE)
            except (ValueError, OSError, RuntimeError, ImportError) as error:
                st.error(f"Perbandingan gagal. {error}")


def _show_results() -> None:
    if "result" not in st.session_state and "comparison" not in st.session_state:
        with st.container(border=True):
            st.subheader("Belum ada hasil analisis", anchor=False)
            st.write("Jalankan video baru atau buka hasil tersimpan dari Riwayat.")
            if st.button("Siapkan video", key="setup_from_results"):
                open_view(SETUP)
        return
    if "comparison" in st.session_state and st.session_state.comparison["results"]:
        batch = st.session_state.comparison
        with st.expander("Perbandingan terakhir", expanded="result" not in st.session_state):
            show_comparison(batch)
            selected = st.selectbox(
                "Buka hasil perbandingan",
                range(len(batch["results"])),
                format_func=lambda i: (
                    f"{batch['rows'][i].get('model', 'YOLO')} - "
                    f"{batch['rows'][i]['preprocess']} - {batch['rows'][i]['tracker']} - "
                    f"{batch['rows'][i]['status']}"
                ),
            )
            if st.button("Tampilkan hasil ini"):
                payload = dict(batch["results"][selected])
                payload["summary"] = CountSummary(**payload["summary"])
                row = batch["rows"][selected]
                st.session_state.result = (
                    RunResult(**payload),
                    batch["source_name"],
                    row["preprocess"],
                    f"Perbandingan - {row['tracker']}",
                )
    if "result" in st.session_state:
        _show_result(*st.session_state.result)


def main() -> None:
    st.set_page_config(page_title="Vehicle Tracking", layout="wide")
    show_header()
    owner = st.session_state.setdefault("job_owner", uuid4().hex)
    queue = JobQueue(ROOT / "outputs" / "jobs", ROOT)
    owned_jobs = queue.owned(owner)
    setup, jobs, results, history = workspace_tabs()
    with setup:
        _show_setup(queue, owner, any(job["status"] in ACTIVE for job in owned_jobs))
    with jobs:
        if owned_jobs:
            show_jobs(str(queue.root), str(ROOT), owner)
        else:
            with st.container(border=True):
                st.subheader("Antrean kosong", anchor=False)
                st.write("Pekerjaan muncul di sini setelah kamu mengirim video untuk dianalisis.")
                if st.button("Siapkan video", key="setup_from_queue"):
                    open_view(SETUP)
    with results:
        _show_results()
    with history:
        _show_history(queue, owned_jobs)


if __name__ == "__main__":
    main()
