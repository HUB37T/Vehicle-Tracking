import csv
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vehicle_counter.contracts import RunConfig, TrackObservation, VideoMetadata
from vehicle_counter.experiments import run_comparison
from vehicle_counter.pipeline import process_video
from vehicle_counter.review import load_reviews, reviews_csv, save_review
from vehicle_counter.video import AnnotatedVideoWriter


class QueueFixture:
    metadata = {"kind": "fixture", "is_model_inference": False, "tracker_name": "Fixture"}
    warnings = []

    def __init__(self, config, metadata):
        pass

    def track_frame(self, frame_bgr, *, frame_index, timestamp_seconds):
        return [
            TrackObservation(
                frame_index, timestamp_seconds, i, "car", 0.9, (15 + i * 40, 65, 40 + i * 40, 95)
            )
            for i in range(3)
        ]


@pytest.fixture
def dashboard_project(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from vehicle_counter.jobs import JobQueue

    monkeypatch.setattr(JobQueue, "ensure_worker", lambda self: None)
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "configs").mkdir()
    (tmp_path / "data" / "videos").mkdir(parents=True)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "yolo11n.pt").write_bytes(b"fixture, no model inference")
    (tmp_path / "models" / "yolo11s.pt").write_bytes(b"fixture, no model inference")
    for name in ("default.yaml", "deepsort.yaml"):
        shutil.copyfile(root / "configs" / name, tmp_path / "configs" / name)
    shutil.copyfile(root / "app.py", tmp_path / "app.py")
    path = tmp_path / "data" / "videos" / "fixture.mp4"
    with AnnotatedVideoWriter(path, VideoMetadata(path.name, 160, 120, 10, 60, 6)) as writer:
        for _ in range(60):
            writer.write(np.zeros((120, 160, 3), np.uint8))
    return tmp_path, path


def fixture_process(
    source,
    config,
    output,
    on_progress=None,
    *,
    max_seconds=None,
    source_label=None,
    should_cancel=None,
):
    return process_video(
        source,
        config,
        output,
        on_progress,
        tracker_factory=QueueFixture,
        max_seconds=max_seconds,
        source_label=source_label,
        should_cancel=should_cancel,
    )


def finish_app_job(app, root):
    from vehicle_counter.jobs import JobQueue

    queue = JobQueue(root / "outputs" / "jobs", root)
    job = queue.claim()
    assert job is not None
    queue.execute(job, processor=fixture_process)
    app.run()


def test_app_short_preview_traffic_review_and_comparison(dashboard_project, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from vehicle_counter import experiments, pipeline

    root, video = dashboard_project
    monkeypatch.setattr(pipeline, "process_video", fixture_process)
    app = AppTest.from_file(str(root / "app.py"), default_timeout=20).run()
    next(w for w in app.selectbox if w.label == "Sumber video").select(video.name).run()
    next(w for w in app.checkbox if w.label == "Aktifkan indikator macet").check().run()
    assert not app.exception
    next(w for w in app.button if w.label == "Uji singkat").click().run()
    assert app.session_state["vehicle_workspace"] == "Antrean"
    finish_app_job(app, root)
    assert not app.exception
    assert app.session_state["vehicle_workspace"] == "Hasil"
    result = app.session_state["result"][0]
    info = json.loads(Path(result.artifact_paths["run_json"]).read_text())
    assert info["processed_frames"] == 50
    assert info["analysis_scope"]["kind"] == "preview"
    assert info["traffic"]["last_status"] == "congested"
    assert any("Hasil uji singkat" in item.value for item in app.info)
    assert any(w.label == "Periksa status lalu lintas" for w in app.selectbox)
    next(w for w in app.text_area if w.label == "Catatan pemeriksa").set_value(
        "Fixture kendaraan terlewat"
    ).run()
    next(w for w in app.button if w.label == "Simpan catatan").click().run()
    assert not app.exception
    assert len(load_reviews(Path(result.output_dir))) == 1
    next(w for w in app.checkbox if w.label == "Tampilkan riwayat tersimpan").check().run()
    next(w for w in app.button if w.label == "Buka hasil tersimpan").click().run()
    assert not app.exception
    assert app.session_state["vehicle_workspace"] == "Hasil"
    assert app.session_state["result"][0].run_id == result.run_id
    monkeypatch.setattr(experiments, "process_video", fixture_process)
    next(w for w in app.button if w.label == "Bandingkan metode").click().run()
    finish_app_job(app, root)
    assert not app.exception
    batch = app.session_state["comparison"]
    assert len(batch["rows"]) == 2 and batch["status"] == "completed"
    assert all(row["processed_frames"] == 50 for row in batch["rows"])
    next(w for w in app.button if w.label == "Tampilkan hasil ini").click().run()
    assert not app.exception


def test_manual_review_keeps_automatic_csv_immutable(dashboard_project):
    root, video = dashboard_project
    result = fixture_process(video, RunConfig(), root / "results")
    directory = Path(result.output_dir)
    original = (directory / "events.csv").read_bytes()
    for note in ("Pertama", "Kedua"):
        save_review(
            directory,
            frame_index=10,
            track_id=None,
            status="terlewat",
            corrected_class="car",
            direction="A_TO_B",
            notes=note,
        )
    assert (directory / "events.csv").read_bytes() == original
    assert len(list(csv.DictReader(reviews_csv(directory).decode("utf-8-sig").splitlines()))) == 2
    with pytest.raises(ValueError):
        save_review(
            directory,
            frame_index=1000,
            track_id=None,
            status="terlewat",
            corrected_class="car",
            direction="A_TO_B",
            notes="",
        )


def line_editor_args(app):
    return next(
        json.loads(component.proto.json_args)
        for component in app.get("component_instance")
        if component.proto.component_name.endswith("vehicle_line_editor")
    )


def test_dragged_geometry_is_rendered_on_the_same_rerun(dashboard_project):
    from streamlit.testing.v1 import AppTest

    root, video = dashboard_project
    app = AppTest.from_file(str(root / "app.py"), default_timeout=20).run()
    next(w for w in app.selectbox if w.label == "Sumber video").select(video.name).run()
    next(w for w in app.checkbox if w.label == "Aktifkan indikator macet").check().run()
    applied = {
        "p1": [0.27, 0.52],
        "p2": [0.81, 0.57],
        "roi": [0.14, 0.39, 0.76, 0.84],
        "edit_id": "release-1",
    }
    app.session_state[line_editor_args(app)["key"]] = applied
    app.run()
    assert not app.exception
    rendered = line_editor_args(app)
    assert {field: rendered[field] for field in ("p1", "p2", "roi")} == {
        field: applied[field] for field in ("p1", "p2", "roi")
    }
    assert rendered["saved_edit_id"] == "release-1"
    next(w for w in app.checkbox if w.label == "Buat heatmap keberadaan kendaraan").check().run()
    assert line_editor_args(app)["roi"] == applied["roi"]
    numeric = next(w for w in app.checkbox if w.label == "Atur area dengan angka")
    numeric.check().run()
    next(w for w in app.slider if w.label == "Area X kiri").set_value(0.24).run()
    next(w for w in app.checkbox if w.label == "Atur area dengan angka").uncheck().run()
    assert line_editor_args(app)["roi"] == [0.24, 0.39, 0.76, 0.84]
    assert line_editor_args(app)["p1"] == applied["p1"]


def test_numeric_line_survives_switch_back_to_editor(dashboard_project):
    from streamlit.testing.v1 import AppTest

    root, video = dashboard_project
    previous_drag = {
        "p1": [0.2, 0.6],
        "p2": [0.8, 0.6],
        "roi": [0.1, 0.4, 0.9, 0.9],
        "edit_id": "first",
    }
    app = AppTest.from_file(str(root / "app.py"), default_timeout=20).run()
    next(w for w in app.selectbox if w.label == "Sumber video").select(video.name).run()
    component_key = line_editor_args(app)["key"]
    app.session_state[component_key] = previous_drag
    app.run()
    next(w for w in app.radio if w.label == "Pengaturan garis").set_value("Koordinat angka").run()
    next(w for w in app.slider if w.label == "P1 · X").set_value(0.35).run()
    next(w for w in app.radio if w.label == "Pengaturan garis").set_value("Geser di gambar").run()
    next(w for w in app.button if w.label == "Uji singkat").click().run()
    finish_app_job(app, root)
    assert not app.exception
    result = app.session_state["result"][0]
    info = json.loads(Path(result.artifact_paths["run_json"]).read_text())
    assert info["config"]["counting_line"]["p1"] == [0.35, 0.6]
    app.session_state[component_key] = {**previous_drag, "edit_id": "second"}
    app.run()
    next(w for w in app.button if w.label == "Uji singkat").click().run()
    finish_app_job(app, root)
    info = json.loads(Path(app.session_state["result"][0].artifact_paths["run_json"]).read_text())
    assert info["config"]["counting_line"]["p1"] == [0.2, 0.6]


def test_saved_camera_preset_applies_to_another_clip_and_analysis(dashboard_project):
    from streamlit.testing.v1 import AppTest

    root, video = dashboard_project
    next_clip = video.with_name("camera-1-next.mp4")
    shutil.copyfile(video, next_clip)
    geometry = {"p1": [0.83, 0.63], "p2": [0.24, 0.58], "roi": [0.19, 0.34, 0.91, 0.88]}
    app = AppTest.from_file(str(root / "app.py"), default_timeout=20).run()
    next(w for w in app.selectbox if w.label == "Sumber video").select(video.name).run()
    next(w for w in app.checkbox if w.label == "Aktifkan indikator macet").check().run()
    app.session_state[line_editor_args(app)["key"]] = {**geometry, "edit_id": "dragged-preset"}
    next(w for w in app.text_input if w.label == "Nama preset").set_value("Kamera 1").run()
    next(w for w in app.button if w.label == "Simpan posisi sekarang").click().run()
    assert not app.exception and not app.error
    assert any(item.proto.label == "Unduh JSON preset" for item in app.get("download_button"))

    app = AppTest.from_file(str(root / "app.py"), default_timeout=20).run()
    next(w for w in app.selectbox if w.label == "Sumber video").select(next_clip.name).run()
    next(w for w in app.radio if w.label == "Pengaturan garis").set_value("Koordinat angka").run()
    next(w for w in app.slider if w.label == "P1 · X").set_value(0.3).run()
    next(w for w in app.selectbox if w.label == "Preset tersimpan").select("kamera 1").run()
    next(w for w in app.button if w.label == "Terapkan preset").click().run()
    assert not app.exception and not app.error
    assert next(w for w in app.checkbox if w.label == "Aktifkan indikator macet").value
    assert next(w for w in app.slider if w.label == "P1 · X").value == geometry["p1"][0]
    next(w for w in app.radio if w.label == "Pengaturan garis").set_value("Geser di gambar").run()
    assert all(line_editor_args(app)[field] == value for field, value in geometry.items())
    next(w for w in app.button if w.label == "Uji singkat").click().run()
    finish_app_job(app, root)
    result = app.session_state["result"][0]
    config = json.loads(Path(result.artifact_paths["run_json"]).read_text())["config"]
    assert config["counting_line"]["p1"] == geometry["p1"]
    assert config["counting_line"]["p2"] == geometry["p2"]
    assert config["congestion"]["roi"] == geometry["roi"]


def test_imported_camera_preset_is_private_in_public_demo(dashboard_project, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from vehicle_counter.camera_presets import CameraPreset

    root, video = dashboard_project
    evaluation_clip = video.with_name("vehicles-3-bytetrack-test.mp4")
    shutil.copyfile(video, evaluation_clip)
    experiment_dir = root / "configs" / "experiments"
    experiment_dir.mkdir()
    config_name = "vehicles-3-coursework.yaml"
    repo = Path(__file__).resolve().parents[1]
    shutil.copyfile(repo / "configs" / "experiments" / config_name, experiment_dir / config_name)
    monkeypatch.setenv("VEHICLE_COUNTER_PUBLIC_DEMO", "1")
    app = AppTest.from_file(str(root / "app.py"), default_timeout=20).run()
    uploader = next(w for w in app.file_uploader if w.label == "File preset JSON")
    uploader.upload("invalid.json", b"{malformed}", "application/json").run()
    next(w for w in app.button if w.label == "Impor preset").click().run()
    assert any("JSON yang valid" in item.value for item in app.error)

    preset = CameraPreset("Kamera teman", (0.3, 0.7), (0.9, 0.58), (0.12, 0.41, 0.82, 0.93))
    uploader = next(w for w in app.file_uploader if w.label == "File preset JSON")
    uploader.upload("camera.json", preset.to_json().encode("utf-8"), "application/json").run()
    next(w for w in app.button if w.label == "Impor preset").click().run()
    assert not app.exception and not app.error
    assert next(w for w in app.button if w.label == "Terapkan preset").disabled
    next(w for w in app.selectbox if w.label == "Sumber video").select(evaluation_clip.name).run()
    assert next(w for w in app.button if w.label == "Terapkan preset").disabled
    next(w for w in app.radio if w.label == "Profil analisis").set_value("Kustom").run()
    next(w for w in app.button if w.label == "Terapkan preset").click().run()
    assert all(line_editor_args(app)[field] == value for field, value in preset.points().items())

    other_session = AppTest.from_file(str(root / "app.py"), default_timeout=20).run()
    assert next(w for w in other_session.selectbox if w.label == "Preset tersimpan").options == []
    assert not (root / "outputs" / "camera-presets.sqlite3").exists()


def test_comparison_holds_geometry_and_records_partial_failure(dashboard_project):
    root, video = dashboard_project
    captured = []

    def process(source, config, output, *, max_seconds=None):
        captured.append(config)
        return fixture_process(source, config, output, max_seconds=max_seconds)

    config = replace(RunConfig(), congestion=replace(RunConfig().congestion, enabled=True))
    batch = run_comparison(
        video,
        config,
        root / "comparison",
        modes=("none", "clahe"),
        trackers=("bytetrack.yaml", "configs/deepsort.yaml"),
        processor=process,
        max_seconds=0.5,
    )
    assert len(batch["rows"]) == 4 and batch["status"] == "completed"
    assert all(
        c.counting_line == config.counting_line and c.congestion == config.congestion
        for c in captured
    )

    def fail(*args, **kwargs):
        raise RuntimeError("Fixture comparison interruption")

    with pytest.raises(RuntimeError):
        run_comparison(
            video,
            config,
            root / "failed-comparison",
            modes=("none",),
            trackers=("bytetrack.yaml",),
            processor=fail,
        )
    info = json.loads(next((root / "failed-comparison").glob("*/comparison.json")).read_text())
    assert info["status"] == "failed" and info["error"]["type"] == "RuntimeError"


def test_preview_is_not_scored_against_full_video_annotations(dashboard_project):
    from vehicle_counter.evaluation import compare_runs

    root, video = dashboard_project
    result = fixture_process(video, RunConfig(), root / "short", max_seconds=1)
    info = json.loads(Path(result.artifact_paths["run_json"]).read_text())
    gt = root / "manual.csv"
    gt.write_text("clip_id,manual_event_id,frame_index,timestamp_seconds,class_name,direction\n")
    gt.with_suffix(".json").write_text(
        json.dumps(
            {
                "clip_id": "fixture",
                "source_sha256": info["source"]["sha256"],
                "counting_line": info["config"]["counting_line"],
            }
        )
    )
    with pytest.raises(ValueError, match="Uji singkat"):
        compare_runs(gt, "fixture", [Path(result.output_dir)])


@pytest.mark.parametrize(
    ("variant", "model_label", "comparison"),
    [
        ("yolo11s", "YOLO11s — pembanding", "YOLO11n vs YOLO11s"),
        ("yolo26n", "YOLO26n — kandidat edge", "YOLO11n vs YOLO26n"),
    ],
)
def test_model_controls_and_comparison_keep_camera_settings(
    dashboard_project, monkeypatch, variant, model_label, comparison
):
    from streamlit.testing.v1 import AppTest

    from vehicle_counter import experiments, pipeline

    root, video = dashboard_project
    monkeypatch.setattr(pipeline, "process_video", fixture_process)
    monkeypatch.setattr(experiments, "process_video", fixture_process)
    app = AppTest.from_file(str(root / "app.py"), default_timeout=25).run()
    next(w for w in app.selectbox if w.label == "Sumber video").select(video.name).run()
    next(w for w in app.selectbox if w.label == "Model deteksi").select(model_label).run()
    if variant == "yolo26n":
        assert next(w for w in app.button if w.label == "Uji singkat").disabled
        assert any("--variant yolo26n" in w.value for w in app.warning)
        (root / "models" / "yolo26n.pt").write_bytes(b"fixture, no model inference")
        app.run()
        next(w for w in app.button if w.label == "Uji singkat").click().run()
        finish_app_job(app, root)
        run = app.session_state["result"][0]
        info = json.loads(Path(run.artifact_paths["run_json"]).read_text())
        assert Path(info["config"]["model_weights"]).name == "yolo26n.pt"
    next(w for w in app.selectbox if w.label == "Ukuran masukan YOLO").select(960).run()
    next(w for w in app.number_input if w.label == "Confidence minimum detektor").set_value(
        0.12
    ).run()
    next(w for w in app.radio if w.label == "Yang dibandingkan").set_value(comparison).run()
    next(w for w in app.button if w.label == "Bandingkan metode").click().run()
    finish_app_job(app, root)
    assert not app.exception
    batch = app.session_state["comparison"]
    assert len(batch["rows"]) == 2
    assert {row["model"] for row in batch["rows"]} == {"yolo11n.pt", f"{variant}.pt"}
    assert all(row["imgsz"] == 960 and row["detector_conf"] == 0.12 for row in batch["rows"])
    manifests = [
        json.loads(Path(r["artifact_paths"]["run_json"]).read_text()) for r in batch["results"]
    ]
    assert manifests[0]["config"]["counting_line"] == manifests[1]["config"]["counting_line"]
    assert manifests[0]["config"]["congestion"] == manifests[1]["config"]["congestion"]


def test_zones_heatmap_quality_and_manual_dataset_in_ui(dashboard_project, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from vehicle_counter.training_labels import latest_labels
    from vehicle_counter.ui import dataset as dataset_dashboard

    root, video = dashboard_project
    app = AppTest.from_file(str(root / "app.py"), default_timeout=25).run()
    next(w for w in app.selectbox if w.label == "Sumber video").select(video.name).run()
    next(w for w in app.checkbox if w.label == "Aktifkan indikator macet").check().run()
    next(w for w in app.checkbox if w.label == "Analisis beberapa zona jalan").check().run()
    next(w for w in app.checkbox if w.label == "Buat heatmap keberadaan kendaraan").check().run()
    next(
        w for w in app.button if w.label == "Periksa kualitas video sebelum analisis"
    ).click().run()
    assert not app.exception
    next(w for w in app.button if w.label == "Uji singkat").click().run()
    finish_app_job(app, root)
    assert not app.exception
    result = app.session_state["result"][0]
    info = json.loads(Path(result.artifact_paths["run_json"]).read_text())
    assert len(info["traffic_zones"]) == 2
    assert Path(result.artifact_paths["heatmap_png"]).is_file()
    assert Path(result.artifact_paths["quality_png"]).is_file()
    with Path(result.artifact_paths["lane_counts_csv"]).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    lanes = next(tab for tab in app.tabs if tab.label == "Per lajur")
    table = lanes.dataframe[0].value
    assert table["Lajur"].tolist() == [row["zone_name"] for row in rows]
    assert table["Total"].tolist() == [int(row["total"]) for row in rows]
    assert table["Total"].sum() == result.summary.total
    heatmap = next(tab for tab in app.tabs if tab.label == "Heatmap")
    assert heatmap.get("image")
    monkeypatch.setattr(
        dataset_dashboard,
        "render_shapes",
        lambda *args, **kwargs: {"boxes": [{"class_name": "car", "bbox": [0.1, 0.2, 0.3, 0.4]}]},
    )
    next(w for w in app.checkbox if w.label == "Buka editor kotak manual").check().run()
    next(w for w in app.checkbox if w.label.startswith("Seluruh kotak dan kelas")).check().run()
    next(w for w in app.button if w.label == "Simpan label frame").click().run()
    assert not app.exception
    assert len(latest_labels(root / "outputs" / "dataset-labels")) == 1
    next(w for w in app.button if w.label == "Siapkan ZIP label YOLO").click().run()
    assert not app.exception
    assert app.session_state[f"dataset-zip-{result.run_id}"].startswith(b"PK")


def test_queue_cancel_button_does_not_start_inference(dashboard_project):
    from streamlit.testing.v1 import AppTest

    from vehicle_counter.jobs import JobQueue

    root, video = dashboard_project
    app = AppTest.from_file(str(root / "app.py"), default_timeout=20).run()
    next(w for w in app.selectbox if w.label == "Sumber video").select(video.name).run()
    next(w for w in app.button if w.label == "Uji singkat").click().run()
    job = app.session_state["pending_job"]
    next(w for w in app.button if w.label == "Batalkan pekerjaan").click().run()
    assert not app.exception
    queue = JobQueue(root / "outputs" / "jobs", root)
    assert queue.get(job)["status"] == "cancelled" and queue.claim() is None


def test_mpv_review_requires_local_weights_and_snapshots_threshold(dashboard_project):
    from streamlit.testing.v1 import AppTest

    from vehicle_counter.jobs import JobQueue

    root, video = dashboard_project
    app = AppTest.from_file(str(root / "app.py"), default_timeout=20).run()
    next(w for w in app.selectbox if w.label == "Sumber video").select(video.name).run()
    next(w for w in app.checkbox if w.label == "Periksa ulang kelas MPV/truk").check().run()
    assert not app.exception
    assert next(w for w in app.button if w.label == "Mulai analisis").disabled
    next(w for w in app.checkbox if w.label == "Fokus koreksi MPV/minivan").check().run()
    next(w for w in app.checkbox if w.label == "Kurangi kotak ganda berbeda kelas").check().run()
    (root / "models" / "mobilenet_v3_large-5c1a4163.pth").write_bytes(b"fixture, no inference")
    next(w for w in app.number_input if w.label == "Ambang bukti kendaraan ringan").set_value(
        0.30
    ).run()
    assert not next(w for w in app.button if w.label == "Mulai analisis").disabled
    next(w for w in app.button if w.label == "Uji singkat").click().run()
    job = JobQueue(root / "outputs" / "jobs", root).get(app.session_state["pending_job"])
    config = job["payload"]["config"]
    assert config["truck_review_light_threshold"] == 0.30
    assert config["truck_review_minivan_only"] is True
    assert config["detector_deduplicate_classes"] is True
    assert Path(config["truck_review_weights"]).name == "mobilenet_v3_large-5c1a4163.pth"


@pytest.mark.parametrize("dense", [False, True])
def test_local_recording_profile_snapshots_the_tuned_settings(
    dashboard_project, monkeypatch, dense
):
    from streamlit.testing.v1 import AppTest

    from vehicle_counter.jobs import JobQueue

    root, video = dashboard_project
    monkeypatch.chdir(root)
    project = Path(__file__).resolve().parents[1]
    (root / "configs" / "experiments").mkdir()
    shutil.copyfile(
        project / "configs" / "experiments" / "local-20261119-mpv.yaml",
        root / "configs" / "experiments" / "local-20261119-mpv.yaml",
    )
    shutil.copyfile(
        project / "configs" / "bytetrack-dense.yaml", root / "configs" / "bytetrack-dense.yaml"
    )
    for name in ("yolo26n.pt", "mobilenet_v3_large-5c1a4163.pth"):
        (root / "models" / name).write_bytes(b"fixture, no model inference")
    video = video.rename(video.with_name("local-20261119-VID_20261119_091928-cfr.mp4"))
    app = AppTest.from_file(str(root / "app.py"), default_timeout=20).run()
    next(w for w in app.selectbox if w.label == "Sumber video").select(video.name).run()
    next(w for w in app.radio if w.label == "Profil analisis").set_value("Pemulihan crossing").run()
    if dense:
        label = "ByteTrack untuk kendaraan rapat (eksperimen)"
        option = next(w for w in app.checkbox if w.label == label)
        option.check().run()
    assert not app.exception
    next(w for w in app.button if w.label == "Uji singkat").click().run()
    job = JobQueue(root / "outputs" / "jobs", root).get(app.session_state["pending_job"])
    config = job["payload"]["config"]
    assert config["guard_dense_crossings"] is dense
    expected_tracker = "bytetrack-dense.yaml" if dense else "bytetrack.yaml"
    assert Path(config["tracker_config"]).name == expected_tracker
    assert Path(config["model_weights"]).name == "yolo26n.pt"
    assert config["detector_deduplicate_classes"] is True
    assert config["truck_review_minivan_only"] is True
    assert config["truck_review_light_threshold"] == 0.30
    assert config["counting_line"]["p1"] == [0.02, 0.80]
    assert config["counting_line"]["p2"] == [0.98, 0.80]
    assert len(config["traffic_zones"]) == 2
    assert config["heatmap_enabled"] is True


def test_fine_tuned_model_is_optional_and_warns_about_experimental_status(dashboard_project):
    from streamlit.testing.v1 import AppTest

    from vehicle_counter.jobs import JobQueue

    root, video = dashboard_project
    (root / "models" / "yolo26n-local-ft.pt").write_bytes(b"fixture, no inference")
    app = AppTest.from_file(str(root / "app.py"), default_timeout=20).run()
    next(w for w in app.selectbox if w.label == "Sumber video").select(video.name).run()
    model = next(w for w in app.selectbox if w.label == "Model deteksi")
    assert model.value == "Sesuai profil"
    model.select("YOLO26n — fine-tuning lokal (eksperimen)").run()
    assert not app.exception
    assert any("belum mengungguli pretrained" in w.value for w in app.warning)
    assert (
        "YOLO26n pretrained vs fine-tuning lokal"
        in next(w for w in app.radio if w.label == "Yang dibandingkan").options
    )
    next(w for w in app.button if w.label == "Uji singkat").click().run()
    job = JobQueue(root / "outputs" / "jobs", root).get(app.session_state["pending_job"])
    assert Path(job["payload"]["config"]["model_weights"]).name == "yolo26n-local-ft.pt"
