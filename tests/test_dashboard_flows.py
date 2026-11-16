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


def test_numeric_line_survives_switch_back_to_editor(dashboard_project, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from vehicle_counter import line_editor, pipeline

    root, video = dashboard_project
    previous_drag = {
        "p1": [0.2, 0.6],
        "p2": [0.8, 0.6],
        "roi": [0.1, 0.4, 0.9, 0.9],
        "edit_id": "first",
    }
    monkeypatch.setattr(line_editor, "render_editor", lambda *args, **kwargs: previous_drag)
    monkeypatch.setattr(pipeline, "process_video", fixture_process)
    app = AppTest.from_file(str(root / "app.py"), default_timeout=20).run()
    next(w for w in app.selectbox if w.label == "Sumber video").select(video.name).run()
    next(w for w in app.radio if w.label == "Pengaturan garis").set_value("Koordinat angka").run()
    next(w for w in app.slider if w.label == "P1 · X").set_value(0.35).run()
    next(w for w in app.radio if w.label == "Pengaturan garis").set_value("Geser di gambar").run()
    next(w for w in app.button if w.label == "Uji singkat").click().run()
    finish_app_job(app, root)
    assert not app.exception
    result = app.session_state["result"][0]
    info = json.loads(Path(result.artifact_paths["run_json"]).read_text())
    assert info["config"]["counting_line"]["p1"] == [0.35, 0.6]
    monkeypatch.setattr(
        line_editor, "render_editor", lambda *args, **kwargs: {**previous_drag, "edit_id": "second"}
    )
    app.run()
    next(w for w in app.button if w.label == "Uji singkat").click().run()
    finish_app_job(app, root)
    info = json.loads(Path(app.session_state["result"][0].artifact_paths["run_json"]).read_text())
    assert info["config"]["counting_line"]["p1"] == [0.2, 0.6]


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


def test_model_controls_and_comparison_keep_camera_settings(dashboard_project, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from vehicle_counter import experiments, pipeline

    root, video = dashboard_project
    monkeypatch.setattr(pipeline, "process_video", fixture_process)
    monkeypatch.setattr(experiments, "process_video", fixture_process)
    app = AppTest.from_file(str(root / "app.py"), default_timeout=25).run()
    next(w for w in app.selectbox if w.label == "Sumber video").select(video.name).run()
    next(w for w in app.selectbox if w.label == "Model deteksi").select(
        "YOLO11s — pembanding"
    ).run()
    next(w for w in app.selectbox if w.label == "Ukuran masukan YOLO").select(960).run()
    next(w for w in app.number_input if w.label == "Confidence minimum detektor").set_value(
        0.12
    ).run()
    next(w for w in app.radio if w.label == "Yang dibandingkan").set_value(
        "YOLO11n vs YOLO11s"
    ).run()
    next(w for w in app.button if w.label == "Bandingkan metode").click().run()
    finish_app_job(app, root)
    assert not app.exception
    batch = app.session_state["comparison"]
    assert len(batch["rows"]) == 2
    assert {row["model"] for row in batch["rows"]} == {"yolo11n.pt", "yolo11s.pt"}
    assert all(row["imgsz"] == 960 and row["detector_conf"] == 0.12 for row in batch["rows"])
    manifests = [
        json.loads(Path(r["artifact_paths"]["run_json"]).read_text()) for r in batch["results"]
    ]
    assert manifests[0]["config"]["counting_line"] == manifests[1]["config"]["counting_line"]
    assert manifests[0]["config"]["congestion"] == manifests[1]["config"]["congestion"]


def test_zones_heatmap_quality_and_manual_dataset_in_ui(dashboard_project, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from vehicle_counter import dataset_dashboard
    from vehicle_counter.training_labels import latest_labels

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
