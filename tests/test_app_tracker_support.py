"""A hot-reloaded UI must not offer a tracker rejected by its cached backend."""

import shutil
from pathlib import Path

import numpy as np
import pytest


def test_deepsort_is_blocked_with_restart_guidance_for_a_cached_old_backend(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    pytest.importorskip("cv2")
    pytest.importorskip("imageio_ffmpeg")
    pytest.importorskip("deep_sort_realtime")
    from streamlit.testing.v1 import AppTest

    from vehicle_counter import tracking
    from vehicle_counter.contracts import VideoMetadata
    from vehicle_counter.video import AnnotatedVideoWriter

    project = Path(__file__).resolve().parents[1]
    (tmp_path / "configs").mkdir()
    (tmp_path / "data" / "videos").mkdir(parents=True)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "yolo11n.pt").write_bytes(b"fixture, not model inference")
    for name in ("default.yaml", "deepsort.yaml"):
        shutil.copyfile(project / "configs" / name, tmp_path / "configs" / name)
    shutil.copyfile(project / "app.py", tmp_path / "app.py")
    video = tmp_path / "data" / "videos" / "fixture.mp4"
    with AnnotatedVideoWriter(video, VideoMetadata(video.name, 160, 120, 10, 3, 0.3)) as writer:
        for _ in range(3):
            writer.write(np.zeros((120, 160, 3), np.uint8))

    current_loader = tracking.load_tracker_settings

    def cached_old_loader(value):
        # The real failure: Streamlit reruns app.py but keeps the imported loader.
        raise ValueError("Baseline ini memerlukan konfigurasi tracker_type: bytetrack.")

    monkeypatch.setattr(tracking, "load_tracker_settings", cached_old_loader)
    app = AppTest.from_file(str(tmp_path / "app.py"), default_timeout=15).run()
    next(w for w in app.selectbox if w.label == "Sumber video").select("fixture.mp4").run()
    assert not next(w for w in app.button if w.label == "Mulai analisis").disabled
    next(w for w in app.selectbox if w.label == "Tracker").select("DeepSORT (eksperimen)").run()
    assert not app.exception
    assert next(w for w in app.button if w.label == "Mulai analisis").disabled
    assert any("Mulai ulang aplikasi" in alert.value for alert in app.error)

    # A fresh backend accepts the very same YAML; the option becomes usable.
    monkeypatch.setattr(tracking, "load_tracker_settings", current_loader)
    app.run()
    assert not app.exception
    assert not next(w for w in app.button if w.label == "Mulai analisis").disabled
