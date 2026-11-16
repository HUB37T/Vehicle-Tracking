import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vehicle_counter.contracts import RunConfig, TrackObservation, VideoMetadata
from vehicle_counter.jobs import JobQueue, worker_lock
from vehicle_counter.pipeline import process_video
from vehicle_counter.video import AnnotatedVideoWriter


class FixtureTracker:
    metadata = {"is_model_inference": False, "tracker_name": "Fixture"}
    warnings = []

    def __init__(self, *args):
        pass

    def track_frame(self, frame_bgr, *, frame_index, timestamp_seconds):
        y = 30 + frame_index * 6
        return [
            TrackObservation(frame_index, timestamp_seconds, 1, "car", 0.9, (60, y - 15, 90, y))
        ]


@pytest.fixture
def queued_video(tmp_path):
    source = tmp_path / "source.mp4"
    with AnnotatedVideoWriter(source, VideoMetadata(source.name, 160, 120, 10, 12, 1.2)) as writer:
        for _ in range(12):
            writer.write(np.zeros((120, 160, 3), np.uint8))
    return source


def fixture_process(*args, **kwargs):
    return process_video(*args, **kwargs, tracker_factory=FixtureTracker)


def test_fifo_freezes_source_and_blocks_parallel_claims(queued_video, tmp_path):
    q = JobQueue(tmp_path / "queue")
    one = q.enqueue(queued_video, RunConfig(), owner="a", source_label="Film A.mp4")
    two = q.enqueue(queued_video, RunConfig(), owner="b", source_label="Film B.mp4")
    assert q.get(two, owner="b")["position"] == 2
    assert (q.root / one / "source.mp4").read_bytes() == queued_video.read_bytes()
    assert q.claim() == one and q.claim() is None
    q.execute(one, processor=fixture_process)
    assert q.get(one)["status"] == "completed"
    assert q.claim() == two
    q.execute(two, processor=fixture_process)
    assert q.get(two)["result"]["run"]["summary"]["total"] == 1
    info = json.loads(Path(q.get(two)["result"]["run"]["artifact_paths"]["run_json"]).read_text())
    assert info["video_metadata"]["source_name"] == "Film B.mp4"


def test_cancel_is_owner_scoped_and_keeps_partial_result(queued_video, tmp_path):
    q = JobQueue(tmp_path / "queue")
    job = q.enqueue(queued_video, RunConfig(), owner="a", source_label="fixture.mp4")
    with pytest.raises(ValueError, match="sesi ini"):
        q.cancel(job, owner="b")
    with pytest.raises(ValueError, match="aktif"):
        q.enqueue(queued_video, RunConfig(), owner="a", source_label="fixture.mp4")
    q.cancel(job, owner="a")
    assert q.get(job)["status"] == "cancelled" and q.claim() is None
    running = q.enqueue(queued_video, RunConfig(), owner="a", source_label="fixture.mp4")
    q.claim()

    def process(source, config, output, on_progress=None, **kwargs):
        def update(step):
            on_progress(step)
            q.cancel(running, owner="a")

        return fixture_process(source, config, output, update, **kwargs)

    q.execute(running, processor=process)
    job = q.get(running)
    assert job["status"] == "cancelled"
    info = json.loads(Path(job["result"]["run"]["artifact_paths"]["run_json"]).read_text())
    assert info["status"] == "cancelled" and info["processed_frames"] == 1
    assert Path(job["result"]["run"]["artifact_paths"]["events_csv"]).is_file()
    assert "annotated_video" not in job["result"]["run"]["artifact_paths"]


def test_comparison_uses_queue_slot_and_can_cancel_between_runs(queued_video, tmp_path):
    q = JobQueue(tmp_path / "queue")
    job = q.enqueue(
        queued_video,
        RunConfig(),
        owner="a",
        source_label="fixture.mp4",
        comparison={"modes": ["none", "clahe"], "trackers": ["bytetrack.yaml"]},
    )
    q.claim()

    def process(*args, **kwargs):
        result = fixture_process(*args, **kwargs)
        q.cancel(job, owner="a")
        return result

    q.execute(job, processor=process)
    result = q.get(job)
    assert result["status"] == "cancelled"
    assert len(result["result"]["batch"]["results"]) == 1


def test_worker_lock_excludes_second_process_and_worker_failure_is_recorded(tmp_path):
    q = JobQueue(tmp_path / "queue")
    script = (
        "from pathlib import Path\nfrom vehicle_counter.jobs import "
        "worker_lock\nimport sys\nwith worker_lock(Path(sys.argv[1])) as ok:"
        "\n print(ok)"
    )
    with worker_lock(q.root) as acquired:
        assert acquired
        child = subprocess.run(
            [sys.executable, "-c", script, str(q.root)], capture_output=True, text=True, timeout=20
        )
        assert child.returncode == 0 and child.stdout.strip() == "False"
    invalid = tmp_path / "invalid.mp4"
    invalid.write_bytes(b"fixture invalid video, no model inference")
    job = q.enqueue(
        invalid, replace(RunConfig(), quality_check=False), owner="a", source_label="invalid.mp4"
    )
    child = subprocess.run(
        [sys.executable, "-m", "vehicle_counter.jobs", "worker", "--root", str(q.root), "--once"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert child.returncode == 0, child.stderr
    assert q.get(job)["status"] == "failed" and q.get(job)["error"]


def test_restart_marks_orphan_failed_instead_of_waiting_forever(queued_video, tmp_path):
    q = JobQueue(tmp_path / "queue")
    job = q.enqueue(queued_video, RunConfig(), owner="a", source_label="fixture.mp4")
    assert q.claim() == job
    q.worker(once=True)
    assert q.get(job)["status"] == "failed"


def test_orphaned_source_copy_does_not_block_fifo_and_capacity_releases(queued_video, tmp_path):
    q = JobQueue(tmp_path / "queue")
    jobs = [
        q.enqueue(queued_video, RunConfig(), owner=str(i), source_label="fixture.mp4")
        for i in range(4)
    ]
    with pytest.raises(ValueError, match="Antrean penuh"):
        q.enqueue(queued_video, RunConfig(), owner="extra", source_label="fixture.mp4")
    q.cancel(jobs[-1], owner="3")
    extra = q.enqueue(queued_video, RunConfig(), owner="extra", source_label="fixture.mp4")
    q._set(jobs[0], status="preparing", pid=999999999, process_created=1)
    # Recovering a dead uploader changes only its orphaned preparation row.
    q._set(jobs[1], status="cancelled")
    q._set(jobs[2], status="cancelled")
    q._set(extra, status="cancelled")
    q.worker(once=True)
    assert q.get(jobs[0])["status"] == "failed"


def test_surviving_executor_keeps_slot_when_manager_restarts(queued_video, tmp_path):
    import psutil

    from vehicle_counter.jobs import _spawn

    q = JobQueue(tmp_path / "queue")
    job = q.enqueue(queued_video, RunConfig(), owner="a", source_label="fixture.mp4")
    q.claim()
    child = _spawn(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(15)",
            "vehicle_counter.jobs",
            "execute",
            job,
        ],
        cwd=Path.cwd(),
        log=q.root / "fixture.log",
    )
    try:
        q._set(job, pid=child.pid, process_created=psutil.Process(child.pid).create_time())
        q.worker(once=True)
        assert q.get(job)["status"] == "running" and q.claim() is None
    finally:
        child.terminate()
        child.wait(timeout=10)
    q.worker(once=True)
    assert q.get(job)["status"] == "failed"
