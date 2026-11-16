"""Persistent FIFO queue with one manager and one inference subprocess at a time."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import psutil

from vehicle_counter.config import config_from_dict, config_to_dict

ACTIVE = ("preparing", "queued", "running", "cancel_requested")
TERMINAL = ("completed", "failed", "cancelled")


def _identity(value):
    if (
        not isinstance(value, str)
        or len(value) != 32
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("ID pekerjaan tidak valid.")
    return value


def _process_alive(pid, created, *needles):
    if not pid or created is None:
        return False
    try:
        process = psutil.Process(pid)
        command = process.cmdline()
        return (
            process.is_running()
            and process.status() != psutil.STATUS_ZOMBIE
            and abs(process.create_time() - created) < 0.01
            and all(str(needle) in command for needle in needles)
        )
    except (psutil.Error, OSError):
        return False


def _spawn(command, *, cwd, log):
    with Path(log).open("ab") as handle:
        return subprocess.Popen(
            command,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=handle,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            start_new_session=os.name != "nt",
        )


@contextmanager
def worker_lock(root):
    """OS releases the advisory lock even when the manager process crashes."""
    path = Path(root) / "worker.lock"
    with path.open("a+b") as handle:
        handle.seek(0)
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        acquired = False
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError:
            pass
        try:
            yield acquired
        finally:
            if acquired:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class JobQueue:
    def __init__(self, root: Path, project_root: Path | None = None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.project_root = Path(project_root or Path.cwd()).resolve()
        self.db = self.root / "jobs.sqlite3"
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("""CREATE TABLE IF NOT EXISTS jobs (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT UNIQUE NOT NULL,
                owner TEXT NOT NULL, status TEXT NOT NULL, created REAL NOT NULL,
                updated REAL NOT NULL, payload TEXT NOT NULL, progress TEXT,
                result TEXT, error TEXT, pid INTEGER, process_created REAL)""")

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.db, timeout=15)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _set(self, job_id, **fields):
        fields["updated"] = time.time()
        with self._connect() as connection:
            connection.execute(
                "UPDATE jobs SET " + ",".join(f"{key}=?" for key in fields) + " WHERE job_id=?",
                [*fields.values(), _identity(job_id)],
            )

    def enqueue(self, source, config, *, owner, source_label, max_seconds=None, comparison=None):
        if not isinstance(owner, str) or not owner.strip():
            raise ValueError("Pemilik sesi tidak tersedia.")
        source = Path(source).resolve()
        size = source.stat().st_size
        if size > 100 * 1024 * 1024:
            raise ValueError(
                "Antrean menerima video maksimal 100 MB. Potong video terlebih dahulu."
            )
        if shutil.disk_usage(self.root).free < size + 128 * 1024 * 1024:
            raise ValueError("Penyimpanan tidak cukup untuk menyalin video ke antrean.")
        job_id = uuid4().hex
        payload = {
            "config": config_to_dict(config),
            "source_label": source_label,
            "max_seconds": max_seconds,
            "comparison": comparison,
            "project_root": str(self.project_root),
        }
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if (
                connection.execute(
                    "SELECT count(*) FROM jobs WHERE status IN "
                    "('preparing','queued','running','cancel_requested')"
                ).fetchone()[0]
                >= 4
            ):
                raise ValueError(
                    "Antrean penuh (maksimal empat pekerjaan aktif). Coba lagi setelah"
                    " ada yang selesai."
                )
            if connection.execute(
                (
                    "SELECT 1 FROM jobs WHERE owner=? AND status IN "
                    "('preparing','queued','running','cancel_requested')"
                ),
                (owner,),
            ).fetchone():
                raise ValueError(
                    "Sesi ini masih memiliki pekerjaan aktif. Tunggu atau batalkan dahulu."
                )
            connection.execute(
                "INSERT INTO jobs(job_id,owner,status,created,updated,payload,pid,process_created) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (
                    job_id,
                    owner,
                    "preparing",
                    time.time(),
                    time.time(),
                    json.dumps(payload),
                    os.getpid(),
                    psutil.Process().create_time(),
                ),
            )
        directory = self.root / job_id
        try:
            directory.mkdir()
            shutil.copyfile(source, directory / "source.mp4")
            with self._connect() as connection:
                connection.execute(
                    (
                        "UPDATE jobs SET status='queued',updated=?,pid=NULL,process_created=NULL "
                        "WHERE job_id=? AND status='preparing'"
                    ),
                    (time.time(), job_id),
                )
        except BaseException as error:
            self._set(job_id, status="failed", error=str(error))
            raise
        return job_id

    def get(self, job_id, *, owner=None):
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id=?", (_identity(job_id),)
            ).fetchone()
            if row is None or owner is not None and row["owner"] != owner:
                raise ValueError("Pekerjaan tidak ditemukan untuk sesi ini.")
            row = dict(row)
            row["position"] = connection.execute(
                (
                    "SELECT count(*) FROM jobs WHERE seq<=? AND status IN "
                    "('preparing','queued','running','cancel_requested')"
                ),
                (row["seq"],),
            ).fetchone()[0]
        for key in ("payload", "progress", "result"):
            row[key] = json.loads(row[key]) if row[key] else None
        row.pop("owner")
        return row

    def owned(self, owner):
        with self._connect() as connection:
            ids = connection.execute(
                "SELECT job_id FROM jobs WHERE owner=? ORDER BY seq DESC LIMIT 12", (owner,)
            ).fetchall()
        return [self.get(row[0], owner=owner) for row in ids]

    def cancel(self, job_id, *, owner):
        self.get(job_id, owner=owner)
        with self._connect() as connection:
            connection.execute(
                (
                    "UPDATE jobs SET status=CASE WHEN status IN ('preparing','queued')"
                    " THEN 'cancelled' ELSE 'cancel_requested' END,updated=? WHERE "
                    "job_id=? AND owner=? AND status IN "
                    "('preparing','queued','running')"
                ),
                (time.time(), job_id, owner),
            )

    def cancelled(self, job_id):
        return self.get(job_id)["status"] in ("cancelled", "cancel_requested")

    def claim(self):
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute(
                "SELECT 1 FROM jobs WHERE status IN ('running','cancel_requested')"
            ).fetchone():
                return None
            row = connection.execute(
                "SELECT job_id,status FROM jobs WHERE status IN "
                "('preparing','queued') ORDER BY seq LIMIT 1"
            ).fetchone()
            if not row or row["status"] == "preparing":
                return None
            connection.execute(
                "UPDATE jobs SET status='running',updated=? WHERE job_id=?",
                (time.time(), row["job_id"]),
            )
            return row["job_id"]

    def execute(self, job_id, *, processor=None):
        from vehicle_counter.execution import AnalysisCancelled
        from vehicle_counter.pipeline import process_video

        job_id = _identity(job_id)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if not row or row["status"] not in ("running", "cancel_requested"):
                return
            connection.execute(
                "UPDATE jobs SET pid=?,process_created=? WHERE job_id=?",
                (os.getpid(), psutil.Process().create_time(), job_id),
            )
        job = self.get(job_id)
        payload = job["payload"]
        source = self.root / job_id / "source.mp4"
        processor = processor or process_video

        def progress(step):
            values = {
                **(self.get(job_id)["progress"] or {}),
                **{
                    k: getattr(step, k)
                    for k in ("run_id", "processed_frames", "total_frames", "elapsed_seconds")
                },
            }
            values["summary"] = asdict(step.summary)
            self._set(job_id, progress=json.dumps(values))
            if step.preview_frame_bgr is not None:
                import cv2

                image = step.preview_frame_bgr
                scale = min(1, 640 / max(image.shape[:2]))
                image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                ok, data = cv2.imencode(".jpg", image)
                if ok:
                    temporary = self.root / job_id / "preview.tmp"
                    temporary.write_bytes(data.tobytes())
                    temporary.replace(self.root / job_id / "preview.jpg")

        def cancel():
            return self.cancelled(job_id)

        try:
            config = config_from_dict(payload["config"])
            if payload["comparison"]:
                from vehicle_counter.experiments import run_comparison

                def on_run(index, total, choice):
                    self._set(
                        job_id,
                        progress=json.dumps(
                            {
                                "comparison_index": index + 1,
                                "comparison_total": total,
                                "model": Path(choice.model_weights).name,
                                "mode": choice.preprocess.mode,
                            }
                        ),
                    )

                batch = run_comparison(
                    source,
                    config,
                    self.root / job_id / "comparison",
                    source_label=payload["source_label"],
                    max_seconds=payload["max_seconds"],
                    on_run=on_run,
                    processor=processor,
                    should_cancel=cancel,
                    on_progress=progress,
                    **payload["comparison"],
                )
                result = {"kind": "comparison", "batch": batch}
                status = (
                    "completed"
                    if batch["status"] == "completed"
                    else ("cancelled" if batch["status"] == "cancelled" else "failed")
                )
            else:
                run = processor(
                    source,
                    config,
                    self.root / job_id / "runs",
                    progress,
                    max_seconds=payload["max_seconds"],
                    source_label=payload["source_label"],
                    should_cancel=cancel,
                )
                result = {
                    "kind": "analysis",
                    "run": asdict(run),
                    "source_label": payload["source_label"],
                    "mode": config.preprocess.mode,
                }
                status = run.status
            self._set(job_id, status=status, result=json.dumps(result))
        except BaseException as error:
            self._set(
                job_id,
                status="cancelled" if isinstance(error, AnalysisCancelled) else "failed",
                error=str(error),
            )
            if not isinstance(error, Exception):
                raise

    def ensure_worker(self):
        path = self.root / "worker.json"
        try:
            info = json.loads(path.read_text())
            if _process_alive(
                info["pid"], info["created"], "vehicle_counter.jobs", "worker", str(self.root)
            ):
                return
        except (OSError, ValueError, KeyError):
            pass
        _spawn(
            [
                sys.executable,
                "-m",
                "vehicle_counter.jobs",
                "worker",
                "--root",
                str(self.root),
                "--project-root",
                str(self.project_root),
            ],
            cwd=self.project_root,
            log=self.root / "worker.log",
        )

    def worker(self, *, once=False):
        from vehicle_counter.exporting import write_json

        with worker_lock(self.root) as acquired:
            if not acquired:
                return
            write_json(
                self.root / "worker.json",
                {"pid": os.getpid(), "created": psutil.Process().create_time()},
            )
            # A surviving executor owns its active slot even if its manager died.
            while True:
                with self._connect() as connection:
                    preparing = connection.execute(
                        "SELECT job_id,pid,process_created FROM jobs WHERE status='preparing'"
                    ).fetchall()
                    active = connection.execute(
                        "SELECT job_id,pid,process_created,status FROM jobs WHERE status "
                        "IN ('running','cancel_requested')"
                    ).fetchall()
                for row in preparing:
                    if not _process_alive(row["pid"], row["process_created"]):
                        self._set(
                            row["job_id"],
                            status="failed",
                            error="Penyalinan video terhenti. Kirim ulang sumber.",
                        )
                for row in active:
                    if not _process_alive(
                        row["pid"],
                        row["process_created"],
                        "vehicle_counter.jobs",
                        "execute",
                        row["job_id"],
                    ):
                        self._set(
                            row["job_id"],
                            status="cancelled" if row["status"] == "cancel_requested" else "failed",
                            error="Proses analisis terhenti. Hasil belum lengkap; jalankan ulang.",
                        )
                job_id = self.claim()
                if job_id:
                    directory = self.root / job_id
                    payload = self.get(job_id)["payload"]
                    child = _spawn(
                        [
                            sys.executable,
                            "-m",
                            "vehicle_counter.jobs",
                            "execute",
                            "--root",
                            str(self.root),
                            "--job",
                            job_id,
                            "--project-root",
                            payload["project_root"],
                        ],
                        cwd=payload["project_root"],
                        log=directory / "process.log",
                    )
                    while child.poll() is None:
                        time.sleep(0.25)
                    if self.get(job_id)["status"] in ("running", "cancel_requested"):
                        self._set(
                            job_id,
                            status="failed",
                            error=f"Proses berhenti dengan kode {child.returncode}; "
                            "lihat process.log.",
                        )
                    if once:
                        return
                elif once:
                    return
                time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(description="Antrean lokal Vehicle Tracking")
    parser.add_argument("mode", choices=["worker", "execute"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--job")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    queue = JobQueue(args.root, args.project_root)
    if args.mode == "worker":
        queue.worker(once=args.once)
    else:
        queue.execute(_identity(args.job))


if __name__ == "__main__":
    main()
