"""Responsive Streamlit polling; all inference stays in the shared job worker."""

from pathlib import Path

import streamlit as st

from vehicle_counter.contracts import CountSummary, RunResult
from vehicle_counter.jobs import ACTIVE, JobQueue
from vehicle_counter.ui.layout import QUEUE, RESULTS, WORKSPACE_KEY, open_view

LABELS = {
    "preparing": "Menyalin video",
    "queued": "Menunggu giliran",
    "running": "Sedang dianalisis",
    "cancel_requested": "Menunggu pembatalan",
    "completed": "Selesai",
    "cancelled": "Dibatalkan",
    "failed": "Gagal",
}
STATUS_COLORS = {"completed": "green", "running": "teal", "failed": "red"}


def restore_job(job):
    result = job["result"]
    if not result:
        return
    if result["kind"] == "comparison":
        st.session_state.comparison = result["batch"]
        st.session_state.pop("result", None)
    else:
        payload = dict(result["run"])
        payload["summary"] = CountSummary(**payload["summary"])
        st.session_state.result = (
            RunResult(**payload),
            result["source_label"],
            result["mode"],
            "Antrean",
        )


def show_jobs(root: str, project_root: str, owner: str):
    queue = JobQueue(Path(root), Path(project_root))
    active = any(job["status"] in ACTIVE for job in queue.owned(owner))
    panel = st.fragment(run_every=1 if active else None)(_render_jobs)
    panel(root, project_root, owner)


def _render_jobs(root: str, project_root: str, owner: str):
    queue = JobQueue(Path(root), Path(project_root))
    jobs = queue.owned(owner)
    if not jobs:
        return
    if any(job["status"] in ACTIVE for job in jobs):
        queue.ensure_worker()
    st.subheader("Antrean analisis sesi ini")
    st.caption(
        "Satu pekerjaan inferensi berjalan pada satu waktu. Pengaturan "
        "pekerjaan yang sudah dikirim tetap tersimpan."
    )
    for job in jobs[:5]:
        with st.container(border=True):
            status = job["status"]
            name, badge = st.columns([3, 2])
            name.text(job["payload"]["source_label"])
            badge.badge(LABELS[status], color=STATUS_COLORS.get(status, "gray"))
            if job["status"] in ("queued", "preparing"):
                st.info(
                    f"Posisi antrean: {job['position']}. "
                    "Halaman boleh tetap dibuka sambil menunggu."
                )
            progress = job["progress"] or {}
            if job["status"] in ("running", "cancel_requested"):
                current, total = progress.get("processed_frames", 0), progress.get("total_frames")
                st.progress(
                    min(0.99, current / total) if total else 0.0,
                    text=f"Frame {current}" + (f" / {total}" if total else " · menyiapkan model"),
                )
                if "comparison_total" in progress:
                    st.caption(
                        f"Percobaan {progress['comparison_index']}/{progress['comparison_total']}"
                    )
                if "summary" in progress:
                    st.caption(f"Crossing sementara: {progress['summary']['total']}")
                preview = queue.root / job["job_id"] / "preview.jpg"
                if preview.is_file():
                    with st.expander("Preview proses terakhir"):
                        try:
                            st.image(preview.read_bytes(), width="stretch")
                        except OSError:
                            pass
            if job["status"] in ACTIVE:
                if st.button(
                    "Batalkan pekerjaan",
                    key=f"cancel-{job['job_id']}",
                    disabled=job["status"] == "cancel_requested",
                ):
                    queue.cancel(job["job_id"], owner=owner)
                    st.rerun()
                st.caption(
                    "Pembatalan diperiksa di sela frame; operasi model yang sedang "
                    "berjalan diselesaikan dahulu."
                )
            elif job["result"]:
                if (
                    st.session_state.get("pending_job") == job["job_id"]
                    and st.session_state.get("loaded_job") != job["job_id"]
                ):
                    restore_job(job)
                    st.session_state.loaded_job = job["job_id"]
                    if st.session_state.get(WORKSPACE_KEY) == QUEUE:
                        open_view(RESULTS)
                    st.rerun()
                if st.button("Buka hasil pekerjaan", key=f"open-job-{job['job_id']}"):
                    restore_job(job)
                    open_view(RESULTS)
            if job["error"]:
                st.warning(job["error"])
