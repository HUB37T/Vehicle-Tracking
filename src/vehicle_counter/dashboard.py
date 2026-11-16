"""Result exploration for the Streamlit interface, using exported run artifacts."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import streamlit as st

from vehicle_counter.contracts import DIRECTIONS, VEHICLE_CLASSES
from vehicle_counter.evaluation import compare_runs
from vehicle_counter.review import REVIEW_STATUSES, load_reviews, reviews_csv, save_review
from vehicle_counter.traffic import TRAFFIC_LABELS


def show_diagnostics(result, manifest, frame_loader):
    diagnostics = manifest.get("diagnostics")
    if diagnostics is None:
        st.info("Analisis ulang video untuk mendapatkan penjelasan kondisi tracking dan garis.")
        return
    for message in diagnostics["messages"]:
        st.info(message)
    st.caption(
        f"{diagnostics['detection_instances']} kemunculan kotak deteksi dan "
        f"{diagnostics['track_observation_instances']} observasi track selama analisis. "
        "Satu kendaraan dapat muncul pada banyak frame."
    )
    st.write(
        {
            "Observasi sisi garis": diagnostics["observations_by_side"],
            "Jeda yang mereset riwayat": diagnostics["gap_resets"],
        }
    )
    examples = diagnostics["examples"]
    if examples:
        selected = st.selectbox(
            "Contoh ID yang sempat hilang",
            range(len(examples)),
            format_func=lambda i: (
                f"ID {examples[i]['track_id']} - frame {examples[i]['frame_index']}"
            ),
            key=f"diagnostic-{result.run_id}",
        )
        example = examples[selected]
        st.caption(f"Jeda {example['gap_seconds']:.2f} detik antara kedua observasi ini.")
        for column, frame in zip(
            st.columns(2), (example["previous_frame_index"], example["frame_index"]), strict=True
        ):
            column.image(
                frame_loader(result.artifact_paths["annotated_video"], int(frame)),
                channels="BGR",
                caption=f"Frame {frame}",
            )


def show_timeline(result):
    path = result.artifact_paths.get("timeline_csv")
    if not path:
        st.info("Analisis ulang video untuk menampilkan grafik per interval.")
        return
    rows = pd.read_csv(path)
    if rows.empty:
        st.info("Belum ada interval video yang diproses.")
        return
    grouped = rows.groupby(["start_seconds", "direction"], as_index=False)["count"].sum()
    st.caption("Jumlah crossing per interval 5 detik, berdasarkan event otomatis.")
    pivot = grouped.pivot(index="start_seconds", columns="direction", values="count").fillna(0)
    st.bar_chart(pivot, x_label="Waktu video (detik)", y_label="Crossing", stack=False)
    st.dataframe(rows, hide_index=True)


def show_traffic(result, manifest, frame_loader=None):
    info = manifest.get("traffic")
    if info is None:
        st.info("Aktifkan indikator macet dan pilih area jalan sebelum memulai analisis.")
        return
    st.caption(
        "Indikator untuk kamera diam berdasarkan track di area jalan yang dipilih. "
        "Gerak memakai perpindahan relatif terhadap ukuran kotak kendaraan. "
        "Ambang perlu diuji untuk sudut kamera dan kondisi jalan ini."
    )
    st.metric("Status pada sampel terakhir", TRAFFIC_LABELS[info["last_status"]])
    durations = info["durations_seconds"]
    st.metric("Durasi teramati dengan indikasi macet", f"{durations['congested']:.1f} detik")
    episodes = info.get("episodes", [])
    if episodes:
        st.subheader("Waktu indikasi macet")
        st.caption("Rentang dimulai setelah syarat durasi macet terpenuhi. Waktu mengikuti video.")
        st.dataframe(pd.DataFrame(episodes), hide_index=True)
        selected = st.selectbox(
            "Lihat awal kejadian macet",
            range(len(episodes)),
            format_func=lambda i: (
                f"{episodes[i]['start_seconds']:.1f}–{episodes[i]['end_seconds']:.1f} detik"
                + (" · masih teramati pada akhir klip" if episodes[i]["ongoing_at_end"] else "")
            ),
            key=f"traffic-episode-{result.run_id}",
        )
        episode = episodes[selected]
        if frame_loader:
            with st.expander("Periksa frame awal indikasi macet"):
                st.image(
                    frame_loader(
                        result.artifact_paths["annotated_video"], episode["start_frame_index"]
                    ),
                    channels="BGR",
                    width="stretch",
                )
        st.video(
            result.artifact_paths["annotated_video"],
            start_time=max(0, int(episode["start_seconds"]) - 1),
        )
    rows = pd.read_csv(result.artifact_paths["traffic_csv"])
    if not rows.empty:
        st.line_chart(
            rows.set_index("timestamp_seconds")[["vehicle_count", "motion_track_count"]].rename(
                columns={
                    "vehicle_count": "Kendaraan teramati",
                    "motion_track_count": "Riwayat gerak cukup",
                }
            ),
            x_label="Waktu video (detik)",
            y_label="Jumlah track",
        )
        st.line_chart(
            rows.set_index("timestamp_seconds")[["occupancy_fraction", "slow_fraction"]].rename(
                columns={
                    "occupancy_fraction": "Keterisian area",
                    "slow_fraction": "Bagian kendaraan lambat",
                }
            ),
            x_label="Waktu video (detik)",
            y_label="Fraksi 0-1",
        )
        selected = st.selectbox(
            "Periksa status lalu lintas",
            range(len(rows)),
            format_func=lambda i: (
                f"{rows.iloc[i]['timestamp_seconds']:.1f} dtk - "
                f"{TRAFFIC_LABELS[rows.iloc[i]['status']]}"
            ),
            key=f"traffic-sample-{result.run_id}",
        )
        row = rows.iloc[selected]
        st.write(row.reason)
        st.dataframe(rows, hide_index=True)
    with st.expander("Ambang indikator macet"):
        st.json(info["config"])


def show_review_form(result, manifest, selected_row, frame_loader):
    directory = Path(result.output_dir)
    with st.expander("Catat hasil pemeriksaan manual"):
        missed = st.checkbox(
            "Catat kendaraan yang terlewat",
            value=selected_row is None,
            key=f"missed-{result.run_id}",
        )
        if selected_row is None and not missed:
            st.info("Belum ada event otomatis untuk dipilih. Gunakan catatan kendaraan terlewat.")
            return
        suffix = f"{result.run_id}-{'missed' if missed else int(selected_row.track_id)}"
        frame = st.number_input(
            "Frame yang diperiksa",
            0,
            manifest["processed_frames"] - 1,
            0 if missed else int(selected_row.frame_index),
            key=f"review-frame-{suffix}",
        )
        st.image(
            frame_loader(result.artifact_paths["annotated_video"], int(frame)),
            channels="BGR",
            width="stretch",
        )
        with st.form(f"review-{suffix}"):
            status = (
                "terlewat"
                if missed
                else st.selectbox(
                    "Hasil pemeriksaan", [s for s in REVIEW_STATUSES if s != "terlewat"]
                )
            )
            corrected_class = st.selectbox(
                "Kelas menurut pemeriksa",
                VEHICLE_CLASSES,
                index=0 if missed else VEHICLE_CLASSES.index(selected_row.class_name),
            )
            direction = st.selectbox(
                "Arah menurut pemeriksa",
                DIRECTIONS,
                index=0 if missed else DIRECTIONS.index(selected_row.direction),
            )
            notes = st.text_area("Catatan pemeriksa", max_chars=2000)
            if st.form_submit_button("Simpan catatan"):
                save_review(
                    directory,
                    frame_index=int(frame),
                    track_id=None if missed else int(selected_row.track_id),
                    status=status,
                    corrected_class=corrected_class,
                    direction=direction,
                    notes=notes,
                )
                st.success("Catatan pemeriksaan tersimpan.")
        st.caption("Catatan pemeriksa tersimpan sebagai riwayat terpisah dari event otomatis.")
        records = load_reviews(directory)
        if records:
            st.dataframe(pd.DataFrame(records), hide_index=True)
            st.download_button(
                "Unduh catatan pemeriksa",
                reviews_csv(directory),
                "reviews.csv",
                "text/csv",
                on_click="ignore",
                key=f"reviews-download-{result.run_id}",
            )


def show_comparison(batch):
    st.subheader("Hasil perbandingan metode")
    st.caption("Seluruh run memakai video, garis, area pengamatan, dan batas waktu yang sama.")
    st.dataframe(pd.DataFrame(batch["rows"]), hide_index=True)
    directory = Path(batch["output_dir"])
    for name in ("comparison.csv", "comparison.json"):
        st.download_button(
            f"Unduh {name}",
            (directory / name).read_bytes(),
            name,
            on_click="ignore",
            key=f"comparison-{directory.name}-{name}",
        )
    results = batch["results"]
    complete = [Path(r["output_dir"]) for r in results if r["status"] == "completed"]
    with st.expander("Bandingkan dengan anotasi manual"):
        st.caption(
            "Unggah CSV crossing dan JSON pendamping dengan checksum video serta garis "
            "yang sesuai. Evaluasi identitas tetap memerlukan pemeriksaan video."
        )
        csv_file = st.file_uploader(
            "CSV anotasi manual", type="csv", key=f"gt-csv-{directory.name}"
        )
        json_file = st.file_uploader(
            "JSON metadata anotasi", type="json", key=f"gt-json-{directory.name}"
        )
        if st.button(
            "Evaluasi anotasi",
            disabled=not (csv_file and json_file and complete),
            key=f"evaluate-{directory.name}",
        ):
            try:
                with TemporaryDirectory(prefix="vehicle-eval-") as temporary:
                    gt = Path(temporary) / "annotations.csv"
                    gt.write_bytes(csv_file.getvalue())
                    gt.with_suffix(".json").write_bytes(json_file.getvalue())
                    clip_id = json.loads(json_file.getvalue())["clip_id"]
                    report = compare_runs(gt, clip_id, complete)
                st.session_state[f"evaluation-{directory.name}"] = report
            except (ValueError, OSError, KeyError) as error:
                st.error(f"Evaluasi belum dapat dijalankan. {error}")
        if report := st.session_state.get(f"evaluation-{directory.name}"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {k: r[k] for k in ("run_id", "tp", "fp", "fn", "precision", "recall", "f1")}
                        for r in report["runs"]
                    ]
                ),
                hide_index=True,
            )
            st.download_button(
                "Unduh evaluasi",
                json.dumps(report, indent=2),
                "evaluation.json",
                "application/json",
                on_click="ignore",
            )
