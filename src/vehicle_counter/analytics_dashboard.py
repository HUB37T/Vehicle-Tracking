"""Optional road-zone setup and explanation of image-processing evidence."""

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import streamlit as st

from vehicle_counter.contracts import TrafficZoneConfig
from vehicle_counter.shape_editor import decode_polygon, render_shapes
from vehicle_counter.traffic import TRAFFIC_LABELS


def configure_zones(image, config, key):
    with st.expander("Zona macet per lajur/arah"):
        enabled = st.checkbox(
            "Analisis beberapa zona jalan", value=bool(config.traffic_zones), key=f"{key}-multi"
        )
        if not enabled:
            return ()
        st.caption(
            "Beri nama sesuai lajur/arah nyata. Sisi A/B garis tidak otomatis "
            "menjadi nama arah jalan. Zona yang tumpang tindih dapat mengamati"
            " kendaraan yang sama."
        )
        count = st.number_input(
            "Jumlah zona", 1, 4, max(1, len(config.traffic_zones) or 2), key=f"{key}-count"
        )
        saved = st.session_state.get(key + "-zones")
        if saved is None:
            saved = [
                {"zone_id": z.zone_id, "name": z.name, "polygon": z.polygon}
                for z in config.traffic_zones
            ]
        left, top, right, bottom = config.congestion.roi
        zones = []
        for index in range(int(count)):
            a = left + (right - left) * index / count
            b = left + (right - left) * (index + 1) / count
            data = (
                saved[index]
                if index < len(saved)
                else {
                    "zone_id": f"zone-{index + 1}",
                    "name": f"Lajur {index + 1}",
                    "polygon": ((a, top), (b, top), (b, bottom), (a, bottom)),
                }
            )
            zones.append(TrafficZoneConfig(**data))
        selected = st.selectbox(
            "Zona yang diedit",
            range(len(zones)),
            format_func=lambda i: zones[i].name,
            key=f"{key}-selected",
        )
        zone = zones[selected]
        name = st.text_input(
            "Nama lajur/arah", zone.name, max_chars=80, key=f"{key}-name-{selected}"
        )
        value = render_shapes(
            image,
            mode="polygon",
            key=f"{key}-shape-{selected}",
            polygon=zone.polygon,
            overlays=[{"polygon": z.polygon} for i, z in enumerate(zones) if i != selected],
        )
        try:
            points = decode_polygon(value) if value is not None else zone.polygon
            zones[selected] = replace(zone, name=name, polygon=points)
        except (ValueError, TypeError) as error:
            st.error(f"Zona belum valid. {error}")
            return None
        st.session_state[key + "-zones"] = [
            {"zone_id": z.zone_id, "name": z.name, "polygon": z.polygon} for z in zones
        ]
        st.dataframe(
            pd.DataFrame([{"Zona": z.name, "Jumlah titik": len(z.polygon)} for z in zones]),
            hide_index=True,
        )
        return tuple(zones)


def show_quality(report, panel_path=None):
    st.caption(report["scope"] + " Ambang adalah petunjuk awal dan perlu disesuaikan per kamera.")
    for warning in report["warnings"]:
        st.warning(warning)
    if not report["warnings"]:
        st.info(
            "Tidak ada ambang peringatan yang terpicu pada sampel. Ini belum "
            "menjamin semua kendaraan terdeteksi."
        )
    metrics = report["median_metrics"]
    columns = st.columns(3)
    columns[0].metric("Kecerahan rata-rata (0–255)", f"{metrics['brightness_mean']:.1f}")
    columns[1].metric("Simpangan kontras", f"{metrics['contrast_std']:.1f}")
    columns[2].metric("Variansi Laplacian", f"{metrics['laplacian_variance']:.1f}")
    st.caption(
        "Laplacian/FFT mengukur detail citra; area polos juga bisa "
        "memiliki nilai rendah. ORB mencocokkan fitur latar, kemudian "
        "RANSAC menyaring pasangan yang tidak konsisten."
    )
    status = report["background_motion"]["status"]
    if status == "insufficient_features":
        st.info("Fitur latar belum cukup untuk memeriksa gerak kamera.")
    elif status == "candidate_camera_motion":
        st.warning("Ada kandidat gerak kamera/latar bersama. Periksa rekamannya.")
    else:
        st.caption("Sampel pasangan fitur tidak memicu ambang gerak latar.")
    if panel_path is not None:
        st.image(
            panel_path,
            caption="Grayscale, histogram, Laplacian, FFT, dan pencocokan ORB",
            width="stretch",
        )
    with st.expander("Angka pemeriksaan kualitas"):
        st.json(report)


def show_image_analytics(result, manifest):
    stages, quality, heat = st.tabs(("Tahapan model", "Kualitas video", "Heatmap"))
    with stages:
        if path := result.artifact_paths.get("process_evidence"):
            st.image(
                path,
                caption="Ruang warna, prapemrosesan, YOLO, dan tracker run ini",
                width="stretch",
            )
        else:
            st.info("Panel bukti belum tersedia pada run ini.")
    with quality:
        if report := manifest.get("quality"):
            show_quality(report, result.artifact_paths.get("quality_png"))
        else:
            st.info("Aktifkan pemeriksaan kualitas dan analisis ulang untuk mendapatkan panel ini.")
    with heat:
        if path := result.artifact_paths.get("heatmap_png"):
            st.image(path, width="stretch", caption="Heatmap waktu keberadaan anchor kendaraan")
            st.caption(
                "Warna hangat menunjukkan akumulasi keberadaan yang lebih tinggi "
                "dalam video ini. Heatmap memakai hasil tracking yang sama; bukan "
                "jumlah kendaraan unik, kecepatan fisik, atau bukti macet."
            )
            if metadata := result.artifact_paths.get("heatmap_json"):
                info = json.loads(Path(metadata).read_text(encoding="utf-8"))
                st.caption(
                    f"Puncak sel mentah: {info['maximum_raw_seconds']:.2f} detik teramati. "
                    "Gaussian smoothing hanya dipakai untuk tampilan."
                )
        else:
            st.info("Aktifkan heatmap sebelum memulai analisis.")


def show_traffic_zones(result, manifest, frame_loader):
    zones = manifest.get("traffic_zones", [])
    if not zones:
        return
    st.subheader("Kondisi per zona/lajur")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Zona": z["name"],
                    "Status terakhir": TRAFFIC_LABELS[z["last_status"]],
                    "Durasi indikasi macet (detik)": round(z["durations_seconds"]["congested"], 2),
                }
                for z in zones
            ]
        ),
        hide_index=True,
    )
    selected = st.selectbox(
        "Periksa zona lalu lintas",
        range(len(zones)),
        format_func=lambda i: zones[i]["name"],
        key=f"zones-{result.run_id}",
    )
    zone = zones[selected]
    rows = pd.read_csv(result.artifact_paths["traffic_zones_csv"])
    rows = rows[rows.zone_id == zone["zone_id"]]
    st.line_chart(
        rows.set_index("timestamp_seconds")[["vehicle_count"]].rename(
            columns={"vehicle_count": "Kendaraan teramati"}
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
        y_label="Fraksi 0–1",
    )
    st.dataframe(rows, hide_index=True)
    if zone["episodes"]:
        episode = st.selectbox(
            "Kejadian macet pada zona ini",
            range(len(zone["episodes"])),
            format_func=lambda i: (
                f"{zone['episodes'][i]['start_seconds']:.1f}–"
                f"{zone['episodes'][i]['end_seconds']:.1f} detik"
            ),
            key=f"zone-episodes-{result.run_id}-{zone['zone_id']}",
        )
        e = zone["episodes"][episode]
        st.video(
            result.artifact_paths["annotated_video"], start_time=max(0, int(e["start_seconds"]) - 1)
        )
    st.divider()
    st.caption(
        "Ringkasan di bawah berasal dari area persegi panjang utama, terpisah dari zona tambahan."
    )
