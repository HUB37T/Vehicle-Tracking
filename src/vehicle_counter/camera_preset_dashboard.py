"""Streamlit controls for saving, sharing, and applying camera geometry."""

import re
import sqlite3
from pathlib import Path

import streamlit as st

from vehicle_counter.camera_presets import (
    MAX_NAME_LENGTH,
    MAX_PRESET_BYTES,
    CameraPreset,
    load_camera_presets,
    save_camera_preset,
)
from vehicle_counter.contracts import BBox, Point


def show_camera_presets(
    library: Path | None,
    geometry: tuple[Point, Point, BBox] | None,
    *,
    apply_hint: str | None,
) -> CameraPreset | None:
    with st.expander("Preset kamera"):
        st.caption("Gunakan preset untuk video dengan sudut dan framing kamera yang sama.")
        if library is None:
            st.caption("Preset demo tersimpan selama sesi ini. Unduh JSON untuk menyimpannya.")
        selection = st.container()
        st.divider()
        name = st.text_input("Nama preset", max_chars=MAX_NAME_LENGTH, placeholder="Simpang A")
        save = st.button("Simpan posisi sekarang", disabled=geometry is None or not name.strip())
        uploaded = st.file_uploader("File preset JSON", type="json", key="camera_preset_upload")
        import_file = st.button("Impor preset", disabled=uploaded is None)
        session_presets = (
            st.session_state.setdefault("camera_presets", {}) if library is None else {}
        )
        if save or import_file:
            try:
                if save:
                    preset = CameraPreset(name, *geometry)
                else:
                    if uploaded.size > MAX_PRESET_BYTES:
                        raise ValueError("File preset maksimal 16 KB.")
                    preset = CameraPreset.from_json(uploaded.getvalue())
                if library is None:
                    if session_presets.get(preset.key, preset) != preset:
                        raise ValueError("Nama preset sudah dipakai. Gunakan nama lain.")
                    session_presets[preset.key] = preset
                else:
                    save_camera_preset(library, preset)
                st.session_state["camera_preset_selected"] = preset.key
                st.success("Preset tersimpan." if save else "Preset berhasil diimpor.")
            except (OSError, sqlite3.Error, ValueError) as error:
                st.error(f"Preset belum tersimpan. {error}")

        with selection:
            try:
                presets = session_presets if library is None else load_camera_presets(library)
            except (OSError, sqlite3.Error, ValueError) as error:
                st.error(f"Daftar preset tidak dapat dibuka. {error}")
                return None
            selected = st.selectbox(
                "Preset tersimpan",
                sorted(presets),
                format_func=lambda key: presets[key].name,
                key="camera_preset_selected",
                index=None,
                placeholder="Pilih preset kamera" if presets else "Belum ada preset",
                disabled=not presets,
            )
            if apply_hint:
                st.caption(apply_hint)
            if selected is not None:
                preset = presets[selected]
                use, export = st.columns(2)
                if use.button("Terapkan preset", disabled=bool(apply_hint), width="stretch"):
                    return preset
                file_stem = re.sub(r"[^a-z0-9]+", "-", preset.key, flags=re.ASCII).strip("-")
                export.download_button(
                    "Unduh JSON preset",
                    preset.to_json().encode("utf-8"),
                    file_name=f"preset-kamera-{file_stem or 'vehicle-tracking'}.json",
                    mime="application/json",
                    on_click="ignore",
                    width="stretch",
                )
    return None
