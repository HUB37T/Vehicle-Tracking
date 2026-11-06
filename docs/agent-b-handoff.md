# Handoff fondasi untuk Agent B

## Status

Kontrak, konfigurasi, counter, prapemrosesan, dan simulasi sudah tersedia. Semua data simulasi adalah sintetis. Modul `tracking.py`, `pipeline.py`, `app.py`, `visualization.py`, `exporting.py`, dan `evaluation.py` belum dibuat. `process_video(...)` pada CONTEXT masih merupakan interface yang direncanakan, bukan fungsi yang dapat dipanggil sekarang.

Ikuti instalasi dan pemeriksaan di [README](../README.md), lalu baca bagian 7, 11–15, dan 17 pada [CONTEXT](../CONTEXT.md).

## Kontrak yang bisa langsung diimpor

```python
from vehicle_counter.contracts import (
    DIRECTIONS,
    VEHICLE_CLASSES,
    CountEvent,
    CountSummary,
    ProgressUpdate,
    RunConfig,
    RunResult,
    TrackObservation,
    VideoMetadata,
)
from vehicle_counter.config import config_to_dict, load_config
```

Field mengikuti CONTEXT. `PreprocessConfig` dan `CountingLineConfig` juga berada di `contracts.py`. Konfigurasi dan observasi adalah dataclass frozen; urutan koordinat adalah `xyxy`, koordinat piksel frame asli, dan anchor tersedia lewat `observation.anchor`.

Untuk mengubah konfigurasi sebelum run, gunakan `dataclasses.replace`; ini juga memvalidasi nilai baru:

```python
from dataclasses import replace
from vehicle_counter.contracts import PreprocessConfig

config = load_config("configs/default.yaml")
config = replace(config, preprocess=PreprocessConfig(mode="clahe"))
```

`summary.by_class_direction` memuat empat nama kelas, masing-masing dengan `A_TO_B` dan `B_TO_A`, termasuk bucket nol. Snapshot dari `counter.summary()` tidak berbagi dictionary dengan state counter. Event frozen dan `counter.events` berupa tuple.

## Fixture untuk tampilan dan ekspor

```python
from vehicle_counter.simulation import run_simulation

fixture = run_simulation()
assert fixture["source_kind"] == "synthetic"
summary_dict = fixture["summary"]
event_dicts = fixture["events"]
```

Fixture adalah dictionary JSON-serializable, bukan `RunResult` dari video. Tampilkan label **Data simulasi** pada UI selama memakai fixture. Lima frame contoh mempunyai ukuran 101 × 101 dan FPS 10; default menghasilkan dua event pada frame 4, timestamp 0,4 detik. Perubahan garis atau aturan konfigurasi dapat mengubah hasil tersebut.

Untuk bekerja dengan objek kontrak:

```python
from vehicle_counter.counting import VehicleCounter
from vehicle_counter.contracts import RunConfig
from vehicle_counter.simulation import synthetic_frames

counter = VehicleCounter("preview-synthetic", RunConfig(), width=101, height=101)
for observations in synthetic_frames():
    new_events = counter.update(observations)
events = counter.events
summary = counter.summary()
```

Satu pemanggilan `update` menerima seluruh observasi untuk satu frame dengan ID berbeda. Frame dan timestamp batch nonkosong harus meningkat; kirim `[]` ketika tidak ada observasi. Counter tidak menerima prediksi track tanpa deteksi aktual atau ID yang kosong. Kelas di luar empat kelas kendaraan diabaikan. Counter baru dibuat untuk setiap run; replay frame pada instance yang sama ditolak agar tidak menggandakan state.

## Tugas yang dapat dimulai

1. Buat tampilan ringkasan dan tabel/grafik dari fixture dengan label simulasi.
2. Implementasikan ekspor `events.csv` dan `counts.csv` sesuai CONTEXT, lalu uji total yang sama dengan event. Semua delapan bucket harus tetap ada, termasuk ketika event kosong.
3. Siapkan renderer garis/box/anchor/lintasan yang menerima frame BGR dan kontrak observasi, tanpa mengimpor tracker ke UI.
4. Siapkan evaluator dengan data sintetis berlabel untuk menguji pencocokan satu-ke-satu dan metrik. Evaluasi model yang sesungguhnya menunggu GT video.

Ekspor fixture tidak diberi status run video `completed`, karena video dan metadata inferensi nyata belum tersedia. Integrasi upload, decoder, callback `ProgressUpdate`, MP4 beranotasi, dan `RunResult` akhir menunggu pipeline A.

## Kepemilikan berkas dan langkah integrasi

Agent A menangani fondasi, geometri/counter, prapemrosesan, dan pipeline CV. Agent B menangani UI, renderer, ekspor, dan evaluator. Berkas bersama (`contracts.py`, konfigurasi, dependensi, README, CONTEXT) diselaraskan sebelum perubahan digabung.

Saat menambahkan Streamlit atau library lain, tambahkan dependensi secara terpusat ke `pyproject.toml`, uji lingkungan gabungan, lalu perbarui `constraints.txt` dengan versi aktual yang diuji. Simpan CLI simulasi dan tes counter agar integrasi dapat diperiksa tanpa video.
