# Handoff pipeline untuk Agent B

## Status

Kontrak, konfigurasi, counter, prapemrosesan, simulasi, adapter YOLO/ByteTrack, reader/writer video, dan `process_video(...)` sudah tersedia. Renderer dan ekspor CSV/JSON dasar juga dibuat agar pipeline dapat diuji lengkap. `app.py` dan `evaluation.py` belum dibuat. Semua fixture dan klip gerakan sintetis tetap diberi label; akurasi video jalan belum dievaluasi.

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

## Menyambungkan UI ke pipeline nyata

```python
from vehicle_counter.pipeline import process_video

config = load_config("configs/default.yaml")
result = process_video(
    source_path="data/videos/jalan.mp4",
    config=config,
    output_root="outputs",
    on_progress=handle_progress,  # fungsi UI yang menerima satu ProgressUpdate
)
```

Contoh di atas memerlukan video aktual, bobot lokal, ekstra tracking, dan `handle_progress` milik UI; contoh ini bukan snippet mandiri sebelum prasyarat tersebut tersedia. Buat input sementara dengan path lokal saat pengguna mengunggah file dan tutup reader setelah pipeline kembali. Pipeline mengelola reader/writer internalnya sendiri.

Fungsi bersifat sinkron. Jalankan sekali ketika pengguna menekan mulai, kemudian simpan `RunResult` di session state. Widget/download rerun menggunakan hasil tersimpan. Setiap pemanggilan membuat tracker/counter baru; model state tidak dibagikan antarrun. Parameter `tracker_factory` khusus untuk injeksi pengujian, bukan untuk UI produksi.

Callback membawa snapshot ringkasan dan salinan `preview_frame_bgr`, termasuk pada frame terakhir. Ubah BGR ke RGB untuk tampilan. `total_frames` dapat `None`; tampilkan jumlah frame terproses jika persentase belum bisa dihitung. Jangan mengasumsikan callback dipanggil pada setiap frame atau bahwa progres 100% berarti semua berkas sudah diverifikasi; status akhir dari `RunResult` menentukan keberhasilan.

Kunci `artifact_paths`: `run_json`, `events_csv`, `counts_csv`, dan `annotated_video` pada run sukses. Status `failed` dapat memiliki CSV parsial; tampilkan kegagalan, bukan hasil akhir lengkap. Detail error ada di `run.json` dan `warnings`. Input video yang tidak valid sebelum run dibuat menaikkan exception yang perlu ditampilkan UI.

Renderer `FrameAnnotator` dan fungsi `export_counts`/`write_json` sudah digunakan pipeline. Agent B dapat mengembangkan modul itu saat menghubungkan UI; pertahankan kontrak serta pengujian CSV/lifecycle yang tersedia. Ekspor memakai event yang sama dengan counter, bukan hitungan baru dari box yang ditampilkan.

## Tugas lanjutan

1. Buat tampilan ringkasan dan tabel/grafik dari fixture dengan label simulasi.
2. Sambungkan ekspor `events.csv` dan `counts.csv` yang sudah tersedia ke tombol unduh; pertahankan semua delapan bucket, termasuk saat event kosong.
3. Sambungkan preview callback/video beranotasi dan kontrol garis ke UI. Renderer sudah menerima frame sumber, observasi, serta ringkasan.
4. Siapkan evaluator dengan data sintetis berlabel untuk menguji pencocokan satu-ke-satu dan metrik. Evaluasi model yang sesungguhnya menunggu GT video.

Ekspor fixture tidak diberi status inferensi nyata. Run sintetis yang memakai factory uji menuliskan `is_model_inference: false` dan peringatan di metadata. Integrasi upload dan interaksi pengguna masih milik UI; decoder, callback, MP4, serta `RunResult` pipeline sudah siap.

## Kepemilikan berkas dan langkah integrasi

Agent A menangani fondasi, geometri/counter, prapemrosesan, dan pipeline CV. Agent B menangani UI, renderer, ekspor, dan evaluator. Berkas bersama (`contracts.py`, konfigurasi, dependensi, README, CONTEXT) diselaraskan sebelum perubahan digabung.

Saat menambahkan Streamlit atau library lain, tambahkan dependensi secara terpusat ke `pyproject.toml`, uji lingkungan gabungan, lalu perbarui `constraints.txt` dengan versi aktual yang diuji. Snapshot integrasi sekarang memakai CPU PyTorch; baca instalasi tracking di README sebelum memasang constraints. Simpan CLI simulasi dan tes counter agar integrasi tetap dapat diperiksa tanpa video.
