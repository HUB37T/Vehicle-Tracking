# Validasi integrasi — 8 November 2026

## Lingkungan

Windows, Python 3.12, CPU PyTorch. Versi aktual direkam di `constraints.txt`. GPU belum diuji. Tidak ada training atau fine-tuning yang dijalankan.

## Pemeriksaan otomatis

Suite meliputi perilaku counter, konfigurasi, prapemrosesan, simulasi, pemetaan indeks deteksi, asosiasi ByteTrack nyata dengan detector fixture, reset tracker antar-run, pembacaan/penulisan video sintetis, callback, kesesuaian CSV dengan event log, serta status gagal ketika inferensi atau verifikasi output bermasalah. Tes tidak mengunduh model; jaringan untuk persiapan bobot dimock pada tes unit.

Hasil pada lingkungan integrasi: 118 tes lulus, lint dan format bersih, serta `pip check` tidak menemukan dependensi rusak. Metadata distribusi PyTorch dibersihkan melalui pemasangan ulang resmi, lalu diverifikasi sesuai runtime CPU; constraints memuat satu versi per dependensi.

Jalankan `python -m pytest -q`, `python -m ruff check .`, `python -m ruff format --check .`, dan `python -m pip check` setelah instalasi ekstra tracking untuk memeriksa lingkungan lengkap. Dependensi opsional yang belum tersedia menyebabkan tes integrasi terkait dilewati; suite fondasi tetap dapat dipakai sendiri.

## Inferensi model nyata yang telah dijalankan

- Model: YOLO11n resmi, SHA256 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`.
- Sumber bobot diverifikasi terhadap [repositori resmi Ultralytics di Hugging Face](https://huggingface.co/Ultralytics/YOLO11). Jalur unduhan GitHub sempat mengalami timeout; salinan distribusi resmi tersebut berhasil diakses dan diverifikasi checksum-nya.
- Sumber gambar: [foto contoh bus dari repositori resmi Ultralytics](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/assets/bus.jpg).
- Klip lokal: foto diubah menjadi 480 × 640 piksel, diulang 12 frame dengan pergeseran vertikal buatan, FPS 10, durasi 1,2 detik. Ini gerakan sintetis, bukan video jalan nyata.
- Mode prapemrosesan: `none`, garis default P1=(0.1,0.6), P2=(0.9,0.6).
- Hasil: 12 frame selesai, satu event `bus / B_TO_A`; tujuh bucket lain nol. MP4 H.264 hasil dapat didekode sampai selesai; CSV/JSON konsisten.
- Run lokal yang pertama diverifikasi: `20261108T101121Z-f8f3c0e2f831`. Video, bobot, dan hasil berada di lokasi yang diabaikan Git; run ID ini bukan artefak yang dapat diunduh dari repositori.

Smoke test tersebut membuktikan sambungan model, tracker, counter, dan ekspor dapat berjalan di perangkat ini. Jumlah satu event bukan ukuran akurasi jalan, bukan metrik deteksi, dan bukan bukti performa real-time. Evaluasi GT, eksperimen none versus CLAHE pada rekaman jalan, GUI Streamlit, serta validasi GPU masih belum dilakukan.
