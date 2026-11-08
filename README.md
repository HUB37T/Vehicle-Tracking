# Vehicle Tracking & Counting

Proyek akhir mata kuliah **Pengolahan Citra Digital dan Visi Komputer** untuk menghitung kendaraan dari video berdasarkan jenis dan arah perjalanan.

Pengguna mengunggah rekaman jalan dari kamera diam, mengatur garis virtual, lalu sistem mendeteksi dan melacak kendaraan. Hitungan bertambah ketika lintasan kendaraan memenuhi aturan penyeberangan garis.

## Status proyek

CLI video mendukung YOLO11n pretrained, ByteTrack, penghitung dua arah, lima mode prapemrosesan, ekspor CSV/JSON, dan MP4 beranotasi. CLI juga menyediakan praktikum pengolahan citra klasik dari dua frame. Integrasi diuji pada Python 3.12 di Windows dengan PyTorch CPU.

Analisis bisa dijalankan lewat CLI atau antarmuka lokal Streamlit. Evaluasi awal terhadap anotasi manual tersedia untuk dua klip jalan. Lihat [hasil evaluasi awal](docs/counting-evaluation.md).

## Rencana fitur

- Deteksi mobil, sepeda motor, bus, dan truk dengan YOLO11n pretrained.
- Pelacakan kendaraan menggunakan ByteTrack.
- Penghitungan satu garis untuk dua arah dengan pencegahan penghitungan ulang per Track ID.
- Antarmuka lokal Streamlit untuk unggah video, pengaturan garis, progres, dan ringkasan.
- Ekspor event dan hitungan CSV, metadata JSON, serta video beranotasi.
- Eksperimen tanpa prapemrosesan versus CLAHE dan evaluasi terhadap anotasi manual.

## Acuan kerja bersama

- [CONTEXT.md](CONTEXT.md): tujuan, lingkup, aturan counting, rancangan teknis, kontrak data, pembagian kerja, evaluasi, dan contoh prompt untuk agent.
- [AGENTS.md](AGENTS.md): instruksi awal bagi AI agent yang mengerjakan repositori.
- [.gitignore](.gitignore): mengecualikan data besar, bobot, hasil pemrosesan, lingkungan lokal, dan rahasia.
- [.gitattributes](.gitattributes): menyamakan akhir baris berkas teks saat bekerja lintas sistem operasi.

Kedua anggota tim menggunakan versi `CONTEXT.md` yang sama. Selaraskan perubahan kontrak bersama sebelum menggabungkan pekerjaan.

## Mengambil repositori

```sh
git clone https://github.com/HUB37T/Vehicle-Tracking.git
cd Vehicle-Tracking
```

## Instalasi fondasi proyek

Gunakan Python 3.12 untuk mengikuti lingkungan yang sudah diuji. Dari folder repositori, jalankan di PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -c constraints.txt
```

Aktivasi lingkungan tidak diperlukan karena perintah menunjuk langsung ke Python proyek. Pada macOS/Linux, gunakan `python3.12 -m venv .venv` dan ganti `.\.venv\Scripts\python.exe` dengan `.venv/bin/python`.

`pyproject.toml` adalah sumber definisi paket dan dependensi. `constraints.txt` mencatat versi yang diuji, termasuk integrasi CPU. Instalasi fondasi tidak mengunduh bobot ataupun memasang YOLO. Untuk analisis video, lanjutkan instalasi berikut.

## Instalasi dan analisis video

Dari lingkungan fondasi yang sudah terpasang, gunakan CPU untuk baseline:

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -c constraints.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-tracking.txt -c constraints.txt
.\.venv\Scripts\python.exe -m vehicle_counter.cli prepare-model
```

Persiapan bobot membutuhkan internet. Default mengunduh dari repositori resmi `Ultralytics/YOLO11` di Hugging Face, mengunci revision dan memverifikasi SHA256 dari metadata resminya. Alternatif distribusi GitHub tersedia melalui `prepare-model --source github`. Bobot disimpan lokal di `models/yolo11n.pt`; analisis video memakai berkas lokal dan tidak memulai unduhan model diam-diam.

Setelah ada video jalan:

```powershell
.\.venv\Scripts\python.exe -m vehicle_counter.cli inspect data/videos/jalan.mp4
.\.venv\Scripts\python.exe -m vehicle_counter.cli analyze data/videos/jalan.mp4 --config configs/default.yaml
```

Ganti `jalan.mp4` dengan nama rekamanmu. Atur koordinat garis ternormalisasi P1/P2 di YAML sebelum analisis. Dengan garis horizontal dari kiri ke kanan, A berada di atas dan B di bawah. `--weights` dapat menunjuk bobot lokal lain; `--output-root` mengubah folder hasil. Opsi `--device` hanya digunakan dengan PyTorch/perangkat yang sesuai; lingkungan CPU yang diuji tidak mengaktifkan GPU meskipun laptop memiliki GPU.

Masukan baseline adalah MP4 lokal CFR dengan dimensi genap untuk ekspor H.264/yuv420p. Metadata dan frame awal divalidasi; VFR harus dikonversi terlebih dahulu. Pipeline mendeteksi ketidaksesuaian jumlah frame jika metadata menyediakannya. Jika jumlah frame tidak diketahui, keterbatasan verifikasi akhir sumber dicatat sebagai peringatan.

Folder `outputs/<run_id>/` berisi `events.csv`, `counts.csv`, `run.json`, `annotated.mp4`, dan `process_evidence.png`. Audio sumber tidak disertakan. Run diberi status `completed` setelah output video berhasil didekode seluruhnya dan hitungan cocok dengan event log. Kegagalan sesudah run dibuat menghasilkan metadata `failed`, penjelasan error, dan hitungan parsial. CLI mengembalikan kode keluar bukan nol.

`process_evidence.png` memperlihatkan satu frame yang sama pada enam panel. Urutannya adalah frame sumber, kanal L setelah konversi BGR ke Lab, kanal L setelah CLAHE, masukan aktual detektor, deteksi YOLO dengan confidence, dan track ByteTrack dengan ID serta jejak. Panel CLAHE menandai status tidak aktif pada mode lain. Bukti memakai crossing pertama jika ada, dengan frame lain yang menampilkan deteksi atau track sebagai cadangan. `run.json` mencatat indeks, waktu, dan alasan pemilihan frame.

Setiap run membuat detector/tracker dan counter baru. Setiap frame menjalankan satu inferensi deteksi, lalu asosiasi ByteTrack eksplisit; anchor counting berasal dari kotak deteksi yang cocok dengan ID pada frame tersebut. Konfigurasi tracker aktual, bobot/checksum, perangkat, versi, dan waktu pemrosesan dicatat di `run.json`.

## Praktikum pengolahan citra

Gunakan `pcd-lab` untuk melihat tahapan pada frame target dan frame sebelumnya. Contoh berikut memilih frame 60 dengan indeks mulai dari 0. Perintah ini cukup memakai dependensi fondasi dan OpenCV.

```powershell
.\.venv\Scripts\python.exe -m vehicle_counter.cli pcd-lab data/videos/vehicles.mp4 --frame 60 --config configs/default.yaml
```

- `classical.png` memperlihatkan grayscale, Gaussian smoothing, selisih antarframe, threshold global tetap versus Otsu, opening, closing, Connected Component Labeling (CCL), Canny, dan Optical Flow Lucas–Kanade.
- `preprocessing.png` membandingkan frame yang sama pada mode `none`, `clahe`, `gamma`, `gaussian`, dan `unsharp`.
- `lab.json` mencatat indeks frame, parameter, ambang Otsu, dan kotak, luas, serta centroid komponen. Komponen menunjukkan area perubahan piksel. Bayangan, goyangan kamera, atau bagian berbeda dari kendaraan yang sama dapat membentuk komponen sendiri. Panah Optical Flow diperbesar tiga kali untuk tampilan.

Untuk memakai salah satu mode pada inferensi, tambahkan `--preprocess`. Pilih satu mode per analisis dan gunakan klip, garis, serta parameter deteksi yang sama untuk membandingkan hasil.

```powershell
.\.venv\Scripts\python.exe -m vehicle_counter.cli analyze data/videos/vehicles.mp4 --config configs/default.yaml --preprocess gamma
```

Konfigurasi `gamma` di bawah 1 mencerahkan midtone. `gaussian_kernel_size` mengatur ukuran kernel Gaussian yang juga dipakai pada `unsharp`. `unsharp_amount` mengatur kekuatan penajaman. Hasil `analyze` tetap menghitung crossing berdasarkan YOLO dan ByteTrack. Pengaruh setiap mode terhadap akurasi memerlukan pembanding berupa hitungan manual.

## Menjalankan antarmuka Streamlit

Setelah menyiapkan CPU PyTorch dan bobot model seperti pada instalasi tracking:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ui.txt -c constraints.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Buka `http://127.0.0.1:8501`. Pilih video dari `data/videos/` atau unggah MP4 hingga 100 MB. Geser koordinat P1/P2 sambil melihat preview, pilih prapemrosesan, lalu tekan **Mulai analisis**. Rekap, video, panel bukti, dan unduhan muncul setelah proses selesai. Folder hasil UI berada di `outputs/ui/<run_id>/`.

## Mengevaluasi hitungan

`vehicle-counter evaluate` menerima satu CSV anotasi, satu JSON pendamping dengan nama yang sama, dan folder hasil run. Evaluator memeriksa checksum video serta garis sebelum mencocokkan event. Contoh perintah, metrik, dan catatan kedua klip ada di [laporan evaluasi](docs/counting-evaluation.md).

## Mencoba tanpa video

```powershell
.\.venv\Scripts\python.exe -m vehicle_counter.simulation --config configs/default.yaml
```

Simulasi berisi lima Track ID buatan. Dengan konfigurasi default, hanya dua crossing sah yang dihitung: satu `car` ke arah `A_TO_B` dan satu `motorcycle` ke arah `B_TO_A`. Track diam, track pada zona netral, dan track di luar ujung garis tidak dihitung. Hasil JSON berlabel `source_kind: synthetic`.

Untuk menyimpan data contoh bagi antarmuka:

```powershell
.\.venv\Scripts\python.exe -m vehicle_counter.simulation --output outputs/simulation.json
```

Lihat [handoff Agent B](docs/agent-b-handoff.md) untuk kontrak yang sudah dapat dipakai temanmu dan batas integrasi yang masih menunggu pipeline video.

## Menjalankan pemeriksaan

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pip check
```

Tes mencakup crossing, konfigurasi, prapemrosesan, simulasi, asosiasi ByteTrack dengan detektor fixture, lifecycle video, callback, validasi unduhan, ekspor, dan penanganan kegagalan. Tes yang memerlukan ekstra tracking dilewati jika dependensinya belum dipasang. Pengujian otomatis tidak mengunduh bobot atau video. Lihat [catatan validasi integrasi](docs/integration-validation.md) untuk inferensi model nyata yang dijalankan secara terpisah; manfaat CLAHE pada rekaman jalan perlu dievaluasi kemudian.

## Struktur yang sudah tersedia

```text
configs/default.yaml             Konfigurasi awal
src/vehicle_counter/contracts.py Kontrak data bersama dan konfigurasi immutable
src/vehicle_counter/config.py    Pembacaan YAML dan snapshot konfigurasi
src/vehicle_counter/geometry.py  Garis berhingga dan arah sisi
src/vehicle_counter/counting.py  Event crossing dan ringkasan hitungan
src/vehicle_counter/preprocessing.py  None, CLAHE, gamma, Gaussian, dan unsharp
src/vehicle_counter/pcd_lab.py         Panel materi pengolahan citra klasik
src/vehicle_counter/simulation.py     Contoh observasi sintetis dan CLI
src/vehicle_counter/tracking.py       Adapter YOLO + ByteTrack per run
src/vehicle_counter/video.py          Reader MP4 dan writer H.264 streaming
src/vehicle_counter/pipeline.py       Orkestrasi analisis dan lifecycle run
src/vehicle_counter/visualization.py  Anotasi frame sumber dan lintasan pendek
src/vehicle_counter/exporting.py      Ekspor CSV dan metadata JSON
src/vehicle_counter/cli.py            Persiapan bobot, inspeksi, analisis video
tests/                          Pengujian otomatis
docs/agent-b-handoff.md          Panduan integrasi untuk teman/Agent B
```

Antarmuka dan evaluator sudah tersambung. Anotasi awal perlu ditinjau ulang oleh anggota kedua sebelum evaluasi akhir.

## Data dan bobot model

Rencana lokasi lokal:

- `data/videos/`: video sumber.
- `data/annotations/`: anotasi manual dan metadata data uji yang dapat dibagikan.
- `models/`: bobot model yang diunduh.
- `outputs/`: keluaran setiap run.

Folder berisi petunjuk lokal; video, bobot, dan keluaran analisis diabaikan Git. Dokumentasikan sumber, izin penggunaan, dan cara memperoleh data/bobot agar percobaan dapat diulang. Berkas anotasi dan dokumentasi eksperimen dapat dimasukkan ke repositori selama layak dibagikan.

Training belum diperlukan untuk baseline pretrained. Mulailah dari video pengembangan dan video uji dengan hitungan manual. Lihat [panduan data video](docs/video-data-guide.md) untuk pemisahan data dan kondisi ketika fine-tuning baru dipertimbangkan.

## Lisensi

Kode dan dokumentasi milik proyek mengikuti [lisensi MIT](LICENSE) yang sudah tersedia di repositori. Library, bobot pretrained, serta data pihak ketiga mengikuti lisensi sumbernya masing-masing.
