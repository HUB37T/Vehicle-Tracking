# Vehicle Tracking & Counting

Proyek akhir mata kuliah **Pengolahan Citra Digital dan Visi Komputer** untuk menghitung kendaraan dari video berdasarkan jenis dan arah perjalanan.

Pengguna mengunggah rekaman jalan dari kamera diam, mengatur garis virtual, lalu sistem mendeteksi dan melacak kendaraan. Hitungan bertambah ketika lintasan kendaraan memenuhi aturan penyeberangan garis.

## Status proyek

Fondasi Python sudah tersedia: kontrak data bersama, konfigurasi YAML tervalidasi, geometri garis, counter dua arah, prapemrosesan none/CLAHE, serta simulasi dengan observasi sintetis. Pengujian dijalankan pada Python 3.12 di Windows.

Integrasi YOLO/ByteTrack, pembacaan video, aplikasi Streamlit, ekspor run video, dan evaluasi pada rekaman nyata masih tahap berikutnya. Simulasi saat ini tidak menjalankan detektor maupun tracker dan tidak menunjukkan akurasi model.

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

`pyproject.toml` adalah sumber definisi paket dan dependensi. `constraints.txt` mencatat versi dependensi yang diuji pada tahap fondasi; kombinasi lengkap dengan YOLO dan Streamlit perlu diuji ulang saat integrasi. Bobot model tidak diunduh pada tahap ini.

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

Tes mencakup aturan crossing, konsistensi ringkasan dengan event log, input yang tidak valid, konfigurasi, prapemrosesan, dan simulasi. Pengujian counter tidak memerlukan video, GPU, atau model YOLO. CLAHE diuji menggunakan citra sintetis; manfaatnya pada kendaraan perlu dievaluasi kemudian.

## Struktur yang sudah tersedia

```text
configs/default.yaml             Konfigurasi awal
src/vehicle_counter/contracts.py Kontrak data bersama dan konfigurasi immutable
src/vehicle_counter/config.py    Pembacaan YAML dan snapshot konfigurasi
src/vehicle_counter/geometry.py  Garis berhingga dan arah sisi
src/vehicle_counter/counting.py  Event crossing dan ringkasan hitungan
src/vehicle_counter/preprocessing.py  Mode none dan CLAHE
src/vehicle_counter/simulation.py     Contoh observasi sintetis dan CLI
tests/                          Pengujian otomatis
docs/agent-b-handoff.md          Panduan integrasi untuk teman/Agent B
```

Tahap berikutnya untuk Agent A adalah adapter YOLO/ByteTrack dan pipeline pembacaan video setelah data tersedia. Agent B dapat mulai mengembangkan tampilan dan ekspor menggunakan fixture berlabel sesuai handoff.

## Data dan bobot model

Rencana lokasi lokal:

- `data/videos/`: video sumber.
- `data/annotations/`: anotasi manual dan metadata data uji yang dapat dibagikan.
- `models/`: bobot model yang diunduh.
- `outputs/`: keluaran setiap run.

Folder berisi petunjuk lokal; video, bobot, dan keluaran analisis diabaikan Git. Dokumentasikan sumber, izin penggunaan, dan cara memperoleh data/bobot agar percobaan dapat diulang. Berkas anotasi dan dokumentasi eksperimen dapat dimasukkan ke repositori selama layak dibagikan.

## Lisensi

Kode dan dokumentasi milik proyek mengikuti [lisensi MIT](LICENSE) yang sudah tersedia di repositori. Library, bobot pretrained, serta data pihak ketiga mengikuti lisensi sumbernya masing-masing.
