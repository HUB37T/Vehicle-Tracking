# Vehicle Tracking & Counting

Proyek akhir mata kuliah **Pengolahan Citra Digital dan Visi Komputer** untuk menghitung kendaraan dari video berdasarkan jenis dan arah perjalanan.

Pengguna mengunggah rekaman jalan dari kamera diam, mengatur garis virtual, lalu sistem mendeteksi dan melacak kendaraan. Hitungan bertambah ketika lintasan kendaraan memenuhi aturan penyeberangan garis.

## Status proyek

CLI video mendukung YOLO11n pretrained, ByteTrack, DeepSORT opsional, penghitung dua arah, tujuh mode prapemrosesan, ekspor CSV/JSON, dan MP4 beranotasi. UI menyediakan antrean/batal, zona macet per lajur, heatmap, kualitas video, dan editor label manual. Integrasi diuji pada Python 3.12 di Windows dengan PyTorch CPU.

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

YOLO11n tetap model bawaan. YOLO11s pretrained tersedia sebagai pembanding yang lebih besar dari [keluarga YOLO11 resmi](https://docs.ultralytics.com/models/yolo11/). Siapkan secara eksplisit:

```powershell
.\.venv\Scripts\python.exe -m vehicle_counter.cli prepare-model --variant yolo11s
```

Pilih melalui **Pengaturan model YOLO**. Ukuran masukan dan confidence dapat diatur sebelum run. CLI menyediakan `--weights models/yolo11s.pt --imgsz 960 --conf 0.1`. Pengunduhan memverifikasi checksum sebelum mempromosikan berkas dan memeriksa checksum cache bila metadata sumber tersedia. Ini penggunaan bobot pretrained; belum ada fine-tuning model tim.

Setelah ada video jalan:

```powershell
.\.venv\Scripts\python.exe -m vehicle_counter.cli inspect data/videos/jalan.mp4
.\.venv\Scripts\python.exe -m vehicle_counter.cli analyze data/videos/jalan.mp4 --config configs/default.yaml
```

Ganti `jalan.mp4` dengan nama rekamanmu. Atur koordinat garis ternormalisasi P1/P2 di YAML sebelum analisis. Dengan garis horizontal dari kiri ke kanan, A berada di atas dan B di bawah. `--weights` dapat menunjuk bobot lokal lain; `--output-root` mengubah folder hasil. Opsi `--device` hanya digunakan dengan PyTorch/perangkat yang sesuai; lingkungan CPU yang diuji tidak mengaktifkan GPU meskipun laptop memiliki GPU.

Masukan baseline adalah MP4 lokal CFR dengan dimensi genap untuk ekspor H.264/yuv420p. Metadata dan frame awal divalidasi; VFR harus dikonversi terlebih dahulu. Pipeline mendeteksi ketidaksesuaian jumlah frame jika metadata menyediakannya. Jika jumlah frame tidak diketahui, keterbatasan verifikasi akhir sumber dicatat sebagai peringatan.

Folder `outputs/<run_id>/` berisi `events.csv`, `counts.csv`, `crossing_audit.csv`, `run.json`, `annotated.mp4`, dan `process_evidence.png`. Audio sumber tidak disertakan. Run diberi status `completed` setelah output video berhasil didekode seluruhnya dan hitungan cocok dengan event log. Kegagalan sesudah run dibuat menghasilkan metadata `failed`, penjelasan error, dan hitungan parsial. CLI mengembalikan kode keluar bukan nol.

`process_evidence.png` memperlihatkan satu frame yang sama pada enam panel. Urutannya adalah frame sumber, kanal L setelah konversi BGR ke Lab, kanal L setelah CLAHE, masukan aktual detektor, deteksi YOLO dengan confidence, dan track ByteTrack dengan ID serta jejak. Panel CLAHE menandai status tidak aktif pada mode lain. Bukti memakai crossing pertama jika ada, dengan frame lain yang menampilkan deteksi atau track sebagai cadangan. `run.json` mencatat indeks, waktu, dan alasan pemilihan frame.

Setiap run membuat detector/tracker dan counter baru. Setiap frame menjalankan satu inferensi deteksi, lalu asosiasi tracker yang dipilih; anchor counting berasal dari kotak deteksi yang cocok dengan ID pada frame tersebut. Konfigurasi tracker aktual, bobot/checksum, perangkat, versi, dan waktu pemrosesan dicatat di `run.json`. Panel keenam menampilkan nama tracker aktual, ByteTrack atau DeepSORT.

`crossing_audit.csv` menambah koordinat box, anchor, dan tanda tumpang tindih untuk setiap event yang diterima counter. Tab **Audit crossing** menampilkan tabel serta frame video hasil untuk ID yang dipilih. Tanda `anchor_inside_other_detection` menunjukkan kandidat oklusi untuk ditinjau, bukan vonis event salah. Event tanpa tanda juga belum terverifikasi benar. Kontrak tujuh kolom `events.csv` tetap sama.

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

Mode `median` menambah eksperimen pengurangan noise impuls. `bilateral_clahe` menjalankan bilateral BGR sebelum CLAHE pada kanal L di Lab; diameter default 5, sigma warna/ruang 35. Bilateral berusaha menghaluskan noise sambil menjaga tepi, tetapi tetap dapat menghilangkan detail yang dibutuhkan detektor. Kedua mode tersedia di praktikum, CLI, dan profil UI **Kustom**; default tetap sama. Gunakan **CLAHE vs prapemrosesan terpilih** untuk membandingkan dalam satu antrean dengan geometri/config lainnya sama.

### Pemeriksaan kualitas tanpa YOLO

```powershell
.\.venv\Scripts\python.exe -m vehicle_counter.cli inspect-quality data/videos/vehicles.mp4
```

Pemeriksaan memakai maksimal enam sampel pada tiga detik awal, ukuran maksimal 640 piksel. Panel menunjukkan grayscale, histogram, Laplacian, FFT, dan pencocokan ORB + Hamming + RANSAC pada fitur latar. `quality.json` menyimpan pengukuran serta peringatan cahaya/kontras/detail rendah dan kandidat gerak latar. Area polos bisa menyerupai blur; kendaraan/pohon bergerak dapat mengganggu pemeriksaan latar. Sedikit fitur menghasilkan keterangan belum cukup data. Ini petunjuk awal, bukan skor akurasi atau jaminan kamera stabil sepanjang video.

Analisis normal menyertakan `quality.png`/`quality.json` secara default. UI juga memiliki tombol **Periksa kualitas video sebelum analisis**. Hilangkan centang pemeriksaan atau gunakan CLI `--no-quality-check` untuk melewatinya. Pemeriksaan tidak memilih prapemrosesan secara otomatis.

## Menjalankan antarmuka Streamlit

Setelah menyiapkan CPU PyTorch dan bobot model seperti pada instalasi tracking:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ui.txt -c constraints.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Buka `http://127.0.0.1:8501`. Di **Siapkan analisis**, pilih video dari `data/videos/` atau unggah MP4 hingga 100 MB. Pada profil **Kustom**, geser titik P1/P2 langsung di pratinjau atau gunakan kontrol angka. Profil evaluasi mempertahankan garis yang dibekukan. Pengaturan YOLO, tracker, prapemrosesan, dan kualitas video ada di **Model dan pelacakan**.

Pakai **Uji singkat** untuk memeriksa 3, 5, atau 10 detik awal, atau **Mulai analisis** untuk video penuh. Pantau proses di **Antrean**. Jika kamu masih di tab Antrean saat proses berakhir, aplikasi membuka **Hasil**. **Riwayat** membuka kembali hasil tersimpan. Hasil baru berada di `outputs/jobs/<job_id>/` dan hasil lama di `outputs/ui/` tetap terbaca.

### Menyimpan dan berbagi preset kamera

1. Pilih video, aktifkan indikator macet untuk melihat kotaknya, lalu atur garis dan kotak pada pratinjau. Buka **Preset kamera**, isi **Nama preset**, dan klik **Simpan posisi sekarang**.
2. Untuk memakai lagi, pilih video dengan sudut dan framing kamera yang sama. Pilih **Kustom** jika video memiliki profil evaluasi. Pilih nama di **Preset tersimpan**, lalu klik **Terapkan preset**. Garis dan kotak akan tampil pada pratinjau dengan indikator macet aktif.
3. Untuk berbagi, pilih preset lalu klik **Unduh JSON preset**. Di perangkat lain, buka **Preset kamera**, unggah file, klik **Impor preset**, lalu **Terapkan preset**. Gunakan nama berbeda jika sudah ada preset dengan nama sama dan posisi berbeda.

Preset berisi nama, posisi P1/P2 dalam urutan asli, serta koordinat kotak utama pada rentang 0–1. Di lokal, daftar tersimpan di `outputs/camera-presets.sqlite3`. Di demo publik, daftar tersedia selama sesi browser, jadi unduh JSON untuk memakainya di sesi lain.

### Antrean dan tombol batal

Antrean aplikasi menjalankan satu pekerjaan inferensi pada satu waktu, termasuk seluruh batch perbandingan. Maksimal empat pekerjaan aktif, satu per sesi. Progres dan preview diperbarui berkala; widget tetap dapat dipakai. **Batalkan pekerjaan** langsung membatalkan permintaan yang menunggu, atau meminta proses berjalan berhenti setelah operasi/frame saat ini. Hitungan parsial dan metadata disimpan sebagai `cancelled`; video parsial tidak disajikan sebagai hasil sukses dan evaluator menolaknya.

Pekerjaan menyimpan salinan video serta konfigurasi saat dikirim; perubahan widget tidak mengubah pekerjaan lama. Worker dapat meneruskan proses saat tab ditutup, tetapi kontrol batal milik sesi pengirim. Jika sesi berganti, hasil selesai dapat dicari lewat riwayat. `outputs/jobs/` berisi SQLite, folder sumber, log, progres, dan hasil; jaga folder ini jika ingin mempertahankan riwayat/sumber pelabelan. Data tidak otomatis dihapus saat tab ditutup, sehingga penyimpanan perlu dikelola tim. CLI `analyze` langsung tetap sinkron dan berada di luar antrean UI.

- **Perbandingan metode** menjalankan none vs CLAHE, ByteTrack vs DeepSORT, atau empat kombinasi dengan video, garis, area, dan durasi yang sama. DeepSORT membutuhkan dependensi opsional di bawah. Tabel membandingkan jumlah serta waktu pemrosesan; penilaian akurasi membutuhkan anotasi manual. Gunakan run penuh untuk evaluasi GT seluruh video.
- **YOLO11n vs YOLO11s** pada perbandingan metode memakai konfigurasi sama dan menampilkan nama model, ukuran masukan, confidence, serta checksum bobot. Kedua bobot harus tersedia lokal. Jangan menilai model hanya dari jumlah deteksi atau crossing yang lebih banyak.
- **Riwayat analisis** dapat mencari dan membuka kembali hingga 100 run terbaru di `outputs/`, termasuk run singkat dan hasil gagal. Hasil yang dibuka memakai konfigurasi run tersimpan; pengaturan percobaan baru tidak mengubahnya. Folder run yang dipindah tetap bisa dibuka jika berkas hasilnya lengkap.
- **Hasil → Rekap** menampilkan crossing per kelas dan arah beserta grafik waktu per interval 5 detik dari event otomatis. Buka **Bukti materi → Diagnostik deteksi dan tracking** untuk memeriksa deteksi kosong, track belum muncul, posisi terhadap garis, dan jeda ID.
- **Audit crossing** menyediakan catatan pemeriksaan manual termasuk kendaraan terlewat. Catatan tersimpan sebagai riwayat terpisah dan dapat diunduh; hitungan otomatis tetap bersumber dari `events.csv`.
- Keluaran tambahan berupa `timeline.csv`, `diagnostics.json`, dan `traffic.csv` jika indikator macet aktif. Hasil uji singkat diberi label khusus dan tidak dianggap hitungan seluruh video. Pembuatan laporan PDF ditunda.

### Mencoba indikator macet

Aktifkan **Indikator macet**, lalu geser kotak biru untuk memindahkan area atau tarik sudutnya untuk mengubah ukurannya. Posisi disimpan saat drag dilepas. Hindari langit, trotoar, atau area kosong yang luas, dan pilih area yang juga dipantau detektor. Atur ambang di **Pengaturan indikator macet** jika sudut kamera berbeda. Status serta grafik muncul pada tab **Lalu lintas** dan overlay video hasil.

Indikator awal menggabungkan minimal 3 track, keterisian area minimal 15%, dan minimal 65% track bergerak lambat selama 3 detik. Kecepatan relatif dihitung dari perpindahan pusat box terhadap diagonal box, dengan riwayat minimal 1 detik. Ini belum mengukur km/jam. Kamera harus diam; ID terputus, perspektif, parkir, antrean lampu merah, dan pemilihan area dapat memengaruhi hasil. Ambang memerlukan validasi per kamera; belum ada metrik akurasi macet.

Tab **Lalu lintas** juga mencatat rentang waktu indikasi macet, durasi, jumlah track tertinggi, dan apakah kondisi masih teramati pada akhir klip. Pilih kejadian untuk memutar video dari dekat awalnya dan memeriksa frame konfirmasi. Rentang dimulai setelah syarat durasi terpenuhi, bukan sejak awal kendaraan melambat. CSV `traffic_episodes.csv` menyimpan rentang yang sama. Durasi status menggunakan status sampel sampai sampel berikutnya atau akhir video yang dianalisis; ketelitian perubahan status mengikuti interval sampel 0.5 detik. Run lama tetap memakai ringkasan yang dibuat saat itu.

Uji lokal pada 15 November 2026 membandingkan kedua model pada dua klip penuh dengan ByteTrack `fuse_score: true` dan CLAHE. Hujan: YOLO11n TP/FP/FN = 7/0/0, YOLO11s = 6/0/1 (masukan 960). Istanbul: 5/5/0 versus 5/4/0 (masukan 640 dan crop detektor yang sama). Keduanya lebih lambat pada YOLO11s dalam uji ini. Angka berasal dari pencocokan kelas/arah/waktu dengan GT awal, `identity_verified: false`; belum merupakan audit identitas ulang atau mAP deteksi. YOLO11n tetap bawaan karena model lebih besar belum unggul konsisten. Artefak percobaan tersimpan lokal di `outputs/model-improvement-validation/`; anotasi tidak diubah.

Jika video lokal `vehicles-3-bytetrack-test.mp4` tersedia, pilih profil **Pemulihan crossing** untuk memakai preset antrean dan lakukan uji 10 detik. CLI setara untuk mencoba awal 8 detik:

```powershell
.\.venv\Scripts\python.exe -m vehicle_counter.cli analyze data/videos/vehicles-3-bytetrack-test.mp4 --config configs/experiments/vehicles-3-congestion.yaml --max-seconds 8
```

Untuk video lain, `--max-seconds 5 --congestion-roi 0.05 0.45 0.95 0.95` mengaktifkan indikator pada area ternormalisasi tersebut. Hilangkan `--max-seconds` untuk analisis penuh. Parameter lengkap berada di `CongestionConfig`/bagian `congestion` YAML; lihat [acuan indikator macet](CONTEXT.md#132-indikator-macet-awal).

### Zona per lajur dan heatmap

Setelah indikator macet aktif, buka **Zona macet per lajur/arah**, centang analisis beberapa zona, pilih 1–4 zona, dan beri nama sesuai jalan yang diamati. Geser titik poligon; tambah/hapus titik jika perlu mengikuti perspektif lajur. Poligon 3–12 titik tidak boleh bersilangan. Setiap zona memiliki riwayat gerak dan status sendiri, sehingga kendaraan bergerak pada lajur lain tidak dicampur ke indikator zona tersebut. Zona yang tumpang tindih boleh mengamati kendaraan sama; jangan menjumlahkannya sebagai kendaraan unik.

Tab **Lalu lintas** menampilkan status/episode per zona dan CSV `traffic_zones.csv`/`traffic_zone_episodes.csv`. Area persegi panjang utama tetap ditampilkan terpisah. Profil **Pemulihan crossing** untuk klip `vehicles-3-bytetrack-test.mp4` kini menggunakan preset [dua zona contoh](configs/experiments/vehicles-3-coursework.yaml); bentuknya perlu disesuaikan untuk kamera lain.

Centang **Buat heatmap keberadaan kendaraan**, lalu buka **Hasil → Heatmap** setelah analisis. Heatmap memakai anchor hasil tracking yang sama, dengan penambahan `1/FPS` detik per sel yang teramati setiap frame. `heatmap.npz` menyimpan grid mentah 128×128, `heatmap.json` menjelaskan satuan/skala, dan `heatmap.png` menampilkan warna relatif pada frame sumber. Gaussian smoothing hanya memperhalus tampilan. Warna hangat bukan bukti macet atau jumlah kendaraan unik. CLI: tambahkan `--heatmap`.

### Menyiapkan label untuk fine-tuning

Pada **Audit crossing**, buka **Beri kotak manual untuk calon dataset YOLO**, centang editor kotak, lalu pilih frame sumber. Pilih kelas dan tarik kotak; klik kotak untuk mengubah kelas atau memperbaiki sudutnya. Konfirmasi setelah seluruh kendaraan diperiksa. Frame tanpa kotak yang dikonfirmasi disimpan sebagai contoh negatif. Jika sumber run lama hilang, unggah ulang video asli; checksum wajib sama.

Label tersimpan sebagai revisi baru di `outputs/dataset-labels/`, terpisah dari event counting dan catatan crossing. Ekspor ZIP berisi PNG sumber tanpa overlay, label YOLO `class cx cy width height` ternormalisasi, metadata, dan petunjuk. Kelas dataset: 0 `car`, 1 `motorcycle`, 2 `bus`, 3 `truck`. Ekspor memakai revisi terakhir tiap video/frame, maksimal 200 frame/150 MB melalui UI. Belum ada split otomatis: pisahkan train/validation/test berdasarkan **video sumber**, dan tinjau label bersama teman sebelum pelatihan. Fitur ini belum melatih bobot baru atau menghasilkan mAP.

### Hasil eksperimen filter tambahan

Uji 16 November 2026 memakai klip penuh, YOLO11n, ByteTrack `fuse_score: true`, dan geometri/config pemulihan yang sama. Angka berikut adalah pencocokan otomatis kelas/arah/waktu, `identity_verified: false`.

| Klip | Prapemrosesan | TP | FP | FN |
| --- | --- | ---: | ---: | ---: |
| Hujan | CLAHE | 7 | 0 | 0 |
| Hujan | Bilateral + CLAHE | 6 | 0 | 1 |
| Hujan | Median | 7 | 0 | 0 |
| Istanbul | CLAHE | 5 | 5 | 0 |
| Istanbul | Bilateral + CLAHE | 3 | 6 | 2 |
| Istanbul | Median | 3 | 6 | 2 |

Filter tambahan belum menunjukkan peningkatan konsisten, sehingga tidak mengganti default. Artefak lokal berada di `outputs/coursework-feature-validation/`. Dua klip tersebut sudah berulang dipakai untuk eksperimen; evaluasi akhir perlu video baru yang belum dipakai memilih parameter. Uji worker nyata terpisah di `outputs/real-worker-validation/` memverifikasi antrean, pembatalan saat menunggu/berjalan, dua zona, heatmap, dan konsistensi satu crossing pada enam detik awal klip antrean. Angka fixture pada pengujian otomatis tidak dimasukkan sebagai metrik akurasi proyek.

## Mengevaluasi hitungan

`vehicle-counter evaluate` menerima satu CSV anotasi, satu JSON pendamping dengan nama yang sama, dan folder hasil run. Evaluator memeriksa checksum video serta garis sebelum mencocokkan event. Contoh perintah, metrik, dan catatan kedua klip ada di [laporan evaluasi](docs/counting-evaluation.md).

Pencocokan otomatis memakai kelas, arah, dan waktu, sehingga dua kendaraan berbeda yang lewat berdekatan bisa tertukar pasangan. Laporan sekarang menyatakan `identity_verified: false`; periksa frame sumber dan audit identitas sebelum menyebut crossing pulih. Lihat [audit Istanbul dan perbandingan tracker](docs/tracker-safety-evaluation.md).

Untuk memilih profil tracker pada satu run CLI, gunakan `--tracker-config configs/bytetrack-no-fuse.yaml`. Profil ini memakai `fuse_score: false` dan disediakan untuk eksperimen terhadap baseline. Lihat [hasil dan cara mengulang eksperimen ByteTrack](docs/bytetrack-fuse-score-evaluation.md).

Profil `configs/experiments/vehicles-3-refined.yaml` menguji pemeriksaan kelas SUV/van dan penolakan crossing ganda dengan `fuse_score: true`. [Petunjuk bobot, perintah, dan hasil perbandingan](docs/bytetrack-fuse-score-evaluation.md#uji-lanjutan-dengan-fuse_score-true) tersedia dalam laporan.

## Mencoba DeepSORT

ByteTrack dengan `fuse_score: true` tetap pilihan utama. DeepSORT adalah metode berbeda untuk eksperimen materi tracking; parameter `fuse_score` tidak berlaku padanya. Pasang setelah dependensi tracking dan PyTorch CPU tersedia:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[deepsort]" -c constraints.txt
.\.venv\Scripts\python.exe -m vehicle_counter.cli analyze data/videos/holdout-hvsd-rain.mp4 --config configs/experiments/hvsd-rain-recovery.yaml --tracker-config configs/deepsort.yaml
```

Di Streamlit, pilih **DeepSORT (eksperimen)** pada pilihan tracker. Tombol analisis dinonaktifkan bila paket opsional belum tersedia. `deep-sort-realtime==1.3.2` memakai MobileNetV2 pretrained ImageNet yang dibundel dalam paket untuk fitur tampilan pada CPU; bobot dan checksum dicatat di metadata. Ini belum merupakan model ReID yang dilatih khusus kendaraan. Analisis hanya meneruskan track terkonfirmasi dengan deteksi aktual pada frame itu; posisi prediksi Kalman saat deteksi hilang tidak menghasilkan crossing. DeepSORT lebih lambat dan menambah hitungan ganda pada klip hujan dalam [uji aktual](docs/tracker-safety-evaluation.md), sehingga belum direkomendasikan menggantikan ByteTrack. Paket lama ini masih memakai `pkg_resources`, sehingga ekstra mengunci `setuptools<81`.

Setelah memperbarui kode saat aplikasi lokal masih berjalan, mulai ulang proses Streamlit agar modul pelacak dimuat ulang. Hentikan proses di terminal dengan Ctrl+C, lalu jalankan kembali perintah Streamlit pada bagian antarmuka. Streamlit dapat memperbarui tampilan sambil mempertahankan modul Python lama di memori; akibatnya pilihan DeepSORT bisa terlihat tetapi pelacak lama masih menolaknya. UI memeriksa konfigurasi melalui pelacak yang sedang dimuat dan menampilkan petunjuk mulai ulang sebelum mengizinkan analisis jika terjadi ketidaksesuaian ini.

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
src/vehicle_counter/tracking.py       Adapter YOLO + pilihan tracker per run
src/vehicle_counter/deepsort.py       Asosiasi DeepSORT opsional
src/vehicle_counter/crossing_audit.py  Bukti posisi crossing untuk tinjauan visual
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
