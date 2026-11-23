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

## Dokumentasi proyek

- [Rancangan teknis](docs/project-design.md) memuat lingkup, aturan penghitungan, arsitektur, kontrak data, dan evaluasi.
- [.gitignore](.gitignore): mengecualikan data besar, bobot, hasil pemrosesan, lingkungan lokal, dan rahasia.
- [.gitattributes](.gitattributes): menyamakan akhir baris berkas teks saat bekerja lintas sistem operasi.

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

YOLO11n adalah model bawaan. Siapkan YOLO11s untuk membandingkan model yang lebih besar, atau YOLO26n untuk menguji kandidat perangkat edge. [YOLO11 resmi](https://docs.ultralytics.com/models/yolo11/) · [YOLO26 resmi](https://docs.ultralytics.com/models/yolo26/)

```powershell
.\.venv\Scripts\python.exe -m vehicle_counter.cli prepare-model --variant yolo11s
.\.venv\Scripts\python.exe -m vehicle_counter.cli prepare-model --variant yolo26n
```

Pilih di **Siapkan analisis → Model dan pelacakan → Model deteksi**. Ukuran masukan dan confidence dapat diatur sebelum run. Untuk YOLO26n lewat CLI, tambahkan `--weights models/yolo26n.pt` pada perintah `analyze`. YOLO26n memakai bobot pretrained PyTorch dalam alur video yang sama dengan YOLO11n. Ukur performa lagi pada perangkat dan runtime yang akan dipakai di lapangan.

Setelah ada video jalan:

```powershell
.\.venv\Scripts\python.exe -m vehicle_counter.cli inspect data/videos/jalan.mp4
.\.venv\Scripts\python.exe -m vehicle_counter.cli analyze data/videos/jalan.mp4 --config configs/default.yaml
```

Ganti `jalan.mp4` dengan nama rekamanmu. Atur koordinat garis ternormalisasi P1/P2 di YAML sebelum analisis. Dengan garis horizontal dari kiri ke kanan, A berada di atas dan B di bawah. `--weights` dapat menunjuk bobot lokal lain; `--output-root` mengubah folder hasil. Opsi `--device` hanya digunakan dengan PyTorch/perangkat yang sesuai; lingkungan CPU yang diuji tidak mengaktifkan GPU meskipun laptop memiliki GPU.

Masukan baseline adalah MP4 lokal CFR dengan dimensi genap untuk ekspor H.264/yuv420p. Metadata dan frame awal divalidasi; VFR harus dikonversi terlebih dahulu. Pipeline mendeteksi ketidaksesuaian jumlah frame jika metadata menyediakannya. Jika jumlah frame tidak diketahui, keterbatasan verifikasi akhir sumber dicatat sebagai peringatan.

Folder `outputs/<run_id>/` berisi `events.csv`, `counts.csv`, `crossing_audit.csv`, `run.json`, `annotated.mp4`, dan `process_evidence.png`. Audio sumber tidak disertakan. Run diberi status `completed` setelah output video berhasil didekode seluruhnya dan hitungan cocok dengan event log. Kegagalan sesudah run dibuat menghasilkan metadata `failed`, penjelasan error, dan hitungan parsial. CLI mengembalikan kode keluar bukan nol.

`process_evidence.png` di **Hasil → Bukti materi → Tahapan model** menampilkan frame sumber, hasil prapemrosesan sesuai mode, deteksi YOLO dengan confidence, dan hasil pelacakan. Gamma, Gaussian, unsharp, dan median menampilkan hasil beserta parameternya untuk dibandingkan dengan frame sumber. CLAHE dan bilateral + CLAHE menampilkan kanal L sebelum dan sesudah penyesuaian serta hasil BGR yang dipakai detektor. Jika detektor memakai ROI, panel masukan memperlihatkan potongan aktualnya. Bukti memakai crossing pertama jika ada, dengan frame lain yang menampilkan deteksi atau track sebagai cadangan. `run.json` mencatat indeks, waktu, dan alasan pemilihan frame.

Setiap run membuat detector/tracker dan counter baru. Setiap frame menjalankan satu inferensi deteksi, lalu asosiasi tracker yang dipilih; anchor counting berasal dari kotak deteksi yang cocok dengan ID pada frame tersebut. Konfigurasi tracker aktual, bobot/checksum, perangkat, versi, dan waktu pemrosesan dicatat di `run.json`. Panel pelacakan menampilkan nama tracker aktual, ByteTrack atau DeepSORT.

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
- **YOLO11n vs YOLO11s** dan **YOLO11n vs YOLO26n** di **Perbandingan metode** memakai konfigurasi sama dan menampilkan nama model, ukuran masukan, confidence, serta checksum bobot. Siapkan kedua bobot untuk pasangan yang dipilih. Gunakan anotasi manual untuk menilai ketepatan crossing.
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

Untuk video lain, `--max-seconds 5 --congestion-roi 0.05 0.45 0.95 0.95` mengaktifkan indikator pada area ternormalisasi tersebut. Hilangkan `--max-seconds` untuk analisis penuh. Parameter lengkap berada di `CongestionConfig`/bagian `congestion` YAML; lihat [acuan indikator macet](docs/project-design.md#132-indikator-macet-awal).

### Kendaraan rapat dan deteksi antrean yang terputus

Untuk enam klip kamera diam lokal, pilih **Pemulihan crossing** agar geometri, YOLO26n pretrained, dan pemeriksa MPV terisi. Pilih **Model dan pelacakan → ByteTrack untuk kendaraan rapat (eksperimen)** untuk mencoba sambungan ID yang lebih longgar. Pasang garis pada bagian jalan tempat kendaraan terlihat jelas di kedua sisi. Tinjau crossing dekat ujung garis dan saat motor saling menutupi dengan **Audit crossing**. Profil eksperimen mempertahankan `fuse_score: true` dan memakai `match_thresh: 0.87`.

Tinjauan lokal menemukan dua crossing motor A→B yang pulih pada `091928` dan tujuh motor B→A pada `092328`. Jumlah event sistem pada empat klip berubah dari 32, 21, 24, dan 60 menjadi 34, 21, 26, dan 70. Terdapat kandidat duplikasi motor dan pergeseran kelas. Gunakan pilihan eksperimen untuk perbandingan per kamera. Tinjauan memeriksa perubahan yang terlihat, dengan anotasi lengkap seluruh kendaraan masih diperlukan.

Di **Pengaturan indikator macet**, **Toleransi deteksi terputus saat macet** menunggu sampel baru jika bukti gerak antrean yang sudah terkonfirmasi hilang sesaat. Profil lokal memakai 0,20 detik. Pada klip `092139`, uji dengan tracker dan garis yang sama memperpanjang indikasi area gabungan dari 4,0–12,0 detik menjadi 4,0–26,2 detik. Area lajur kanan tetap mendeteksi antrean sampai sekitar 26,5 detik pada kedua run. Hasil visual menunjukkan lajur kanan masih mengantre setelah detik 12. Untuk menilai kapan antrean mulai dan terurai, lihat status per lajur pada kamera diam. Rekap crossing tetap 24 pada kedua uji toleransi.

Keluaran eksperimen dan tinjauan kejadian tersimpan lokal di `outputs/crossing-traffic-20261119/`.

Pilihan **ByteTrack untuk kendaraan rapat (eksperimen)** juga memeriksa kelas dari deteksi terbaru menjelang crossing dan menyaring pasangan ID searah ketika kotaknya tumpang tindih kuat selama beberapa frame. Ringkasan memakai kelas final pada event. Untuk melihat kapan keputusan terjadi, buka **Unduhan → Unduh koreksi crossing kendaraan rapat**. Jalur CLI yang setara untuk rekaman pada sudut jalan terbagi:

```powershell
.\.venv\Scripts\python.exe -m vehicle_counter.cli analyze data/videos/local-20261119-VID_20261119_091928-cfr.mp4 --config configs/experiments/local-20261119-mpv.yaml --tracker-config configs/bytetrack-dense.yaml --guard-dense-crossings
```

Rekap arah mengikuti posisi garis yang tersimpan. Video dari kamera bergerak masih perlu diperiksa secara terpisah karena posisi garis terhadap jalan berubah selama panning.

Pada empat klip yang sebelumnya diuji dengan opsi kendaraan rapat, hasil perbaikan adalah:

| Klip | A→B sebelum → sesudah | B→A sebelum → sesudah | Crossing ganda disaring | Penyesuaian kelas |
| --- | ---: | ---: | ---: | ---: |
| `091928` | 30 → 30 | 4 → 4 | 0 | 1 |
| `092102` | 2 → 2 | 19 → 19 | 0 | 1 |
| `092139` | 19 → 18 | 7 → 7 | 1 | 1 |
| `092328` | 38 → 36 | 32 → 32 | 2 | 1 |

Tinjauan visual mengonfirmasi duplikasi motor pada `092139` detik 29,97, motor pada `092328` detik 8,90, dan kotak kabin/kendaraan yang sama pada `092328` detik 72,17. Empat kelas disesuaikan setelah melihat kendaraan pada frame sumber. Dua crossing motor A→B dan tujuh B→A yang sebelumnya dipulihkan tetap tercatat. Ini audit kasus, dengan anotasi lengkap dan peninjau kedua masih diperlukan. Rincian lokal tersedia di `outputs/dense-crossing-repair-20261119/`.

Jika pemeriksa crop aktif, hasilnya juga dipakai saat voting seluruh track menunjukkan mobil dan konsensus dekat garis menunjukkan truk. Bukti SUV dari crop dapat mempertahankan kelas mobil. Keputusan ini tercatat dengan alasan `passenger_class_conflict` di `truck_review.csv`. Pemilihan posisi garis dan keterlihatan kendaraan tetap memengaruhi crossing, termasuk motor yang berhenti dekat garis.

Uji lokal mencakup 10 rekaman penuh. Pada delapan rekaman kamera diam terdapat tujuh penyaringan duplikasi dan empat penyesuaian kelas, dengan kasusnya diperiksa pada frame sumber. Dua rekaman kamera bergerak juga diproses, dengan `camera_motion_limit` pada tabel perbandingan lokal. Untuk empat klip pada tabel, pembanding memakai ByteTrack kendaraan rapat sebelum perbaikan. Enam klip lainnya dibandingkan dengan run ByteTrack biasa yang sudah tersedia. `direction-comparison.csv` dan `visual-review.csv` di folder keluaran mencatat kedua pembanding dan kasus yang ditinjau.

### Zona per lajur dan heatmap

Buka **Zona per lajur/arah**, centang **Analisis beberapa zona jalan**, pilih 1–4 zona, lalu beri nama sesuai lajur atau arah. Atur titik poligon agar tiap zona mencakup bagian garis hitung yang dilalui kendaraan di lajur tersebut. Poligon memakai 3–12 titik dan sisinya tidak boleh bersilangan.

**Hasil → Per lajur** menampilkan jumlah mobil, motor, bus, dan truk yang melewati garis di tiap zona. Pakai **Unduhan → Unduh hitungan per lajur** untuk mendapatkan `lane_counts.csv`. Saat titik potong masuk beberapa zona, crossing diberikan satu kali ke zona dengan nomor prioritas terkecil. Crossing yang tidak masuk zona tampil di **Di luar zona** dan tetap masuk total. Run lama perlu dianalisis ulang untuk membuat rekap ini.

Jika indikator macet aktif, **Lalu lintas** juga menampilkan status per zona dan menyediakan `traffic_zones.csv` serta `traffic_zone_episodes.csv`. Pemantauan per zona mengamati kendaraan di dalam setiap poligon, termasuk kendaraan yang berada di area tumpang tindih. Area persegi panjang utama tetap ditampilkan terpisah. Profil **Pemulihan crossing** untuk `vehicles-3-bytetrack-test.mp4` menyediakan [dua zona contoh](configs/experiments/vehicles-3-coursework.yaml).

Centang **Buat heatmap keberadaan kendaraan**, lalu buka **Hasil → Heatmap** setelah analisis. Heatmap memakai anchor hasil tracking yang sama, dengan penambahan `1/FPS` detik per sel yang teramati setiap frame. `heatmap.npz` menyimpan grid mentah 128×128, `heatmap.json` menjelaskan satuan/skala, dan `heatmap.png` menampilkan warna relatif pada frame sumber. Gaussian smoothing hanya memperhalus tampilan. Warna hangat bukan bukti macet atau jumlah kendaraan unik. CLI: tambahkan `--heatmap`.

### Pemeriksaan MPV yang terbaca truk

Di **Model dan pelacakan**, aktifkan **Periksa ulang kelas MPV/truk**, centang **Fokus koreksi MPV/minivan**, lalu gunakan **Ambang bukti kendaraan ringan: 0,30** untuk mencoba setelan rekaman lokal. Aktifkan juga **Kurangi kotak ganda berbeda kelas** untuk mengatasi satu kendaraan yang terdeteksi ganda sebagai mobil/bus/truk. Siapkan bobot pemeriksa sekali:

```powershell
curl.exe -fL https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth -o models/mobilenet_v3_large-5c1a4163.pth
```

Analisis memverifikasi checksum bobot dan tidak mengunduhnya diam-diam. Kandidat crossing hasil voting `truck` diperiksa dengan MobileNetV3-Large pada crop frame sumber. Dalam mode fokus MPV, kelas teratas wajib `minivan` dan jumlah skor kelompok kendaraan ringan minimal 0,30. Minibus, moving van, dan truk tidak otomatis disamakan dengan MPV. Skor ini merupakan heuristik, bukan confidence YOLO atau kepastian klasifikasi.

Pada video hasil, `car (review)` menandai koreksi sejak kendaraan melewati garis; sebelumnya label masih mengikuti detektor. **Unduhan → Unduh pemeriksaan MPV dan truk** menyediakan `truck_review.csv` berisi keputusan dan skor tiap kandidat. Rekap, daftar crossing, dan hitungan per lajur memakai kelas final yang sama. Panel bukti YOLO/tracker menampilkan kelas asli detektor/tracker, bukan kelas hasil koreksi crop; kotak pada panel YOLO sudah melewati filter kotak ganda jika diaktifkan.

Filter kotak ganda memakai IoU minimal 0,90 dan mempertahankan kotak dengan confidence lebih tinggi sebelum tracking. Hanya konflik antarkelas mobil/bus/truk yang diperiksa; kotak motor tidak disaring. Filter ini berbeda dari `deduplicate_crossings`, yang membandingkan kandidat crossing sesudah tracking. Filter tidak menjamin seluruh ID switch atau hitung ganda teratasi. Jumlah kotak yang disaring dicatat pada `tracker.duplicate_box_filter` di `run.json`.

Profil [local-20261119-mpv.yaml](configs/experiments/local-20261119-mpv.yaml) menyediakan YOLO26n, masukan 640, confidence 0,10, tanpa prapemrosesan, ByteTrack `fuse_score: true`, filter kotak antarkelas, dan fokus minivan. Garis/zona khusus sudut jalan terbagi pada rekaman lokal; periksa posisinya untuk kamera lain. Jalankan dari folder proyek pada video CFR yang sudah disiapkan:

Pada UI, pilih salah satu dari enam rekaman kamera diam `local-20261119-VID_20261119_*-cfr.mp4` lalu **Profil analisis → Pemulihan crossing** untuk mengisi profil ini. Klip persimpangan dengan kamera bergerak tidak mendapat preset tersebut.

```powershell
.\.venv\Scripts\python.exe -m vehicle_counter.cli analyze data/videos/local-20261119-VID_20261119_091928-cfr.mp4 --config configs/experiments/local-20261119-mpv.yaml
```

Tuning ini memilih parameter inferensi dan pemeriksa kelas. Bobot pretrained belum dilatih ulang; fine-tuning YOLO memerlukan label kotak lengkap serta pemisahan video train/validation/test.

Hasil lokal 19 November pada `VID_20261119_091928.mp4` versi CFR:

| Hasil | Mobil | Motor | Bus | Truk | Total crossing |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline YOLO26n 640 | 17 | 12 | 0 | 5 | 34 |
| Fokus MPV + filter kotak antarkelas | 18 | 12 | 0 | 2 | 32 |

Tiga MPV pada 1,50; 2,63; dan 13,50 detik dikoreksi ke mobil. Dua crossing ganda pada 1,50 dan 10,17 detik dihilangkan; pasangan kotak baseline memiliki IoU 0,973 dan 0,970 serta menampilkan kendaraan fisik yang sama. Pada `092328`, duplikasi minibus pada 68,20 detik dihilangkan sehingga total 61 menjadi 60. Jumlah motor pada keenam klip kamera diam tetap sama dengan baseline. Ini pemeriksaan kasus tertentu, belum GT lengkap atau mAP deteksi.

Percobaan ukuran 960 pada klip pertama menghasilkan 42 crossing dan 22 event motor, tetapi salah menggolongkan satu kendaraan barang sebagai mobil. Percobaan `agnostic_nms` juga membuat satu motor nyata terlewat pada `092102`; filter terbatas pada profil akhir mempertahankan 10 event motor baseline klip itu. Ukuran lebih besar dan jumlah crossing lebih tinggi belum otomatis lebih akurat. Manifest/konfigurasi/keluaran lokal berada di `outputs/local-tuning-20261119/`; baseline tetap tersedia di `outputs/local-recordings-20261119/`.

Sepuluh rekaman diuji ulang lokal. Delapan klip kamera diam diuji dengan filter akhir; dua klip kamera bergerak dipakai sebagai kontrol tanpa filter kotak. Tiga MPV serta 10 contoh truk/pickup diperiksa pada crossing video lengkap: MPV menjadi mobil dan truk/pickup tetap truk. Tiga crossing yang hilang cocok dengan duplikasi terverifikasi; event lain memiliki pasangan spasial/temporal pada baseline, tetapi pencocokan itu belum merupakan audit identitas lengkap. `comparison.csv`, `verified-class-cases.json`, dan manifest tiap run menyimpan rincian lokal. Hitungan dari kamera bergerak tetap memiliki keterbatasan garis tetap. Pengujian kode/UI terbaru mencakup 256 tes yang lulus.

### Menyiapkan label untuk fine-tuning

Pada **Audit crossing**, buka **Beri kotak manual untuk calon dataset YOLO**, centang editor kotak, lalu pilih frame sumber. Pilih kelas dan tarik kotak; klik kotak untuk mengubah kelas atau memperbaiki sudutnya. Konfirmasi setelah seluruh kendaraan diperiksa. Frame tanpa kotak yang dikonfirmasi disimpan sebagai contoh negatif. Jika sumber run lama hilang, unggah ulang video asli; checksum wajib sama.

Label tersimpan sebagai revisi baru di `outputs/dataset-labels/`, terpisah dari event counting dan catatan crossing. Ekspor ZIP berisi PNG sumber tanpa overlay, label YOLO `class cx cy width height` ternormalisasi, metadata, dan petunjuk. Kelas dataset: 0 `car`, 1 `motorcycle`, 2 `bus`, 3 `truck`. MPV, SUV, angkot, dan van kecil masuk `car`; bus besar masuk `bus`. Ekspor memakai revisi terakhir tiap video/frame, maksimal 200 frame/150 MB melalui UI. Pisahkan train/validation/test berdasarkan **video sumber**, dan tinjau label bersama teman sebelum pelatihan. Editor tidak otomatis melatih bobot; perintah pelatihan terpisah dijelaskan berikut.

### Fine-tuning bobot YOLO lokal

Fine-tuning bobot sudah dijalankan sebagai eksperimen pada 19 November 2026 dengan RTX 4070 laptop, PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121, dan Ultralytics 8.4.207. Lingkungan `.venv-train` dipisahkan dari `.venv` aplikasi CPU. Instalasi baru dapat memakai Python 3.12 dan [paket CUDA resmi PyTorch](https://pytorch.org/get-started/previous-versions/#v251):

```powershell
py -3.12 -m venv .venv-train
.\.venv-train\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
.\.venv-train\Scripts\python.exe -m pip install -r requirements-training.txt
```

Pada laptop pengembangan, `.venv-train` dibuat dengan `--system-site-packages` untuk menggunakan instalasi CUDA yang sudah tersedia. Tidak perlu mengganti paket `.venv` aplikasi. [Panduan fine-tuning Ultralytics](https://docs.ultralytics.com/guides/finetuning-guide/) menjelaskan transfer bobot dan pemilihan lapisan yang dibekukan.

Dataset awal terdiri dari **17 crop area jalan / 76 kotak** untuk training, **5 crop / 30 kotak** untuk validasi, dan **6 crop / 18 kotak** untuk uji, termasuk dua crop jalan kosong. Proposal detektor diperiksa dan dikoreksi dengan bantuan AI. Label masih memerlukan tinjauan independen oleh manusia. Ini dataset awal yang sangat kecil, belum mencakup bus besar. Training memakai tiga video jalan terbagi; validasi satu video lain; uji dua video WhatsApp dari sudut kamera lain. Video-video itu sudah dipakai pada eksperimen inferensi sebelumnya, sehingga bukan koleksi uji akhir proyek yang sama sekali belum pernah dilihat.

Manifest label tersimpan lokal di `outputs/fine-tuning-20261119/reviewed-labels.json`. Setiap entri berisi `image`, `source_group` (checksum video asli), `split`, `reviewed`, `reviewer`, dan `boxes` berupa kelas serta kotak piksel `[x1,y1,x2,y2]` relatif pada gambar. Asal video, frame, ROI crop, dan cara pelabelan ikut dicatat. `prepare` menolak frame belum ditinjau, gambar identik yang berulang, dan video sumber yang melintasi split. Folder tujuan harus baru:

```powershell
.\.venv-train\Scripts\python.exe -m vehicle_counter.fine_tuning prepare outputs/fine-tuning-20261119/reviewed-labels.json --output outputs/fine-tuning-repeat/dataset
.\.venv-train\Scripts\python.exe -m vehicle_counter.fine_tuning train outputs/fine-tuning-repeat/dataset/dataset.yaml --weights models/yolo26n.pt --project outputs/fine-tuning-repeat/training --epochs 30 --device 0
```

Parameter awal: ukuran 640, batch 4, seed 37, AdamW, learning rate 0,0002, freeze 10 lapisan, dan early stopping 10 epoch. Kandidat empat kelas berhenti pada epoch 23 dan memilih `best.pt` dari epoch 13. Kandidat kedua mempertahankan 80 kelas pretrained, freeze 22 dan learning rate 0,0001, memakai opsi `--keep-pretrained-head --freeze 22 --learning-rate 0.0001`. Opsi itu mengubah **indeks label berdasarkan nama kelas**, misalnya car 0→2 dan truck 3→7; tidak memakai label empat kelas langsung pada head COCO.

Kandidat dipilih berdasarkan validasi, kemudian hanya kandidat terpilih dan pretrained dievaluasi pada split uji. Evaluasi AP memakai ukuran/batch/confidence/IoU yang sama, dengan label baseline dipetakan ke indeks COCO. Hasil berikut hanya pada crop berlabel awal, bukan akurasi counting atau jaminan kinerja CCTV:

| Bobot | mAP50–95 validasi | mAP50 uji | mAP50–95 uji |
| --- | ---: | ---: | ---: |
| Pretrained | 0,9360 | 0,9308 | 0,8170 |
| Fine-tuning empat kelas | 0,8193 | 0,8428 | 0,6369 |
| Fine-tuning head pretrained | 0,8044 | Tidak dipilih | Tidak dipilih |

**Bobot eksperimen belum mengungguli pretrained dan tidak mengganti default.** `models/yolo26n-local-ft.pt` tersedia lokal sebagai **YOLO26n — fine-tuning lokal (eksperimen)** di UI, beserta pilihan perbandingannya. Sidecar bobot menyimpan checksum dan metadata training. Bobot, gambar, label lokal, serta artefak diabaikan Git; teman perlu memperoleh dataset/bobot secara terpisah.

Uji integrasi video membandingkan pretrained dan kandidat dengan MobileNet dimatikan, filter kotak aktif, dan ByteTrack `fuse_score: true`. Klip `091928` diproses penuh; masing-masing video WhatsApp diuji pada 15 detik awal. Hitungan bertambah pada sebagian run, tetapi belum ada GT lengkap untuk menyatakan crossing tambahan benar. Dua contoh MPV yang diminta juga belum berubah menjadi mobil pada prediksi full-frame kandidat tanpa MobileNet. Gunakan model utama dan profil pemeriksa MPV untuk demo saat ini. Eksperimen selanjutnya membutuhkan lebih banyak anotasi full-frame yang ditinjau manusia, contoh bus, dan koleksi uji baru.

Artefak lokal: `training/`, `training-retained/`, `dataset/provenance.json`, `detection-evaluation.json`, `development-cases.json`, dan `video-comparison.json` di `outputs/fine-tuning-20261119/`. Test ini sudah dibuka untuk diagnosis; hasil eksperimen berikutnya tidak boleh dianggap evaluasi final independen pada split yang sama.

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
docs/project-design.md          Rancangan teknis dan kontrak data
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
