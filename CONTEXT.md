# CONTEXT — Penghitung Kendaraan dari Video

Versi dokumen: 0.3 | Dibuat: 6 November 2026 | Diperbarui: 8 November 2026 | Bahasa antarmuka dan dokumentasi: Indonesia.

Dokumen ini adalah acuan bersama pemilik proyek, rekan satu tim, dan AI agent yang membantu pengerjaan. Baca sebelum merancang, menulis kode, menguji, atau mengubah lingkup proyek. Isinya mencakup tujuan, istilah, spesifikasi perilaku, rancangan teknis awal, pembagian pekerjaan, dan kriteria selesai agar hasil dari agent yang berbeda dapat digabungkan.

## 1. Status dan cara menggunakan dokumen

**Yang sudah dipilih pengguna:** proyek akhir mata kuliah Pengolahan Citra Digital dan Visi Komputer dengan topik penghitung kendaraan. Proyek dikerjakan bersama teman dengan bantuan AI agent masing-masing.

**Usulan awal dokumen ini:** video unggahan, aplikasi lokal berbasis Python dan Streamlit, deteksi YOLO11n, tracking ByteTrack, satu garis penghitung dua arah, empat kelas kendaraan, serta eksperimen CLAHE. Pilihan ini menjadi baseline kerja bersama sampai pemilik proyek menggantinya; pilihan tersebut belum merupakan persyaratan resmi dosen.

**Status implementasi saat dokumen dibuat:** folder kerja belum berisi implementasi aplikasi. Struktur direktori, kontrak modul, parameter, dan target di bawah adalah rancangan, bukan laporan fitur yang sudah tersedia atau hasil pengujian.

**Status terbaru:** pipeline video, panel bukti, antarmuka Streamlit, dan evaluator crossing tersedia. Lingkungan yang diuji adalah Python 3.12 di Windows dengan CPU PyTorch. Hasil awal dua klip jalan dicatat di `docs/counting-evaluation.md`. Profil ByteTrack `fuse_score: false` tersedia untuk eksperimen pada klip terpisah. Peninjau kedua serta evaluasi akhir dengan cakupan kelas dan kondisi yang lebih beragam masih diperlukan.

Sebelum mengerjakan suatu tugas, agent harus:

1. Membaca dokumen ini dan memeriksa kode, instruksi repositori, serta perubahan lokal yang sudah ada.
2. Menyebutkan bagian yang akan dikerjakan dan berkas yang menjadi tanggung jawabnya.
3. Mengikuti istilah serta kontrak data di sini. Jika kode menyimpang, jelaskan perbedaannya sebelum memperluas perubahan.
4. Mengerjakan tugas yang diberikan beserta pemeriksaan yang relevan; laporkan hasil aktual dan batasannya.
5. Memperbarui bagian dokumen yang terdampak jika ada perubahan rancangan bersama. Catat keputusan di bagian 21.

Instruksi terbaru dari pemilik proyek mengungguli dokumen ini. Keputusan kecil di dalam modul boleh dibuat agent; perubahan lingkup, makna hitungan, model/tracker, kontrak bersama, dan ukuran keberhasilan perlu diselaraskan dengan tim terlebih dahulu. Fakta yang belum diketahui tetap ditulis sebagai asumsi atau hal yang perlu dikonfirmasi.

## 2. Identitas dan tujuan proyek

**Judul kerja:** Sistem Penghitung Kendaraan Dua Arah Berdasarkan Jenis Menggunakan YOLO dan Multi-Object Tracking.

**Masalah:** menghitung kendaraan secara manual dari rekaman jalan membutuhkan waktu, terutama ketika harus dipisahkan berdasarkan jenis dan arah perjalanan.

**Solusi:** pengguna mengunggah video jalan, menentukan sebuah garis virtual, lalu aplikasi mendeteksi dan melacak kendaraan. Hitungan bertambah saat lintasan kendaraan memenuhi aturan penyeberangan garis. Hasil ditampilkan pada video beranotasi, ringkasan hitungan, dan berkas ekspor yang dapat diperiksa.

**Tujuan utama:** menghasilkan hitungan kendaraan yang melewati lokasi pengamatan per jenis dan arah, serta mengukur kesalahannya terhadap anotasi manual.

**Tujuan pembelajaran:** menjelaskan perbedaan deteksi, tracking, dan counting; menerapkan pemrosesan citra yang relevan; serta mengevaluasi pengaruh prapemrosesan terhadap hasil sistem.

Kontribusi tim berada pada integrasi pipeline, aturan crossing, antarmuka, eksperimen, evaluasi, dan analisis kegagalan. Jika menggunakan bobot pretrained, sebutkan secara jelas bahwa tim menggunakan model yang sudah dilatih pihak lain.

## 3. Istilah bersama

| Istilah | Definisi dan pemakaian |
| --- | --- |
| Video sumber | Rekaman yang dianalisis dalam satu run. |
| Frame | Satu citra dari video; `frame_index` dimulai dari 0. |
| Detection | Bounding box, kelas, dan confidence hasil model pada satu frame; belum menyatakan identitas antarframe. |
| Track | Rangkaian observasi yang diberi identitas sama oleh tracker selama satu run. |
| Track ID | ID lokal dari tracker; bukan nomor kendaraan, pelat, atau identitas permanen. |
| Track observation | Data posisi dan kelas suatu track pada frame tertentu. |
| Anchor | Titik tengah bawah bounding box yang digunakan untuk menentukan crossing. |
| Counting line | Segmen garis virtual berhingga dari P1 ke P2. Bukan seluruh perpanjangan garis. |
| Sisi A / sisi B | Dua sisi geometris garis sesuai rumus bagian 7. |
| Direction | `A_TO_B` atau `B_TO_A`, sesuai perpindahan sisi. |
| Crossing event | Satu kejadian penyeberangan sah yang dicatat ke event log. |
| Count | Agregasi crossing event; berbeda dari jumlah box pada frame. |
| Run | Satu eksekusi analisis dengan video dan konfigurasi tetap, diidentifikasi oleh `run_id`. |
| Ground truth / GT | Anotasi manual yang menjadi pembanding hasil sistem. |
| Pasangan otomatis | Dua event dengan kelas, arah, dan waktu yang memenuhi toleransi evaluator; identitas kendaraan belum terverifikasi. |
| Crossing terverifikasi | Event yang identitas kendaraannya sudah dicocokkan lewat tinjauan visual dengan anotasi. |
| Tanda audit | Kandidat masalah yang perlu ditinjau, misalnya anchor berada dalam kotak deteksi lain; tanda bukan bukti event salah dan ketiadaan tanda bukan bukti benar. |
| Baseline | Konfigurasi pembanding awal dengan prapemrosesan `none`. |
| Klip pengembangan | Rekaman untuk memilih parameter dan konfigurasi sebelum uji akhir. |
| Klip uji akhir | Rekaman dengan GT manual yang dievaluasi setelah parameter dibekukan. |
| Dataset fine-tuning | Frame dan bounding box berkelas untuk pelatihan tambahan jika evaluasi baseline menunjukkan kebutuhan; belum menjadi prasyarat proyek. |
| CLAHE | Mode peningkatan kontras lokal untuk eksperimen pembanding. |
| MVP | Versi minimum yang memenuhi fitur wajib pada bagian 5. |

Gunakan istilah `crossing event` untuk kejadian lewat garis, bukan `detection`. Gunakan arah A/B, bukan masuk/keluar, sampai konteks lokasi benar-benar menentukan makna masuk/keluar.

## 4. Lingkup pengamatan dan asumsi

- Satu video dari satu kamera diam pada satu run; jalan terlihat cukup jelas.
- Prioritas pengujian awal: siang atau pencahayaan cukup, objek tidak terlalu kecil, serta kepadatan ringan–sedang.
- Video demo disarankan berdurasi 30–120 detik dengan resolusi hingga 1280 × 720 agar mudah diulang. Ini saran, bukan klaim batas kemampuan.
- MVP memakai video constant frame rate yang FPS-nya valid. Variable frame rate perlu dikonversi dahulu atau kelak didukung dengan timestamp media yang benar.
- Semua frame diproses berurutan untuk menjaga kontinuitas tracking. Pembaruan preview UI boleh lebih jarang tanpa melewati analisis frame.
- Kelas dihitung: `car`, `motorcycle`, `bus`, dan `truck`.
- Sepeda, pejalan kaki, serta objek lain tidak masuk hitungan. Kendaraan lokal seperti angkot atau pikap mengikuti kelas model; klasifikasi khusus memerlukan data dan lingkup tambahan.
- Sistem berjalan lokal. CPU harus dapat menjalankan baseline; GPU bersifat akselerator opsional. Kecepatan ditentukan melalui pengukuran pada perangkat tim.
- Internet dapat dibutuhkan saat pemasangan dependensi dan pengunduhan bobot. Demo lokal sesudah penyiapan memakai bobot yang sudah tersedia.

## 5. Fitur wajib dan batas pengembangan

### Fitur MVP

1. Mengunggah satu video MP4, memvalidasi bahwa video dapat dibaca, dan menampilkan informasi dasarnya.
2. Menampilkan preview frame awal untuk mengatur P1 dan P2 melalui kontrol koordinat ternormalisasi; garis serta label sisi A/B terlihat sebelum proses dimulai. Drag-and-drop garis adalah peningkatan opsional.
3. Memilih mode `none` atau `clahe` dan mengatur parameter yang disediakan aplikasi sebelum run dimulai.
4. Mendeteksi empat kelas kendaraan dan mempertahankan ID menggunakan tracker.
5. Menghitung crossing satu garis untuk dua arah sesuai bagian 7.
6. Menampilkan box, kelas, ID, anchor, garis, lintasan pendek, dan hitungan pada preview/video hasil.
7. Menampilkan jumlah per kelas dan arah serta total keseluruhan; grafik batang sederhana mendukung pembacaan ringkasan.
8. Mengekspor `events.csv`, `counts.csv`, `run.json`, dan video beranotasi `annotated.mp4` untuk run yang sukses.
9. Menampilkan progres dan pesan kesalahan yang membantu pengguna memperbaiki input atau konfigurasi.
10. Memiliki pengujian logika counting dan evaluasi terhadap anotasi manual.

Eksperimen `none` versus `clahe` dan pembahasan hasilnya termasuk keluaran akademik proyek. Implementasi dimulai dari baseline yang benar, kemudian menambahkan mode eksperimen.

### Pengembangan opsional setelah MVP

- Sumber webcam/RTSP, ROI, banyak garis, dan banyak kamera.
- Perbandingan ByteTrack dengan DeepSORT atau tracker lain.
- Pelatihan/fine-tuning untuk kelas kendaraan lokal.
- Gamma correction atau filter spasial sebagai eksperimen tambahan yang terukur.
- Ekspor grafik, analisis per interval waktu, dan kontrol gambar garis yang interaktif.

### Di luar lingkup awal

Pengenalan pelat, identifikasi pemilik kendaraan, pelacakan lintas kamera, deteksi pelanggaran, estimasi kecepatan dalam km/jam, segmentasi jalan, database pengguna, login, deployment publik, dan aplikasi seluler. Kecepatan fisik membutuhkan kalibrasi jarak/perspektif; panjang lintasan piksel saja tidak cukup.

## 6. Kaitan dengan materi kuliah

| Materi | Penerapan | Bukti untuk laporan/demo |
| --- | --- | --- |
| 1. Representasi citra | Frame, resolusi, koordinat piksel, kanal warna | Penjelasan format frame dan konsistensi koordinat |
| 2. Konversi ruang warna | BGR–LAB–BGR untuk CLAHE dan BGR–grayscale pada praktikum | Panel grayscale dan hasil CLAHE |
| 3. Transformasi intensitas | Gamma correction sebagai pilihan input YOLO | Panel perbandingan mode dan konfigurasi run |
| 4. Histogram dan peningkatan kualitas | CLAHE pada kanal luminansi | Perbandingan visual, histogram sampel, dan metrik counting |
| 5–6. Filtering spasial | Gaussian smoothing dan unsharp masking sebagai pilihan input YOLO | Panel perbandingan mode dan konfigurasi run |
| 9–10. Tepi dan thresholding | Canny pada grayscale yang dihaluskan, serta threshold tetap dan Otsu pada selisih antarframe | Panel tepi dan mask biner |
| 11–12. Morfologi dan CCL | Opening dan closing mask Otsu, lalu pelabelan komponen dengan konektivitas 8 | Panel mask dan kotak kandidat, luas, serta centroid pada `lab.json` |
| 15. Optical Flow | Lucas–Kanade untuk perpindahan titik sudut antara dua frame berurutan | Panel vektor gerak pada frame target |
| 16. Tracking objek modern | ByteTrack sebagai pilihan utama dan DeepSORT sebagai eksperimen opsional | ID dan lintasan antarframe; analisis ID switch dan perbandingan tracker |
| 17–18. Deteksi deep learning dan YOLO | Detektor pretrained untuk jenis kendaraan | Box, kelas, confidence, serta penjelasan inferensi |

ByteTrack dan DeepSORT adalah dua metode berbeda. Atas izin pengguna pada 8 November 2026, DeepSORT tersedia sebagai eksperimen dengan fitur tampilan MobileNetV2 pretrained ImageNet, belum dilatih khusus identitas kendaraan. ByteTrack `fuse_score: true` tetap pilihan utama; parameter itu tidak berlaku untuk DeepSORT. Hasil eksperimen mengikuti [laporan perbandingan tracker](docs/tracker-safety-evaluation.md).

`pcd-lab` membuat panel dari frame N−1 dan N. Selisih intensitas menunjukkan area perubahan, dan CCL menghasilkan kandidat area gerak yang dapat terpecah atau tergabung. Optical Flow memperkirakan perpindahan titik citra. `analyze` tetap memakai deteksi YOLO dan asosiasi ByteTrack sebagai sumber event crossing.

## 7. Aturan counting yang menjadi acuan bersama

### 7.1 Koordinat dan garis

- Koordinat citra asli: titik asal di kiri atas, x ke kanan, y ke bawah.
- Bounding box memakai `xyxy = [x1, y1, x2, y2]` dalam piksel citra asli.
- Anchor: `p = ((x1 + x2) / 2, y2)`.
- Konfigurasi garis menyimpan P1 dan P2 sebagai `[x, y]` pada rentang `[0, 1]`. Konversi ke piksel memakai `x * (width - 1)` dan `y * (height - 1)`.
- Untuk `v = P2 - P1`, hitung signed distance dalam piksel:

```text
d(p) = (v.x * (p.y - P1.y) - v.y * (p.x - P1.x)) / length(v)
```

- Dengan lebar toleransi `epsilon_px`: sisi A jika `d < -epsilon_px`, sisi B jika `d > epsilon_px`, dan zona netral jika `abs(d) <= epsilon_px`.
- Zona netral mengurangi hitungan akibat jitter di sekitar garis. Nilainya dihitung dari fraksi tinggi frame pada konfigurasi.
- P1 dan P2 harus berbeda. Urutan keduanya menentukan tanda sisi; menukarnya juga menukar makna arah. Preview harus memperlihatkan hal ini.
- Contoh: garis horizontal dari kiri ke kanan membuat sisi A di atas dan B di bawah. Kendaraan turun melewati garis menghasilkan `A_TO_B`.

### 7.2 Syarat sebuah crossing event

Event diterbitkan hanya jika seluruh kondisi berikut terpenuhi:

1. Observasi memiliki Track ID dan kelas kendaraan yang termasuk lingkup.
2. Track sudah memiliki sedikitnya `min_track_hits` observasi valid sejak inisialisasi riwayat geraknya.
3. Ada observasi sisi stabil sebelumnya dan observasi terbaru pada sisi stabil yang berlawanan. Stabil di sini berarti berada di luar zona netral.
4. Lintasan anchor yang teramati sejak sisi stabil terakhir berpotongan dengan segmen garis P1–P2, termasuk kedua ujungnya. Melintasi perpanjangan di luar segmen tidak memenuhi syarat.
5. Jeda antarobservasi sepanjang transisi tidak melebihi `max_crossing_gap_seconds`.
6. Kunci `(run_id, line_id, track_id)` belum pernah menghasilkan event.

Observasi dalam zona netral menjaga sisi stabil sebelumnya. Ketika kembali ke sisi stabil yang sama, perbarui titik acuan dan hapus kandidat crossing yang belum selesai. Ketika mencapai sisi berlawanan, selesaikan pemeriksaan crossing lalu jadikan sisi tersebut acuan berikutnya, baik event diterima maupun ditolak.

Perpotongan dapat diakumulasi sebagai flag dari segmen antarobservasi selama transisi; tidak perlu menyimpan lintasan tak terbatas ketika kendaraan berhenti pada garis. Sampel track prediksi tanpa observasi deteksi yang diasosiasikan tidak dipakai untuk menerbitkan event.

### 7.3 Kebijakan satu kali hitung

- Satu Track ID hanya dihitung sekali pada satu garis selama satu run, terlepas dari arah. U-turn dengan ID yang sama tidak menambah event kedua pada MVP.
- Ini adalah hitungan crossing yang teramati, bukan jaminan jumlah kendaraan fisik unik. ID baru untuk kendaraan yang sama tetap berisiko menghasilkan duplikasi.
- Profil eksperimen dapat mengaktifkan `deduplicate_crossings`. Kandidat dengan ID baru perlu cocok arah, tumpang tindih kotak, dan tampilan dengan crossing baru-baru ini sebelum dibuang. Riwayat pembanding berakhir setelah 15 detik atau kendaraan lama terlihat cukup jauh dari garis.
- Count bersumber dari event log. `total = jumlah event`, dan jumlah seluruh bucket kelas–arah harus sama dengan total.
- Kelas event dipilih dari kelas yang paling sering muncul pada observasi valid track hingga event. Jika seri, gunakan jumlah confidence terbesar; jika masih seri, urutan alfabet nama kelas. Kelas event dibekukan setelah dicatat.
- Profil eksperimen dapat mengisi `truck_review_weights` untuk memeriksa ulang crop kandidat `truck` dengan MobileNetV3-Large. Bukti kendaraan ringan yang cukup dapat mengubah kelas final menjadi `car` sebelum event dicatat.
- Detection tanpa ID boleh ditampilkan dengan label yang jelas, tetapi belum menambah count.

### 7.4 Kasus khusus

| Kejadian | Perilaku yang diharapkan |
| --- | --- |
| Kendaraan baru muncul di sisi A lalu tetap di A | Tidak ada event |
| Kendaraan pertama terlihat di zona netral | Tunggu sisi stabil awal, lalu perpindahan yang sah |
| Menyentuh garis lalu kembali ke sisi asal | Tidak ada event |
| Bergerak sejajar atau berhenti pada garis | Tidak ada event tambahan |
| Anchor berpindah sisi tetapi di luar ujung garis | Tidak ada event |
| Perpindahan A–netral–B yang sah | Satu event `A_TO_B` |
| Track yang sudah dihitung bergetar atau berbalik | Count tetap |
| Observasi hilang lebih lama dari batas gap | Hapus riwayat transisi dan hit gerak, mulai acuan sisi baru; pertahankan catatan sudah dihitung |
| Tracker mengganti ID di dekat garis | Catat sebagai keterbatasan; evaluasi duplikasi atau event terlewat |
| Kelas berubah setelah crossing | Event lama dan ringkasan tidak berubah |
| Video diproses ulang | Buat run baru dan reset seluruh state tracker/counter |
| Tidak ada kendaraan | Ekspor tetap valid dengan event kosong dan delapan bucket count bernilai nol |

Timestamp event adalah waktu video pada frame pertama yang memenuhi syarat event, bukan perkiraan presisi saat roda melintasi garis. Untuk video CFR: `timestamp_seconds = frame_index / source_fps`.

## 8. Alur pemrosesan

```text
Unggah video dan validasi metadata
    -> preview dan konfigurasi garis
    -> mulai run dengan konfigurasi tetap
    -> baca frame secara berurutan
    -> prapemrosesan none / clahe / gamma / gaussian / unsharp
    -> deteksi YOLO + asosiasi tracking ByteTrack
    -> normalisasi hasil menjadi TrackObservation
    -> evaluasi crossing dan catat CountEvent
    -> agregasi ringkasan dan gambar anotasi
    -> tulis video/output secara bertahap
    -> finalisasi, tampilkan, dan unduh hasil
```

Prapemrosesan `none` mempertahankan isi frame untuk baseline. `clahe` bekerja pada kanal L di LAB. `gamma` mengubah intensitas dengan pangkat gamma, `gaussian` menghaluskan citra, dan `unsharp` menambahkan selisih antara citra asli dan citra yang dihaluskan. Setiap run memilih satu mode. Anotasi tetap digambar di frame sumber agar hasil dapat dibandingkan.

Konversi RGB untuk komponen tampilan dilakukan pada batas UI. Kontrak internal frame OpenCV tetap BGR. Koordinat hasil inferensi harus dipetakan ke citra asli sebelum masuk counter dan renderer.

Satu adapter boleh memakai API tracking Ultralytics yang sekaligus menjalankan deteksi dan asosiasi. Hindari dua inferensi model pada frame yang sama hanya untuk memisahkan nama modul. Counter menerima data biasa dan dapat diuji tanpa model, GPU, maupun Streamlit.

## 9. Teknologi dan konfigurasi awal

| Komponen | Pilihan baseline | Alasan rancangan |
| --- | --- | --- |
| Bahasa | Python 3.12 pada lingkungan fondasi yang diuji | Satu bahasa untuk CV, evaluasi, dan UI; Python 3.11 adalah usulan awal |
| Video dan citra | OpenCV, NumPy | Pembacaan frame, konversi warna, CLAHE, anotasi |
| Model | Ultralytics YOLO11n, bobot `yolo11n.pt` | Model kecil pretrained untuk memulai pengujian lokal |
| Tracker | ByteTrack, `bytetrack.yaml` dipilih eksplisit | Baseline asosiasi untuk kamera diam; hasil ID diuji pada data tim |
| UI | Streamlit | Unggah video, kontrol parameter, preview, dan unduh hasil |
| Data dan grafik | pandas dan grafik bawaan Streamlit | CSV dan ringkasan sederhana |
| Pengujian | pytest | Logika geometri, state counting, dan konsistensi ekspor |

Model dan tracker dipilih eksplisit, bukan mengikuti default library yang dapat berubah. DeepSORT merupakan dependensi opsional melalui ekstra `deepsort` dan `configs/deepsort.yaml`, bukan dependensi wajib MVP. Training dari nol dan GPU bukan prasyarat. Kedua tracker meneruskan hanya deteksi aktual yang terasosiasi; prediksi posisi tanpa deteksi pada frame itu tidak boleh membuat event.

Berikut bentuk konfigurasi bersama yang diusulkan; nilai awal adalah titik mulai eksperimen, bukan parameter optimal:

```yaml
model_weights: yolo11n.pt
tracker_config: bytetrack.yaml
device: cpu
imgsz: 640
detector_conf: 0.10
vehicle_classes: [car, motorcycle, bus, truck]
preprocess:
  mode: none
  clahe_clip_limit: 2.0
  clahe_tile_grid_size: [8, 8]
  gamma: 0.75
  gaussian_kernel_size: 5
  unsharp_amount: 0.7
counting_line:
  line_id: main
  p1: [0.10, 0.60]
  p2: [0.90, 0.60]
  epsilon_height_fraction: 0.01
min_track_hits: 3
max_crossing_gap_seconds: 0.5
```

`detector_conf` adalah ambang kandidat masuk tracker, bukan jaminan kualitas setiap track. Nilai rendah membantu mempertahankan kandidat untuk asosiasi; ambang internal dan buffer ByteTrack perlu diperiksa bersama agar deteksi berconfidence rendah tidak disaring terlalu awal. Simpan isi konfigurasi tracker yang benar-benar dipakai, bukan hanya nama berkasnya.

Petakan kelas melalui nama kelas model dan validasi keempat nama tersedia; hindari asumsi ID numerik berlaku untuk semua model. Pilihan perangkat GPU harus menyebut perangkat aktual pada metadata. Jika GPU tidak tersedia, tawarkan CPU atau tampilkan alasan kegagalan secara jelas.

Pada implementasi pertama, uji satu lingkungan yang bekerja lalu pin versi dependensi serta catat versi Python, library, bobot/checksum, perangkat, dan konfigurasi dalam metadata. Jangan mengarang versi yang sudah diuji. Uji pula encoding MP4 yang dapat diputar di browser; kebutuhan FFmpeg/codec tambahan ditulis di README jika benar-benar dipakai.

## 10. Arsitektur dan batas tanggung jawab modul

Struktur berikut adalah usulan untuk dibuat bertahap sesuai tugas:

```text
project/
├── CONTEXT.md
├── AGENTS.md
├── README.md
├── LICENSE
├── .gitignore
├── .gitattributes
├── pyproject.toml
├── requirements.txt
├── requirements-tracking.txt
├── constraints.txt
├── app.py
├── configs/default.yaml
├── src/vehicle_counter/
│   ├── contracts.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── tracking.py
│   ├── geometry.py
│   ├── counting.py
│   ├── simulation.py
│   ├── video.py
│   ├── cli.py
│   ├── pipeline.py
│   ├── visualization.py
│   ├── exporting.py
│   └── evaluation.py
├── tests/
├── data/videos/
├── data/annotations/
├── models/
├── outputs/
└── docs/experiments/
```

Struktur ini mencakup rencana akhir; berkas yang benar-benar tersedia dirangkum di README. Definisi dependensi berada di `pyproject.toml`; `requirements.txt` memanggil instalasi editable dengan ekstra tahap fondasi, dan `constraints.txt` merekam versi dependensi yang diuji.

`tracking.py` memiliki lifecycle model/tracker per run dan mengubah keluaran library menjadi kontrak internal. `geometry.py` dan `counting.py` berisi aturan domain. `pipeline.py` mengatur pembacaan dan pemrosesan video. `app.py` mengelola interaksi pengguna; rumus hitungan berada di counter. `exporting.py` menulis hasil dari event yang sama dengan yang dipakai UI. `evaluation.py` membandingkan hasil terhadap GT.

Video besar, bobot model, lingkungan virtual, dan hasil run tidak masuk version control secara default. Simpan cara mendapatkan data/bobot dan metadata yang diperlukan untuk mengulang percobaan.

## 11. Kontrak data untuk integrasi antaragent

Nama field berikut menjadi acuan implementasi `contracts.py`; gunakan tipe data terstruktur seperti dataclass. Perubahan field harus diselaraskan sebelum kedua bagian digabungkan.

| Kontrak | Field wajib dan makna |
| --- | --- |
| `VideoMetadata` | `source_name: str`, `width: int`, `height: int`, `source_fps: float`, `frame_count: int \| null`, `duration_seconds: float \| null` |
| `RunConfig` | Field pada konfigurasi bagian 9; setelah run dimulai isinya tetap |
| `TrackObservation` | `frame_index: int`, `timestamp_seconds: float`, `track_id: int`, `class_name: str`, `confidence: float`, `bbox_xyxy: tuple[float, float, float, float]` |
| `CountEvent` | `run_id: str`, `line_id: str`, `track_id: int`, `frame_index: int`, `timestamp_seconds: float`, `class_name: str`, `direction: str` |
| `CountSummary` | `total: int`, `by_class_direction: dict[str, dict[str, int]]` dengan empat kelas × dua arah, termasuk bucket nol |
| `ProgressUpdate` | `run_id: str`, `processed_frames: int`, `total_frames: int \| null`, `elapsed_seconds: float`, `summary: CountSummary`, `preview_frame_bgr: ndarray \| null` |
| `RunResult` | `run_id: str`, `status: str`, `summary: CountSummary`, `output_dir: str`, `artifact_paths: dict[str, str]`, `warnings: list[str]` |

`TrackObservation` mewakili hasil track yang didukung observasi aktual; keluaran tanpa ID ditangani di adapter/renderer. Semua kontrak memakai koordinat frame asli, nama kelas kanonis, timestamp detik video, dan frame index 0-based.

Titik integrasi yang disepakati dalam rancangan:

```text
process_video(source_path, config, output_root, on_progress=None) -> RunResult
counter.update(observations_for_frame) -> list[CountEvent]
counter.summary() -> CountSummary
```

Counter dibangun dengan `VehicleCounter(run_id, config, width=..., height=...)`. `counter.events` memberikan tuple event. `update` menerima satu batch observasi per frame; ID dalam batch harus unik, frame/timestamp sama, dan batch nonkosong harus meningkat terhadap sebelumnya. Batch kosong diperbolehkan. Untuk reset run, buat instance baru. `process_video(...)` sudah tersedia di `pipeline.py`; keyword tambahan `tracker_factory` adalah seam pengujian. UI menggunakan default YOLO/ByteTrack, bukan fixture.

Adapter menggunakan satu `YOLO.predict` per frame lalu `BYTETracker.update` eksplisit. Nama kelas dipetakan melalui model. Kotak `TrackObservation` diambil dari deteksi yang cocok dengan ID pada frame tersebut berdasarkan detection index, bukan posisi prediksi Kalman tanpa observasi. Tracker diperbarui juga ketika deteksi kosong. Isi konfigurasi ByteTrack aktual dan unit track_buffer (frame) masuk metadata.

`process_video` bersifat sinkron untuk baseline. Callback progres membawa snapshot yang aman dibaca UI, bukan objek tracker yang dapat dimutasi. `preview_frame_bgr` adalah salinan frame beranotasi atau `null` ketika pembaruan tidak membawa gambar; UI mengonversinya ke RGB untuk ditampilkan. Pipeline boleh membatasi frekuensi callback/preview agar tampilan tidak memperlambat analisis. Pengaturan widget baru berlaku pada run berikutnya. Pemrosesan model dilakukan sekali per run; interaksi UI setelah selesai memakai hasil tersimpan, bukan menjalankan video ulang tanpa perintah pengguna.

## 12. Format keluaran dan lifecycle run

Setiap run memakai folder unik `outputs/<run_id>/`. Berkas video sumber tetap utuh. Nama file unggahan untuk tampilan boleh dipertahankan, sedangkan path kerja dibuat aplikasi agar tidak bergantung pada nama dari pengguna.

**`events.csv`:** satu baris per crossing event, dengan urutan kolom:

```text
run_id,line_id,track_id,frame_index,timestamp_seconds,class_name,direction
```

Contoh ilustratif, bukan hasil pengujian:

```csv
run_id,line_id,track_id,frame_index,timestamp_seconds,class_name,direction
demo-001,main,12,150,5.0,motorcycle,A_TO_B
```

**`counts.csv`:** delapan baris untuk kombinasi empat kelas dan dua arah, dengan header `run_id,line_id,class_name,direction,count`. Total diperoleh dari penjumlahan baris; jangan menambah baris total yang ikut terjumlah sebagai kendaraan.

**`run.json`:** mencatat status, sumber/checksum, metadata video, konfigurasi lengkap termasuk tracker, versi perangkat lunak dan bobot, perangkat pemrosesan, jumlah frame aktual, total waktu, processing FPS, ringkasan, peringatan, serta path artefak. Informasi yang tidak tersedia ditulis `null` beserta alasan bila memengaruhi interpretasi.

**`annotated.mp4`:** video sumber dengan overlay, dimensi asli, dan FPS sumber. Audio tidak dipertahankan pada MVP; beri keterangan di UI/README. Video hasil menampilkan isi frame sumber, termasuk ketika inferensi memakai CLAHE.

**`process_evidence.png`:** panel satu frame yang menampilkan sumber BGR, kanal L dari Lab, CLAHE pada L jika aktif, masukan aktual model, deteksi YOLO sebelum asosiasi, dan track ByteTrack beserta ID dan jejak. Frame crossing pertama diprioritaskan. Metadata `process_evidence` pada `run.json` mencatat frame, waktu, alasan pemilihan, status CLAHE, dan jumlah deteksi serta track. Kunci artefaknya juga `process_evidence`.

Status run: `running`, `completed`, atau `failed`. Tulis `completed` setelah semua keluaran wajib selesai ditulis dan dapat dibaca. Jika pembacaan berhenti tidak wajar atau writer gagal, simpan status `failed` dan penjelasan; hasil parsial tidak disajikan sebagai hitungan akhir yang lengkap.

Reader, writer, dan berkas sementara harus ditutup setelah sukses maupun gagal. Video diproses dan ditulis bertahap, tanpa menyimpan seluruh frame di memori. Batas ukuran unggahan dibuat jelas dan dapat dikonfigurasi; validasi mencakup decoder, bukan hanya ekstensi file.

Implementasi CLI sekarang menerima MP4 lokal CFR dengan dimensi genap untuk H.264/yuv420p. Verifikasi output mendekode seluruh frame dan memeriksa dimensi, FPS, serta jumlahnya. Pemeriksaan input membaca frame awal dan membandingkan jumlah decode dengan metadata jika tersedia; VFR harus dikonversi oleh pengguna, belum dideteksi secara menyeluruh. Error sebelum pembuatan run menaikkan exception; error sesudahnya menghasilkan RunResult `failed`, metadata error, dan CSV parsial yang diberi status tersebut. Artefak sukses memakai kunci `run_json`, `events_csv`, `counts_csv`, `annotated_video`, dan `process_evidence`.

## 13. Alur antarmuka pengguna

1. Halaman awal menjelaskan kegunaan aplikasi dan asumsi kamera diam.
2. Pengguna mengunggah MP4; aplikasi menampilkan nama, resolusi, FPS, durasi jika tersedia, dan preview awal.
3. Pengguna mengatur garis dan memeriksa label A/B. Kontrol lanjutan menyediakan mode prapemrosesan dan parameter yang relevan.
4. Pengguna menekan **Mulai analisis**. Aplikasi membuat run baru dan mengunci konfigurasi run tersebut.
5. Selama analisis, tampilkan status, jumlah frame/progres jika diketahui, preview berkala, dan ringkasan sementara yang diberi label sementara.
6. Setelah sukses, tampilkan video hasil, total kendaraan, tabel kelas–arah, grafik batang, serta tombol unduh keluaran.
7. Untuk percobaan lain, pengguna memulai run baru; hasil run sebelumnya tetap terpisah.

Contoh pesan yang harus informatif: video tidak dapat dibaca, FPS tidak valid, garis bertitik sama, bobot belum tersedia, perangkat GPU tidak tersedia, atau video keluaran gagal ditulis. Tampilan untuk pengguna menjelaskan tindakan perbaikan; detail exception dapat dicatat untuk diagnosis.

## 14. Data dan anotasi manual

Gunakan rekaman sendiri atau sumber video yang mengizinkan penggunaan untuk tugas akademik. Catat asal, URL bila ada, tanggal pengambilan/akses, izin/lisensi yang diketahui, resolusi, FPS, durasi, dan kondisi pengamatan.

Pisahkan klip pengembangan untuk penyetelan parameter dari klip uji akhir. Hindari memakai dua potongan dari rentang video yang sama pada kedua kelompok; usahakan variasi rekaman atau waktu pengambilan. Parameter yang dipilih pada data pengembangan dibekukan sebelum evaluasi akhir.

Set awal yang disarankan: setidaknya satu klip pengembangan dan tiga klip uji, termasuk satu klip tanpa crossing. Usahakan total sedikitnya 30 crossing manual pada klip uji yang berisi kendaraan. Klip malam/padat menjadi uji tantangan tambahan dan dilaporkan terpisah dari kondisi utama.

GT event memakai `clip_id,manual_event_id,frame_index,timestamp_seconds,class_name,direction,notes`. `manual_event_id` bukan Track ID hasil sistem. Simpan konfigurasi garis per klip agar penilaian manual dan program menggunakan lokasi serta arah yang sama.

Kedua anggota tim menghitung atau meninjau crossing pada video lambat, menyelesaikan perbedaan, lalu membekukan GT. Gunakan anchor tengah bawah box secara konseptual agar definisi lewat garis konsisten. Tandai kejadian yang ambigu/tertutup berat sebelum melihat hasil akhir model; jelaskan aturan penyertaannya. Untuk U-turn, anotasi manual mengikuti kebijakan satu kali per kendaraan per garis selama klip, sejauh identitas dapat ditinjau manusia.

## 15. Evaluasi dan eksperimen

### 15.1 Metrik counting utama

Untuk setiap klip i, kelas k, dan arah d, definisikan `P_i,k,d` sebagai count prediksi dan `G_i,k,d` sebagai count manual. Laporkan tabel lengkap, termasuk bucket nol.

```text
AE_i,k,d = abs(P_i,k,d - G_i,k,d)
MAE_bucket = sum(AE_i,k,d) / (jumlah_klip * 4 * 2)
MAE_total = mean_i(abs(total_prediksi_i - total_GT_i))
E_count = 100 * sum(AE_i,k,d) / sum(G_i,k,d)
```

`E_count` adalah persentase error agregat berbasis bucket, bukan akurasi deteksi. Jika seluruh GT nol, `E_count` tidak terdefinisi; laporkan jumlah false count. Error ini bisa melebihi 100%. Tampilkan pula selisih total karena kesalahan kelas/arah dapat saling menutupi jika hanya melihat jumlah keseluruhan.

Untuk menilai kejadian, bukan hanya kecocokan total, cocokkan event prediksi dan manual secara satu-ke-satu per klip, kelas, dan arah, dengan selisih waktu maksimal 0,5 detik sebagai toleransi awal. Pilih pencocokan dengan jumlah pasangan maksimum, lalu jumlah selisih waktu minimum. Event berlebih menjadi FP, event manual tanpa pasangan menjadi FN. Laporkan precision, recall, dan F1 dengan penanganan denominator nol yang dijelaskan. Metrik ini membantu menemukan kasus satu event terlewat tetapi event lain terhitung ganda.

Label FP/FN dari evaluator merupakan hasil pencocokan temporal, bukan audit identitas. Pada lalu lintas padat, event kendaraan lain bisa menjadi pasangan otomatis yang keliru. Metadata hasil menyatakan `matching_basis: class_direction_time` dan `identity_verified: false`. Laporkan crossing terverifikasi secara terpisah setelah melihat video sumber. `crossing_audit.csv` dan tab Audit crossing menyediakan koordinat dan frame event; tanda audit tidak mengubah event log atau hitungan. Artefak tambahan memakai kunci `crossing_audit`; kontrak `events.csv` tetap tujuh kolom. CSV audit berisi tujuh kolom event ditambah `x1,y1,x2,y2,anchor_x,anchor_y,overlapping_boxes,review_reason`. Jumlah barisnya sama dengan event pada run sukses, termasuk header tanpa baris untuk run nol crossing.

mAP memerlukan anotasi bounding box; IDF1/MOTA/HOTA memerlukan anotasi tracking yang sesuai. Metrik tersebut opsional jika tim benar-benar menyediakan anotasi dan evaluatornya. Confidence bukan akurasi dan bukan pengganti evaluasi.

### 15.2 Kecepatan dan keterulangan

- `processing_fps = processed_frames / processing_wall_seconds` untuk pipeline dari pembacaan frame sampai finalisasi output; unduh bobot dan unggah file di luar pengukuran ini.
- Catat waktu pemuatan model secara terpisah, perangkat, resolusi, versi, konfigurasi, serta durasi video.
- Pada CLI, inspeksi awal/checksum sumber berada di luar timer processing; model_load_seconds mencatat pembuatan detector/tracker. Timer processing mencakup loop, encoding/finalisasi video, verifikasi decode output, ekspor CSV, dan callback. total_wall_seconds mencakup penyiapan; penulisan metadata final berada sesudah pengukuran.
- Pemrosesan video unggahan tidak harus real-time. Sebutan real-time hanya dipakai jika pipeline terukur memenuhi kebutuhan FPS sumber pada perangkat yang dilaporkan.
- Jika mengulang benchmark, simpan semua hasil dan jelaskan ringkasannya; jangan memilih hanya run tercepat atau paling akurat.

### 15.3 Eksperimen wajib: none versus CLAHE

1. Jalankan `none` pada klip pengembangan sampai pipeline dan aturan counting terverifikasi.
2. Jalankan `clahe` pada klip yang sama dengan model, garis, tracker, dan parameter lain yang tetap.
3. Tentukan konfigurasi akhir memakai data pengembangan.
4. Jalankan kedua mode pada klip uji yang sama dengan tracker baru untuk setiap run.
5. Bandingkan error count, precision/recall/F1 event, processing FPS, serta frame/histogram sampel. Simpan run ID untuk setiap percobaan.
6. Bahas kondisi ketika CLAHE membantu, tidak berpengaruh, atau memperburuk hasil. Peningkatan kontras visual tidak otomatis memperbaiki deteksi.

### 15.4 Eksperimen asosiasi ByteTrack

Profil `configs/bytetrack-no-fuse.yaml` menonaktifkan penggabungan confidence dengan biaya asosiasi. Baseline tetap memakai `bytetrack.yaml`. Bandingkan kedua profil dengan video, YOLO, prapemrosesan, dan garis yang sama. Laporkan crossing yang pulih serta event ekstra setelah mengecek identitas kendaraan di frame sumber. Catat kasus dengan titik acuan tertutup secara terpisah. Lihat [evaluasi eksperimen ByteTrack](docs/bytetrack-fuse-score-evaluation.md).

Profil lanjutan `configs/experiments/vehicles-3-refined.yaml` memakai ByteTrack dengan `fuse_score: true`, pemeriksaan kelas, dan penolakan crossing ganda berbasis tampilan. Gunakan profil ini untuk eksperimen antrean kamera diam dan uji dengan klip lain sebelum mengaktifkannya pada profil demo.

Eksperimen deteksi pada klip antrean dapat memilih `detection_roi` sebagai `[x1, y1, x2, y2]` ternormalisasi. Potongan harus memuat seluruh segmen garis dan memberi ruang observasi di kedua sisi. Setelah deteksi, semua kotak dikembalikan ke koordinat frame sumber sebelum asosiasi dan counting. Profil hujan menguji kenaikan `imgsz` dengan detektor frame penuh. Kedua profil memakai pengaturan ByteTrack yang sama. Lihat [audit pemulihan deteksi dan ID](docs/bytetrack-holdout-evaluation.md#uji-pemulihan-deteksi-dan-id).

## 16. Pengujian dan kriteria selesai

### Pengujian perilaku yang wajib

Counter diuji dengan observasi sintetis agar tidak bergantung pada unduhan model:

- Crossing A–B dan B–A yang sah; arah dan bucket benar.
- Banyak observasi satu track menghasilkan paling banyak satu event.
- Track diam, jitter zona netral, serta menyentuh lalu kembali tidak menambah count.
- Perpindahan sisi di luar segmen tidak dihitung.
- Track pertama muncul di zona netral dan track yang belum cukup hit tidak dihitung prematur.
- Gap di atas batas mereset riwayat gerak, sedangkan catatan sudah dihitung tetap terjaga.
- Perubahan kelas menggunakan kebijakan voting dan tie-break yang ditentukan.
- Reset run menghapus state lama; ekspor dan summary cocok dengan event log.
- Frame tanpa observasi dan run tanpa kendaraan tetap menghasilkan keluaran valid.
- Validasi garis nol panjang, koordinat, video rusak, dan FPS tidak valid memberikan kegagalan yang jelas.

Integrasi minimal diuji pada satu video pendek nyata: inferensi/tracking berjalan, hasil CSV dapat dibaca, video beranotasi dapat diputar, dan hasil UI sama dengan ekspor. Pengujian unit saja belum membuktikan model menghitung kendaraan dengan benar.

### Kriteria selesai produk dan tugas akademik

- [ ] Instalasi dan cara menjalankan aplikasi tersedia di README dan berhasil diikuti dari lingkungan baru yang didukung.
- [ ] Seluruh fitur MVP bekerja pada video demo, termasuk run ulang dan input tanpa kendaraan.
- [x] Pengujian logika counting yang relevan lulus; suite integrasi 8 November 2026 mencakup 118 tes keseluruhan.
- [ ] Semua artefak run sukses valid dan totalnya konsisten.
- [ ] Evaluasi manual dan perbandingan none/CLAHE selesai pada data yang dicatat.
- [ ] Ada contoh kegagalan serta penjelasan keterbatasan, bukan hanya video terbaik.
- [ ] Kedua anggota dapat menjelaskan alur, garis/arah, ID, metrik, dan kontribusi masing-masing.

**Target kualitas awal, belum hasil:** usahakan `E_count <= 10%` dan F1 event setidaknya 0,90 pada klip uji kondisi utama yang memenuhi asumsi. Target ini harus ditinjau setelah uji awal dan disesuaikan dengan rubrik dosen; hasil aktual tetap dilaporkan walaupun target tidak tercapai. Kategori kendaraan yang belum muncul pada data uji dinyatakan belum teruji.

## 17. Pembagian kerja untuk dua anggota dan dua agent

| Area | Anggota/Agent A | Anggota/Agent B |
| --- | --- | --- |
| Fokus utama | Pipeline CV dan aturan counting | Antarmuka, keluaran, dan evaluasi |
| Berkas utama | `config.py`, `preprocessing.py`, `tracking.py`, `geometry.py`, `counting.py`, `pipeline.py` | `app.py`, `visualization.py`, `exporting.py`, `evaluation.py` |
| Pengujian | Geometri, state, adapter, smoke test pipeline | Ekspor, agregasi tampilan, evaluator, alur UI |
| Keluaran | Pipeline nyata yang menghasilkan event sesuai kontrak | UI tersambung ke pipeline, unduhan, tabel evaluasi |

Berkas bersama: `CONTEXT.md`, `contracts.py`, `configs/default.yaml`, `requirements.txt`, dan `README.md`. Tentukan satu penanggung jawab pada setiap perubahan berkas bersama. Agent A mengoordinasikan bentuk kontrak awal; Agent B meninjau kebutuhan UI/ekspor sebelum integrasi. Pembagian ini saran dan boleh ditukar oleh tim.

Agent B dapat memakai fixture sesuai kontrak lalu menyambungkan pipeline nyata. Fixture diberi label data contoh; demo akhir memakai video nyata. Preview dan ringkasan masuk melalui `on_progress` sesuai kontrak bagian 11. Pada tahap integrasi CLI, Agent A membuat renderer dan ekspor minimal yang dibutuhkan pipeline karena belum ada implementasi rekan pada berkas tersebut. Agent B melanjutkan integrasi UI/evaluator dan pengembangan renderer/ekspor dengan mempertahankan kontrak serta tes.

Gunakan branch terpisah atau folder kerja terisolasi jika bekerja serentak. Periksa perubahan yang belum digabung sebelum mengedit. Hindari menghapus atau menimpa pekerjaan rekan. Setiap handoff menyertakan perubahan, pemeriksaan yang dijalankan, contoh penggunaan, dan keterbatasan.

## 18. Urutan pengerjaan dan hasil tiap tahap

| Tahap | Pekerjaan | Selesai ketika |
| --- | --- | --- |
| 1. Selaraskan kontrak | Periksa rubrik, pilih klip awal, buat tipe data dan konfigurasi | Kedua bagian memakai nama field/arah yang sama |
| 2. Baseline deteksi/tracking | Baca video, jalankan model/tracker, render ID | Satu video selesai diproses dengan ID yang dapat diperiksa |
| 3. Counting | Implementasi geometri, state, event, summary | Skenario counting lulus dan satu klip ditinjau manual |
| 4. UI dan keluaran | Upload, garis, progres, anotasi, CSV/JSON/video | Run dapat dilakukan dari UI dan artefaknya konsisten |
| 5. Eksperimen | Tambahkan CLAHE dan evaluasi GT | Tabel kedua mode beserta run ID dan analisis tersedia |
| 6. Penyiapan presentasi | Dokumentasi, diagram, demo, batasan | Instalasi diikuti ulang dan kedua anggota memahami sistem |

Jadwal kalender baru dibuat setelah tenggat dan waktu kerja tim diketahui. Urutan ini menentukan dependensi, bukan estimasi durasi yang sudah disepakati.

## 19. Bahan laporan dan demonstrasi

Laporan memuat masalah, tujuan, lingkup, teori singkat metode yang benar-benar digunakan, sumber data, pipeline, rumus crossing, konfigurasi, rancangan UI, prosedur GT, hasil evaluasi, eksperimen CLAHE, analisis kegagalan, dan kesimpulan berdasarkan hasil aktual.

Untuk demo, siapkan video pendek, bobot lokal, konfigurasi garis yang tersimpan, dan hasil cadangan dari run yang dapat direproduksi. Tunjukkan video asli, deteksi/ID, crossing beserta arah, ringkasan dan CSV, lalu contoh perbandingan kedua mode. Jika memakai hasil yang sudah diproses, nyatakan bahwa itu rekaman hasil sebelumnya.

Pertanyaan yang harus dapat dijawab tim:

- Mengapa box pada tiap frame tidak langsung dijumlahkan?
- Bagaimana ID berbeda dari kendaraan fisik unik?
- Mengapa anchor tengah bawah, zona netral, dan garis berhingga dipakai?
- Bagaimana satu ID dicegah terhitung lagi?
- Bagaimana occlusion dan ID switch menyebabkan error?
- Apa pengaruh CLAHE berdasarkan data, dan bagaimana hitungan manual dibuat?
- Apa yang dilakukan tim sendiri dan apa yang berasal dari model/library pretrained?

## 20. Contoh prompt untuk AI agent

### Prompt awal untuk kedua agent

```text
Baca CONTEXT.md di root proyek dan instruksi repositori sebelum bekerja.
Jelaskan singkat tujuan, lingkup MVP, status kode, dan kontrak yang relevan
untuk tugasku. Ikuti definisi crossing dan sumber hitungan pada dokumen.
Kerjakan tugas yang aku berikan sampai dapat diperiksa. Laporkan berkas yang
berubah, pemeriksaan yang benar-benar dijalankan, hasil, dan keterbatasan.
Sinkronkan perubahan kontrak bersama sebelum mengubah pihak pemanggilnya.
```

### Prompt tugas Agent A

```text
Mengacu pada CONTEXT.md, kerjakan pipeline CV, konfigurasi, geometri, dan
counting. Mulai dengan memeriksa kode serta menyelaraskan contracts.py.
Gunakan YOLO11n + ByteTrack eksplisit dan koordinat citra asli. Implementasikan
baseline none dahulu, lalu mode CLAHE. Sediakan process_video dan counter
sesuai kontrak, serta uji perilaku crossing tanpa memerlukan model. Sambungkan
ekspor/render melalui batas modul bersama yang sudah disepakati dengan B.
Hindari mengedit UI dan pekerjaan B tanpa koordinasi. Sertakan cara mencoba
satu video nyata dan nyatakan bila pengujian inferensi belum dapat dijalankan.
```

### Prompt tugas Agent B

```text
Mengacu pada CONTEXT.md, kerjakan antarmuka Streamlit, visualisasi, ekspor,
dan evaluator. Baca kontrak serta kode pipeline A terlebih dahulu. Gunakan
fixture berlabel jika pipeline belum tersedia, kemudian sambungkan hasil
nyata. UI dan CSV mengambil hitungan dari event/summary yang sama. Siapkan
unggah video, pengaturan garis dan label arah, progres/preview melalui
on_progress, tabel/grafik, serta unduhan. Uji konsistensi ekspor dan evaluator.
Koordinasikan perubahan contracts.py serta dependensi dengan A sebelum integrasi.
```

### Prompt integrasi dan pemeriksaan bersama

```text
Baca CONTEXT.md dan perubahan terbaru kedua bagian. Periksa kecocokan
kontrak, koordinat, arah, lifecycle tracker/run, dan sumber count. Jalankan
pengujian relevan serta satu alur video nyata jika data/model tersedia.
Perbaiki masalah integrasi dalam lingkup MVP. Cocokkan event, ringkasan UI,
CSV, metadata, dan video hasil. Laporkan hasil aktual, masalah tersisa, serta
status setiap kriteria selesai tanpa mengarang metrik atau fitur yang belum ada.
```

## 21. Keputusan awal dan hal yang perlu dikonfirmasi

| Tanggal | Status | Keputusan / alasan |
| --- | --- | --- |
| 2026-11-06 | Pilihan pengguna | Topik proyek: penghitung kendaraan, dengan acuan bersama untuk dua anggota dan agent |
| 2026-11-06 | Usulan baseline | Video unggahan dan kamera diam agar demo dapat diulang serta dievaluasi |
| 2026-11-06 | Usulan baseline | YOLO11n pretrained + ByteTrack; fokus pada pipeline dan counting sebelum pelatihan tambahan |
| 2026-11-06 | Usulan baseline | Satu segmen garis, dua arah, satu event per Track ID per garis per run |
| 2026-11-06 | Usulan baseline | Empat kelas, Streamlit lokal, dan keluaran CSV/JSON/video |
| 2026-11-06 | Usulan eksperimen | Bandingkan none dengan CLAHE tanpa mengasumsikan peningkatan hasil |
| 2026-11-06 | Instruksi pengguna | Inisialisasi dan push repositori ke `HUB37T/Vehicle-Tracking`; tambahkan README, aturan Git, dan arahan agent. Implementasi aplikasi tetap tahap berikutnya |
| 2026-11-06 | Implementasi fondasi | Pengguna meminta mulai dari bagian yang dapat dikerjakan tanpa video. Kontrak, konfigurasi, counter, prapemrosesan, dan simulasi tersedia; simulasi selalu berlabel sintetis |
| 2026-11-06 | Lingkungan aktual | Memakai Python 3.12 yang sudah tersedia, dengan dependensi fondasi tercatat di constraints.txt. Integrasi dependensi YOLO/Streamlit diuji pada tahap berikutnya |
| 2026-11-08 | Integrasi CLI | Pengguna meminta implementasi deteksi/tracking/pipeline. YOLO11n pretrained + ByteTrack, MP4 H.264, CSV/JSON, dan callback tersedia. Renderer/ekspor dasar dibuat untuk uji end-to-end; UI/evaluator tetap tahap Agent B |
| 2026-11-08 | Data | Training belum dilakukan; kumpulkan video pengembangan dan uji dengan GT crossing dahulu. Fine-tuning baru dipertimbangkan berdasarkan kegagalan baseline yang terukur |
| 2026-11-08 | Distribusi bobot | Default prepare-model memakai repositori resmi Ultralytics di Hugging Face dengan revision/checksum; alternatif GitHub tersedia. Analisis tidak mengunduh bobot diam-diam |
| 2026-11-08 | Materi tambahan | Mode gamma, Gaussian, dan unsharp tersedia di `analyze`. Thresholding, morfologi, CCL, Canny, dan Optical Flow dapat diperiksa melalui `pcd-lab`. Dampak terhadap hitungan diukur terpisah dengan GT |
| 2026-11-08 | Bukti proses | Run sukses menyertakan panel satu frame untuk konversi warna, CLAHE, YOLO, dan ByteTrack, dengan frame serta status metode tercatat di metadata |
| 2026-11-08 | UI dan evaluasi awal | Streamlit memakai pipeline yang sama dengan CLI. Evaluator memverifikasi video dan garis terhadap metadata anotasi. Dua klip sudah ditinjau visual awal dan menunggu validasi kedua |
| 2026-11-08 | Eksperimen yang diizinkan pengguna | DeepSORT ditambahkan sebagai opsi pembanding. ByteTrack `fuse_score: true` tetap pilihan utama. Uji aktual dan keterbatasan dicatat di docs/tracker-safety-evaluation.md |
| 2026-11-08 | Audit crossing | Pasangan temporal dibedakan dari identitas terverifikasi. Audit koordinat dan frame ditambahkan tanpa mengubah sumber hitungan, kontrak events.csv, atau anotasi yang sudah dibekukan |

Hal yang masih perlu diketahui, tetapi tidak menghalangi pembuatan baseline:

- Tenggat, format laporan, rubrik, serta apakah bobot pretrained diperbolehkan.
- Apakah dosen menerima ByteTrack atau mensyaratkan DeepSORT/metode tertentu.
- Nama anggota dan pembagian tanggung jawab final.
- Spesifikasi perangkat, ketersediaan GPU, serta batas penyimpanan.
- Video yang akan dipakai, izin sumber, dan ketersediaan tiap kelas kendaraan.
- Apakah ada kewajiban webcam, jumlah metode tertentu, atau training model.

Jika jawaban mengubah lingkup, perbarui bagian terkait dan catat keputusan baru di tabel ini. Checkbox pada bagian 16 diubah berdasarkan bukti implementasi dan verifikasi.

## 22. Referensi teknis awal

Dokumentasi berikut diperiksa pada 6 November 2026. Ini rujukan penggunaan library, bukan bukti kinerja pada video proyek. Detail API mengikuti versi dependensi yang nanti benar-benar diuji tim.

- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/): nama bobot deteksi dan dukungan model.
- [Ultralytics Object Detection](https://docs.ultralytics.com/tasks/detect/): keluaran deteksi berupa box, kelas, dan confidence.
- [Ultralytics Multi-Object Tracking](https://docs.ultralytics.com/modes/track/): ID tracking, pemilihan ByteTrack, dan parameter tracker. ByteTrack dan DeepSORT adalah algoritma yang berbeda.
- [OpenCV Histogram Equalization dan CLAHE](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html): dasar eksperimen peningkatan kontras.
- [Streamlit File Uploader](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader): unggahan file dan pengaturan batas ukuran. Pipeline tetap harus memeriksa keterbacaan video.
- [Repositori model resmi Ultralytics di Hugging Face](https://huggingface.co/Ultralytics/YOLO11): distribusi bobot yang dipakai prepare-model dengan revision/checksum; diperiksa 8 November 2026.

Tambahkan sumber primer atau paper algoritma yang benar-benar dibahas saat menulis laporan akademik. Catat lisensi library, bobot, dan video yang dipakai sesuai sumbernya.
