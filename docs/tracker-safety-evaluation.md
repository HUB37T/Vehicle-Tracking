# Audit crossing dan eksperimen DeepSORT

ByteTrack dengan `fuse_score: true` tetap pilihan utama. DeepSORT tersedia sebagai opsi pembanding materi kuliah, dengan konfigurasi dibekukan sebelum menjalankan perbandingan ini. Hasilnya belum mendukung penggantian ByteTrack. Semua hasil di bawah berasal dari inferensi video lokal CPU, bukan fixture. Klip sudah dipakai untuk pengembangan, sehingga ini bukan uji akhir pada data baru.

## Temuan Istanbul

YOLO pada antrean padat dapat membentuk kotak yang mencakup kendaraan kuning dan putih sekaligus, atau hanya bagian kendaraan yang terlihat. Titik tengah bawah kotak tersebut bukan titik bawah satu kendaraan yang dapat diikuti dengan pasti. Tracker juga bisa memecah identitas pada kotak-kotak tersebut. Mengganti algoritma asosiasi tidak menciptakan deteksi yang hilang atau memisahkan kotak gabungan.

Replay ByteTrack terhadap 372 frame deteksi tersimpan mengulang tepat sepuluh event run sebelumnya. Menambahkan NMS antarsemua kelas pada IoU 0,7 tetap menghasilkan sepuluh event dengan frame yang sama. Percobaan itu tidak dipasang pada pipeline karena belum memperbaiki penyebabnya.

[Audit per event](istanbul-crossing-audit.csv) membedakan empat identitas terverifikasi, pasangan temporal yang salah, dan kejadian ambigu. I01 truk, I02 minivan, I03 van gelap, serta I05 mobil gelap pulih pada kedua tracker. I04 taksi belum punya event yang identitasnya dapat diverifikasi. Evaluator otomatis memasangkan event kendaraan putih di belakang taksi pada frame 193 ke I04 karena kelas, arah, dan waktunya dekat. F1 temporal tidak membuktikan taksi sudah dihitung.

Pada ByteTrack, frame 151 dan 180 mengandung kotak gabungan pasangan kendaraan yang sama: indikasi hitung ulang, dengan identitas tunggal masih ambigu. Pada DeepSORT, muncul kandidat tambahan di frame 82 dan 103 pada area oklusi awal. Frame 262 memiliki anchor y265,6, jauh dari garis y330; ini kandidat perpindahan ID antarobjek, belum crossing fisik terverifikasi. Audit ini satu tinjauan visual dan masih memerlukan peninjau kedua.

## Konfigurasi yang dibandingkan

- Detector YOLO11n lokal yang sama, confidence 0,10, CLAHE, garis dan video yang sama.
- Istanbul: ROI sumber `[640,205,995,460]`, input YOLO 640. Hujan: frame penuh, input 960. Vehicles-3: frame penuh, input 640.
- Pemeriksaan crop `truck` dan deduplikasi tampilan aktif pada kedua tracker, mengikuti profil perbaikan sebelumnya.
- ByteTrack memakai konfigurasi bawaan terpasang dengan `fuse_score: true`.
- DeepSORT memakai [konfigurasi tetap](../configs/deepsort.yaml): `max_age=30`, `n_init=3`, `max_cosine_distance=0.20`, `max_iou_distance=0.70`, `nn_budget=100`. `fuse_score` bukan parameter DeepSORT.
- Fitur tampilan DeepSORT: MobileNetV2 bottleneck pretrained ImageNet, CPU float32. Bobot dibundel `deep-sort-realtime==1.3.2`, SHA256 `2f518e773d4402dde55f981ae3078a72ba95c3adccae1d55051a4be844d50197`. Belum ada pelatihan ReID khusus kendaraan.

Adapter DeepSORT hanya meneruskan track terkonfirmasi dengan deteksi aktual pada frame tersebut. Indeks deteksi disimpan sebagai data tambahan asosiasi. Counter menerima box deteksi asli yang cocok, bukan posisi prediksi Kalman ketika deteksi hilang. Setiap run memakai tracker baru. CPU tanpa kemampuan instruksi lanjutan memakai kernel MobileNet tanpa MKLDNN; cabang ini diuji lokal dengan kemampuan CPU disimulasikan, belum diuji pada VM.

## Hasil inferensi ulang

`TP/FP/FN otomatis` di tabel memakai kelas, arah, serta toleransi waktu ±0,5 detik. Khusus Istanbul, cakupan anotasi mengecualikan sejumlah oklusi: event ekstra perlu audit dan tidak semuanya dapat dinyatakan hitung palsu. FPS mencakup pemrosesan, encoding, verifikasi decode, ekspor dan callback sesuai pipeline. Angka diukur pada laptop lokal; beberapa pemeriksaan lain berjalan bersamaan, sehingga bukan benchmark terisolasi.

| Klip | Tracker | Event | TP/FP/FN otomatis | F1 temporal | Processing FPS |
| --- | --- | ---: | --- | ---: | ---: |
| Istanbul | ByteTrack | 10 | 5 / 5 / 0 | 0,667 | 22,03 |
| Istanbul | DeepSORT | 12 | 5 / 7 / 0 | 0,588 | 3,68 |
| HVSD hujan | ByteTrack | 7 | 7 / 0 / 0 | 1,000 | 17,99 |
| HVSD hujan | DeepSORT | 8 | 7 / 1 / 0 | 0,933 | 9,85 |
| Vehicles-3 | ByteTrack | 6 | 4 / 2 / 0 | 0,800 | 21,96 |
| Vehicles-3 | DeepSORT | 5 | 4 / 1 / 0 | 0,889 | 4,02 |
| Vehicles | ByteTrack | 9 | 9 / 0 / 0 | 1,000 | 4,60 |
| Vehicles-2 | ByteTrack | 14 | 13 / 1 / 15 | 0,619 | 19,43 |

Pada hujan, SUV hitam yang sama mendapat ID 39 dan 40 pada frame 59 dan 63 di DeepSORT: hitung ganda terverifikasi. Sedan putih pada frame 50 tetap kendaraan berbeda, dan van putih pada frame 181 serta truk pada frame 182 tetap masing-masing dihitung. ByteTrack mempertahankan ketujuh crossing tanpa hitung ganda yang ditemukan pada tinjauan ini.

Pada Vehicles-3, DeepSORT mempertahankan empat event lama pada frame 20, 253, 364, dan 623, serta tidak menghasilkan event van ganda frame 545. Masih ada event tambahan frame 913 dengan kotak yang hanya mencakup bagian belakang sedan putih yang sebelumnya dihitung pada frame 623: indikasi penghitungan ulang karena ID dan bentuk box berubah. Event ekstra ByteTrack frame 1027 tidak muncul pada run DeepSORT. Ini perbaikan pada satu klip, disertai perlambatan dan regresi hitung ganda pada hujan. ByteTrack pada kelima klip mempertahankan jumlah serta metrik sebelumnya; Vehicles-2 masih kehilangan 15 pasangan otomatis dan belum layak disebut akurat pada kondisi itu.

| Klip | ByteTrack run | DeepSORT run |
| --- | --- | --- |
| Istanbul | `20261108T144308Z-9eb243558308` | `20261108T144327Z-29410d2fe422` |
| HVSD hujan | `20261108T144509Z-76d368012236` | `20261108T144520Z-0c306ba934a8` |
| Vehicles-3 | `20261108T144540Z-02dd2ad721f7` | `20261108T144633Z-b0624d492a47` |
| Vehicles | `20261108T145132Z-8d6ef4a894c9` | Tidak dijalankan pada regresi ini. |
| Vehicles-2 | `20261108T145120Z-996d8f55f690` | Tidak dijalankan pada regresi ini. |

## Bukti dan cara mengulang

Hasil lokal berada di `outputs/safety-audit/<clip_id>-<tracker>/<run_id>/`. Setiap run sukses memverifikasi seluruh frame output, menulis event/rekap yang konsisten, dan mengekspor `crossing_audit.csv` dengan box serta anchor setiap event. Panel bukti keenam mengikuti nama tracker aktual. Tab Audit crossing menyediakan pemilihan ID/frame pada video hasil. Tanda `anchor_inside_other_detection` hanya kandidat tinjauan; seluruh event Istanbul dapat bertanda termasuk empat crossing yang benar. Tanda tidak dipakai untuk menghapus event otomatis.

Contoh menjalankan kedua tracker pada hujan setelah menyiapkan video dan bobot sesuai README serta laporan sebelumnya:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[deepsort]" -c constraints.txt
.\.venv\Scripts\python.exe -m vehicle_counter.cli analyze data/videos/holdout-hvsd-rain.mp4 --config configs/experiments/hvsd-rain-recovery.yaml --output-root outputs/compare-rain/bytetrack
.\.venv\Scripts\python.exe -m vehicle_counter.cli analyze data/videos/holdout-hvsd-rain.mp4 --config configs/experiments/hvsd-rain-recovery.yaml --tracker-config configs/deepsort.yaml --output-root outputs/compare-rain/deepsort
.\.venv\Scripts\python.exe -m vehicle_counter.cli evaluate data/annotations/holdout-hvsd-rain.csv outputs/compare-rain/bytetrack/ISI_RUN_ID outputs/compare-rain/deepsort/ISI_RUN_ID --clip-id holdout-hvsd-rain
```

Untuk Istanbul dan Vehicles-3, gunakan masing-masing profil `istanbul-recovery.yaml` dan `vehicles-3-refined.yaml` di direktori konfigurasi eksperimen, serta anotasi dengan clip ID yang sesuai. Sumber video, stabilisasi, cakupan GT, dan bobot MobileNetV3 pemeriksa truck mengikuti [uji holdout](bytetrack-holdout-evaluation.md). Tidak ada video, bobot, atau keluaran besar yang dimasukkan Git.

Implementasi DeepSORT mengikuti kemampuan library [deep-sort-realtime](https://github.com/levan92/deep_sort_realtime), termasuk deteksi asli dan informasi tambahan track. Sumber algoritma: [Deep SORT, Wojke dkk.](https://arxiv.org/abs/1703.07402). Fitur tampilan generik pada eksperimen ini tidak membuktikan akurasi ReID kendaraan.

## Pemeriksaan perangkat lunak

149 tes lulus pada lingkungan Python 3.12 Windows dengan PyTorch CPU dan paket DeepSORT terpasang. Cakupan tambahan meliputi validasi parameter, lifecycle tracker/ROI, deteksi kosong, penolakan posisi prediksi tanpa deteksi, audit yang konsisten dengan event, serta penandaan batas evaluasi temporal. Lint, format, dan pemeriksaan dependensi lulus. Uji antarmuka Streamlit melalui AppTest membuka profil pemulihan, memilih kedua tracker, menampilkan hasil nyata, memeriksa peringatan 12 event Istanbul, dan mengganti frame audit. Tes otomatis tidak mengunduh video atau bobot. Peringatan dependensi `pkg_resources` yang usang tetap muncul; batas versi setuptools dicatat di ekstra paket.

Delapan run lengkap pada matriks di atas selesai dan jumlah frame, event, audit, rekap, serta ringkasan cocok. Keempat checksum CSV/JSON anotasi Istanbul dan hujan tetap sama dengan berkas pengunci sebelumnya. Event ByteTrack Istanbul dan hujan identik dengan run pemulihan terdahulu; keenam frame event Vehicles-3 juga sama. Uji tambahan lewat CLI publik proyek menghasilkan delapan event DeepSORT hujan yang identik dengan run perbandingan (`20261108T145427Z-fad6417aca6f`).

## Batas dan prioritas berikutnya

Hasil Istanbul belum bebas kesalahan. Prioritas akurasi adalah anotasi bounding box dan identitas pada area garis yang tertutup, lalu memperbaiki kualitas deteksi/pemisahan objek dan menilai asosiasi terhadap anotasi itu. Menambahkan metode tracking sendiri belum cukup. Video hujan dan klip lama tetap dipakai sebagai uji regresi agar perbaikan pada antrean tidak menghapus dua kendaraan berbeda yang lewat rapat. Bus, motor, malam hari, dan kamera bergerak belum tercakup penuh pada evaluasi ini. Perubahan ini belum dideploy ke VM.
