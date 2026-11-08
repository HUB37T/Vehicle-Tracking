# Eksperimen `fuse_score` pada ByteTrack

Perbandingan memakai 1.152 frame awal `vehicles-3.mp4`, rekaman antrean lalu lintas saat senja dengan kamera diam. Video diperkecil ke 1280 × 720 dan durasinya 48,048 detik pada 23,976 fps. [Anotasi CSV](../data/annotations/vehicles-3-bytetrack-test.csv) dan [metadata klip](../data/annotations/vehicles-3-bytetrack-test.json) menyimpan cakupan, checksum, dan konfigurasi garis.

Kedua run memakai YOLO11n pada CPU, CLAHE, ukuran input 640, confidence deteksi 0,10, dan segmen yang sama. Satu-satunya perbedaan pengaturan efektif tracker adalah `fuse_score`. Pencocokan otomatis menggunakan toleransi ±0,5 detik dengan kelas dan arah yang sama. Pemeriksaan frame sumber dan video beranotasi dipakai untuk membedakan kendaraan yang sama, crossing palsu, dan kasus ambigu.

## Hasil

| Profil | Event program | Cocok | Ekstra | Manual tanpa pasangan | F1 crossing |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fuse_score: true` | 7 | 1 | 6 | 3 | 0,182 |
| `fuse_score: false` | 9 | 1 | 8 | 3 | 0,154 |

Baseline tercatat sebagai `20261108T125804Z-7282d1db8d4f` dan eksperimen sebagai `20261108T125804Z-1e19317dfb99`. Evaluator memakai empat crossing manual dengan titik acuan yang dapat diperiksa. Keempat kendaraan menghasilkan event B → A di kedua run setelah dipasangkan berdasarkan identitas visual. Hanya J04, sedan putih pada frame 623, yang cocok kelasnya. J01 (SUV hitam, frame 20), J02 (van ringan putih, frame 253), dan J03 (SUV biru gelap, frame 364) tercatat sebagai `truck` di kedua profil dan diberi kelas `car` pada anotasi awal. Karena itu, ketiganya masuk kolom ekstra sekaligus manual tanpa pasangan.

| Frame event | `true` | `false` | Cek visual |
| --- | --- | --- | --- |
| 267 | — | ID 61, `truck`, B → A | Hitungan ganda. ID 61 kembali menghitung van J02 yang sudah dihitung sebagai ID 1 pada frame 253. |
| 544–545 | ID 275, `truck`, B → A pada 545 | ID 85, `car`, B → A pada 544 | Hitungan ganda van J02 pada kedua profil. Sedan di depan menutupi sebagian van. Sedan sendiri menghasilkan crossing pada frame 623. |
| 913 | ID 347, `car`, B → A | ID 136, `car`, B → A | Minivan putih tertutup sebagian oleh sedan. Titik tengah bawah kendaraan utuh sulit dipastikan pada frame ini. |
| 1017 | — | ID 170, `car`, A → B | Crossing palsu arah balik. Frame sebelum dan sesudahnya tidak memperlihatkan kendaraan bergerak menyeberang dari kanan ke kiri pada segmen. |
| 1027 | ID 453, `car`, B → A | ID 114, `car`, B → A | Kendaraan di lajur belakang saling menutupi. Identitas dan titik tengah bawah kendaraan yang menghasilkan event masih ambigu. |

Tidak ada crossing terverifikasi yang pulih pada klip ini. Profil eksperimen menambah dua hitungan palsu yang terverifikasi. Jumlah ekstra evaluator juga mencakup tiga kelas yang berbeda dari anotasi serta dua event ambigu pada masing-masing profil. Pertahankan profil bawaan untuk demo. Untuk klip antrean ini, prioritas perbaikan adalah kestabilan kelas SUV dan van, lalu pencegahan hitungan ganda saat deteksi tertutup. Hasil masih berasal dari satu klip dengan anotasi satu peninjau.

## Mengulang eksperimen

Gunakan `vehicles-3.mp4` lokal dengan checksum pada metadata. URL asal berkas lokal belum terverifikasi. Perintah berikut memakai FFmpeg bawaan `imageio-ffmpeg` dan CLI proyek dari direktori akar.

```powershell
$ffmpeg = .\.venv\Scripts\python.exe -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())'
$clip = 'data/videos/vehicles-3-bytetrack-test.mp4'
$config = 'configs/experiments/vehicles-3-bytetrack.yaml'
$root = 'outputs/bytetrack-experiment'
& $ffmpeg -hide_banner -threads 6 -i data/videos/vehicles-3.mp4 -map 0:v:0 -frames:v 1152 -vf scale=1280:720 -c:v libx264 -threads 4 -preset fast -an $clip
$baseline = .\.venv\Scripts\python.exe -m vehicle_counter.cli analyze $clip --config $config --output-root "$root/fuse-on" | ConvertFrom-Json
$experiment = .\.venv\Scripts\python.exe -m vehicle_counter.cli analyze $clip --config $config --tracker-config configs/bytetrack-no-fuse.yaml --output-root "$root/fuse-off" | ConvertFrom-Json
.\.venv\Scripts\python.exe -m vehicle_counter.cli evaluate data/annotations/vehicles-3-bytetrack-test.csv $baseline.output_dir $experiment.output_dir --clip-id vehicles-3-bytetrack-test --output "$root/evaluation.json"
```

Verifikasi checksum klip sebelum memakai CSV yang ada. Encoder atau versi FFmpeg yang berbeda dapat menghasilkan checksum berbeda.

## Uji lanjutan dengan `fuse_score: true`

Profil [vehicles-3-refined.yaml](../configs/experiments/vehicles-3-refined.yaml) memakai video, YOLO11n, CLAHE, garis, dan setelan ByteTrack baseline. Kandidat crossing berkelas `truck` diperiksa dengan crop frame sumber. MobileNetV3-Large perlu menempatkan salah satu label kendaraan ringan di posisi teratas dan memberi jumlah probabilitas kelompok minimal 0,40 sebelum kelas event diubah ke `car`. Kotak dan label deteksi pada video tetap memperlihatkan keluaran detektor, sementara rekap dan CSV mengikuti kelas final event. [Dokumentasi model Torchvision](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_large.html)

Kandidat crossing dengan ID baru dibandingkan dengan event searah dalam 15 detik terakhir. Kandidat dibuang jika kotaknya beririsan dengan IoU minimal 0,65 dan korelasi dua arah dari bagian atas kotak minimal 0,80. Kotak yang kurang bertekstur dilewati. Riwayat pembanding dihapus lebih awal jika kendaraan lama terlihat meninggalkan garis sejauh bentang kotaknya pada arah tegak lurus garis. Aturan ini ditujukan untuk antrean kamera diam. Kendaraan yang tampil serupa dan saling berdekatan masih perlu diuji pada lebih banyak klip.

| Profil | Event program | Cocok | Ekstra evaluator | Manual tanpa pasangan | Hitungan palsu terverifikasi | F1 crossing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fuse_score: true` baseline | 7 | 1 | 6 | 3 | 1 | 0,182 |
| `fuse_score: false` | 9 | 1 | 8 | 3 | 3 | 0,154 |
| `fuse_score: true` dengan pemeriksaan kelas dan duplikasi | 6 | 4 | 2 | 0 | 0 | 0,800 |

Run `20261108T131419Z-9cef552c7755` memproses seluruh 1.152 frame. SUV hitam J01 (frame 20), van putih J02 (253), dan SUV biru gelap J03 (364) berubah dari `truck` menjadi `car`. Duplikat van pada frame 545 tidak masuk event log, dan sedan putih J04 pada frame 623 tetap tercatat. Dua event ekstra pada frame 913 dan 1027 tetap ambigu karena titik acuan kendaraan tertutup. Anotasi empat crossing berasal dari satu peninjau dan dipakai saat menyusun profil ini, sehingga hasilnya adalah evaluasi pengembangan.

Kontrol pada rekaman `vehicles-2-test` memakai replay observasi YOLO dan ByteTrack yang direkam sebelumnya dengan frame video asli untuk pemeriksaan crop. Ke-14 event tetap sama, termasuk keenam event berkelas `truck`. Kontrol tersebut memakai deteksi tersimpan dan tanpa inferensi ulang YOLO.

Siapkan bobot Torchvision sebelum menjalankan profil. SHA-256 bobot adalah `5c1a416349c4cf298f2a6a5e2600ed0ee55e604713578f5e74e6bc8bcaef7997` dan diperiksa saat run dimulai.

```powershell
curl.exe -fL https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth -o models/mobilenet_v3_large-5c1a4163.pth
$refined = .\.venv\Scripts\python.exe -m vehicle_counter.cli analyze $clip --config configs/experiments/vehicles-3-refined.yaml --output-root "$root/refined" | ConvertFrom-Json
.\.venv\Scripts\python.exe -m vehicle_counter.cli evaluate data/annotations/vehicles-3-bytetrack-test.csv $baseline.output_dir $experiment.output_dir $refined.output_dir --clip-id vehicles-3-bytetrack-test --output "$root/evaluation-refined.json"
```
