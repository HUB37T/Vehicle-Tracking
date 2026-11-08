# Uji profil ByteTrack pada rekaman baru

Garis, kelas, dan perkiraan frame crossing ditulis dari video sumber sebelum kedua run. Hash CSV dan JSON dibekukan di `outputs/holdout/<clip_id>/annotation-lock.json` dan diperiksa lagi saat evaluasi. Satu peninjau mengerjakan anotasi. Perubahan profil dan ambang dihentikan sebelum uji ini. SUV, van ringan, dan pikap ringan masuk `car`. Truk boks masuk `truck`.

| Klip | Kondisi | Lingkup manual |
| --- | --- | --- |
| `holdout-istanbul-stabilized` | Antrean padat siang hari, 372 frame, 1280 × 720 | Dua lajur dalam segmen y ≈ 330 px. Titik tengah bawah yang tidak terlihat karena kendaraan bertumpuk ditinjau terpisah. |
| `holdout-hvsd-rain` | Kamera diam saat hujan, 194 frame, 704 × 432 | Tujuh crossing di kedua arah. Termasuk SUV hitam, van putih, truk boks, dan beberapa mobil yang lewat rapat. |

Sumber antrean padat berasal dari Ahmet Akpolat. Zoom video distabilkan memakai pencocokan fitur SIFT dan estimasi affine parsial RANSAC pada tepi dan bagian atas gambar. Semua frame dipetakan ke perspektif frame terakhir sebelum membuat anotasi. Penyesuaian ini menggeser koordinat gambar, sehingga metadata menyimpan checksum video hasil stabilisasi yang dipakai kedua run. [Traffic on Highway, Pexels](https://www.pexels.com/video/traffic-on-highway-14359967/)

Klip hujan memakai `Dataset/rain.mp4` dari `Dataset.zip`, lalu dikonversi ke 30 FPS konstan tanpa perubahan resolusi. [Highway Vehicle Surveillance Dataset, Hadi Sadoghi Yazdi, CC BY 4.0](https://data.mendeley.com/datasets/zdgmdg7p25/1)

## Konfigurasi

Kedua profil memakai bobot YOLO11n, CLAHE, CPU, ukuran input 640, dan pengaturan ByteTrack yang sama dengan `fuse_score: true`. Profil perbaikan menambahkan `deduplicate_crossings: true` dan `truck_review_weights: models/mobilenet_v3_large-5c1a4163.pth`. Garis diambil dari JSON pendamping tiap klip. Evaluator memakai toleransi waktu ±0,5 detik dan kecocokan kelas serta arah, lalu event diperiksa kembali pada video beranotasi.

Hasil dan video lokal tersedia di `outputs/holdout/<clip_id>/`. `baseline.yaml` dan `refined.yaml` di direktori itu berisi konfigurasi yang benar-benar digunakan. Video hasil stabilisasi dan konversi ada di `data/videos/`. Untuk menyiapkan bobot MobileNetV3-Large, ikuti [panduan eksperimen](bytetrack-fuse-score-evaluation.md#uji-lanjutan-dengan-fuse_score-true).

## Hasil

| Klip | Profil | Cocok | Ekstra | Manual tanpa pasangan | F1 crossing |
| --- | --- | ---: | ---: | ---: | ---: |
| Istanbul | Baseline | 0 | 0 | 5 | 0,000 |
| Istanbul | Perbaikan | 0 | 0 | 5 | 0,000 |
| HVSD hujan | Baseline | 6 | 0 | 1 | 0,923 |
| HVSD hujan | Perbaikan | 6 | 0 | 1 | 0,923 |

| Klip | Run baseline | Run perbaikan |
| --- | --- | --- |
| Istanbul | `20261108T132839Z-37809487c91c` | `20261108T132901Z-f22d89c67948` |
| HVSD hujan | `20261108T132635Z-c0afb352ab70` | `20261108T132647Z-32a96d3899bf` |

### Cek identitas dan penyebab event terlewat

- **HVSD, kendaraan berbeda yang lewat rapat.** Sedan putih R02 dan SUV hitam R03 dicatat sebagai ID 3 dan ID 10 pada frame 50 dan 59 di kedua run. Jaraknya 0,30 detik. Sedan gelap R04 dan van putih R06 juga tercatat satu kali. Tidak ada event ganda yang ditemukan di kedua run ini.
- **HVSD, truk boks R07.** Detektor melihat truk dari jalur seberang. ID 47 punya observasi di sisi A, tetapi tidak melanjutkan sampai sisi B. ID 52 mulai pada frame 179 dengan anchor x ≈ 605,5 px, di dalam zona netral garis x ≈ 604,6 px, lalu bergerak ke sisi B. Counter tidak punya observasi stabil di sisi A untuk ID 52. Ini terjadi sebelum pemeriksaan kelas maupun duplikasi.
- **Istanbul, truk boks I01.** Di sekitar garis, YOLO menghasilkan kotak `truck` dengan confidence 0,13–0,24 pada contoh frame 88–101. Kotak di wilayah itu tidak menghasilkan observasi ByteTrack yang dapat dipakai untuk crossing.
- **Istanbul, I02–I05.** Pada jendela ±12 frame di sekitar keempat anotasi, tidak ada deteksi yang titik tengah bawahnya berada di sekitar posisi kendaraan pada garis. Kendaraan saling menutupi dan terlihat kecil dari kamera. Di seluruh klip tidak ada observasi ByteTrack dengan anchor di x 684–940 dan y 310–350. Karena baseline tidak menghasilkan crossing, klip ini tidak menguji penolakan duplikasi setelah event terbentuk.

Perbaikan menghasilkan event yang identik dengan baseline pada kedua klip baru. Tidak ditemukan kendaraan berbeda yang hilang khusus karena perbaikan. Pada sampel ini pemeriksaan kelas dan duplikasi belum mendapat kesempatan memperbaiki crossing yang terlewat. Hambatan berikutnya adalah memperoleh deteksi yang cukup dan mempertahankan Track ID di kedua sisi garis.

## Uji pemulihan deteksi dan ID

Anotasi, garis, bobot YOLO11n, CLAHE, dan pengaturan ByteTrack dikunci seperti run sebelumnya. Kedua profil pemulihan tetap memakai `fuse_score: true`, pemeriksaan crop `truck`, dan penolakan duplikasi dari profil perbaikan sebelumnya. Profil [Istanbul](../configs/experiments/istanbul-recovery.yaml) memberi YOLO potongan 355 × 255 piksel yang mencakup garis dan kedua sisinya. Deteksi dipetakan kembali ke koordinat sumber sebelum ByteTrack. Profil [HVSD hujan](../configs/experiments/hvsd-rain-recovery.yaml) memakai frame penuh dengan `imgsz: 960`.

Menaikkan ambang biaya asosiasi ByteTrack dengan input lama menghasilkan event truk hujan berkelas `bus` pada replay dan tidak memulihkan crossing Istanbul. Memperbesar input penuh ke 1280 pada Istanbul juga tidak menghasilkan pasangan pada replay. Setelan ByteTrack bawaan dipakai di run akhir.

| Klip | Event sebelum → sesudah | Pasangan otomatis | Pasangan terverifikasi | Event lain setelah audit |
| --- | ---: | ---: | ---: | --- |
| Istanbul | 0 → 10 | 0 → 5 dari 5 | 0 → 4 dari 5 | Enam event dengan titik acuan tertutup atau kotak yang mencakup lebih dari satu kendaraan. |
| HVSD hujan | 6 → 7 | 6 → 7 dari 7 | 6 → 7 dari 7 | Tidak ditemukan hitungan ganda. |

- **Istanbul, empat crossing pulih.** Truk I01 tercatat pada frame 93, minivan I02 pada 118, van gelap I03 pada 170, dan mobil gelap I05 pada 210. Evaluator otomatis memasangkan I03 dengan event frame 180 di lajur sebelah. Kotak frame 193 yang dipasangkan ke taksi I04 mengikuti kendaraan putih di belakang taksi. Taksi sendiri belum punya event yang dapat diverifikasi.
- **Istanbul, audit hitung ganda.** Frame 151 dan 180 memiliki dua event yang kotaknya mencakup pasangan kendaraan kuning dan putih yang sama. Ini indikasi hitung ulang pada area tumpang tindih. Identitas satu kendaraan untuk kedua kotak belum bisa dipastikan. Event pada frame 77, 193, 304, dan 334 juga tidak ditambahkan ke jumlah terverifikasi karena titik acuan tertutup atau identitas tidak cocok dengan anotasi.
- **HVSD hujan, satu crossing pulih.** Truk R07 memperoleh ID 78 di sisi A sejak frame 151 dan mempertahankan ID yang sama saat mencapai sisi B pada frame 182. Sedan dan SUV yang hanya berjarak 0,30 detik tetap dihitung masing-masing, dan keenam crossing lama tetap cocok.

Run akhir Istanbul adalah `20261108T134026Z-268478fb3e8a` dan HVSD hujan adalah `20261108T134026Z-cbf6c829f9c5`. CSV crossing, CSV rekap, video beranotasi, dan audit frame tersimpan di `outputs/recover-crossings/`. Hash anotasi sebelum dan sesudah identik. Karena pemilihan profil dilakukan setelah melihat kedua klip ini, hasil pemulihan merupakan uji pengembangan dengan satu peninjau.

```powershell
.\.venv\Scripts\python.exe -m vehicle_counter.cli analyze data/videos/holdout-istanbul-stabilized.mp4 --config configs/experiments/istanbul-recovery.yaml --output-root outputs/recover-crossings/holdout-istanbul-stabilized
.\.venv\Scripts\python.exe -m vehicle_counter.cli analyze data/videos/holdout-hvsd-rain.mp4 --config configs/experiments/hvsd-rain-recovery.yaml --output-root outputs/recover-crossings/holdout-hvsd-rain
```
