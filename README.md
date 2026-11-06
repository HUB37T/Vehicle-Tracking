# Vehicle Tracking & Counting

Proyek akhir mata kuliah **Pengolahan Citra Digital dan Visi Komputer** untuk menghitung kendaraan dari video berdasarkan jenis dan arah perjalanan.

Pengguna mengunggah rekaman jalan dari kamera diam, mengatur garis virtual, lalu sistem mendeteksi dan melacak kendaraan. Hitungan bertambah ketika lintasan kendaraan memenuhi aturan penyeberangan garis.

## Status proyek

Saat ini repositori berisi dokumentasi dan rancangan awal. Aplikasi, dependensi, bobot model, serta hasil evaluasi belum tersedia. Pilihan teknis berikut merupakan baseline yang dapat disesuaikan dengan ketentuan dosen dan kesepakatan tim.

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

Petunjuk pemasangan dependensi dan menjalankan aplikasi ditambahkan setelah implementasi pertama berhasil diuji. Untuk tugas agent pertama, gunakan contoh prompt pada bagian 20 di `CONTEXT.md`.

## Data dan bobot model

Rencana lokasi lokal:

- `data/videos/`: video sumber.
- `data/annotations/`: anotasi manual dan metadata data uji yang dapat dibagikan.
- `models/`: bobot model yang diunduh.
- `outputs/`: keluaran setiap run.

Folder tersebut dibuat saat dibutuhkan. Video, bobot, dan keluaran analisis diabaikan Git. Dokumentasikan sumber, izin penggunaan, dan cara memperoleh data/bobot agar percobaan dapat diulang. Berkas anotasi dan dokumentasi eksperimen dapat dimasukkan ke repositori selama layak dibagikan.

## Lisensi

Kode dan dokumentasi milik proyek mengikuti [lisensi MIT](LICENSE) yang sudah tersedia di repositori. Library, bobot pretrained, serta data pihak ketiga mengikuti lisensi sumbernya masing-masing.
