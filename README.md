# TRAQ — Penghitung Kendaraan dan Indikator Kemacetan

TRAQ adalah proyek akhir mata kuliah **Pengolahan Citra Digital dan Visi Komputer**. Aplikasi menganalisis rekaman jalan dari kamera diam untuk menghitung **mobil, motor, bus, dan truk** yang melintasi garis virtual. Hasilnya mencakup hitungan dua arah dan indikasi kemacetan pada area yang dipilih.

## Demo online

[Buka demo TRAQ](https://pred-autoscaler.bccdev.id/vehicle-tracking/demo/) di browser.

1. Buka tab **Siapkan analisis**. Pada **Sumber video**, pilih `vehicles-3-bytetrack-test.mp4`, yaitu klip antrean yang tersedia di demo.
2. Pada **Profil analisis**, pilih **Pemulihan crossing** untuk memakai posisi garis dan zona pengamatan yang sudah disiapkan. Pastikan **Aktifkan indikator macet** tercentang.
3. Atur **Durasi uji singkat** menjadi **10 detik** dan klik **Uji singkat**. Progres muncul di tab **Antrean**. Gunakan **Mulai analisis** untuk memproses video penuh.
4. Setelah selesai, buka tab **Hasil** untuk melihat rekap, video beranotasi, dan status lalu lintas.

## Menjalankan demo di laptop Windows

Siapkan **Git**, **Python 3.12**, dan koneksi internet untuk instalasi awal. Perintah berikut dijalankan di **PowerShell**. Jika repositori sudah tersedia, buka foldernya dan mulai dari pembuatan `.venv`.

```powershell
git clone https://github.com/HUB37T/Vehicle-Tracking.git
cd Vehicle-Tracking
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -c constraints.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-ui.txt -c constraints.txt
.\.venv\Scripts\python.exe -m vehicle_counter.cli prepare-model
```

Perintah terakhir menyiapkan bobot **YOLO11n**, model bawaan untuk demo lokal. Untuk membuka aplikasi pada penggunaan pertama maupun berikutnya, jalankan perintah ini dari folder repositori.

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

1. Buka [demo lokal](http://127.0.0.1:8501) di browser. Biarkan PowerShell terbuka selama aplikasi digunakan.
2. Pada **Sumber video**, pilih **Unggah MP4** dan masukkan rekaman jalan dari kamera diam. Untuk percobaan pertama, [unduh klip contoh dari Roboflow](https://media.roboflow.com/supervision/video-examples/vehicles.mp4). Video lokal juga muncul di daftar jika disimpan di `data/videos/`.
3. Di **Pratinjau dan garis hitung**, geser kedua ujung garis supaya kendaraan bisa melintasinya. Untuk mencoba indikator macet, centang **Aktifkan indikator macet**, lalu tempatkan kotak pengamatan di bagian jalan yang ingin diamati.
4. Klik **Uji singkat**, lalu buka **Hasil**. Untuk menutup aplikasi, tekan **Ctrl+C** di PowerShell.

### Memakai YOLO26n

Siapkan bobotnya sekali dengan perintah berikut, lalu pilih **YOLO26n — kandidat edge** pada **Model dan pelacakan → Model deteksi**.

```powershell
.\.venv\Scripts\python.exe -m vehicle_counter.cli prepare-model --variant yolo26n
```

## Membaca hasil

| Tab | Yang bisa dilihat |
| --- | --- |
| **Rekap** | Jumlah kendaraan per jenis dan arah **A→B** atau **B→A**. Label A dan B terlihat di pratinjau garis. |
| **Per lajur** | Jumlah kendaraan per zona lajur jika zona sudah dikonfigurasi. |
| **Video** | Rekaman dengan kotak deteksi, ID kendaraan, dan garis penghitungan. |
| **Lalu lintas** | Status, durasi, dan rentang waktu indikasi macet pada area atau zona yang diamati. |
| **Bukti materi** | Gambar sumber, hasil prapemrosesan yang dipilih, kotak YOLO, dan jejak pelacakan pada frame yang sama. |
| **Unduhan** | Video beranotasi dan berkas CSV untuk memeriksa hasil analisis. |

## Cara kerja

Setiap frame melewati prapemrosesan pilihan, misalnya CLAHE. YOLO mendeteksi kendaraan dan ByteTrack menghubungkan deteksi dengan ID antarframe. Sistem mencatat satu perlintasan saat jejak kendaraan menyeberangi garis sesuai aturan. Indikasi macet dihitung dari keterisian zona dan gerak lambat kendaraan yang bertahan selama durasi tertentu.

## Dokumentasi

- [Rancangan teknis](docs/project-design.md) menjelaskan alur, aturan penghitungan, dan format keluaran.
- [Panduan teknis dan eksperimen](docs/usage-and-experiments.md) memuat CLI, preset kamera, pengaturan lanjutan, dan pengujian.
- [Evaluasi penghitungan kendaraan](docs/counting-evaluation.md) memuat perbandingan dengan anotasi manual.
- [Panduan data video](docs/video-data-guide.md) memuat format video dan pemisahan data pengembangan serta uji.

## Lisensi

Kode dan dokumentasi proyek menggunakan [lisensi MIT](LICENSE). Library, bobot model, dan data pihak ketiga mengikuti lisensi sumbernya masing-masing.
