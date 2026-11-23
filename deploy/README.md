# Deployment demo Vehicle Tracking

Image CPU menyiapkan Python, OpenCV, PyTorch, dan Streamlit di dalam kontainer. Nginx mempertahankan awalan `/vehicle-tracking/demo/` saat meneruskan HTTP dan WebSocket. Konfigurasi Streamlit lokal tetap menggunakan URL akar.

Image demo memakai NumPy 2.2.6, OpenCV 4.12.0.88, dan runtime kompatibilitas Polars untuk CPU VM yang belum mendukung x86-64-v2. PyTorch memakai jalur CPU generik saat NNPACK tidak tersedia.

Pemeriksaan crop MobileNet memakai kernel CPU portabel untuk mendukung depthwise convolution pada CPU VM ini.

Image menyertakan ekstra DeepSORT (`deep-sort-realtime==1.3.2`, `scipy==1.18.1`) sehingga pilihan **DeepSORT (eksperimen)** dapat digunakan di UI. ByteTrack tetap pilihan awal. Fitur tampilan DeepSORT juga memakai kernel portabel pada CPU generik; uji inisialisasi dan analisis klip pendek sebelum mengganti kontainer aktif. DeepSORT lebih berat dan belum direkomendasikan untuk klip panjang di VM bersama ini.

Dari checkout GitHub pada VM, siapkan bobot resmi `models/yolo11n.pt`, direktori `data/videos` untuk contoh opsional, dan `outputs` yang bisa ditulis UID 1000.

Versi dengan pilihan model memerlukan `models/yolo11s.pt` agar perbandingan YOLO11n/YOLO11s aktif. Siapkan dengan `python -m vehicle_counter.cli prepare-model --variant yolo11s` sebelum menjalankan kontainer, lalu verifikasi checksum metadata sumber. Pilihan YOLO11s tetap eksperimen dan memerlukan waktu lebih lama pada VM. Untuk preset antrean, sediakan `data/videos/vehicles-3-bytetrack-test.mp4` sesuai metadata lokal tim; jangan memasukkan video/bobot ke Git.

Pertahankan bind mount `outputs` ketika mengganti kontainer agar riwayat dan catatan manual tetap tersedia. Editor garis memerlukan paket `src/vehicle_counter/ui/editor/index.html`, yang harus ikut dalam konteks build dan penyalinan sumber. Uji halaman editor melalui awalan URL publik, bukan hanya healthcheck. Jalankan uji singkat CLI pada CPU VM sebelum aktivasi; keberhasilan pengujian lokal saja belum memeriksa kompatibilitas CPU server.

Untuk menampilkan profil **Pemulihan crossing**, siapkan `models/mobilenet_v3_large-5c1a4163.pth` dan salah satu klip `holdout-hvsd-rain.mp4` atau `holdout-istanbul-stabilized.mp4` di `data/videos`. Verifikasi checksum klip terhadap JSON anotasi dan gunakan [petunjuk bobot](../docs/bytetrack-fuse-score-evaluation.md#uji-lanjutan-dengan-fuse_score-true). Di UI, **Baseline** dan **Pemulihan crossing** memakai garis dan CLAHE yang sama untuk klip tersebut. Pilih **Kustom** untuk mengatur garis sendiri.

```sh
docker build -f deploy/Dockerfile -t vehicle-tracking:demo .
docker run -d --name vehicle-tracking-demo --restart unless-stopped \
  --cpus=1.25 --memory=1400m --memory-swap=1400m --pids-limit=256 \
  --read-only --tmpfs /tmp:size=160m,mode=1777 \
  --cap-drop=ALL --security-opt=no-new-privileges \
  -e STREAMLIT_BROWSER_SERVER_ADDRESS=pred-autoscaler.bccdev.id \
  -e STREAMLIT_BROWSER_SERVER_PORT=443 \
  -p 127.0.0.1:18501:8501 \
  -v "$PWD/models:/app/models:ro" \
  -v "$PWD/data/videos:/app/data/videos:ro" \
  -v "$PWD/outputs:/app/outputs" vehicle-tracking:demo
```

Cek endpoint `/vehicle-tracking/demo/_stcore/health` di port 18501. Pasang `vehicle-tracking-nginx.conf` sebagai snippet pada server Nginx yang sudah menerima trafik domain demo. Validasi dengan `nginx -t`, lalu muat ulang konfigurasi. Verifikasi halaman, aset dan WebSocket melalui HTTPS. Semua perubahan rute berada di bawah `/vehicle-tracking/demo/`. Untuk demo VM bersama, gunakan klip pendek dengan resolusi sekitar 720p.

## Pembaruan dengan runtime yang sudah tersedia

Jika ruang VM terbatas, [Dockerfile.update](Dockerfile.update) dapat memakai image Vehicle Tracking yang sudah teruji di host tersebut. Image sumber harus memiliki runtime kompatibel, UID 1000, healthcheck, dan perintah Streamlit dari Dockerfile utama. Gunakan tag sumber yang tetap, simpan image sebelumnya untuk pemulihan, dan lakukan uji sebelum mengganti kontainer. Contoh dari akar checkout pada VM:

```sh
docker build -f deploy/Dockerfile.update \
  --build-arg RUNTIME_IMAGE=vehicle-tracking:38648bf \
  --build-arg SOURCE_COMMIT="$(git rev-parse HEAD)" \
  -t "vehicle-tracking:$(git rev-parse --short HEAD)" .
```

Pembaruan ini menyalin kode/config/UI dan memasang DeepSORT tanpa mengganti NumPy/OpenCV/PyTorch kompatibel yang telah diuji. Untuk host baru, gunakan Dockerfile utama. Pertahankan batas sumber daya, mount, dan rute dari perintah kontainer di atas; pembaruan Vehicle Tracking tidak memerlukan perubahan konfigurasi atau restart layanan WasteNot.

## Deployment antrean dan heatmap 16 November 2026

Demo publik menggunakan image `vehicle-tracking:queue-90b9fc6565de` (ID diawali `0547600379b5`). Snapshot berasal dari workspace lokal. Arsip dan checksum sumber tersimpan di `/home/dev/vehicle-tracking-releases/queue-90b9fc6565de/`. Kontainer sebelumnya disimpan dengan nama `vehicle-tracking-backup-queue-90b9fc6565de`.

Worker menulis status SQLite, sumber pekerjaan, dan hasil ke bind mount `outputs`. Pengaturan `VEHICLE_COUNTER_PUBLIC_DEMO=1` membatasi riwayat dan ekspor label pada sesi browser pengirim. Slot inferensi berlaku bersama untuk pekerjaan dari UI. Analisis CLI yang dijalankan langsung berlangsung di luar antrean.

Kandidat pada CPU VM memproses 120 frame dari lima detik awal klip antrean. Hasilnya satu crossing, dua zona, dan heatmap PNG yang valid. Pekerjaan yang dibatalkan saat menunggu tidak menghasilkan run. Pekerjaan berikutnya selesai setelah slot tersedia. Pemeriksaan ini memakai model nyata YOLO11n dan ByteTrack dengan `fuse_score: true`.

FFT pemeriksaan kualitas memakai DFT `float32` OpenCV karena FFT NumPy menghasilkan nilai non-finite pada CPU VM. Uji kandidat diulang setelah perbaikan. Pemeriksaan eksternal HTTPS untuk halaman, aset JavaScript, dan healthcheck lulus. Konfigurasi Nginx WasteNot memiliki checksum yang sama dan endpoint tersebut merespons 502 sebelum maupun sesudah deployment.

Pengujian antarmuka dengan Streamlit AppTest di kontainer aktif mengirim uji tiga detik melalui tombol UI. Run `20261116T053055Z-23695d32db82` selesai melalui antrean dan menampilkan tab serta tombol unduh Heatmap. Sesi pengujian kedua tidak memiliki akses ke riwayat sesi pertama.

## Deployment sebelumnya pada 16 November 2026

Image `vehicle-tracking:170f2ab-snapshot-fe3aa4fe684b` (ID diawali `bf0fb30a1402`) memuat snapshot sebelum fitur antrean dan heatmap. Sumber beserta checksum per berkas tersimpan di `/home/dev/vehicle-tracking-releases/170f2ab-snapshot-fe3aa4fe684b/`. Cadangan dari pembaruan sebelumnya tersimpan sebagai `vehicle-tracking-backup-20261116-fe3aa4fe684b`.

185 pengujian lokal, lint, format, dan pemeriksaan diff lulus sebelum build. Kandidat diuji terpisah pada port loopback 18502 sebelum mengganti layanan utama. Uji inferensi aktual pada CPU VM:

- ByteTrack + YOLO11n + preset antrean: 120 frame / 5.005 detik, satu crossing, status akhir `congested`, satu episode 4.004–5.005 detik. Run `20261116T023424Z-eb86f018fb0b`; `fuse_score: true` terverifikasi.
- YOLO11s + ByteTrack: tiga frame awal klip hujan selesai. Ini smoke test kompatibilitas, bukan evaluasi akurasi atau analisis video penuh.
- DeepSORT + YOLO11n: tiga frame awal klip hujan selesai. Ini smoke test pemuatan dan inferensi, bukan evaluasi tracking video penuh.

Healthcheck kontainer dan HTTPS publik lulus. Melalui browser publik, preview/editor pada awalan URL, pembukaan hasil dari riwayat, indikator/durasi macet, dan video dari detik ketiga berhasil diperiksa. Bobot YOLO11s dan klip antrean tersedia lokal di VM; data hasil lama tetap berada pada bind mount yang sama.

WasteNot sudah merespons HTTP 502 sebelum pembaruan ini. Ketiga checksum konfigurasi Nginx WasteNot dan respons awal tersebut identik setelah aktivasi. Deployment ini tidak mengubah konfigurasi Nginx atau memulai ulang layanan WasteNot; gangguan awal itu belum ditangani.

## Validasi VM 8 November 2026

Kode aplikasi `bbd83b7` aktif sebagai image `vehicle-tracking:bbd83b7` (image ID diawali `bd31d67b6de5`) dengan batas CPU/memori dan mount yang sama. Image pembaruan berbasis `vehicle-tracking:38648bf`; kontainer lama disimpan dalam keadaan berhenti sebagai `vehicle-tracking-backup-38648bf-20261108` untuk pemulihan.

Pada CPU VM yang melaporkan kemampuan `DEFAULT`, inisialisasi fitur MobileNetV2 dan asosiasi DeepSORT berhasil tanpa akses jaringan. Analisis penuh 194 frame klip hujan selesai untuk kedua tracker: ByteTrack tujuh event (`20261108T151527Z-6a3f50348ce7`) dan DeepSORT delapan event (`20261108T151903Z-4ed2a8edae12`). Jumlah frame dan audit sesuai metadata; hasil sama dengan pengujian lokal, termasuk hitung ganda SUV pada DeepSORT yang dicatat di [laporan evaluasi](../docs/tracker-safety-evaluation.md). Ini pemeriksaan deploy, bukan bukti peningkatan akurasi. Uji dijalankan dengan satu CPU dan memori 1000 MB; layanan aktif tetap dibatasi 1,25 CPU dan 1400 MB.

Healthcheck kontainer lulus. Checksum ketiga konfigurasi Nginx WasteNot dan PID ketiga layanannya identik sebelum/sesudah aktivasi. Tidak ada konfigurasi Nginx atau layanan WasteNot yang diubah maupun dimulai ulang.

Pemeriksaan publik melalui HTTPS dan WebSocket memverifikasi halaman/aset, pilihan DeepSORT yang tersedia, dan analisis ByteTrack profil pemulihan hujan. Run publik `20261108T152604Z-4c7df75d9ae1` selesai dengan tujuh event dan 194 frame. Keenam unduhan (metadata, event, rekap, audit, video, panel bukti) merespons HTTP 200; jumlah event/audit/rekap/summary sama. Video memiliki header MP4 dan panel bukti memiliki header PNG yang valid; pipeline telah memverifikasi decode video penuh sebelum run selesai. Endpoint publik WasteNot tetap HTTP 200.
