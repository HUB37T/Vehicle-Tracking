# Video sumber lokal

Letakkan rekaman MP4 jalan dari kamera diam di folder ini. Video diabaikan Git; README ini tetap dibagikan. Mulai dengan klip siang 30–60 detik dan kendaraan yang terlihat jelas.

Catat sumber/izin, resolusi, FPS, durasi, kondisi pengamatan, serta pemisahan data pengembangan dan uji di metadata anotasi. Klip `synthetic_bus_translation.mp4` yang dibuat lokal hanya menggeser foto contoh secara buatan untuk menguji integrasi model.

Pada 19 November 2026, pemilik proyek memberikan delapan rekaman `VID_20261119_*.mp4` dan dua video WhatsApp untuk pengujian lokal. Unduh melalui folder Drive yang dibagikan dalam percakapan proyek. Berkas asli dipertahankan di folder ini. Salinan `local-20261119-*-cfr.mp4` memakai 30 FPS dan resolusi 1280×720. Daftar sumber, ID Drive, checksum, konfigurasi, dan hasil pengujian tersedia lokal di `outputs/local-recordings-20261119/`. Dua rekaman persimpangan memiliki gerakan kamera besar yang perlu diperiksa saat menilai crossing.

Baca [panduan data](../../docs/video-data-guide.md). Baseline memerlukan MP4 CFR dengan dimensi genap untuk ekspor H.264; CLI `inspect` memeriksa metadata dan frame awal.
