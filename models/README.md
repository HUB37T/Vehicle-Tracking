# Bobot model lokal

Folder ini menyimpan bobot lokal YOLO11n dan YOLO11s opsional. Di perangkat pengembangan, kedua bobot resmi sudah diunduh dan diuji. File bobot serta metadata sumber lokal diabaikan Git; rekan yang clone repositori perlu menyiapkan sendiri melalui perintah di README utama.

Catat nama/checksum, asal, versi library, dan lisensi bobot yang benar-benar dipakai saat implementasi. Simulasi counting tidak menggunakan bobot model.

`python -m vehicle_counter.cli prepare-model` memakai distribusi resmi Hugging Face, revision yang dicatat, dan verifikasi SHA256. `--source github` menyediakan distribusi resmi GitHub v8.3.0 sebagai alternatif. Inferensi memakai bobot lokal dan merekam checksum dalam metadata run.

Gunakan `prepare-model --variant yolo11s` untuk pembanding. Bobot berasal dari [Ultralytics/YOLO11](https://huggingface.co/Ultralytics/YOLO11), belum dilatih ulang oleh tim. YOLO11n tetap default; perbandingan dua klip awal belum menunjukkan keunggulan counting YOLO11s yang konsisten. Berkas cache dengan metadata sumber diperiksa checksum saat `prepare-model` dipanggil ulang.
