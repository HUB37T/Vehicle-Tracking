# Bobot model lokal

Folder ini menyimpan bobot lokal YOLO11n, YOLO11s, dan YOLO26n. Bobot serta metadata sumber lokal diabaikan Git. Siapkan varian yang dibutuhkan dengan perintah di README utama.

Catat nama/checksum, asal, versi library, dan lisensi bobot yang benar-benar dipakai saat implementasi. Simulasi counting tidak menggunakan bobot model.

`python -m vehicle_counter.cli prepare-model` memakai distribusi resmi Hugging Face, mencatat revision, dan memverifikasi SHA256. Untuk `--source github`, YOLO11 memakai rilis `v8.3.0` dan YOLO26n memakai `v8.4.0`. Inferensi memakai bobot lokal dan merekam checksum dalam metadata run.

Gunakan `prepare-model --variant yolo11s` atau `prepare-model --variant yolo26n` untuk menyiapkan pembanding. Ketiga varian menggunakan bobot pretrained resmi. Berkas cache dengan metadata sumber diperiksa checksum saat `prepare-model` dipanggil ulang. [Ultralytics/YOLO11](https://huggingface.co/Ultralytics/YOLO11) · [Ultralytics/YOLO26](https://huggingface.co/Ultralytics/YOLO26)
