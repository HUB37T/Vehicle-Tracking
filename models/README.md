# Bobot model lokal

Folder ini menyimpan bobot lokal YOLO11n. Di perangkat pengembangan, bobot resmi sudah diunduh dan diuji. File bobot serta metadata sumber lokal diabaikan Git; rekan yang clone repositori perlu menyiapkan sendiri melalui perintah di README utama.

Catat nama/checksum, asal, versi library, dan lisensi bobot yang benar-benar dipakai saat implementasi. Simulasi counting tidak menggunakan bobot model.

`python -m vehicle_counter.cli prepare-model` memakai distribusi resmi Hugging Face, revision yang dicatat, dan verifikasi SHA256. `--source github` menyediakan distribusi resmi GitHub v8.3.0 sebagai alternatif. Inferensi memakai bobot lokal dan merekam checksum dalam metadata run.
