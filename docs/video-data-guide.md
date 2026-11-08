# Video evaluasi dan kemungkinan fine-tuning

## Kebutuhan sekarang

Baseline memakai YOLO11n pretrained dan empat kelas yang sudah tersedia: car, motorcycle, bus, truck. Tidak ada training dalam pipeline saat ini. [Dokumentasi model YOLO11](https://docs.ultralytics.com/models/yolo11/) menjelaskan bobot pretrained dan tugas deteksi yang didukung.

Siapkan satu klip pengembangan terlebih dahulu, kemudian sedikitnya tiga klip uji jika memungkinkan. Klip 30–120 detik cukup untuk memulai; usahakan total minimal 30 crossing pada data uji berisi kendaraan, serta satu klip tanpa crossing untuk memeriksa false count. Jumlah ini rencana awal tugas, bukan ukuran dataset yang menjamin performa.

Pilih rekaman MP4 dari kamera diam, pencahayaan cukup, kendaraan terlihat jelas, dan lintasan melewati garis yang bisa ditempatkan pada area pengamatan. Prioritaskan CFR, resolusi hingga 720p untuk baseline CPU, serta lebar/tinggi genap. Gunakan sumber yang mengizinkan penggunaan akademik dan catat asalnya. Simpan video di `data/videos/`; file video diabaikan Git.

## Pisahkan peran data

| Kelompok | Penggunaan |
| --- | --- |
| Pengembangan | Menempatkan garis dan menyetel ambang detector/tracker, ukuran input, serta parameter CLAHE |
| Uji akhir | Mengukur hasil setelah parameter dibekukan; berisi GT manual yang ditinjau tim |
| Fine-tuning, jika kelak diperlukan | Melatih ulang model dengan frame dan bounding box berlabel |

Pisahkan pengembangan dan uji berdasarkan rekaman atau waktu pengambilan yang berbeda. Potongan berdekatan dari video yang sama dapat sangat mirip; jangan memperlakukannya sebagai bukti generalisasi lintas kondisi.

Untuk evaluasi counting, anotasi awal berupa kejadian crossing, bukan box setiap frame:

```csv
clip_id,manual_event_id,frame_index,timestamp_seconds,class_name,direction,notes
```

Gunakan garis, kelas, arah, dan kebijakan U-turn yang sama dengan CONTEXT. Kedua anggota menghitung/meninjau secara manual lalu menyelesaikan perbedaan. Simpan metadata sumber dan konfigurasi garis bersama anotasi, lalu jalankan evaluasi model pada data tersebut. Metadata model atau confidence bukan pengganti GT.

## Kapan fine-tuning masuk akal

Pertimbangkan fine-tuning jika evaluasi memperlihatkan kegagalan deteksi yang konsisten, misalnya kendaraan lokal sering salah kelas, objek pada sudut kamera tertentu sering terlewat, atau kondisi pencahayaan target berbeda jauh dari data pretrained. Bedakan dahulu error deteksi dari error asosiasi ID atau aturan crossing; training detector tidak otomatis menyelesaikan dua masalah terakhir.

Fine-tuning membutuhkan frame beranotasi bounding box dan kelas, serta pemisahan train/validation/test berdasarkan video sebelum mengekstrak frame. Jenis anotasi ini berbeda dari GT crossing. Lingkup, jumlah data, izin sumber, dan waktu komputasi disepakati setelah baseline dievaluasi. Pelatihan dari nol belum direncanakan.

## Klip sintetis pengujian integrasi

Klip `synthetic_bus_translation.mp4` yang dibuat lokal berasal dari foto contoh resmi bus yang digeser secara buatan. Ia hanya menguji inferensi, tracking, crossing, dan keluaran pipeline. Ia bukan rekaman kendaraan bergerak di jalan, bukan data training, dan tidak dipakai untuk mengklaim akurasi proyek.
