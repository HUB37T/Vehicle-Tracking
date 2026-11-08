# Anotasi manual

`vehicles.csv` dan `vehicles-2-test.csv` berisi anotasi awal dari cuplikan frame sumber. JSON pendamping mencatat checksum, segmen garis, cakupan, dan kasus ambigu. Lihat [hasil dan cara evaluasi](../../docs/counting-evaluation.md). Status anotasi masih menunggu peninjau kedua.

`vehicles-3-bytetrack-test.csv` memuat crossing yang dapat diperiksa dari klip antrean senja untuk [eksperimen ByteTrack](../../docs/bytetrack-fuse-score-evaluation.md). JSON pendamping mencatat cakupan dan kasus dengan titik acuan tertutup.

`holdout-istanbul-stabilized.csv` dan `holdout-hvsd-rain.csv` memuat anotasi yang dibuat sebelum [uji dua klip baru](../../docs/bytetrack-holdout-evaluation.md). JSON pendamping mencatat sumber, checksum, garis, dan batas tinjauan.

Simpan ground truth dan metadata yang boleh dibagikan di folder ini. Berkas CSV di sini dapat masuk version control.

Header GT event sesuai CONTEXT:

```csv
clip_id,manual_event_id,frame_index,timestamp_seconds,class_name,direction,notes
```

Simpan konfigurasi garis setiap klip bersama metadata. Anotasi memakai frame 0-based, waktu detik video, kelas car/motorcycle/bus/truck, serta A_TO_B/B_TO_A. Kedua anggota meninjau hitungan sebelum data uji dibekukan. Beri label eksplisit untuk anotasi sintetis yang dipakai menguji evaluator.
