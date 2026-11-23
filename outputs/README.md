# Keluaran lokal

Keluaran di folder ini diabaikan Git; README ini tetap dibagikan. Simulasi dapat menghasilkan `outputs/simulation.json`, berlabel sintetis dan bukan hasil inferensi.

Pipeline video memakai `outputs/<run_id>/` untuk events.csv, counts.csv, run.json, dan annotated.mp4 sesuai [rancangan teknis](../docs/project-design.md#12-format-keluaran-dan-lifecycle-run). Status selesai hanya diberikan setelah verifikasi keluaran. Run gagal dapat memiliki CSV parsial dan log encoder; periksa status metadata sebelum memakai hasil.

UI baru memakai antrean di `outputs/jobs/`: `jobs.sqlite3`, log worker, dan satu folder per pekerjaan berisi salinan video, progres/preview, serta folder run atau batch perbandingan. Pertahankan folder ini jika ingin membuka riwayat atau memberi label frame sumber. Hasil `cancelled` tetap parsial; video parsial tidak diterbitkan sebagai sukses. Hasil lama di `outputs/ui/` tetap dapat dibaca.

`heatmap.png/.npz/.json`, `quality.png/.json`, `traffic_zones.csv`, dan `traffic_zone_episodes.csv` berada di folder run yang mengaktifkan fitur terkait. Nilai heatmap adalah waktu keberadaan anchor per sel, bukan hitungan unik/kecepatan fisik.

`outputs/dataset-labels/` menyimpan revisi kotak manual, frame sumber, dan label YOLO. Ini terpisah dari event crossing. Ekspor ZIP belum menentukan split train/validation/test; pisahkan menurut video sumber sebelum melatih model. Video, bobot, pekerjaan, hasil, dan dataset tidak masuk Git.
