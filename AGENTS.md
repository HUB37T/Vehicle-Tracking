# Instruksi untuk AI agent

## Acuan proyek

Sebelum merancang, mengimplementasikan, menguji, atau meninjau perubahan, baca [CONTEXT.md](CONTEXT.md). Dokumen itu memuat tujuan, lingkup MVP, istilah, aturan crossing, kontrak data, pembagian kerja, dan kriteria selesai. Baca [README.md](README.md) untuk status aktual dan petunjuk menjalankan proyek ketika tersedia.

Instruksi terbaru pemilik proyek mengungguli rancangan awal. Tandai asumsi yang berubah dan perbarui bagian `CONTEXT.md` yang terkait agar kedua anggota tim memakai acuan yang sama.

## Koordinasi perubahan

- Periksa kode dan perubahan lokal sebelum mengedit; pertahankan pekerjaan rekan yang belum digabung.
- Ikuti pembagian tugas A/B pada bagian 17 `CONTEXT.md` ketika tugas diberikan kepada salah satu agent tersebut.
- Selaraskan perubahan berkas bersama, kontrak data, makna hitungan, dan model/tracker dengan rekan yang terdampak sebelum integrasi.
- Simpan logika counting di modul domain, dengan event log sebagai sumber ringkasan UI dan ekspor.
- Catat video, bobot, dan keluaran analisis di lokasi lokal sesuai dokumentasi. Masukkan sumber dan cara memperolehnya ke dokumentasi, bukan berkas besar atau rahasia ke Git.

## Verifikasi dan handoff

Untuk perubahan kode, jalankan pemeriksaan yang relevan dengan perilaku yang diubah. Untuk counting, gunakan skenario pada bagian 16 `CONTEXT.md`; saat integrasi, periksa kesesuaian event, UI, CSV, dan video hasil. Untuk perubahan dokumentasi, periksa konsistensi isi, tautan lokal, dan diff.

Laporkan perubahan, pemeriksaan yang benar-benar dijalankan, hasil aktual, serta keterbatasan. Sebutkan secara jelas jika inferensi model atau alur video nyata belum diuji. Angka ilustrasi dan fixture diberi label; metrik proyek berasal dari evaluasi yang benar-benar dijalankan.
