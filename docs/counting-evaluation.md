# Evaluasi awal penghitungan kendaraan

| Klip | Mode | Manual | Program | TP | FP | FN | F1 crossing | Error per jenis dan arah |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `vehicles` | `none` | 9 | 9 | 9 | 0 | 0 | 1,00 | 0,00% |
| `vehicles` | `clahe` | 9 | 9 | 9 | 0 | 0 | 1,00 | 0,00% |
| `vehicles-2-test` | `none` | 28 | 11 | 10 | 1 | 18 | 0,51 | 67,86% |
| `vehicles-2-test` | `clahe` | 28 | 14 | 13 | 1 | 15 | 0,62 | 57,14% |

Anotasi ini adalah tinjauan visual awal oleh satu pemeriksa. Frame diperkirakan dari rangkaian gambar sumber. Lakukan tinjauan kedua sebelum memakai hasil sebagai ground truth final. Pencocokan memakai toleransi ±0,5 detik, kelas dan arah yang sama, serta satu pasangan per event. Evaluator memaksimalkan jumlah pasangan lalu meminimalkan selisih waktu. Nilai tak terdefinisi, seperti error ketika jumlah manual nol, ditulis sebagai `null`.

`vehicles` adalah klip pengembangan penuh (538 frame, 21,52 detik). Garis dari `[0.10, 0.60]` ke `[0.90, 0.60]` meliputi kedua arah. Truk yang sudah berada di zona netral pada awal video dikeluarkan dari anotasi. Pada mobil putih yang sebagian tertutup, CLAHE mencatat crossing pada 0,36 detik dan `none` pada 0,80 detik. Keduanya masuk toleransi terhadap perkiraan manual 0,80 detik.

`vehicles-2-test` adalah uji tambahan dari rekaman berbeda. Delapan detik awal dikonversi ke 1280 × 720. Segmen `[0.25, 0.80]` ke `[0.36, 0.80]` hanya meliputi satu lajur untuk membuat peninjauan kendaraan padat lebih terarah. Anotasi awal memuat 23 mobil dan 5 truk, semuanya A → B. Posisi kendaraan dekat ujung garis dan definisi kendaraan utilitas ringan perlu ditinjau ulang. Pada potongan ini CLAHE mencocokkan tiga crossing tambahan. Perbandingan pada dua klip ini belum mencakup bus, motor, kondisi malam, atau uji akhir dengan peninjau kedua.

Video berasal dari [koleksi contoh Roboflow Supervision](https://supervision.roboflow.com/0.24.0/assets/). CSV dan JSON pendamping menyimpan anotasi, parameter garis, URL, checksum, serta catatan cakupan. Video dan keluaran lokal berada di direktori yang diabaikan Git.

| Klip | Run `none` | Run `clahe` | Laporan JSON lokal |
| --- | --- | --- | --- |
| `vehicles` | `20261108T105212Z-8bf7b0c998f1` | `20261108T105422Z-f898fc13648c` | `outputs/evaluation/vehicles.json` |
| `vehicles-2-test` | `20261108T120818Z-77847c7ebaa2` | `20261108T120836Z-1b0521d3602f` | `outputs/evaluation/vehicles-2-test.json` |

Untuk mengulang, gunakan video dengan checksum pada JSON pendamping dan `counting_line` yang sama untuk kedua mode. Contoh evaluasi untuk dua folder run yang tersedia di perangkat ini:

```powershell
$runs = @(Get-ChildItem outputs/compare-vehicles -Directory | Select-Object -ExpandProperty FullName)
.\.venv\Scripts\python.exe -m vehicle_counter.cli evaluate data/annotations/vehicles.csv @runs --clip-id vehicles --output outputs/evaluation/vehicles.json
```

Klip kedua dibuat dari delapan detik awal `vehicles-2.mp4` dengan filter FFmpeg `scale=1280:720`, encoder `libx264`, preset `fast`, dan tanpa audio. Encoding ulang yang berbeda dapat menghasilkan checksum berbeda. Cocokkan isi klip dan anotasi sebelum membuat metadata baru.
