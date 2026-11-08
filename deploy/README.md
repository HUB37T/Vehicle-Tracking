# Deployment demo Vehicle Tracking

Image CPU menyiapkan Python, OpenCV, PyTorch, dan Streamlit di dalam kontainer. Nginx mempertahankan awalan `/vehicle-tracking/demo/` saat meneruskan HTTP dan WebSocket. Konfigurasi Streamlit lokal tetap menggunakan URL akar.

Dari checkout GitHub pada VM, siapkan bobot resmi `models/yolo11n.pt`, direktori `data/videos` untuk contoh opsional, dan `outputs` yang bisa ditulis UID 1000. Jalankan:

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
