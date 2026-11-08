FROM python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    HOME=/tmp YOLO_CONFIG_DIR=/tmp/ultralytics

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY constraints.txt ./
RUN python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -c constraints.txt

COPY README.md pyproject.toml requirements-ui.txt ./
COPY src/ src/
RUN python -m pip install -r requirements-ui.txt -c constraints.txt && python -m pip check

COPY app.py ./
COPY configs/ configs/
COPY .streamlit/config.toml .streamlit/config.toml
RUN mkdir -p models data/videos outputs && chown 1000:1000 outputs
USER 1000:1000
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/vehicle-tracking/demo/_stcore/health', timeout=4).close()"]
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.headless=true", "--server.address=0.0.0.0", "--server.baseUrlPath=vehicle-tracking/demo", "--server.fileWatcherType=none"]
