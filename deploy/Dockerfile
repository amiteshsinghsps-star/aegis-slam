# AEGIS-SLAM runtime image.
#
# Targets AWS Graviton (arm64) but builds as multi-arch so the same Dockerfile
# works locally (x86) and on a c8g instance. The heavy, aarch64-fragile deps
# (pyceres, pycolmap, torch, lightglue) are OPTIONAL: the pipeline imports them
# inside try/except and falls back to a pure OpenCV-5 path with a simpler
# optimiser when they are absent. This directly addresses the PRD risk
# "pyceres/pycolmap/torch fail on aarch64 → ship the OpenCV-only path".
#
# Build (from the REPO ROOT, required — the build context must contain aegis/):
#                       docker build -t aegis-slam -f deploy/Dockerfile .
# Run (local):          docker run -p 8000:8000 aegis-slam
# Run (Graviton):       see deploy/README.md (compose sets context: ..)

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OPENCV_HEADLESS=1 \
    AEGIS_RUNS=/app/runs \
    AEGIS_DATASET=/app/Dataset/synthetic

WORKDIR /app

# OS deps: libgl/glib are needed by opencv even headless on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libglib2.0-0 curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

COPY deploy/requirements-server.txt /tmp/requirements-server.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements-server.txt \
    # Optional accelerator stack — best effort; ignored if no wheel for arch.
    && (pip install --no-cache-dir pyceres pycolmap || echo "[build] pyceres/pycolmap unavailable -> OpenCV-only optimiser") \
    && (pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu || echo "[build] torch unavailable -> classical features only") \
    && (pip install --no-cache-dir "lightglue @ git+https://github.com/cvg/LightGlue.git" || echo "[build] lightglue unavailable -> ALIKED backend disabled")

COPY . /app
# .dockerignore keeps Dataset/ out of the image; regenerate the small deterministic benchmark scene so the
# default job and judge mode work on a fresh instance.
RUN python -m aegis.synth --out /app/Dataset/synthetic --frames 200 --no_mp4

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -fsS http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "aegis.server:app", "--host", "0.0.0.0", "--port", "8000"]
