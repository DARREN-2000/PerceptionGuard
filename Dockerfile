# CPU image. Reproduces the exact environment the reported numbers were
# measured in: no GPU, no network at runtime, headless OpenCV.
#
# The GPU/TensorRT image is NOT provided as a Dockerfile because it has never
# been built or benchmarked here. Writing one would imply a validation that
# did not happen. docs/DEPLOYMENT.md documents the intended path instead.
FROM python:3.11-slim

# libglib2 is the one system library opencv-python-headless still links against.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency layer first so source edits do not invalidate the pip cache.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY tests/ ./tests/

RUN pip install --no-cache-dir -e .

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Default: run the gate suite. It is fully self-contained (the scenes are
# rendered, not downloaded), so `docker run <image>` is a complete
# verification of the build with no dataset mounting required.
CMD ["python", "scripts/smoke_m1.py"]
