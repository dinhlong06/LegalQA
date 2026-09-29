# LegalQA (UIT DSC 2026 · Task 2) — image tái lập kết quả private test. Đội Artiz.
#
#   docker build -t uitdsc2026-legalqa:private .
#   docker run --rm --gpus '"device=0"' -v /duong/dan/data:/data \
#     uitdsc2026-legalqa:private /data/private-official.json /data/out
#   # -> /data/out/submission.zip  (và submission_nollm.zip)
#
# Một môi trường Python duy nhất (task 2 không có xung đột version như task 1):
#   /opt/venv  transformers 5.16.1 · sentence-transformers 6.0.1 (+ datasets/accelerate/peft
#              cho scripts/train_encoders.py)
# torch 2.6.0+cu124 dùng chung từ base image (--system-site-packages).
# Console script `legalqa` cài editable vào venv -> ENTRYPOINT.

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime AS env
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
RUN python -m venv --system-site-packages /opt/venv \
 && /opt/venv/bin/pip install -r /tmp/requirements.txt

# Code (MIT) — nhỏ, đặt trước trọng số để sửa code không phải chép lại ~10 GB.
COPY LICENSE README.md pyproject.toml /app/
COPY legalqa/ /app/legalqa/
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/
COPY training_logs/ /app/training_logs/
RUN /opt/venv/bin/pip install --no-deps -e .

FROM env AS full
# Trọng số + index nằm TRONG image: lúc inference không tải gì từ mạng.
COPY index/ /app/index/
COPY models/ /app/models/
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

ENTRYPOINT ["/opt/venv/bin/legalqa"]