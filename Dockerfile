FROM python:3.12-slim

# ffmpeg is a hard runtime dependency (audio normalization + chunking)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY frontend ./frontend

RUN mkdir -p /app/uploads
ENV UPLOAD_DIR=/app/uploads

EXPOSE 8000

# --workers 1 because the job store is in-process memory (see app/job_manager.py).
# Scale horizontally only after swapping in a shared job store (Redis/DB).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
