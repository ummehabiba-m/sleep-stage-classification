FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (CPU-only torch to keep image small)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html \
    && pip install --no-cache-dir -r requirements.txt

# Copy source and model
COPY src/ ./src/
COPY models/best_model.pth ./models/
COPY configs/default.yaml ./configs/

# Expose inference API
EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
