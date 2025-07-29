# Dockerfile
# ──────────────────────────────────────────────────────────────────────────────
# 1) Base image (pick cuda vs cpu as you need)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# 2) Working dir
WORKDIR /app

# 3) Copy & install dependencies
#    Make sure your requirements.txt includes fastapi, uvicorn[standard], python-multipart, pillow, torchvision, etc.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4) Copy your inference code, model definition, losses, dataset, weights & static assets
COPY inference.py     .
COPY model.py         .
COPY losses.py        .
COPY dataset.py       .
COPY deblur_model_ema.pth .
COPY static           ./static

# 5) Expose and run
EXPOSE 8000
CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8000"]
