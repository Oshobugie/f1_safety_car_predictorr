FROM python:3.11-slim

# Install C++ runtime library required by XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirement list and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files and pre-trained models
COPY app.py .
COPY artifacts/ artifacts/

# Hugging Face Spaces listens on port 7860 by default
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]