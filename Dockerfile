FROM python:3.11-slim

LABEL maintainer="Tokuma Gebissa <tokumag>"
LABEL description="Edge Computer Vision & Industrial Defect Detection Pipeline"

WORKDIR /app

# Install minimal OS dependencies for computer vision and headless display
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and configurations
COPY . .

# Run internal test suite to verify container integrity
RUN python -m pytest tests/ -v

# Streamlit port
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
