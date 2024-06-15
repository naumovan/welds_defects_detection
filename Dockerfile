FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN --mount=source=./,target=/source,rw pip install /source --no-cache-dir

COPY bin.py /app/bin.py
COPY models /app/models

EXPOSE 8501

CMD ["streamlit", "run", "bin.py"]
