FROM python:3.10-slim

# 1. Cài đặt thư viện hệ thống
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /AIFilmZone

# 2. Copy requirements ra thư mục tạm (tmp)
COPY app/requirements.txt /tmp/requirements.txt

# 3. Cài đặt thư viện với Cache Mounthe
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt

# 4. Copy toàn bộ source code
COPY app /AIFilmZone/app

# 5. Tạo thư mục storage 
RUN mkdir -p /AIFilmZone/storage

# 6. Env & CMD
ENV PYTHONPATH=/AIFilmZone
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]