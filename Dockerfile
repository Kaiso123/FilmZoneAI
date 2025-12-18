FROM python:3.10-slim

# 1. Cài đặt các thư viện hệ thống cần thiết (ffmpeg cho audio, curl để check health)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Thiết lập thư mục làm việc
WORKDIR /AIFilmZone

# 3. Copy requirements trước để tận dụng Docker Cache 
COPY app/requirements.txt /AIFilmZone/app/requirements.txt
RUN pip install --no-cache-dir -r /AIFilmZone/app/requirements.txt

# 4. Copy toàn bộ source code vào
COPY app /AIFilmZone/app

# 5. Tạo thư mục storage 
RUN mkdir -p /AIFilmZone/storage

# 6. Biến môi trường mặc định 
ENV PYTHONPATH=/AIFilmZone

# 7. Expose port 
EXPOSE 8000

# Lệnh mặc định (override bởi docker-compose)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]