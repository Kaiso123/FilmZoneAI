FROM python:3.10-slim
RUN apt-get update && apt-get install -y ffmpeg
WORKDIR /AIFilmZone
COPY app/ /AIFilmZone/app
RUN pip install --no-cache-dir -r /AIFilmZone/app/requirements.txt
COPY storage/ /AIFilmZone/storage
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
