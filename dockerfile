FROM python:3.10-slim
EXPOSE 8000

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY app_backend .

CMD ["fastapi", "run", "main.py"]
