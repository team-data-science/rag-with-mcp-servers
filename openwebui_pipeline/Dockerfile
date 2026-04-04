FROM python:3.12-slim
WORKDIR /app

COPY openwebui_pipeline/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY openwebui_pipeline/. .

CMD ["python", "main.py"]
