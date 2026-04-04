FROM python:3.12-slim
WORKDIR /app
COPY mcp_rag_server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY mcp_rag_server/. .
CMD ["python", "server.py"]
