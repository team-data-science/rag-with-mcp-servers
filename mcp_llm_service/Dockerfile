FROM python:3.10-slim
WORKDIR /app

# Install curl for in-container testing
RUN apt-get update \
 && apt-get install -y curl \
 && rm -rf /var/lib/apt/lists/*


COPY mcp_llm_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mcp_llm_service/. .

CMD ["python", "main.py"]
