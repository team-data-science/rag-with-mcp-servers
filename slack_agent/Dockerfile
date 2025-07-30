FROM python:3.10-slim
WORKDIR /app
COPY slack_agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY slack_agent/. .
CMD ["python", "bot.py"]
