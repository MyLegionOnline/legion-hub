FROM python:3.15-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Chromium and WebKit render the agent `screenshot` tool (app/core/screenshot.py);
# room workspaces carry no browsers.
RUN playwright install chromium webkit --with-deps

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
