FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git xfsprogs && rm -rf /var/lib/apt/lists/*

# Suppress safe.directory warnings when running as root over foreign-owned volumes
RUN git config --global --add safe.directory '*' && \
    git config --global user.name "Legion" && \
    git config --global user.email "hub@mylegion.online" && \
    git config --global init.defaultBranch main

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8100"]
