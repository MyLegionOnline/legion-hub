FROM python:3.12-slim
COPY smoke.py /smoke.py
CMD ["python", "/smoke.py"]
