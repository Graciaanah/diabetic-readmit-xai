FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY tests/ ./tests/

# data/ is intentionally NOT copied into the image -- raw data stays out of
# the container per the Data Privacy Plan (docs/vision-document.md, Sec. 3.4).
# Mount it as a volume at runtime instead:
#   docker run -v $(pwd)/data:/app/data diabetic-readmission-pipeline
RUN mkdir -p data/raw data/processed reports

CMD ["python", "-m", "src.preprocessing.dag"]
