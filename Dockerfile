FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/documents/ ./data/documents/

RUN python -m src.build_index

EXPOSE 7860

CMD ["python", "-m", "src.app_ui"]