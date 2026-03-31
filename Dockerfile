FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860
ENV APP_MODE=uvicorn

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy everything, including both app.py files
COPY . /app

# Install after source is copied
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

EXPOSE 7860 8501

CMD ["sh", "-c", "if [ \"$APP_MODE\" = \"streamlit\" ]; then streamlit run streamlit_app.py --server.port ${PORT:-7860} --server.address 0.0.0.0; else uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}; fi"]
