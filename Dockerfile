# Works on Hugging Face Spaces (Docker SDK, port 7860), Fly.io, Railway, any container host.
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PORT=7860
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p uploads && useradd -m app && chown -R app /app
USER app
EXPOSE 7860
CMD ["sh", "-c", "gunicorn app:app --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:${PORT}"]
