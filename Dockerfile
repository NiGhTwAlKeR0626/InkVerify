
# Python base image
FROM python:3.12-slim

# Environment configuration
ENV PYTHONUNBUFFERED=1

# Application directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create uploads directory and application user
RUN mkdir -p uploads \
    && useradd -m app \
    && chown -R app:app /app

# Run as non-root user
USER app

# Railway assigns PORT dynamically
EXPOSE 8080

# Start Flask application using Gunicorn
CMD ["sh", "-c", "exec gunicorn app:app --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:${PORT:-8080}"]