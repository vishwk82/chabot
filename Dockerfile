# Dockerfile

FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the remaining code
COPY . /app

# Expose port 8000
EXPOSE 8000

# Set environment variable for openai key in production (or pass via CLI)
# ENV OPENAI_API_KEY="sk-..."

# By default, run the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
