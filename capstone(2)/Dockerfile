# Use official Python image
FROM python:3.12-slim

# Set working directory in container
WORKDIR /app

# Copy all files into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8000
EXPOSE 8000

# Run FastAPI app (main.py)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
