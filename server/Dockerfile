# Base Image 
FROM python:3.11-slim

# Set Working Directory 
WORKDIR /app

# Install System Dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Project Files
COPY . .

# Install Python Dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose Port (IMPORTANT for HF)
EXPOSE 7860

# Run Server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]