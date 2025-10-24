# Use a Python base image compatible with Hugging Face Spaces
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for Playwright + Crawl4AI
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    unzip \
    git \
    xvfb \
    libnss3 \
    libatk-bridge2.0-0 \
    libx11-xcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm-dev \
    libasound2 \
    libatk1.0-0 \
    libxkbcommon0 \
    libcups2 \
    libgtk-3-0 \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright Chromium (used by Crawl4AI)
RUN playwright install --with-deps chromium

# Copy the entire app (including .env)
COPY . .

# Expose the port expected by Hugging Face (7860)
EXPOSE 7860

# Hugging Face expects the app to listen on port 7860
ENV PORT=7860

# Command to run FastAPI with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
