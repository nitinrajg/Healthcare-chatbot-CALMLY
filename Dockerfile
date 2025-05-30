FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create environment file
RUN touch .env

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create a directory for temporary file uploads
RUN mkdir -p temp_uploads && chmod 777 temp_uploads

# Expose port for the Gradio app
EXPOSE 7860

# Set empty environment variables for API keys
# ENV GROQ_API_KEY="" \
#     ELEVENLABS_API_KEY=""

# Run the application with server name and host specified for external access
CMD ["python", "app.py"]
