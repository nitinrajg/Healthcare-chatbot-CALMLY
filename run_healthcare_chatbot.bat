@echo off
echo 🚀 Building Docker image...
docker build -t healthcare-chatbot .

if %errorlevel% equ 0 (
    echo ✅ Build successful. Running the container on port 7860...
    docker run -p 7860:7860 healthcare-chatbot
) else (
    echo ❌ Build failed. Please check your Dockerfile and source code.
)
pause
