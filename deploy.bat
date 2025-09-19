@echo off
REM Fetii Chatbot Deployment Script for Windows

echo 🚀 Starting Fetii Chatbot Deployment...

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not running. Please start Docker Desktop and try again.
    echo 💡 Start Docker Desktop from the Start menu
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist ".env" (
    echo ⚠️  .env file not found. Creating from example...
    if exist "env.example" (
        copy env.example .env >nul
        echo 📝 Please edit .env file with your OpenAI API key before continuing.
        echo 🔑 Set OPENAI_API_KEY=your_actual_api_key_here
        pause
    ) else (
        echo ❌ No env.example file found. Please create .env file manually.
        pause
        exit /b 1
    )
)

REM Build the Docker image
echo 📦 Building Docker image...
docker build -t fetii-chatbot:latest .

if %errorlevel% neq 0 (
    echo ❌ Failed to build Docker image
    pause
    exit /b 1
)

echo ✅ Docker image built successfully!

REM Stop existing container if running
echo 🛑 Stopping existing container (if any)...
docker stop fetii-chatbot >nul 2>&1
docker rm fetii-chatbot >nul 2>&1

REM Run the container
echo 🏃 Starting container...
docker run -d ^
    --name fetii-chatbot ^
    -p 8501:8501 ^
    --env-file .env ^
    -v "%cd%/database:/app/database" ^
    -v "%cd%/embeddings:/app/embeddings" ^
    -v "%cd%/data:/app/data" ^
    fetii-chatbot:latest

if %errorlevel% neq 0 (
    echo ❌ Failed to start container
    pause
    exit /b 1
)

echo ✅ Container started successfully!
echo 🌐 Application is available at: http://localhost:8501
echo 📊 To view logs: docker logs fetii-chatbot
echo 🛑 To stop: docker stop fetii-chatbot
echo 🔄 To restart: docker restart fetii-chatbot
echo 🗑️  To remove: docker rm -f fetii-chatbot
pause
