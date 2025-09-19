#!/bin/bash

# Fetii Chatbot Deployment Script

echo "🚀 Starting Fetii Chatbot Deployment..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop and try again."
    echo "💡 On Windows: Start Docker Desktop from the Start menu"
    echo "💡 On macOS: Start Docker Desktop from Applications"
    echo "💡 On Linux: Start Docker service with 'sudo systemctl start docker'"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating from example..."
    if [ -f "env.example" ]; then
        cp env.example .env
        echo "📝 Please edit .env file with your OpenAI API key before continuing."
        echo "🔑 Set OPENAI_API_KEY=your_actual_api_key_here"
        read -p "Press Enter after updating .env file..."
    else
        echo "❌ No env.example file found. Please create .env file manually."
        exit 1
    fi
fi

# Build the Docker image
echo "📦 Building Docker image..."
docker build -t fetii-chatbot:latest .

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully!"
else
    echo "❌ Failed to build Docker image"
    exit 1
fi

# Stop existing container if running
echo "🛑 Stopping existing container (if any)..."
docker stop fetii-chatbot 2>/dev/null || true
docker rm fetii-chatbot 2>/dev/null || true

# Run the container
echo "🏃 Starting container..."
docker run -d \
    --name fetii-chatbot \
    -p 8501:8501 \
    --env-file .env \
    -v "$(pwd)/database:/app/database" \
    -v "$(pwd)/embeddings:/app/embeddings" \
    -v "$(pwd)/data:/app/data" \
    fetii-chatbot:latest

if [ $? -eq 0 ]; then
    echo "✅ Container started successfully!"
    echo "🌐 Application is available at: http://localhost:8501"
    echo "📊 To view logs: docker logs fetii-chatbot"
    echo "🛑 To stop: docker stop fetii-chatbot"
    echo "🔄 To restart: docker restart fetii-chatbot"
    echo "🗑️  To remove: docker rm -f fetii-chatbot"
else
    echo "❌ Failed to start container"
    exit 1
fi
