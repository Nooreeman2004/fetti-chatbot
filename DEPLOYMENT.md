# 🚀 Fetii Chatbot Deployment Guide

This guide will help you deploy the Fetii Chatbot using Docker.

## 📋 Prerequisites

1. **Docker Desktop** installed and running
   - [Download Docker Desktop](https://www.docker.com/products/docker-desktop/)
   - Make sure Docker is running before proceeding

2. **OpenAI API Key**
   - Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)

## 🛠️ Quick Deployment

### Option 1: Using Deployment Scripts (Recommended)

#### For Windows:
```bash
# Run the deployment script
deploy.bat
```

#### For Linux/macOS:
```bash
# Make script executable and run
chmod +x deploy.sh
./deploy.sh
```

### Option 2: Manual Docker Commands

1. **Create environment file:**
   ```bash
   # Copy the example environment file
   cp env.example .env
   
   # Edit .env file and add your OpenAI API key
   # Set OPENAI_API_KEY=your_actual_api_key_here
   ```

2. **Build the Docker image:**
   ```bash
   docker build -t fetii-chatbot:latest .
   ```

3. **Run the container:**
   ```bash
   docker run -d \
     --name fetii-chatbot \
     -p 8501:8501 \
     --env-file .env \
     -v "$(pwd)/database:/app/database" \
     -v "$(pwd)/embeddings:/app/embeddings" \
     -v "$(pwd)/data:/app/data" \
     fetii-chatbot:latest
   ```

### Option 3: Using Docker Compose

1. **Create environment file:**
   ```bash
   cp env.example .env
   # Edit .env with your API key
   ```

2. **Start with Docker Compose:**
   ```bash
   docker-compose up -d
   ```

## 🌐 Accessing the Application

Once deployed, the application will be available at:
- **URL:** http://localhost:8501
- **Port:** 8501

## 📊 Managing the Container

### View Logs
```bash
docker logs fetii-chatbot
```

### Stop the Container
```bash
docker stop fetii-chatbot
```

### Restart the Container
```bash
docker restart fetii-chatbot
```

### Remove the Container
```bash
docker rm -f fetii-chatbot
```

### Update the Application
```bash
# Stop and remove existing container
docker stop fetii-chatbot
docker rm fetii-chatbot

# Rebuild and restart
docker build -t fetii-chatbot:latest .
docker run -d --name fetii-chatbot -p 8501:8501 --env-file .env \
  -v "$(pwd)/database:/app/database" \
  -v "$(pwd)/embeddings:/app/embeddings" \
  -v "$(pwd)/data:/app/data" \
  fetii-chatbot:latest
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Database Configuration
DATABASE_PATH=./database/transportation.db

# ChromaDB Configuration
CHROMA_PERSIST_DIR=./embeddings/chromadb

# Streamlit Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### Volume Mounts

The container uses the following volume mounts:
- `./database` → `/app/database` (Database files)
- `./embeddings` → `/app/embeddings` (Vector embeddings)
- `./data` → `/app/data` (Raw data files)

## 🐛 Troubleshooting

### Common Issues

1. **Docker not running:**
   - Start Docker Desktop
   - On Linux: `sudo systemctl start docker`

2. **Port already in use:**
   - Change the port mapping: `-p 8502:8501`
   - Or stop the service using port 8501

3. **OpenAI API key not working:**
   - Check your `.env` file
   - Verify the API key is correct
   - Ensure you have credits in your OpenAI account

4. **Container fails to start:**
   - Check logs: `docker logs fetii-chatbot`
   - Verify all required files are present
   - Check if database and embeddings are properly initialized

### Health Check

The container includes a health check that verifies the application is running:
```bash
# Check container health
docker inspect fetii-chatbot --format='{{.State.Health.Status}}'
```

## 📁 File Structure

```
Fetii-chatbot/
├── Dockerfile              # Docker image definition
├── docker-compose.yml      # Docker Compose configuration
├── .dockerignore          # Files to exclude from Docker build
├── deploy.sh              # Linux/macOS deployment script
├── deploy.bat             # Windows deployment script
├── env.example            # Environment variables template
├── task.json              # AWS ECS task definition
├── requirements.txt       # Python dependencies
├── chatbot/               # Core chatbot modules
├── database/              # Database files and scripts
├── embeddings/            # Vector embeddings
├── ui/                    # Streamlit UI
└── data/                  # Raw data files
```

## 🚀 Production Deployment

### AWS ECS Deployment

1. **Build and push to ECR:**
   ```bash
   # Build image
   docker build -t fetii-chatbot .
   
   # Tag for ECR
   docker tag fetii-chatbot:latest YOUR_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com/fetii-chatbot:latest
   
   # Push to ECR
   docker push YOUR_ACCOUNT_ID.dkr.ecr.YOUR_REGION.amazonaws.com/fetii-chatbot:latest
   ```

2. **Update task.json:**
   - Replace `YOUR_ACCOUNT_ID` with your AWS account ID
   - Replace `YOUR_REGION` with your AWS region
   - Update the image URI in the task definition

3. **Deploy to ECS:**
   ```bash
   # Register task definition
   aws ecs register-task-definition --cli-input-json file://task.json
   
   # Create or update service
   aws ecs create-service --cluster your-cluster --service-name fetii-chatbot --task-definition fetii-chatbot
   ```

## 📞 Support

If you encounter any issues:
1. Check the container logs: `docker logs fetii-chatbot`
2. Verify your environment variables
3. Ensure all required files are present
4. Check the troubleshooting section above
