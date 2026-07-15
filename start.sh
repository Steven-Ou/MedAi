#!/bin/bash
# Start Ollama in the background
ollama serve &

# Wait until Ollama is actually responding before trying to pull
echo "Waiting for Ollama to be ready..."
until curl -s http://localhost:11434/api/tags > /dev/null; do
  sleep 5
done

# Pull the model. 
# Note: Pulling llama3.2 is enough to start; others can be pulled later.
echo "Pulling llama3.2..."
ollama pull llama3.2

# Start the FastAPI app
echo "Starting FastAPI..."
python -m uvicorn app:app --host 0.0.0.0 --port 7860