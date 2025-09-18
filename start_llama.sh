#!/bin/bash
# Script de démarrage du chatbot Qwen avec RAG

# Variables d'environnement
export BACKEND=${BACKEND:-llama_cpp}
export RAG_ENABLED=true
export GGUF_PATH="./models/qwen2.5-0.5b-instruct-q4_k_m.gguf"

echo "Démarrage du chatbot Qwen avec backend: $BACKEND"
echo "RAG activé: $RAG_ENABLED"
echo "Modèle GGUF: $GGUF_PATH"

cd backend
source .venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8000