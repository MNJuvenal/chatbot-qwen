# Dockerfile simple pour Chatbot Qwen
FROM python:3.12-slim

# Variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV BACKEND=llama_cpp
ENV RAG_ENABLED=true
ENV GGUF_PATH=./models/qwen2.5-0.5b-instruct-q4_k_m.gguf
ENV PORT=10000

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gcc \
    g++ \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Répertoire de travail
WORKDIR /app

# Copier les fichiers de l'application
COPY backend/ ./
COPY frontend/ ../frontend/

# Installer les dépendances Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Créer le répertoire pour les modèles
RUN mkdir -p models

# Exposer le port
EXPOSE $PORT

# Démarrer l'application
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port $PORT"]