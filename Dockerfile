# Dockerfile pour Render - Chatbot Qwen avec RAG
FROM python:3.12-slim

# Variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV BACKEND=llama_cpp
ENV RAG_ENABLED=true
ENV GGUF_PATH=./models/qwen2.5-0.5b-instruct-q4_k_m.gguf
ENV PORT=10000

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Répertoire de travail
WORKDIR /app

# Copier les requirements en premier pour le cache Docker
COPY backend/requirements.txt ./

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste de l'application
COPY backend/ ./
COPY frontend/ ../frontend/

# Créer les répertoires nécessaires
RUN mkdir -p models admin_ui

# Copier l'interface d'admin si elle existe
RUN if [ -d "admin_ui" ]; then cp -r admin_ui/* admin_ui/; fi

# Exposer le port
EXPOSE $PORT

# Script de démarrage
CMD ["sh", "-c", "python download_model_render.py && uvicorn app:app --host 0.0.0.0 --port $PORT"]