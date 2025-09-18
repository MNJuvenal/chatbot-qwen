# Dockerfile pour Render - Chatbot Qwen avec RAG
FROM python:3.12-slim

# Variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV BACKEND=llama_cpp
ENV RAG_ENABLED=true
ENV GGUF_PATH=./models/qwen2.5-0.5b-instruct-q4_k_m.gguf
ENV PORT=10000

# Installer les dépendances système (compilateurs pour llama-cpp-python)
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

# Copier les requirements en premier pour le cache Docker
COPY backend/requirements.txt ./

# Installer les dépendances Python (séparément pour optimiser le cache)
RUN pip install --no-cache-dir --upgrade pip

# Installer d'abord les packages sans compilation
RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    uvicorn[standard]==0.30.6 \
    python-dotenv>=1.0.1 \
    numpy>=1.21.0 \
    python-multipart>=0.0.5

# Installer llama-cpp-python avec wheel pré-compilé
RUN pip install --no-cache-dir \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    llama-cpp-python>=0.2.86

# Installer le reste des dépendances
RUN pip install --no-cache-dir \
    transformers>=4.41.0 \
    torch>=2.0.0 \
    accelerate>=0.33.0 \
    sentencepiece \
    sentence-transformers>=2.2.0 \
    faiss-cpu>=1.7.0

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