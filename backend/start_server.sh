#!/bin/bash

# Script de lancement du chatbot Qwen avec RAG
cd "$(dirname "$0")"

echo "🚀 Lancement du Chatbot Qwen avec RAG..."
echo "📁 Répertoire: $(pwd)"

# Activer l'environnement virtuel
source .venv/bin/activate

echo "🔧 Vérification des dépendances..."
# Vérifier et installer les dépendances si nécessaire
pip install -q python-multipart faiss-cpu sentence-transformers

echo "🤖 Démarrage du serveur backend..."
echo "   - API principale: http://localhost:8000"
echo "   - Documentation: http://localhost:8000/docs"
echo "   - Interface Admin: http://localhost:8000/admin"
echo "   - Interface Chat: http://localhost:5500 (démarrer séparément)"
echo ""
echo "Appuyez sur Ctrl+C pour arrêter le serveur"

# Lancer uvicorn
uvicorn app:app --host 0.0.0.0 --port 8000 --reload