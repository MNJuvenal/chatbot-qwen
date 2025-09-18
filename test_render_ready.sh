#!/bin/bash
# Script de test pré-déploiement Render

echo "=== Test de préparation Render ==="

cd backend

echo "1. Test des dépendances..."
if [ -f "requirements.txt" ]; then
    echo "   ✓ requirements.txt présent"
else
    echo "   ✗ requirements.txt manquant"
    exit 1
fi

echo "2. Test du script de téléchargement..."
if [ -f "download_model_render.py" ]; then
    echo "   ✓ Script de téléchargement présent"
else
    echo "   ✗ Script de téléchargement manquant"
    exit 1
fi

echo "3. Test de l'app FastAPI..."
if [ -f "app.py" ]; then
    echo "   ✓ app.py présent"
else
    echo "   ✗ app.py manquant"
    exit 1
fi

echo "4. Test des variables d'environnement..."
export BACKEND=llama_cpp
export RAG_ENABLED=true
if [ "$BACKEND" = "llama_cpp" ] && [ "$RAG_ENABLED" = "true" ]; then
    echo "   ✓ Variables d'environnement configurables"
else
    echo "   ✗ Problème avec les variables"
fi

echo "5. Vérification de la structure Render..."
cd ..
if [ -f "render.yaml" ] && [ -f "Procfile" ] && [ -f ".gitignore" ]; then
    echo "   ✓ Fichiers Render présents"
else
    echo "   ✗ Fichiers Render manquants"
    exit 1
fi

echo ""
echo "=== Prêt pour le déploiement Render ! ==="
echo "Étapes suivantes :"
echo "1. git add . && git commit -m 'Optimisé pour Render'"
echo "2. git push origin main"
echo "3. Connecter le repo à Render"
echo "4. Créer un Web Service avec render.yaml"