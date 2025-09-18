#!/bin/bash

# Script de téléchargement de modèles Qwen GGUF optimisés pour CPU
# Usage: ./download_models.sh

cd "$(dirname "$0")/models"

echo "🔥 Téléchargement de modèles Qwen GGUF optimisés CPU..."
echo "📁 Répertoire: $(pwd)"

# URL des modèles Qwen GGUF populaires
declare -A MODELS=(
    ["qwen2.5-0.5b-instruct-q4_k_m.gguf"]="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    ["qwen2.5-1.5b-instruct-q4_k_m.gguf"]="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    ["qwen2.5-3b-instruct-q4_k_m.gguf"]="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
)

echo "Modèles disponibles :"
echo "1. Qwen2.5-0.5B (léger, ~350MB)"
echo "2. Qwen2.5-1.5B (équilibré, ~1GB)"  
echo "3. Qwen2.5-3B (performant, ~2GB)"
echo "4. Télécharger tous"

read -p "Choisissez un modèle (1-4): " choice

download_model() {
    local filename=$1
    local url=$2
    
    if [ -f "$filename" ]; then
        echo "$filename déjà présent"
        return
    fi
    
    echo "Téléchargement de $filename..."
    if command -v wget &> /dev/null; then
        wget -O "$filename" "$url" --progress=bar
    elif command -v curl &> /dev/null; then
        curl -L -o "$filename" "$url" --progress-bar
    else
        echo "wget ou curl requis pour télécharger"
        exit 1
    fi
    
    if [ $? -eq 0 ]; then
        echo "$filename téléchargé avec succès"
    else
        echo "Erreur lors du téléchargement de $filename"
        rm -f "$filename"
    fi
}

case $choice in
    1)
        download_model "qwen2.5-0.5b-instruct-q4_k_m.gguf" "${MODELS[qwen2.5-0.5b-instruct-q4_k_m.gguf]}"
        ;;
    2)
        download_model "qwen2.5-1.5b-instruct-q4_k_m.gguf" "${MODELS[qwen2.5-1.5b-instruct-q4_k_m.gguf]}"
        ;;
    3)
        download_model "qwen2.5-3b-instruct-q4_k_m.gguf" "${MODELS[qwen2.5-3b-instruct-q4_k_m.gguf]}"
        ;;
    4)
        for filename in "${!MODELS[@]}"; do
            download_model "$filename" "${MODELS[$filename]}"
        done
        ;;
    *)
        echo "Choix invalide"
        exit 1
        ;;
esac

echo ""
echo "Modèles disponibles :"
ls -lh *.gguf 2>/dev/null || echo "Aucun modèle GGUF trouvé"

echo ""
echo "Pour utiliser un modèle, configurez les variables d'environnement :"
echo "export BACKEND=llama_cpp"
echo "export GGUF_PATH=./models/qwen2.5-1.5b-instruct-q4_k_m.gguf"