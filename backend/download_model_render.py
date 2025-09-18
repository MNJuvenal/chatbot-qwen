#!/usr/bin/env python3
"""
Script de téléchargement de modèle pour Render
Télécharge le modèle GGUF au démarrage pour éviter de stocker 700MB dans le repo
"""

import os
import urllib.request
import sys
from pathlib import Path

def download_model():
    """Télécharge le modèle GGUF si pas présent"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_file = models_dir / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    
    if model_file.exists():
        print(f"Modèle déjà présent: {model_file}")
        return
    
    print("Téléchargement du modèle GGUF...")
    model_url = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    
    try:
        print("Début du téléchargement...")
        def show_progress(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100.0, (block_num * block_size / total_size) * 100)
                print(f"\rProgrès: {percent:.1f}% ({block_num * block_size // 1024 // 1024} MB)", end="", flush=True)
        
        urllib.request.urlretrieve(model_url, model_file, reporthook=show_progress)
        print(f"\nModèle téléchargé: {model_file}")
        print(f"Taille: {model_file.stat().st_size // 1024 // 1024} MB")
    except Exception as e:
        print(f"\nErreur téléchargement: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_model()