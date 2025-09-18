# Chatbot Qwen avec RAG - Déploiement Render

Un chatbot intelligent basé sur Qwen 2.5 avec système RAG (Retrieval-Augmented Generation) optimisé pour CPU.

## Fonctionnalités

- **Modèle Qwen 2.5** (0.5B) optimisé CPU avec quantification GGUF
- **Système RAG** avec FAISS et embeddings multilingues
- **Interface web** responsive et moderne
- **Admin RAG** pour gérer documents et FAQ
- **Backend hybride** : Transformers ou llama-cpp
- **Déploiement Render** avec téléchargement automatique des modèles

## Architecture

```
Frontend (HTML/JS) → FastAPI → llama-cpp/GGUF → Qwen 2.5
                  ↗ RAG System → FAISS Index → Documents/FAQ
```

## Déploiement sur Render

### 1. Configuration automatique
Le fichier `render.yaml` configure automatiquement :
- Plan Starter (1GB RAM requis)
- Variables d'environnement
- Commandes de build et démarrage

### 2. Processus de déploiement
1. **Fork/Clone** ce repo
2. **Connecter à Render** (GitHub integration)
3. **Créer Web Service** avec détection automatique du `render.yaml`
4. **Deploy** - Le modèle GGUF sera téléchargé automatiquement

### 3. Variables d'environnement (pré-configurées)
```env
BACKEND=llama_cpp
RAG_ENABLED=true
GGUF_PATH=./models/qwen2.5-0.5b-instruct-q4_k_m.gguf
RAG_MODEL=Alibaba-NLP/gte-multilingual-base
```

## Endpoints API

- `GET /` - Page d'accueil
- `GET /health` - Health check 
- `POST /chat` - API de chat
- `/admin` - Interface d'administration RAG
- `/docs` - Documentation API automatique

## Développement local

```bash
# Installation
cd backend
pip install -r requirements.txt

# Télécharger modèle (optionnel, fait automatiquement)
python download_model_render.py

# Lancer le serveur
uvicorn app:app --reload
```

## Optimisations Render

- **Téléchargement dynamique** : Les modèles (700MB) ne sont pas dans Git
- **Quantification Q4** : Modèle léger mais performant
- **CPU optimisé** : llama-cpp avec support AVX2
- **Health checks** : Monitoring intégré
- **Frontend unifié** : Servi depuis le backend

## Coûts

- **Plan Starter** : $7/mois (recommandé - 1GB RAM)
- **Plan Free** : Possible mais avec limitations mémoire

## Support

Le chatbot supporte :
- Conversations multi-tours
- Recherche RAG dans documents
- FAQ intégrée
- Upload de documents (.txt, .pdf, .md)
- Interface d'administration web

---

**Déployé avec ❤️ sur Render**