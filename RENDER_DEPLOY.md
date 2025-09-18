# Configuration Render pour Chatbot Qwen

## Configuration des services Render

### Web Service Configuration :
- **Build Command**: `cd backend && pip install -r requirements.txt`
- **Start Command**: `cd backend && python download_model_render.py && uvicorn app:app --host 0.0.0.0 --port $PORT`
- **Environment**: `Python 3.12`

### Variables d'environnement Render :
```
BACKEND=llama_cpp
RAG_ENABLED=true
GGUF_PATH=./models/qwen2.5-0.5b-instruct-q4_k_m.gguf
RAG_MODEL=Alibaba-NLP/gte-multilingual-base
MAX_NEW_TOKENS=192
TEMPERATURE=0.7
```

### Plan Free Render - Limites :
- **RAM** : 512 MB (Insuffisant pour GGUF + RAG)
- **Build time** : 15 min max
- **Deploy size** : 500 MB

### Plan Starter ($7/mois) - Recommandé :
- **RAM** : 1 GB (Suffisant)
- **CPU** : 0.5 vCPU 
- **Build time** : 15 min
- **Deploy size** : 1 GB

## Optimisations pour Render :

### 1. Modèle léger recommandé
Le modèle Qwen 0.5B (469MB) est optimal pour Render.

### 2. Alternative : Backend Transformers
Si llama-cpp pose problème, basculer sur :
```
BACKEND=transformers
MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
```

### 3. Structure déploiement
```
backend/           # Racine du service Render
├── app.py        # FastAPI
├── rag_system.py # RAG
├── requirements.txt
├── download_model_render.py
└── models/       # Créé au runtime
    └── *.gguf   # Téléchargé dynamiquement
```

### 4. Frontend statique
Le frontend HTML/CSS/JS peut être déployé séparément sur :
- **Netlify/Vercel** (gratuit)
- **Render Static Site** (gratuit)

## Commandes de déploiement :

1. **Connecter repo GitHub** à Render
2. **Configurer Web Service** :
   - Root Directory: `backend`
   - Build: `pip install -r requirements.txt`
   - Start: `python download_model_render.py && uvicorn app:app --host 0.0.0.0 --port $PORT`
3. **Ajouter variables environnement**
4. **Deploy** !