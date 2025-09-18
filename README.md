# Chatbot Qwen (CPU) — Backend FastAPI + Frontend HTML/JS

## Démarrage rapide (Hugging Face Transformers en CPU)

1) Backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
cp .env.example .env  # optionnel: ajustez MODEL_ID
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

2) Frontend
```bash
cd ../frontend
python -m http.server 5500
# Ouvrez http://localhost:5500
```

## Démarrage rapide avec llama.cpp (Recommandé - Plus rapide)

**Option 1: Script automatique**
```bash
# Démarrage en une commande avec le script start_llama.sh
./start_llama.sh
```

Le script `start_llama.sh` configure automatiquement :
- Backend: `llama_cpp` (plus rapide que transformers)
- RAG activé pour la recherche documentaire
- Modèle GGUF: `qwen2.5-0.5b-instruct-q4_k_m.gguf`
- Serveur sur `http://localhost:8000`
- Interface chat directement accessible sur `/`
- Interface admin sur `/admin`

**Option 2: Manuel**
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export BACKEND=llama_cpp
export RAG_ENABLED=true
export GGUF_PATH="./models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
uvicorn app:app --host 0.0.0.0 --port 8000
```

Puis ouvrez `http://localhost:8000` dans votre navigateur.

## Configuration
- `.env` (backend) :
```
BACKEND=transformers
MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
MAX_NEW_TOKENS=192
TEMPERATURE=0.7
TOP_P=0.95
REPETITION_PENALTY=1.1
```
- Pour **llama.cpp** (option CPU rapide), mettez `BACKEND=llama_cpp` et `GGUF_PATH` vers un fichier `.gguf`.

**Avantages de llama.cpp :**
- ⚡ Plus rapide que transformers sur CPU
- 💾 Utilise moins de mémoire RAM
- 🔧 Modèles GGUF quantifiés (Q4_K_M recommandé)
- 🚀 Démarrage plus rapide avec `./start_llama.sh`

## Test API
```bash
curl -X POST http://localhost:8000/chat   -H 'Content-Type: application/json'   -d '{
    "messages": [
      {"role":"system","content":"Tu es un assistant utile et concis. Réponds en français."},
      {"role":"user","content":"Explique-moi ce qu\'est Qwen."}
    ],
    "max_new_tokens": 128,
    "temperature": 0.7
  }'
```

## Remarques
- Choisissez un modèle léger pour CPU (ex. `Qwen/Qwen2.5-0.5B-Instruct`). 
- Pour des réponses plus courtes: baissez `MAX_NEW_TOKENS`. 
- Historique limité côté frontend; pour des prompts longs, envisagez un résumé côté backend.
