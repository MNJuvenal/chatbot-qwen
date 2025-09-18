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
