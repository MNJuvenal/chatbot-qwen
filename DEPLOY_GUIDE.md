# Guide de Déploiement Render - Chatbot Qwen

## Options de déploiement

### Option 1: Plan Starter avec llama-cpp (Recommandée)

**Avantages :** Plus rapide, modèles GGUF optimisés
**Coût :** $7/mois

### Option 2: Plan Free avec Transformers

**Avantages :** Gratuit
**Limitations :** Plus lent, mémoire limitée

---

## Déploiement Plan Starter (llama-cpp)

### 1. Configuration sur Render

**Paramètres :**
- **Runtime** : `Docker`
- **Plan** : `Starter`
- **Dockerfile** : `Dockerfile` (par défaut)

**Variables d'environnement :**
```
BACKEND=llama_cpp
RAG_ENABLED=true
GGUF_PATH=./models/qwen2.5-0.5b-instruct-q4_k_m.gguf
RAG_MODEL=Alibaba-NLP/gte-multilingual-base
MAX_NEW_TOKENS=192
TEMPERATURE=0.7
```

### 2. Temps de build
- **Durée** : 15-20 minutes (compilation llama-cpp + téléchargement modèle)
- **RAM utilisée** : ~800MB

---

## Déploiement Plan Free (Transformers)

### 1. Utiliser Dockerfile alternatif

Sur Render, dans les paramètres avancés :
- **Docker Command** : `docker build -f Dockerfile.free .`

**Variables d'environnement :**
```
BACKEND=transformers
RAG_ENABLED=true
MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
RAG_MODEL=Alibaba-NLP/gte-multilingual-base
```

### 2. Limitations
- **RAM** : 512MB (limite Free)
- **Performance** : Plus lent que llama-cpp
- **Pas de modèles GGUF** : Utilise Transformers standard

1. Cliquer sur "Create Web Service"
2. Le build va démarrer automatiquement
3. **Temps d'attente** : ~10-15 minutes (téléchargement du modèle 469MB)

### 4. Vérification

Une fois déployé :
- **URL principale** : `https://votre-app.onrender.com`
- **Health check** : `https://votre-app.onrender.com/health`
- **Interface admin** : `https://votre-app.onrender.com/admin`
- **API docs** : `https://votre-app.onrender.com/docs`

### 5. Résolution de problèmes

**Si le build échoue :**
- Vérifier que le plan est "Starter" (Free n'a pas assez de RAM)
- Vérifier les logs de build pour erreurs spécifiques
- Le téléchargement du modèle peut prendre 5-10 minutes

**Si l'application ne démarre pas :**
- Vérifier les variables d'environnement
- Consulter les logs de runtime
- Vérifier que le port $PORT est bien configuré

### 6. Optimisations post-déploiement

- **Custom Domain** : Configurer votre domaine personnalisé
- **SSL** : Activé automatiquement par Render
- **Monitoring** : Utiliser l'endpoint `/health`
- **Logs** : Accessible depuis le dashboard Render

## Structure finale déployée

```
https://votre-app.onrender.com/
├── /              # Interface de chat
├── /admin         # Administration RAG
├── /health        # Health check
├── /docs          # Documentation API
└── /chat          # API endpoint
```

**Le chatbot est maintenant accessible publiquement !** 🚀