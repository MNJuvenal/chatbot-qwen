# Guide de Déploiement Render - Chatbot Qwen

## Étapes de déploiement

### 1. Créer un nouveau Web Service sur Render

1. Aller sur [render.com](https://render.com)
2. Cliquer sur "New +" → "Web Service"
3. Connecter votre repository GitHub : `MNJuvenal/chatbot-qwen`

### 2. Configuration du service

**Paramètres principaux :**
- **Name** : `qwen-chatbot` (ou nom de votre choix)
- **Runtime** : `Docker`
- **Plan** : `Starter` ($7/mois - requis pour 1GB RAM)
- **Branch** : `main`

**Variables d'environnement à ajouter :**
```
BACKEND=llama_cpp
RAG_ENABLED=true
GGUF_PATH=./models/qwen2.5-0.5b-instruct-q4_k_m.gguf
RAG_MODEL=Alibaba-NLP/gte-multilingual-base
MAX_NEW_TOKENS=192
TEMPERATURE=0.7
```

### 3. Déploiement

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