import os
from typing import List, Literal, Optional
from dataclasses import dataclass
import json
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Import du système RAG
from rag_system import QwenRAGSystem

load_dotenv()

# =================== Config ===================
BACKEND = os.getenv("BACKEND", "llama_cpp")  # "transformers" ou "llama_cpp" (défaut pour Render)
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
GGUF_PATH = os.getenv("GGUF_PATH", "./models/qwen2.5-0.5b-instruct-q4_k_m.gguf")

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 192))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
TOP_P = float(os.getenv("TOP_P", 0.95))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", 1.1))

# Configuration RAG
RAG_ENABLED = os.getenv("RAG_ENABLED", "true").lower() == "true"
RAG_MODEL = os.getenv("RAG_MODEL", "Alibaba-NLP/gte-multilingual-base")  # Modèle d'embeddings Qwen

SYSTEM_DEFAULT = "Tu es un assistant utile, factuel et concis. Réponds toujours en français."

# =================== FastAPI ===================
app = FastAPI(title="Qwen Chat API (CPU)", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =================== I/O Schemas ===================
class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    repetition_penalty: float | None = None
    use_rag: bool = True  # Active le RAG par défaut

class ChatResponse(BaseModel):
    reply: str

class FAQItem(BaseModel):
    question: str
    answer: str

class FAQBatchRequest(BaseModel):
    faq_items: List[FAQItem]

# =================== Prompt utils (Qwen chat) ===================
def build_qwen_prompt(msgs: List[Message], rag_context: str = "") -> str:
    lines = []
    
    # Ajouter le contexte RAG au prompt système si présent
    system_content = SYSTEM_DEFAULT
    if rag_context:
        system_content = f"{SYSTEM_DEFAULT}\n\n{rag_context}Utilise ces informations pour répondre si elles sont pertinentes pour la question."
    
    # Vérifier s'il y a déjà un message système
    has_system = any(m.role == "system" for m in msgs)
    if not has_system:
        lines += ["<|im_start|>system", system_content, "<|im_end|>"]
    
    for m in msgs:
        if m.role == "system":
            # Enrichir le message système existant avec le contexte RAG
            content = m.content
            if rag_context:
                content = f"{content}\n\n{rag_context}Utilise ces informations pour répondre si elles sont pertinentes pour la question."
            lines += [f"<|im_start|>{m.role}", content, "<|im_end|>"]
        else:
            lines += [f"<|im_start|>{m.role}", m.content, "<|im_end|>"]
    
    lines.append("<|im_start|>assistant")
    return "\n".join(lines)

# =================== Backends ===================
@dataclass
class Backend:
    name: str
    def generate(self, messages: List[Message], **gen):
        raise NotImplementedError

class TransformersBackend(Backend):
    def __init__(self, model_id: str):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id)
        self.model.eval()
        self.device = torch.device("cpu")  # CPU forcé
        self.model.to(self.device)
        self.name = f"transformers:{model_id}"

    def generate(self, messages: List[Message], rag_context: str = "", **gen):
        prompt = build_qwen_prompt(messages, rag_context)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=gen.get("max_new_tokens", 192),
                do_sample=True,
                temperature=gen.get("temperature", 0.7),
                top_p=gen.get("top_p", 0.95),
                repetition_penalty=gen.get("repetition_penalty", 1.1),
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(output[0], skip_special_tokens=False)
        after = text.split("<|im_start|>assistant")[-1]
        reply = after.split("<|im_end|>")[0].strip()
        return reply

class LlamaCppBackend(Backend):
    def __init__(self, gguf_path: str):
        from llama_cpp import Llama
        # n_ctx = 4096 par défaut ; ajustez selon vos besoins
        self.llm = Llama(model_path=gguf_path, n_ctx=4096, vocab_only=False)
        self.name = f"llama_cpp:{gguf_path}"

    def generate(self, messages: List[Message], rag_context: str = "", **gen):
        prompt = build_qwen_prompt(messages, rag_context)
        out = self.llm(
            prompt,
            max_tokens=gen.get("max_new_tokens", 192),
            temperature=gen.get("temperature", 0.7),
            top_p=gen.get("top_p", 0.95),
            repeat_penalty=gen.get("repetition_penalty", 1.1),
            stop=["<|im_end|>", "<|im_start|>"],
        )
        reply = out["choices"][0]["text"].strip()
        return reply

# Sélection
if BACKEND == "llama_cpp":
    engine = LlamaCppBackend(GGUF_PATH)
else:
    engine = TransformersBackend(MODEL_ID)

print(f"[Boot] Backend prêt: {engine.name}")

# Initialisation RAG
rag_system = None
if RAG_ENABLED:
    try:
        rag_system = QwenRAGSystem(model_name=RAG_MODEL)
        print(f"[Boot] RAG système prêt : {RAG_MODEL}")
    except Exception as e:
        print(f"[Boot] RAG désactivé : {e}")
        rag_system = None

# =================== Routes ===================

@app.get("/", response_class=HTMLResponse)
def get_frontend():
    """Servir la page d'accueil du chat"""
    try:
        with open("../frontend/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Frontend non trouvé</h1><p>API disponible sur <a href='/docs'>/docs</a></p>",
            status_code=404
        )

@app.get("/admin", response_class=HTMLResponse)
def get_admin():
    """Servir l'interface d'administration"""
    try:
        with open("admin_ui/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Interface admin non trouvée</h1><p>API disponible sur <a href='/docs'>/docs</a></p>",
            status_code=404
        )

@app.get("/api")
def api_info():
    """Informations sur l'API"""
    return {"message": "Chatbot Qwen API", "frontend": "/", "admin": "/admin", "docs": "/docs"}

@app.get("/health")
def health_check():
    """Health check pour Render et monitoring"""
    return {
        "status": "healthy",
        "backend": engine.name if 'engine' in globals() else "not initialized",
        "rag_enabled": rag_system is not None,
        "rag_documents": getattr(rag_system, 'index_size', 0) if rag_system else 0
    }

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    gen = dict(
        max_new_tokens=req.max_new_tokens or MAX_NEW_TOKENS,
        temperature=req.temperature or TEMPERATURE,
        top_p=req.top_p or TOP_P,
        repetition_penalty=req.repetition_penalty or REPETITION_PENALTY,
    )
    
    # Récupération du contexte RAG si activé
    rag_context = ""
    if req.use_rag and rag_system and len(req.messages) > 0:
        # Utiliser la dernière question de l'utilisateur pour la recherche RAG
        user_messages = [m for m in req.messages if m.role == "user"]
        if user_messages:
            last_question = user_messages[-1].content
            rag_context = rag_system.get_context_for_query(last_question)
            if rag_context:
                print(f"[RAG] Contexte trouvé pour: {last_question[:50]}...")
    
    reply = engine.generate(req.messages, rag_context=rag_context, **gen)
    return ChatResponse(reply=reply)

# Route pour gérer la FAQ dynamiquement
@app.post("/rag/add_faq")
def add_faq(question: str, answer: str):
    """Ajoute une nouvelle FAQ au système RAG"""
    if not rag_system:
        return {"error": "RAG system not enabled"}
    
    rag_system.add_faq_item(question, answer)
    return {"success": True, "message": f"FAQ ajoutée: {question}"}

# Route pour ré-indexer les documents
@app.post("/rag/reindex")
def reindex_documents():
    """Ré-indexe tous les documents RAG"""
    if not rag_system:
        return {"error": "RAG system not enabled"}
    
    rag_system.reindex_all()
    return {"success": True, "message": "Documents ré-indexés"}

# Route pour tester la recherche RAG
@app.post("/rag/search")
def search_documents(query: str, k: int = 3):
    """Recherche dans les documents RAG"""
    if not rag_system:
        return {"error": "RAG system not enabled"}
    
    results = rag_system.search(query, k)
    return {
        "query": query,
        "results": [
            {
                "content": doc.content,
                "source": doc.source,
                "metadata": doc.metadata
            }
            for doc in results
        ]
    }

# =================== Upload Routes ===================
@app.post("/upload/document")
async def upload_document(file: UploadFile = File(...)):
    """Upload d'un document (.txt, .md) pour indexation RAG"""
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not enabled")
    
    # Vérifier le type de fichier
    allowed_extensions = ['.txt', '.md']
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Type de fichier non supporté. Utilisez: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Lire le contenu du fichier
        content = await file.read()
        text_content = content.decode('utf-8')
        
        # Utiliser la nouvelle méthode pour ajouter le document
        chunks_count = rag_system.add_text_document(text_content, file.filename)
        
        return {
            "success": True,
            "message": f"Document '{file.filename}' uploadé et indexé avec succès",
            "filename": file.filename,
            "size": len(text_content),
            "chunks": chunks_count
        }
        
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Fichier non valide (encodage UTF-8 requis)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'upload: {str(e)}")

@app.post("/upload/faq")
async def upload_faq_batch(request: FAQBatchRequest):
    """Upload d'une liste de FAQ au format JSON"""
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not enabled")
    
    try:
        # Ajouter chaque FAQ
        added_count = 0
        for faq_item in request.faq_items:
            rag_system.add_faq_item(faq_item.question, faq_item.answer)
            added_count += 1
        
        return {
            "success": True,
            "message": f"{added_count} FAQ ajoutées avec succès",
            "count": added_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'ajout des FAQ: {str(e)}")

@app.post("/upload/faq_single")
async def upload_single_faq(request: dict):
    """Ajoute une FAQ unique"""
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not enabled")
    
    try:
        question = request.get("question", "").strip()
        answer = request.get("answer", "").strip()
        
        if not question or not answer:
            raise HTTPException(status_code=400, detail="Question et réponse requises")
            
        rag_system.add_faq_item(question, answer)
        return {
            "success": True,
            "message": "FAQ ajoutée avec succès",
            "question": question,
            "answer": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.get("/rag/status")
def get_rag_status():
    """Obtient le statut du système RAG"""
    if not rag_system:
        return {"enabled": False, "documents": 0}
    
    return {
        "enabled": True,
        "documents": len(rag_system.documents),
        "model": rag_system.model_name,
        "index_size": rag_system.index.ntotal if rag_system.index else 0
    }

@app.get("/rag/documents")
def list_documents():
    """Liste tous les documents indexés"""
    if not rag_system:
        return {"error": "RAG system not enabled"}
    
    docs_info = []
    for doc in rag_system.documents:
        docs_info.append({
            "doc_id": doc.doc_id,
            "source": doc.source,
            "type": doc.metadata.get("type", "unknown"),
            "preview": doc.content[:100] + "..." if len(doc.content) > 100 else doc.content
        })
    
    return {"documents": docs_info, "total": len(docs_info)}

@app.delete("/rag/clear")
def clear_all_documents():
    """Supprime tous les documents indexés"""
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not enabled")
    
    try:
        # Vider l'index et les documents
        rag_system.documents.clear()
        rag_system._create_empty_index()
        rag_system._save_index()
        
        return {"success": True, "message": "Tous les documents ont été supprimés"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")
