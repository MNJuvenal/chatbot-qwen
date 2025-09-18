import os
import json
import pickle
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
from dataclasses import dataclass

# RAG avec modèle d'embeddings Qwen
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

@dataclass
class Document:
    """Représente un document dans l'index RAG"""
    content: str
    metadata: Dict[str, Any]
    source: str
    doc_id: str

class QwenRAGSystem:
    """
    Système RAG "Full Qwen" avec modèle d'embeddings multilingue
    Compatible avec votre FastAPI existant
    """
    
    def __init__(
        self,
        data_dir: str = "./rag_data",
        model_name: str = "Alibaba-NLP/gte-multilingual-base",  # Modèle Qwen multilingue
        index_file: str = "faiss_index.bin",
        docs_file: str = "documents.pkl"
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.model_name = model_name
        self.index_file = self.data_dir / index_file
        self.docs_file = self.data_dir / docs_file
        
        # Initialisation du modèle d'embeddings
        self.model = None
        self.index = None
        self.documents: List[Document] = []
        
        self._check_dependencies()
        self._load_or_create_index()
    
    def _check_dependencies(self):
        """Vérifie les dépendances nécessaires"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("pip install sentence-transformers")
        if not FAISS_AVAILABLE:
            raise ImportError("pip install faiss-cpu")
    
    def _load_or_create_index(self):
        """Charge l'index existant ou en crée un nouveau"""
        print(f"[RAG] Initialisation du modèle d'embeddings : {self.model_name}")
        # Ajouter trust_remote_code=True pour les modèles personnalisés
        self.model = SentenceTransformer(self.model_name, trust_remote_code=True)
        
        # Charger l'index existant
        if self.index_file.exists() and self.docs_file.exists():
            try:
                self.index = faiss.read_index(str(self.index_file))
                with open(self.docs_file, 'rb') as f:
                    self.documents = pickle.load(f)
                print(f"[RAG] Index chargé : {len(self.documents)} documents")
                return
            except Exception as e:
                print(f"[RAG] Erreur chargement index : {e}")
        
        # Créer un nouvel index
        self._create_empty_index()
        
        # Auto-indexation des fichiers existants
        self._auto_index_files()
    
    def _create_empty_index(self):
        """Crée un index FAISS vide"""
        # Dimension d'embedding du modèle
        test_embedding = self.model.encode(["test"])
        dimension = test_embedding.shape[1]
        
        # Index FAISS avec Inner Product (normalisé = cosine similarity)
        self.index = faiss.IndexFlatIP(dimension)
        print(f"[RAG] Nouvel index FAISS créé (dimension: {dimension})")
    
    def _auto_index_files(self):
        """Auto-indexation des fichiers dans le dossier rag_data"""
        faq_file = self.data_dir / "faq.json"
        docs_dir = self.data_dir / "docs"
        
        # Indexer FAQ si présente
        if faq_file.exists():
            self._index_faq(faq_file)
        
        # Indexer documents si présents
        if docs_dir.exists():
            for file_path in docs_dir.glob("*.txt"):
                self._index_text_file(file_path)
            for file_path in docs_dir.glob("*.md"):
                self._index_text_file(file_path)
        
        self._save_index()
    
    def _index_faq(self, faq_file: Path):
        """Indexe un fichier FAQ JSON"""
        try:
            with open(faq_file, 'r', encoding='utf-8') as f:
                faq_data = json.load(f)
            
            for item in faq_data.get('faq', []):
                question = item.get('question', '')
                answer = item.get('answer', '')
                
                # Indexer la question pour la recherche
                doc = Document(
                    content=f"Q: {question}\nR: {answer}",
                    metadata={
                        "type": "faq",
                        "question": question,
                        "answer": answer
                    },
                    source="faq.json",
                    doc_id=f"faq_{len(self.documents)}"
                )
                self._add_document(doc)
            
            print(f"[RAG] FAQ indexée : {len(faq_data.get('faq', []))} Q&R")
        except Exception as e:
            print(f"[RAG] Erreur indexation FAQ : {e}")
    
    def _index_text_file(self, file_path: Path):
        """Indexe un fichier texte en chunks"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Découpage en chunks de 500 caractères avec overlap
            chunks = self._split_text(content, chunk_size=500, overlap=50)
            
            for i, chunk in enumerate(chunks):
                doc = Document(
                    content=chunk,
                    metadata={
                        "type": "document",
                        "file": file_path.name,
                        "chunk_id": i
                    },
                    source=str(file_path),
                    doc_id=f"{file_path.stem}_{i}"
                )
                self._add_document(doc)
            
            print(f"[RAG] Document indexé : {file_path.name} ({len(chunks)} chunks)")
        except Exception as e:
            print(f"[RAG] Erreur indexation {file_path.name} : {e}")
    
    def _split_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Découpe le texte en chunks avec overlap"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Éviter de couper au milieu d'un mot
            if end < len(text):
                last_space = chunk.rfind(' ')
                if last_space > chunk_size // 2:
                    chunk = chunk[:last_space]
                    end = start + last_space
            
            chunks.append(chunk.strip())
            start = max(start + chunk_size - overlap, end)
        
        return chunks
    
    def _add_document(self, doc: Document):
        """Ajoute un document à l'index"""
        # Générer l'embedding
        embedding = self.model.encode([doc.content])
        
        # Normaliser pour utiliser Inner Product comme cosine similarity
        faiss.normalize_L2(embedding)
        
        # Ajouter à l'index FAISS
        self.index.add(embedding.astype('float32'))
        
        # Ajouter aux documents
        self.documents.append(doc)
    
    def _save_index(self):
        """Sauvegarde l'index et les documents"""
        if self.index and len(self.documents) > 0:
            faiss.write_index(self.index, str(self.index_file))
            with open(self.docs_file, 'wb') as f:
                pickle.dump(self.documents, f)
            print(f"[RAG] Index sauvegardé : {len(self.documents)} documents")
    
    def search(self, query: str, k: int = 3) -> List[Document]:
        """Recherche les documents les plus pertinents"""
        if not self.index or len(self.documents) == 0:
            return []
        
        # Générer embedding de la requête
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Recherche dans l'index
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Récupérer les documents
        results = []
        for i, score in zip(indices[0], scores[0]):
            if i < len(self.documents) and score > 0.3:  # Seuil de pertinence
                results.append(self.documents[i])
        
        return results
    
    def get_context_for_query(self, query: str, max_tokens: int = 1000) -> str:
        """Génère le contexte RAG pour une requête"""
        relevant_docs = self.search(query, k=5)
        
        if not relevant_docs:
            return ""
        
        context_parts = []
        total_length = 0
        
        for doc in relevant_docs:
            content = doc.content
            
            # Éviter de dépasser la limite de tokens
            if total_length + len(content) > max_tokens:
                remaining = max_tokens - total_length
                if remaining > 100:  # Assez d'espace pour un bout utile
                    content = content[:remaining] + "..."
                else:
                    break
            
            context_parts.append(content)
            total_length += len(content)
        
        if context_parts:
            context = "\n\n".join(context_parts)
            return f"Informations pertinentes :\n{context}\n\n"
        
        return ""
    
    def add_faq_item(self, question: str, answer: str):
        """Ajoute une nouvelle FAQ à l'index"""
        doc = Document(
            content=f"Q: {question}\nR: {answer}",
            metadata={
                "type": "faq",
                "question": question,
                "answer": answer,
                "uploaded": True  # Marquer comme uploadé dynamiquement
            },
            source="uploaded",
            doc_id=f"faq_uploaded_{len(self.documents)}"
        )
        self._add_document(doc)
        self._save_index()
        print(f"[RAG] FAQ ajoutée : {question}")
    
    def add_text_document(self, content: str, filename: str, metadata: dict = None):
        """Ajoute un document texte à l'index"""
        if metadata is None:
            metadata = {}
        
        # Découpage en chunks
        chunks = self._split_text(content, chunk_size=500, overlap=50)
        
        for i, chunk in enumerate(chunks):
            doc = Document(
                content=chunk,
                metadata={
                    "type": "document",
                    "file": filename,
                    "chunk_id": i,
                    "uploaded": True,
                    **metadata
                },
                source=f"uploaded/{filename}",
                doc_id=f"upload_{filename}_{i}"
            )
            self._add_document(doc)
        
        self._save_index()
        print(f"[RAG] Document uploadé indexé : {filename} ({len(chunks)} chunks)")
        return len(chunks)
    
    def reindex_all(self):
        """Ré-indexe tous les documents"""
        print("[RAG] Ré-indexation complète...")
        
        # Sauvegarder les documents uploadés dynamiquement
        uploaded_docs = []
        for doc in self.documents:
            if doc.metadata.get("uploaded", False):
                uploaded_docs.append(doc)
        
        self.documents.clear()
        self._create_empty_index()
        self._auto_index_files()
        
        # Restaurer les documents uploadés
        for doc in uploaded_docs:
            self.documents.append(doc)
            embedding = self.model.encode([doc.content])
            faiss.normalize_L2(embedding)
            self.index.add(embedding.astype('float32'))
        
        self._save_index()
        print(f"[RAG] Ré-indexation terminée - {len(self.documents)} documents")
    
    def clear_all_documents(self):
        """Supprime tous les documents de l'index"""
        print("[RAG] Suppression de tous les documents...")
        self.documents.clear()
        self._create_empty_index()
        self._save_index()
        print("[RAG] Tous les documents supprimés")
    
    def get_documents_info(self):
        """Retourne les informations sur tous les documents"""
        docs_info = []
        for doc in self.documents:
            docs_info.append({
                "doc_id": doc.doc_id,
                "source": doc.source,
                "type": doc.metadata.get("type", "unknown"),
                "uploaded": doc.metadata.get("uploaded", False),
                "preview": doc.content[:100] + "..." if len(doc.content) > 100 else doc.content,
                "metadata": doc.metadata
            })
        return docs_info
