import os
import faiss
import numpy as np
import json
import re
import pickle
from typing import List, Any
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline

class FaissVectorStore:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        try:
            self.model = SentenceTransformer(embedding_model)
        except Exception as e:
            print(f"[WARN] Online model load failed ({e}). Falling back to local cache only.")
            self.model = SentenceTransformer(embedding_model, local_files_only=True)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"[INFO] Loaded embedding model: {embedding_model}")

    def _compact_text(self, text: str, max_sentences: int = 2, max_chars: int = 400) -> str:
        if not text:
            return ""
        cleaned = re.sub(r"\s+", " ", text).strip()
        # Sentence split (simple)
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        if sentences and sentences[0]:
            compact = " ".join(sentences[:max_sentences]).strip()
        else:
            compact = cleaned
        if len(compact) > max_chars:
            compact = compact[:max_chars].rsplit(" ", 1)[0] + "..."
        return compact

    def _compact_metadata_view(self) -> List[Any]:
        compact = []
        for m in self.metadata:
            text = m.get("text", "") if isinstance(m, dict) else ""
            src = m.get("source") if isinstance(m, dict) else None
            src_name = os.path.basename(src) if isinstance(src, str) else src
            compact.append({
                "summary": self._compact_text(text),
                "source": src_name,
                "page": m.get("page") if isinstance(m, dict) else None,
                "chunk_id": m.get("chunk_id") if isinstance(m, dict) else None,
            })
        return compact

    def build_from_documents(self, documents: List[Any]):
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        emb_pipe = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)
        metadatas = []
        for i, chunk in enumerate(chunks):
            meta = getattr(chunk, "metadata", {}) or {}
            metadatas.append({
                "text": chunk.page_content,
                "source": str(meta.get("source")) if meta.get("source") is not None else None,
                "page": meta.get("page"),
                "chunk_id": meta.get("chunk_id", i),
                "start_sec": meta.get("start_sec"),
                "end_sec": meta.get("end_sec"),
                "confidence": meta.get("confidence"),
            })
        self.add_embeddings(np.array(embeddings).astype('float32'), metadatas)
        self.save()
        print(f"[INFO] Vector store built and saved to {self.persist_dir}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        print(f"[INFO] Added {embeddings.shape[0]} vectors to Faiss index.")

    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        meta_json_path = os.path.join(self.persist_dir, "metadata.json")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        with open(meta_json_path, "w", encoding="utf-8") as f:
            json.dump(self._compact_metadata_view(), f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved Faiss index and metadata to {self.persist_dir}")

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        meta_json_path = os.path.join(self.persist_dir, "metadata.json")
        self.index = faiss.read_index(faiss_path)
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                self.metadata = pickle.load(f)
        elif os.path.exists(meta_json_path):
            with open(meta_json_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = []
        # Always refresh JSON to keep it compact and consistent
        if self.metadata:
            with open(meta_json_path, "w", encoding="utf-8") as f:
                json.dump(self._compact_metadata_view(), f, ensure_ascii=False, indent=2)
        print(f"[INFO] Loaded Faiss index and metadata from {self.persist_dir}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        D, I = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({"index": idx, "distance": dist, "metadata": meta})
        return results

    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Querying vector store for: '{query_text}'")
        query_emb = self.model.encode([query_text]).astype('float32')
        return self.search(query_emb, top_k=top_k)

# Example usage
if __name__ == "__main__":
    from data_loader import load_all_documents
    docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    store.build_from_documents(docs)
    store.load()
    print(store.query("What is attention mechanism?", top_k=3))
