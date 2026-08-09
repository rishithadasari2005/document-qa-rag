import hashlib
import chromadb
from sentence_transformers import SentenceTransformer
from app.config import CHROMA_DIR, EMBEDDING_MODEL, COLLECTION_PREFIX

class VectorStore:
    def __init__(self, strategy="recursive"):
        self.strategy = strategy
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.collection = self.client.get_or_create_collection(
            f"{COLLECTION_PREFIX}_{strategy}"
        )

    def reset(self):
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(name)

    def add_chunks(self, chunks):
        if not chunks:
            return 0
        texts = [c.text for c in chunks]
        embeddings = self.model.encode(texts, normalize_embeddings=True).tolist()
        ids = []
        for i, c in enumerate(chunks):
            raw = f"{c.metadata.get('source')}|{c.metadata.get('page')}|{i}|{c.text}"
            ids.append(hashlib.md5(raw.encode()).hexdigest())
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=[c.metadata for c in chunks]
        )
        return len(chunks)

    def search(self, query, top_k=5):
        emb = self.model.encode([query], normalize_embeddings=True).tolist()
        result = self.collection.query(
            query_embeddings=emb,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        docs = result["documents"][0] if result["documents"] else []
        metas = result["metadatas"][0] if result["metadatas"] else []
        distances = result["distances"][0] if result["distances"] else []
        return [
            {"text": d, "metadata": m, "distance": dist}
            for d, m, dist in zip(docs, metas, distances)
        ]
