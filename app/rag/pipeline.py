from pathlib import Path
from app.ingestion.loader import load_pdf
from app.rag.chunker import make_chunks
from app.rag.vectorstore import VectorStore
from app.rag.generator import generate_answer

def ingest_file(path, strategy="recursive", reset=False):
    pages = load_pdf(path)
    store = VectorStore(strategy)
    if reset:
        store.reset()
    chunks = make_chunks(pages, strategy)
    count = store.add_chunks(chunks)
    return {"pages": len(pages), "chunks": count, "strategy": strategy}

def ingest_directory(directory, strategy="recursive", reset=False):
    directory = Path(directory)
    store = VectorStore(strategy)
    if reset:
        store.reset()
    total_pages = total_chunks = 0
    for path in directory.glob("*.pdf"):
        pages = load_pdf(path)
        chunks = make_chunks(pages, strategy)
        total_pages += len(pages)
        total_chunks += store.add_chunks(chunks)
    return {"pages": total_pages, "chunks": total_chunks, "strategy": strategy}

def ask(question, strategy="recursive", top_k=5):
    store = VectorStore(strategy)
    contexts = store.search(question, top_k)
    answer = generate_answer(question, contexts)
    return {"answer": answer, "contexts": contexts}
