from fastapi import FastAPI, UploadFile, File, Form
from pathlib import Path
import shutil
from app.config import DOCS_DIR
from app.rag.pipeline import ingest_file, ask

app = FastAPI(title="Document QA with RAG")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    strategy: str = Form("recursive")
):
    if not file.filename.lower().endswith(".pdf"):
        return {"error": "Only PDF files are supported."}
    target = DOCS_DIR / Path(file.filename).name
    with target.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    result = ingest_file(str(target), strategy=strategy)
    return {"filename": target.name, **result}

@app.post("/ask")
def question(question: str, strategy: str = "recursive", top_k: int = 5):
    return ask(question, strategy, top_k)
