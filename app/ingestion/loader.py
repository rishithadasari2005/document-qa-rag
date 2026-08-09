from pathlib import Path
from pypdf import PdfReader

def load_pdf(path: str) -> list[dict]:
    path = Path(path)
    reader = PdfReader(str(path))
    pages = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())
        if text:
            pages.append({
                "text": text,
                "source": path.name,
                "page": page_no
            })
    return pages
