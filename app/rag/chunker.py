import re
from dataclasses import dataclass

@dataclass
class Chunk:
    text: str
    metadata: dict

def fixed_chunks(pages, size=800, overlap=120):
    out = []
    for p in pages:
        text = p["text"]
        start = 0
        while start < len(text):
            end = min(len(text), start + size)
            out.append(Chunk(text[start:end], dict(p)))
            if end == len(text):
                break
            start = max(0, end - overlap)
    return out

def recursive_chunks(pages, size=800, overlap=120):
    separators = ["\n\n", "\n", ". ", " ", ""]
    def split_text(text):
        if len(text) <= size:
            return [text]
        for sep in separators:
            if sep and sep in text:
                pieces = text.split(sep)
                result, current = [], ""
                for piece in pieces:
                    candidate = (current + sep + piece).strip() if current else piece.strip()
                    if len(candidate) <= size:
                        current = candidate
                    else:
                        if current:
                            result.append(current)
                        current = piece.strip()
                if current:
                    result.append(current)
                if all(len(x) <= size for x in result):
                    final = []
                    for i, x in enumerate(result):
                        if i == 0 or overlap <= 0:
                            final.append(x)
                        else:
                            prev = result[i-1]
                            final.append((prev[-overlap:] + " " + x).strip())
                    return final
        return [text[i:i+size] for i in range(0, len(text), max(1, size-overlap))]

    out = []
    for p in pages:
        for text in split_text(p["text"]):
            out.append(Chunk(text, dict(p)))
    return out

def sentence_chunks(pages, sentences_per_chunk=5):
    out = []
    for p in pages:
        sentences = re.split(r"(?<=[.!?])\s+", p["text"])
        for i in range(0, len(sentences), sentences_per_chunk):
            text = " ".join(sentences[i:i+sentences_per_chunk]).strip()
            if text:
                out.append(Chunk(text, dict(p)))
    return out

def parent_child_chunks(pages, parent_size=1600, child_size=500):
    out = []
    parents = fixed_chunks(pages, size=parent_size, overlap=0)
    for parent_id, parent in enumerate(parents):
        text = parent.text
        for start in range(0, len(text), child_size):
            child = text[start:start+child_size]
            if child.strip():
                meta = dict(parent.metadata)
                meta["parent_id"] = parent_id
                meta["parent_text"] = text[:2000]
                out.append(Chunk(child, meta))
    return out

def make_chunks(pages, strategy):
    if strategy == "fixed":
        return fixed_chunks(pages)
    if strategy == "recursive":
        return recursive_chunks(pages)
    if strategy == "sentence":
        return sentence_chunks(pages)
    if strategy == "parent_child":
        return parent_child_chunks(pages)
    raise ValueError(f"Unknown strategy: {strategy}")
