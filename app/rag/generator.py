import requests
from app.config import (
    LLM_PROVIDER, OPENAI_API_KEY, OPENAI_MODEL,
    OLLAMA_MODEL, OLLAMA_BASE_URL
)

SYSTEM = """You answer questions using only the supplied document context.
If the answer is not supported by the context, say you cannot find it in the
provided documents. Do not invent facts. Cite sources as [filename, page N]."""

def generate_answer(question, contexts):
    context = "\n\n".join(
        f"[{i+1}] {x['metadata'].get('source')} page "
        f"{x['metadata'].get('page')}\n{x['text']}"
        for i, x in enumerate(contexts)
    )
    prompt = f"""{SYSTEM}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    if LLM_PROVIDER == "ollama":
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["response"]

    if not OPENAI_API_KEY:
        raise RuntimeError("Set OPENAI_API_KEY or use LLM_PROVIDER=ollama.")

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content
