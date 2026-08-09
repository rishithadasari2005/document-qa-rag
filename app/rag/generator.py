import requests

from app.config import (
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
)


SYSTEM = """You answer questions using only the supplied document context.

If the answer is not supported by the context, say you cannot find it in
the provided documents.

Do not invent facts.

Cite sources as [filename, page N].
"""


def build_prompt(question, contexts):
    context = "\n\n".join(
        f"[{i + 1}] {x['metadata'].get('source')} "
        f"page {x['metadata'].get('page')}\n{x['text']}"
        for i, x in enumerate(contexts)
    )

    return f"""{SYSTEM}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


def generate_with_ollama(prompt):
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )

    response.raise_for_status()

    return response.json()["response"]


def generate_with_openai(prompt):
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )

    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": SYSTEM,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    return response.choices[0].message.content


def generate_answer(question, contexts):
    prompt = build_prompt(question, contexts)

    if LLM_PROVIDER.lower() == "ollama":
        return generate_with_ollama(prompt)

    if LLM_PROVIDER.lower() == "openai":
        return generate_with_openai(prompt)

    raise RuntimeError(
        f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}"
    )