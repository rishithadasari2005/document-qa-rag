import json
import re
from pathlib import Path

from pypdf import PdfReader

from app.config import DOCS_DIR


# ============================================================
# LLM HELPERS
# ============================================================

def get_llm_provider():
    """
    Reads the LLM provider from environment variables.
    Defaults to Ollama so the project works without OpenAI credits.
    """
    import os

    return os.getenv("LLM_PROVIDER", "ollama").lower()


def call_ollama(prompt: str) -> str:
    """
    Calls a locally running Ollama model.
    """

    import os
    import requests

    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    url = os.getenv(
        "OLLAMA_URL",
        "http://localhost:11434/api/generate"
    )

    response = requests.post(
        url,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3
            }
        },
        timeout=180
    )

    response.raise_for_status()

    data = response.json()

    return data.get("response", "")


def call_openai(prompt: str) -> str:
    """
    Optional OpenAI support.
    """

    import os
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )

    client = OpenAI(
        api_key=api_key
    )

    response = client.chat.completions.create(
        model=os.getenv(
            "OPENAI_MODEL",
            "gpt-4o-mini"
        ),
        messages=[
            {
                "role": "system",
                "content": (
                    "You generate evaluation questions "
                    "for document retrieval systems."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content


def call_llm(prompt: str) -> str:

    provider = get_llm_provider()

    if provider == "ollama":
        return call_ollama(prompt)

    if provider == "openai":
        return call_openai(prompt)

    raise ValueError(
        f"Unsupported LLM_PROVIDER: {provider}"
    )


# ============================================================
# JSON PARSER
# ============================================================

def extract_json(text: str):
    """
    Extract JSON from an LLM response.
    Handles ```json ... ``` responses as well.
    """

    text = text.strip()

    # Remove markdown fences
    text = re.sub(
        r"```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"```\s*",
        "",
        text
    )

    # Find JSON array
    start = text.find("[")

    end = text.rfind("]")

    if start == -1 or end == -1:
        raise ValueError(
            "Could not find JSON array in LLM response."
        )

    json_text = text[start:end + 1]

    return json.loads(json_text)


# ============================================================
# PDF READING
# ============================================================

def read_pdf_pages(pdf_path: Path):
    """
    Reads a PDF and returns:

    [
        {
            "page": 1,
            "text": "..."
        }
    ]
    """

    reader = PdfReader(str(pdf_path))

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text() or ""

        text = text.strip()

        if not text:
            continue

        pages.append(
            {
                "page": page_number,
                "text": text
            }
        )

    return pages


# ============================================================
# QUESTION GENERATION
# ============================================================

def generate_questions_for_page(
    text: str,
    page_number: int,
    source_name: str,
    questions_per_page: int = 2
):
    """
    Generate evaluation questions from a single PDF page.
    """

    # Avoid sending extremely large pages to the LLM
    text = text[:12000]

    prompt = f"""
You are an expert RAG evaluation dataset generator.

You are given one page from a document.

Your task is to create {questions_per_page}
useful questions that can test semantic retrieval.

Document source:
{source_name}

Page number:
{page_number}

Page content:
----------------
{text}
----------------

Create questions that are:

1. Answerable from this page.
2. Natural questions a user might ask.
3. Not copied word-for-word from the document.
4. A mixture of factual and paraphrased questions.
5. Specific enough that the correct page can be identified.
6. Different from each other.

Return ONLY valid JSON.

Required format:

[
  {{
    "question": "question here",
    "expected_sources": [
      {{
        "source": "{source_name}",
        "page": {page_number}
      }}
    ]
  }}
]
"""

    raw_response = call_llm(prompt)

    try:
        questions = extract_json(raw_response)
    except Exception as exc:
        print(
            f"Could not parse generated questions "
            f"for page {page_number}: {exc}"
        )
        return []

    valid_questions = []

    for item in questions:

        if not isinstance(item, dict):
            continue

        question = item.get("question")

        if not question:
            continue

        valid_questions.append(
            {
                "question": str(question).strip(),
                "expected_sources": [
                    {
                        "source": source_name,
                        "page": page_number
                    }
                ]
            }
        )

    return valid_questions


# ============================================================
# DYNAMIC DATASET GENERATION
# ============================================================

def generate_dataset_from_directory(
    docs_dir: Path = DOCS_DIR,
    questions_per_page: int = 2,
    max_questions: int = 10
):
    """
    Dynamically generates an evaluation dataset
    from all PDFs currently present in the documents directory.

    No questions.json is required.
    """

    docs_dir = Path(docs_dir)

    pdf_files = sorted(
        docs_dir.glob("*.pdf")
    )

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF documents found in {docs_dir}"
        )

    dataset = []

    for pdf_path in pdf_files:

        print(
            f"\nAnalyzing: {pdf_path.name}"
        )

        pages = read_pdf_pages(
            pdf_path
        )

        for page in pages:

            if len(dataset) >= max_questions:
                break

            generated = generate_questions_for_page(
                text=page["text"],
                page_number=page["page"],
                source_name=pdf_path.name,
                questions_per_page=questions_per_page
            )

            for question in generated:

                if len(dataset) >= max_questions:
                    break

                dataset.append(question)

        if len(dataset) >= max_questions:
            break

    # Remove duplicate questions
    unique = []
    seen = set()

    for item in dataset:

        question = item["question"].lower()

        if question in seen:
            continue

        seen.add(question)

        unique.append(item)

    return unique[:max_questions]


# ============================================================
# CLI TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("DYNAMIC RAG QUESTION GENERATOR")
    print("=" * 60)

    try:

        dataset = generate_dataset_from_directory(
            questions_per_page=2,
            max_questions=10
        )

        print(
            f"\nGenerated {len(dataset)} questions.\n"
        )

        for index, item in enumerate(
            dataset,
            start=1
        ):

            print(
                f"{index}. {item['question']}"
            )

            print(
                f"   Ground truth: "
                f"{item['expected_sources']}"
            )

        print("\n" + "=" * 60)

    except Exception as exc:

        print(
            f"\nERROR: {exc}"
        )