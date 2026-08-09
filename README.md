# Document QA with RAG

An end-to-end Retrieval-Augmented Generation system for document question answering with an evaluation harness for comparing retrieval performance across chunking strategies.

## Features

- PDF ingestion
- Semantic embeddings with Sentence Transformers
- Persistent Chroma vector database
- RAG question answering
- Source/page citations
- Fixed, recursive, sentence and parent-child chunking
- Hit@1, Hit@3, Hit@5 and MRR
- CSV/JSON evaluation reports
- Streamlit UI
- FastAPI backend
- Automated tests
- Docker support
- OpenAI or local Ollama generation

## 1. Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install:

```bash
pip install -r requirements.txt
```

Create `.env` from `.env.example`.

For OpenAI:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-4o-mini
```

Or use Ollama:

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2
```

## 2. Run the UI

```bash
streamlit run frontend/streamlit_app.py
```

Open the displayed local URL.

## 3. Run the API

```bash
uvicorn app.main:app --reload
```

## 4. Ingestion from command line

Put PDFs in `data/documents/`.

```bash
python scripts/ingest.py --strategy recursive --reset
```

Try:

```bash
python scripts/ingest.py --strategy fixed --reset
python scripts/ingest.py --strategy sentence --reset
python scripts/ingest.py --strategy parent_child --reset
```

## 5. Evaluation

Edit:

`data/evaluation/questions.json`

Example:

```json
[
  {
    "question": "What was the company's revenue in 2025?",
    "expected_sources": [
      {
        "source": "annual_report.pdf",
        "page": 24
      }
    ]
  }
]
```

Then run:

```bash
python scripts/evaluate.py --strategy recursive
```

Run all strategies:

```bash
python scripts/evaluate.py --strategy fixed
python scripts/evaluate.py --strategy recursive
python scripts/evaluate.py --strategy sentence
python scripts/evaluate.py --strategy parent_child
```

Results are written to:

`data/evaluation/results/`

Each strategy produces:

- `<strategy>_details.csv`
- `<strategy>_summary.json`

## Metrics

**Hit@K** is 1 when at least one relevant source appears in the top K retrieved chunks.

**MRR (Mean Reciprocal Rank)** rewards a relevant result appearing near the top:

`MRR = average(1 / rank_of_first_relevant_result)`

## Suggested experiment

Use 20–50 questions from the same document collection and compare:

| Strategy | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---:|---:|---:|---:|
| Fixed | run experiment | run experiment | run experiment | run experiment |
| Recursive | run experiment | run experiment | run experiment | run experiment |
| Sentence | run experiment | run experiment | run experiment | run experiment |
| Parent-child | run experiment | run experiment | run experiment | run experiment |

Do not manually enter performance numbers. Use the generated evaluation results.

## Tests

```bash
pytest -q
```

## Docker

Create `.env`, then:

```bash
docker compose up --build
```

Open:

`http://localhost:8501`

## Architecture

```text
PDF
 │
 ▼
PyMuPDF/PyPDF
 │
 ▼
Chunking Strategy
 │
 ▼
Sentence Transformer
 │
 ▼
ChromaDB
 │
 ▼
Top-K Retriever
 │
 ├──► Context + Sources
 │
 ▼
LLM
 │
 ▼
Grounded Answer

Evaluation Dataset
 │
 ▼
Retriever
 │
 ▼
Hit@K + MRR
 │
 ▼
Compare Chunking Strategies
```

## Resume description

**Document QA with RAG**
- Built a RAG-based document question-answering system using semantic embeddings, ChromaDB, and LLM-based response generation.
- Developed an evaluation harness to measure Hit@1/3/5 and MRR across a curated question set.
- Compared fixed, recursive, sentence-based, and parent-child chunking strategies to analyze their impact on retrieval performance.
