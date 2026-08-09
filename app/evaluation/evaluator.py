import json
from pathlib import Path

import pandas as pd

from app.config import EVAL_DIR
from app.evaluation.metrics import (
    evaluate_queries,
    aggregate
)
from app.rag.vectorstore import VectorStore


def run_evaluation(
    strategy="recursive",
    dataset=None
):
    """
    Run retrieval evaluation using a supplied
    dynamic evaluation dataset.

    The evaluator no longer depends on questions.json.
    """

    # --------------------------------------------------------
    # Validate dataset
    # --------------------------------------------------------

    if dataset is None:

        raise ValueError(
            "No evaluation dataset supplied. "
            "Generate dynamic questions first."
        )

    if not dataset:

        raise ValueError(
            "Evaluation dataset is empty."
        )

    # --------------------------------------------------------
    # Create vector store
    # --------------------------------------------------------

    store = VectorStore(
        strategy
    )

    # --------------------------------------------------------
    # Evaluate retrieval
    # --------------------------------------------------------

    rows = evaluate_queries(
        store,
        dataset
    )

    summary = aggregate(
        rows
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    result_dir = EVAL_DIR / "results"

    result_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    details_path = (
        result_dir /
        f"{strategy}_details.csv"
    )

    summary_path = (
        result_dir /
        f"{strategy}_summary.json"
    )

    pd.DataFrame(
        rows
    ).to_csv(
        details_path,
        index=False
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2
        ),
        encoding="utf-8"
    )

    return summary, rows