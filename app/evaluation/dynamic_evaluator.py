from app.evaluation.question_generator import (
    generate_dataset_from_directory
)

from app.evaluation.evaluator import (
    run_evaluation
)


STRATEGIES = [
    "fixed",
    "recursive",
    "sentence",
    "parent_child"
]


def generate_dynamic_dataset(
    questions_per_page=2,
    max_questions=10
):
    """
    Generate a fresh evaluation dataset
    from the current documents.
    """

    return generate_dataset_from_directory(
        questions_per_page=questions_per_page,
        max_questions=max_questions
    )


def run_dynamic_evaluation(
    questions_per_page=2,
    max_questions=10
):
    """
    Generate questions dynamically and
    evaluate all chunking strategies.
    """

    # --------------------------------------------------------
    # Step 1: Generate questions
    # --------------------------------------------------------

    dataset = generate_dynamic_dataset(
        questions_per_page=questions_per_page,
        max_questions=max_questions
    )

    if not dataset:
        raise RuntimeError(
            "No questions were generated."
        )

    # --------------------------------------------------------
    # Step 2: Evaluate strategies
    # --------------------------------------------------------

    results = {}

    for strategy in STRATEGIES:

        print(
            f"\nRunning {strategy}..."
        )

        try:

            summary, rows = run_evaluation(
                strategy=strategy,
                dataset=dataset
            )

            results[strategy] = {
                "summary": summary,
                "rows": rows
            }

        except Exception as exc:

            results[strategy] = {
                "summary": {},
                "rows": [],
                "error": str(exc)
            }

    return dataset, results