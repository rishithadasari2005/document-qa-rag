import sys
from pathlib import Path

# ------------------------------------------------------------
# Add project root to Python path
# ------------------------------------------------------------

ROOT_DIR = Path(
    __file__
).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT_DIR)
    )

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------

from app.evaluation.dynamic_evaluator import (
    run_dynamic_evaluation
)


if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("DYNAMIC RAG EVALUATION")
    print("=" * 60)

    try:

        dataset, results = run_dynamic_evaluation(
            questions_per_page=2,
            max_questions=10
        )

        print(
            f"\nGenerated questions: "
            f"{len(dataset)}"
        )

        print("\nPerformance:")

        for strategy, result in results.items():

            print(
                f"\n{strategy}"
            )

            if result.get("error"):

                print(
                    f"ERROR: {result['error']}"
                )

                continue

            print(
                result["summary"]
            )

    except Exception as exc:

        print(
            f"\nEvaluation failed: {exc}"
        )