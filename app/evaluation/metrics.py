def is_hit(metadata, expected_sources):
    source = metadata.get("source")
    page = metadata.get("page")

    for item in expected_sources:

        # Exact source match only
        if isinstance(item, str):
            if source == item:
                return True

        # Source + page match
        elif isinstance(item, dict):
            expected_source = item.get("source")
            expected_page = item.get("page")

            if source != expected_source:
                continue

            # If page is specified, require exact page
            if expected_page is not None:
                if page == expected_page:
                    return True
            else:
                return True

    return False


def hit_at_k(results, expected_sources, k):
    top_results = results[:k]

    return int(
        any(
            is_hit(result["metadata"], expected_sources)
            for result in top_results
        )
    )


def reciprocal_rank(results, expected_sources):

    for rank, result in enumerate(results, start=1):

        if is_hit(result["metadata"], expected_sources):
            return 1.0 / rank

    return 0.0


def evaluate_queries(store, dataset, k_values=(1, 3, 5)):

    rows = []

    for item in dataset:

        question = item["question"]
        expected_sources = item["expected_sources"]

        results = store.search(
            question,
            max(k_values)
        )

        # Find first correct result
        correct_rank = None

        for rank, result in enumerate(results, start=1):

            if is_hit(
                result["metadata"],
                expected_sources
            ):
                correct_rank = rank
                break

        row = {
            "question": question,
            "expected_sources": expected_sources,
            "correct_rank": correct_rank,
            "mrr": reciprocal_rank(
                results,
                expected_sources
            ),
        }

        for k in k_values:

            row[f"hit@{k}"] = hit_at_k(
                results,
                expected_sources,
                k
            )

        rows.append(row)

    return rows


def aggregate(rows):

    if not rows:
        return {}

    metrics = [
        key
        for key in rows[0]
        if key.startswith("hit@")
        or key == "mrr"
    ]

    summary = {}

    for metric in metrics:

        summary[metric] = round(
            sum(row[metric] for row in rows)
            / len(rows),
            4
        )

    return summary