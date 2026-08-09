from app.evaluation.metrics import hit_at_k, reciprocal_rank

def test_hit_at_k():
    results = [
        {"metadata": {"source": "wrong.pdf", "page": 1}},
        {"metadata": {"source": "right.pdf", "page": 3}},
    ]
    expected = [{"source": "right.pdf", "page": 3}]
    assert hit_at_k(results, expected, 1) == 0
    assert hit_at_k(results, expected, 3) == 1
    assert reciprocal_rank(results, expected) == 0.5
