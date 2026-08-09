from app.rag.chunker import make_chunks

PAGES = [{"text": "This is sentence one. This is sentence two. " * 100,
          "source": "test.pdf", "page": 1}]

def test_strategies_create_chunks():
    for strategy in ["fixed", "recursive", "sentence", "parent_child"]:
        chunks = make_chunks(PAGES, strategy)
        assert len(chunks) > 0
        assert all(c.text.strip() for c in chunks)
