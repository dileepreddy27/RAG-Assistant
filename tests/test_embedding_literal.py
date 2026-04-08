from app.services.embeddings import EmbeddingService


def test_pgvector_literal_format() -> None:
    literal = EmbeddingService.to_pgvector_literal([0.1, -0.2, 0.3])
    assert literal.startswith("[")
    assert literal.endswith("]")
    assert "," in literal
