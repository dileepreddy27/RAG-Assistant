from app.services.chunking import TextChunker


def test_chunker_splits_text_and_keeps_metadata() -> None:
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    text = " ".join(["RAG improves relevance."] * 50)

    chunks = chunker.chunk_document(text, {"source_filename": "sample.txt"})

    assert len(chunks) > 1
    assert chunks[0].metadata["source_filename"] == "sample.txt"
    assert chunks[0].metadata["chunk_index"] == 0
