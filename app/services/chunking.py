from dataclasses import dataclass
from typing import Any

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document as LlamaDocument


@dataclass(slots=True)
class ChunkRecord:
    text: str
    metadata: dict[str, Any]


class TextChunker:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self._splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk_document(self, text: str, metadata: dict[str, Any]) -> list[ChunkRecord]:
        document = LlamaDocument(text=text, metadata=metadata)
        nodes = self._splitter.get_nodes_from_documents([document])

        chunks: list[ChunkRecord] = []
        for index, node in enumerate(nodes):
            chunk_text = (node.text or "").strip()
            if not chunk_text:
                continue

            node_metadata = dict(metadata)
            node_metadata.update(
                {
                    "chunk_index": index,
                    "node_id": getattr(node, "node_id", None),
                    "start_char_idx": getattr(node, "start_char_idx", None),
                    "end_char_idx": getattr(node, "end_char_idx", None),
                }
            )
            chunks.append(ChunkRecord(text=chunk_text, metadata=node_metadata))

        return chunks
