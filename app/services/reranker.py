class RerankerService:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        if not candidates:
            return []

        pairs = [[query, row["chunk_text"]] for row in candidates]
        scores = self._model.predict(pairs)

        for row, score in zip(candidates, scores):
            row["score"] = float(score)

        ranked = sorted(candidates, key=lambda row: row["score"], reverse=True)
        return ranked[:top_k]
