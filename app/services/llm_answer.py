from app.core.config import Settings
from app.models.schemas import ConversationMessage

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


class LLMAnswerService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._llm = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=0.1,
        )
        self._prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a RAG assistant that answers questions strictly from provided context. "
                        "If context is insufficient, say what is missing and suggest what document is needed."
                    ),
                ),
                (
                    "human",
                    (
                        "Conversation history:\n{history}\n\n"
                        "Retrieved context:\n{context}\n\n"
                        "Question: {question}\n\n"
                        "Return a concise, factual answer and include short source references in brackets "
                        "like [doc:<document_id> chunk:<chunk_id>]."
                    ),
                ),
            ]
        )

    async def generate_answer(
        self,
        question: str,
        history: list[ConversationMessage],
        retrieved_chunks: list[dict],
    ) -> str:
        context_parts = []
        total_characters = 0

        for row in retrieved_chunks:
            block = (
                f"doc={row['document_id']} chunk={row['chunk_id']} score={float(row['score']):.4f}\n"
                f"{row['chunk_text']}"
            )
            total_characters += len(block)
            if total_characters > self._settings.max_context_characters:
                break
            context_parts.append(block)

        context = "\n\n".join(context_parts) if context_parts else "No retrieved context."

        history_text = "\n".join(
            f"{message.role}: {message.content}" for message in history[-self._settings.conversation_window :]
        )
        if not history_text:
            history_text = "No previous conversation."

        chain = self._prompt | self._llm
        response = await chain.ainvoke(
            {
                "history": history_text,
                "context": context,
                "question": question,
            }
        )
        return response.content.strip()
