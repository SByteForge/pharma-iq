from fastapi import HTTPException, status


class OllamaUnavailableError(HTTPException):
    """Raised when Ollama service is not reachable."""
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "ollama_unavailable",
                "message": "Local LLM service is not running. Start with: ollama serve",
                "action": "ensure_ollama_running"
            }
        )


class DocumentParseError(HTTPException):
    """Raised when a PDF cannot be parsed."""
    def __init__(self, filename: str, reason: str):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "document_parse_failed",
                "filename": filename,
                "reason": reason,
                "action": "verify_pdf_is_valid_and_not_encrypted"
            }
        )


class EmptyRetrievalError(HTTPException):
    """Raised when vector search returns no relevant results."""
    def __init__(self, query: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "no_relevant_context_found",
                "query": query,
                "message": "No contract data matched this query. Ingest relevant documents first.",
                "action": "post_to_ingest_endpoint"
            }
        )


class EmbeddingError(HTTPException):
    """Raised when embedding generation fails."""
    def __init__(self, reason: str):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "embedding_failed",
                "reason": reason,
                "action": "verify_nomic_embed_text_model_is_pulled"
            }
        )