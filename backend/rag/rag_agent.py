# backend/rag/rag_agent.py
#
# RAG pipeline'inin dis dunyaya acilan yuzu.
# Opsiyonel bagimliliklar veya API anahtari hazir degilse sistem
# chat akisini dusurmeden no-op moda gecer.

from __future__ import annotations

import logging

from backend.core.schemas import NoteUploadResponse, RAGResult

logger = logging.getLogger(__name__)


class RAGAgent:
    """
    RAG pipeline'inin koordinatoru.
    """

    def __init__(self):
        self.indexer = None
        self.retriever = None
        self.available = False

        try:
            from backend.rag.indexer import PDFIndexer
            from backend.rag.retriever import FAISSRetriever

            self.indexer = PDFIndexer()
            self.retriever = FAISSRetriever()
            self.available = True
        except Exception as exc:
            logger.warning("RAG bagimliliklari hazir degil, no-op mod aktif: %s", exc)

    def index(
        self,
        user_id: str,
        filename: str,
        pdf_bytes: bytes,
    ) -> NoteUploadResponse:
        if not self.available or self.indexer is None:
            return NoteUploadResponse(
                user_id=user_id,
                filename=filename,
                chunk_count=0,
                indexed=False,
                message="RAG servisi hazir degil. Bagimliliklar veya OpenAI ayari eksik.",
            )
        return self.indexer.index_pdf(user_id, filename, pdf_bytes)

    def search(
        self,
        user_id: str,
        query: str,
    ) -> RAGResult:
        if not self.available or self.retriever is None:
            return RAGResult(found=False)
        return self.retriever.search(user_id, query)

    def has_notes(self, user_id: str) -> bool:
        if not self.available or self.retriever is None:
            return False
        return self.retriever.user_has_notes(user_id)
