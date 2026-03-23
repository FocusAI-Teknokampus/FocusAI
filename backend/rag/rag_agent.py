# backend/rag/rag_agent.py
#
# RAG pipeline'ının dış dünyaya açılan yüzü.
# Indexer ve Retriever'ı tek bir arayüz altında toplar.
#
# Neden ayrı bir katman?
# graph.py'deki rag_node sadece bu sınıfı bilsin.
# Indexer veya Retriever değişirse rag_node etkilenmesin.
#
# Sahip: K2
# Bağımlılıklar: indexer.py, retriever.py

from backend.rag.indexer import PDFIndexer
from backend.rag.retriever import FAISSRetriever
from backend.core.schemas import NoteUploadResponse, RAGResult


class RAGAgent:
    """
    RAG pipeline'ının koordinatörü.

    İki görevi var:
    1. index() — yeni PDF geldiğinde indexler
    2. search() — chat sırasında alakalı notları arar

    graph.py'deki rag_node bu sınıfı kullanır:
        agent = RAGAgent()
        result = agent.search(user_id, query)
        return {"rag_context": result.source_chunk if result.found else None}

    upload.py'deki endpoint bu sınıfı kullanır:
        agent = RAGAgent()
        response = agent.index(user_id, filename, pdf_bytes)
    """

    def __init__(self):
        self.indexer = PDFIndexer()
        self.retriever = FAISSRetriever()

    def index(
        self,
        user_id: str,
        filename: str,
        pdf_bytes: bytes,
    ) -> NoteUploadResponse:
        """
        PDF'i indexler. Upload endpoint'i bu metodu çağırır.

        Parametreler:
            user_id   : kimin notu
            filename  : orijinal dosya adı
            pdf_bytes : PDF'in binary içeriği

        Döner:
            NoteUploadResponse (chunk_count, indexed, message)
        """
        return self.indexer.index_pdf(user_id, filename, pdf_bytes)

    def search(
        self,
        user_id: str,
        query: str,
    ) -> RAGResult:
        """
        Kullanıcının notlarında arama yapar. rag_node bu metodu çağırır.

        Parametreler:
            user_id : kimin notlarında aranacak
            query   : kullanıcının sorusu

        Döner:
            RAGResult
                found=True  → source_chunk LLM'e verilecek
                found=False → not yok veya alakasız, LLM kendi bilgisiyle cevaplar
        """
        return self.retriever.search(user_id, query)

    def has_notes(self, user_id: str) -> bool:
        """Kullanıcının indexlenmiş notu var mı?"""
        return self.retriever.user_has_notes(user_id)