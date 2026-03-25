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

    def search_with_history(
    self,
    user_id: str,
    query: str,
    history: list[dict],  # [{"role": "user", "content": "..."}, ...]
    max_chars: int = 1000,  # kaç karaktere kadar geçmiş dahil edilsin
    ) -> RAGResult:
    """
    Kullanıcının tüm sohbet geçmişini + son soruyu birleştirerek
    FAISS index'inde arama yapar.

    Neden sadece son soruya bakmıyoruz?
    "peki bu nasıl hesaplanır?" gibi belirsiz sorularda
    önceki mesajlar olmadan ne sorulduğu anlaşılmaz.
    Geçmişi de katarak daha alakalı chunk getirilir.

    Neden tüm geçmişi değil, max_chars kadar alıyoruz?
    OpenAI embedding API'sinin token limiti var (8191 token).
    Çok uzun sorgu hem limiti aşar hem maliyeti artırır.
    Bu yüzden sondan başa doğru mesajları ekliyoruz,
    max_chars dolunca duruyoruz.

    Parametreler:
        user_id   : kimin notlarında aranacak
        query     : kullanıcının son sorusu
        history   : tüm sohbet geçmişi
                    [{"role": "user", "content": "..."},
                     {"role": "assistant", "content": "..."}]
        max_chars : geçmişten kaç karakter dahil edilsin (varsayılan 1000)

    Döner:
        RAGResult
            found=True  → source_chunk LLM'e verilecek
            found=False → not yok veya alakasız, LLM kendi bilgisiyle cevaplar
    """

    # Geçmişten sadece kullanıcı mesajlarını al
    # Asistan mesajları soruyla alakasız chunk getirebilir, gürültü yaratır
    user_messages = [
        m["content"] for m in history if m["role"] == "user"
    ]

    # Sondan başa doğru mesajları ekle, max_chars'ı aşınca dur
    # Neden sondan başa? En yakın bağlam daha önemli.
    # Neden insert(0)? Sıralamayı korumak için — eski mesaj önde kalır.
    selected = []
    total = 0
    for msg in reversed(user_messages):
        if total + len(msg) > max_chars:
            break
        selected.insert(0, msg)
        total += len(msg)

    # Geçmiş + son soru tek stringe birleşiyor
    # Örnek: "türev nedir zincir kuralı nasıl çalışır peki bunu integralde kullanabilir miyiz"
    enriched_query = " ".join(selected + [query]).strip()

    return self.retriever.search(user_id, enriched_query)

    def delete_notes(self, user_id: str) -> bool:
    """
    Kullanıcının tüm FAISS index'ini siler.
    Dashboard'daki 'notları temizle' butonu bunu çağırır.
    """
    import shutil
    from pathlib import Path
    index_path = Path(settings.faiss_index_path) / user_id
    if index_path.exists():
        shutil.rmtree(index_path)
        return True
    return False


    
