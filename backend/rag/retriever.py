# backend/rag/retriever.py
#
# FAISS index'inden alakalı chunk'ları getirir.
# Her chat isteğinde çağrılır.
#
# Tek sorumluluğu:
#   "Bu soruyla en alakalı not parçaları hangileri?" sorusunu yanıtlamak.
#
# Sahip: K2
# Bağımlılıklar: config.py, schemas.py, faiss-cpu, langchain

from pathlib import Path
from typing import Optional

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from backend.core.config import settings
from backend.core.schemas import RAGResult


class FAISSRetriever:
    """
    Kullanıcının FAISS index'inden alakalı chunk'ları arar.

    Nasıl çalışır?
    1. Gelen soruyu embedding API ile vektöre çevirir
    2. Kullanıcının index'ini yükler
    3. Vektörler arasında cosine similarity ile en yakın K chunk'ı bulur
    4. RAGResult olarak döner

    Kullanım:
        retriever = FAISSRetriever()
        result = retriever.search(
            user_id="user_001",
            query="türev nasıl hesaplanır?"
        )
        if result.found:
            print(result.source_chunk)  # en alakalı not parçası
    """

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
        )

    # ─────────────────────────────────────────────────────────────────
    # ANA METOD
    # ─────────────────────────────────────────────────────────────────

    def search(
        self,
        user_id: str,
        query: str,
        top_k: Optional[int] = None,
    ) -> RAGResult:
        """
        Kullanıcının notlarında sorguya en yakın chunk'ları arar.

        Parametreler:
            user_id : kimin notlarında aranacak
            query   : kullanıcının sorusu (ham metin)
            top_k   : kaç chunk dönsün? (None → config'deki değer)

        Döner:
            RAGResult
                found=True  → source_chunk ve filename dolu
                found=False → index yok veya eşik altında sonuç

        Eşik (threshold) neden var?
            FAISS her zaman bir sonuç döner, alakasız bile olsa.
            "futbol topu kaç gram?" sorusuna türev notlarından
            bir şeyler getirir. Bunu önlemek için similarity
            score'una bir alt sınır koyuyoruz.
        """
        k = top_k or settings.rag_top_k
        index_path = self._get_index_path(user_id)

        # Index yoksa kullanıcı henüz not yüklememişi demektir
        if not index_path.exists():
            return RAGResult(found=False)

        try:
            # Index'i yükle
            index = FAISS.load_local(
                str(index_path),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

            # similarity_search_with_score:
            # Normal similarity_search sadece chunk metinlerini döner.
            # _with_score versiyonu yanına similarity score da ekler.
            # Score: 0.0 = tamamen farklı, 1.0 = birebir aynı
            # (Cosine similarity kullanılıyor)
            results = index.similarity_search_with_score(query, k=k)

            if not results:
                return RAGResult(found=False)

            # En alakalı chunk (en yüksek score)
            # results listesi (Document, score) tuple'larından oluşur
            # FAISS'te düşük score = yüksek benzerlik (L2 distance)
            # Bu yüzden min score'u alıyoruz
            best_doc, best_score = min(results, key=lambda x: x[1])

            # Eşik kontrolü:
            # FAISS L2 distance kullanıyor (düşük = benzer).
            # 1.0'dan büyük score'lar genellikle alakasız.
            # Bu eşiği config'e taşıyabiliriz ilerleyen haftalarda.
            SIMILARITY_THRESHOLD = 1.0
            if best_score > SIMILARITY_THRESHOLD:
                return RAGResult(found=False)

            # Tüm chunk'ları birleştir — LLM daha fazla bağlam görsün
            all_chunks_text = self._merge_chunks(results)
            filename = best_doc.metadata.get("filename", "not")

            return RAGResult(
                found=True,
                source_chunk=all_chunks_text,
                filename=filename,
                relevance_score=round(float(1 - best_score), 3),
                # Score'u tersine çeviriyoruz: düşük L2 → yüksek benzerlik
            )

        except Exception as e:
            # Index bozuk, yüklenemedi vs.
            # Hata fırlatmak yerine "not bulunamadı" dön
            # Chat akışı bozulmasın
            return RAGResult(found=False)

    # ─────────────────────────────────────────────────────────────────
    # YARDIMCI METODLAR
    # ─────────────────────────────────────────────────────────────────

    def _merge_chunks(
        self, results: list[tuple]
    ) -> str:
        """
        Birden fazla chunk'ı tek bir metin bloğuna birleştirir.

        Neden birleştiriyoruz?
        LLM'e tek chunk vermek yerine alakalı tüm parçaları vermek
        daha iyi cevap üretilmesini sağlar.
        Ama çok fazla metin vermek de token israfı olur.
        config'deki rag_top_k (=3) bu dengeyi kuruyor.

        Format:
            [Not 1]
            türev bir fonksiyonun...

            [Not 2]
            zincir kuralına göre...
        """
        parts = []
        for i, (doc, score) in enumerate(results):
            parts.append(f"[Not {i + 1}]\n{doc.page_content}")
        return "\n\n".join(parts)

    def _get_index_path(self, user_id: str) -> Path:
        """Kullanıcıya özel index klasörünün yolunu döner."""
        return Path(settings.faiss_index_path) / user_id

    def user_has_notes(self, user_id: str) -> bool:
        """
        Kullanıcının notu var mı? graph.py'deki rag_node bunu kullanabilir.
        Index klasörü yoksa False döner.
        """
        return self._get_index_path(user_id).exists()


   def get_all_filenames(self, user_id: str) -> list[str]:
    """
    FAISS index'indeki tüm benzersiz dosya adlarını döner.
    Dashboard'da 'yüklediğin notlar: türev.pdf, integral.pdf' için.
    """
        index_path = self._get_index_path(user_id)
        if not index_path.exists():
            return []

        try:
            index = FAISS.load_local(
                str(index_path),
                self.embeddings,
                allow_dangerous_deserialization=True,
        )
        # FAISS'in iç store'undan metadata'yı çek
            filenames = set()
            for doc_id in index.docstore._dict:
                doc = index.docstore._dict[doc_id]
                fname = doc.metadata.get("filename")
                if fname:
                    filenames.add(fname)
            return list(filenames)
        except Exception:
            return []
