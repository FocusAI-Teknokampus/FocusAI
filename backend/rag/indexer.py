# backend/rag/indexer.py
#
# PDF → FAISS index pipeline'ı.
# Kullanıcı bir PDF yüklediğinde bu dosya çalışır.
#
# Sorumlulukları:
#   1. PDF'ten ham metni çıkar (PyMuPDF)
#   2. Metni chunk'lara böl (LangChain TextSplitter)
#   3. Her chunk'ı vektöre çevir (OpenAI Embeddings)
#   4. Vektörleri kullanıcıya özel FAISS index'ine kaydet
#
# Sahip: K2
# Bağımlılıklar: config.py, schemas.py, faiss-cpu, pymupdf, langchain

import os
import pickle
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF — pip install pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from backend.core.config import settings
from backend.core.schemas import NoteUploadResponse


class PDFIndexer:
    """
    PDF dosyasını alır, işler ve FAISS index'ine kaydeder.

    Her kullanıcının index'i ayrı klasörde tutulur:
        data/faiss_index/user_001/
        data/faiss_index/user_002/

    Bu sayede kullanıcıların notları birbirine karışmaz.

    Kullanım:
        indexer = PDFIndexer()
        response = indexer.index_pdf(
            user_id="user_001",
            filename="turev_notlari.pdf",
            pdf_bytes=...  # dosyanın binary içeriği
        )
    """

    def __init__(self):
        # OpenAI'ın embedding modeli — her chunk'ı vektöre çevirecek
        # Model: text-embedding-3-small (config'den geliyor)
        # Bu model 1536 boyutlu vektör üretiyor
        self.embeddings = OpenAIEmbeddings(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
        )

        # Metni parçalara bölen splitter
        # chunk_size: her parça max kaç karakter?
        # chunk_overlap: ardışık parçalar kaç karakter örtüşsün?
        # Örtüşme neden var? Cümle bir chunk'ın sonunda, devamı bir sonrakinde
        # kalmasın diye. Anlam bütünlüğü korunsun.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.rag_chunk_size,        # 500 karakter
            chunk_overlap=settings.rag_chunk_overlap,  # 50 karakter
            separators=["\n\n", "\n", ".", " ", ""],
            # Splitter önce çift satır sonu (\n\n = paragraf sonu),
            # bulamazsa tek satır sonu, bulamazsa nokta, bulamazsa boşluk
            # sırasıyla dener. Bu sıra sayesinde cümle ortasından kesmez.
        )

    # ─────────────────────────────────────────────────────────────────
    # ANA METOD
    # ─────────────────────────────────────────────────────────────────

    def index_pdf(
        self,
        user_id: str,
        filename: str,
        pdf_bytes: bytes,
    ) -> NoteUploadResponse:
        """
        PDF'i alır, işler, index'e kaydeder.

        Parametreler:
            user_id   : kimin notu olduğu
            filename  : orijinal dosya adı (metadata için)
            pdf_bytes : PDF dosyasının binary içeriği

        Döner:
            NoteUploadResponse — kaç chunk indexlendi, başarılı mı?
        """
        # 1. PDF'ten metni çıkar
        raw_text = self._extract_text_from_pdf(pdf_bytes)

        if not raw_text.strip():
            return NoteUploadResponse(
                user_id=user_id,
                filename=filename,
                chunk_count=0,
                indexed=False,
                message="PDF'ten metin çıkarılamadı. Dosya taranmış (scanned) olabilir.",
            )

        # 2. Metni chunk'lara böl
        chunks = self._split_into_chunks(raw_text, filename)

        if not chunks:
            return NoteUploadResponse(
                user_id=user_id,
                filename=filename,
                chunk_count=0,
                indexed=False,
                message="Metin parçalara bölünemedi.",
            )

        # 3. Chunk'ları vektörleştir ve FAISS'e kaydet
        self._save_to_faiss(user_id, chunks)

        return NoteUploadResponse(
            user_id=user_id,
            filename=filename,
            chunk_count=len(chunks),
            indexed=True,
            message=f"{len(chunks)} parça başarıyla indexlendi.",
        )

    # ─────────────────────────────────────────────────────────────────
    # ADIM 1: PDF'TEN METİN ÇIKAR
    # ─────────────────────────────────────────────────────────────────

    def _extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """
        PyMuPDF (fitz) ile PDF'in tüm sayfalarından metni çıkarır.

        PyMuPDF neden? Hız ve güvenilirlik.
        pdfplumber daha esnek ama daha yavaş.
        PyPDF2 eski ve bazı PDF'lerde encoding sorunları var.

        Sınırlama: Taranmış (scanned/image-based) PDF'lerde metin yok,
        bu durumda boş string döner. OCR bu projenin kapsamı dışında.
        """
        text_parts = []

        # pdf_bytes'ı bellekte açıyoruz, diske yazmıyoruz
        # stream=pdf_bytes → dosya gibi davran ama RAM'de tut
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                if page_text.strip():
                    # Sayfa numarasını işaret olarak ekle
                    # Bu bilgi chunk metadata'sına girecek
                    text_parts.append(f"[Sayfa {page_num + 1}]\n{page_text}")

        return "\n\n".join(text_parts)

    # ─────────────────────────────────────────────────────────────────
    # ADIM 2: METNİ CHUNK'LARA BÖL
    # ─────────────────────────────────────────────────────────────────

    def _split_into_chunks(
        self, text: str, filename: str
    ) -> list[dict]:
        """
        Ham metni anlamlı parçalara böler.

        Her chunk bir dict:
            {
                "text": "türev, bir fonksiyonun...",
                "metadata": {
                    "filename": "turev_notlari.pdf",
                    "chunk_index": 0
                }
            }

        Metadata neden önemli?
        Retriever bir chunk bulduğunda "bu hangi dosyadan geldi?"
        sorusunu yanıtlayabilmek için. ChatResponse'daki rag_source
        alanı bu bilgiyi UI'a taşır.
        """
        raw_chunks = self.splitter.split_text(text)

        return [
            {
                "text": chunk,
                "metadata": {
                    "filename": filename,
                    "chunk_index": i,
                }
            }
            for i, chunk in enumerate(raw_chunks)
        ]

    # ─────────────────────────────────────────────────────────────────
    # ADIM 3: FAISS'E KAYDET
    # ─────────────────────────────────────────────────────────────────

    def _save_to_faiss(
        self, user_id: str, chunks: list[dict]
    ) -> None:
        """
        Chunk'ları OpenAI ile vektörleştirir ve FAISS index'ine kaydeder.

        FAISS index yapısı:
            data/faiss_index/
                user_001/
                    index.faiss  ← vektörlerin binary hali
                    index.pkl    ← vektöre karşılık gelen metinler

        Kullanıcının önceki index'i varsa üstüne yazar değil,
        yeni chunk'ları eskiyle birleştirir (merge).
        Bu sayede birden fazla PDF yüklenebilir.

        Nasıl çalışır?
        1. LangChain's FAISS.from_texts() her chunk'ı embedding API'ye gönderir
        2. API 1536 boyutlu vektör döner
        3. FAISS bu vektörleri kendi binary formatında saklar
        4. save_local() diske yazar
        """
        index_path = self._get_index_path(user_id)

        # Metin ve metadata listelerini ayır
        # LangChain FAISS wrapper bu formatı bekliyor
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]

        # Kullanıcının mevcut index'i var mı?
        if index_path.exists():
            # Varsa yükle ve yeni chunk'ları ekle (merge)
            existing_index = FAISS.load_local(
                str(index_path),
                self.embeddings,
                allow_dangerous_deserialization=True,
                # allow_dangerous_deserialization: pickle dosyası okurken
                # güvenlik uyarısını geçmek için. Local dosya olduğu için
                # bu projede güvenli.
            )
            existing_index.add_texts(texts, metadatas=metadatas)
            existing_index.save_local(str(index_path))
        else:
            # Yoksa yeni index oluştur
            index_path.mkdir(parents=True, exist_ok=True)
            new_index = FAISS.from_texts(
                texts,
                self.embeddings,
                metadatas=metadatas,
            )
            new_index.save_local(str(index_path))

    def _get_index_path(self, user_id: str) -> Path:
        """
        Kullanıcıya özel index klasörünün yolunu döner.
        Örnek: data/faiss_index/user_001
        """
        return Path(settings.faiss_index_path) / user_id