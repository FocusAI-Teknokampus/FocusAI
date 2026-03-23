# backend/api/routers/upload.py
#
# Kullanıcının PDF notlarını sisteme yükleyen endpoint.
#
# POST /upload  → PDF al → RAGAgent.index() → NoteUploadResponse
#
# Sahip: K3
# Bağımlılıklar: rag_agent.py, schemas.py

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.rag.rag_agent import RAGAgent
from backend.core.schemas import NoteUploadResponse

router = APIRouter()
_rag_agent = RAGAgent()


# ── POST /upload ──────────────────────────────────────────────────────────────

@router.post("", response_model=NoteUploadResponse)
async def upload_pdf(
    user_id: str = Form(...),
    file: UploadFile = File(...),
) -> NoteUploadResponse:
    """
    Kullanıcının PDF notunu sisteme yükler ve indexler.

    Form alanları:
        user_id : kullanıcı kimliği
        file    : PDF dosyası

    Döner:
        NoteUploadResponse
            chunk_count : kaç parçaya bölündü
            indexed     : başarılı mı?
            message     : sonuç açıklaması

    Hata durumları:
        400 → PDF değil başka dosya yüklendi
        500 → indexleme sırasında beklenmedik hata

    Swagger UI'da test etmek için:
        /docs → POST /upload → "Try it out"
        user_id alanına "user_001" yaz
        file alanından PDF seç
    """
    # ── Dosya türü kontrolü ───────────────────────────────────────────
    # UploadFile.content_type her zaman güvenilir değil (tarayıcıya göre değişir)
    # Bu yüzden hem content_type hem dosya uzantısını kontrol ediyoruz
    is_pdf_content_type = file.content_type in [
        "application/pdf",
        "application/x-pdf",
    ]
    is_pdf_extension = (file.filename or "").lower().endswith(".pdf")

    if not (is_pdf_content_type or is_pdf_extension):
        raise HTTPException(
            status_code=400,
            detail=f"Sadece PDF dosyaları kabul edilir. "
                   f"Gelen dosya türü: {file.content_type}",
        )

    # ── PDF içeriğini oku ─────────────────────────────────────────────
    # UploadFile bir stream — tamamını RAM'e çekiyoruz
    # Büyük PDF'lerde (>50MB) bu sorun olabilir
    # Hafta 3'te streaming veya background task eklenebilir
    try:
        pdf_bytes = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Dosya okunurken hata oluştu: {str(e)}",
        )

    # ── RAG pipeline'ını çalıştır ─────────────────────────────────────
    try:
        result = _rag_agent.index(
            user_id=user_id,
            filename=file.filename or "upload.pdf",
            pdf_bytes=pdf_bytes,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"PDF indexlenirken hata oluştu: {str(e)}",
        )

    # Indexleme başarısız ama exception fırlatmadıysa
    # (örn: metin çıkarılamadı) 400 döner
    if not result.indexed:
        raise HTTPException(
            status_code=400,
            detail=result.message,
        )

    return result