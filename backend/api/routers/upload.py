# backend/api/routers/upload.py

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Depends
from sqlalchemy.orm import Session

from backend.rag.rag_agent import RAGAgent
from backend.core.schemas import NoteUploadResponse, UploadedDocumentSummary
from backend.core.database import get_db, UploadedDocumentRecord

router = APIRouter()
_rag_agent = RAGAgent()


@router.post("", response_model=NoteUploadResponse)
async def upload_pdf(
    user_id: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> NoteUploadResponse:
    """
    Kullanıcının PDF notunu sisteme yükler, indexler ve metadata'sını DB'ye kaydeder.
    """
    is_pdf_content_type = file.content_type in [
        "application/pdf",
        "application/x-pdf",
    ]
    is_pdf_extension = (file.filename or "").lower().endswith(".pdf")

    if not (is_pdf_content_type or is_pdf_extension):
        raise HTTPException(
            status_code=400,
            detail=f"Sadece PDF dosyaları kabul edilir. Gelen dosya türü: {file.content_type}",
        )

    try:
        pdf_bytes = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Dosya okunurken hata oluştu: {str(e)}",
        )

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

    if not result.indexed:
        raise HTTPException(
            status_code=400,
            detail=result.message,
        )

    # Yeni: metadata DB kaydı
    doc_row = UploadedDocumentRecord(
        user_id=user_id,
        file_name=file.filename or "upload.pdf",
        file_type=file.content_type or "application/pdf",
        file_size_bytes=len(pdf_bytes),
        chunk_count=result.chunk_count,
        indexed=result.indexed,
        source_path=None,
    )
    db.add(doc_row)
    db.commit()

    return result


@router.get("/documents/{user_id}", response_model=list[UploadedDocumentSummary])
def list_uploaded_documents(
    user_id: str,
    db: Session = Depends(get_db),
) -> list[UploadedDocumentSummary]:
    """
    KullanÄ±cÄ±nÄ±n yÃ¼klediÄŸi PDF listesini dÃ¶ner.

    Neden ayrÄ± endpoint?
    Chat akÄ±ÅŸÄ± ile dosya listesini karÄ±ÅŸtÄ±rmak istemiyoruz.
    UI bu endpoint'i Ã§aÄŸÄ±rÄ±p yalnÄ±zca kullanÄ±cÄ±nÄ±n notlarÄ±nÄ± listeleyebilir.
    """
    rows = (
        db.query(UploadedDocumentRecord)
        .filter(UploadedDocumentRecord.user_id == user_id)
        .order_by(UploadedDocumentRecord.uploaded_at.desc())
        .all()
    )

    return [
        UploadedDocumentSummary(
            filename=row.file_name,
            file_type=row.file_type,
            file_size_bytes=row.file_size_bytes,
            chunk_count=row.chunk_count,
            indexed=row.indexed,
            uploaded_at=row.uploaded_at,
        )
        for row in rows
    ]
