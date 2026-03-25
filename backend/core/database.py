# backend/core/database.py

import os
import uuid
from datetime import datetime
from typing import Generator

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    Index,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from backend.core.config import settings


# ─────────────────────────────────────────────────────────────────
# BAĞLANTI
# ─────────────────────────────────────────────────────────────────

is_sqlite = settings.database_url.startswith("sqlite")

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if is_sqlite else {},
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)

Base = declarative_base()


def generate_uuid() -> str:
    """
    Yeni benzersiz ID üretir.
    """
    return str(uuid.uuid4())


def utcnow() -> datetime:
    """
    UTC zamanı döner.
    """
    return datetime.utcnow()


# ─────────────────────────────────────────────────────────────────
# TABLOLAR
# ─────────────────────────────────────────────────────────────────

class UserRecord(Base):
    """
    Sistemdeki ana kullanıcı kaydı.
    """
    __tablename__ = "users"

    user_id = Column(String, primary_key=True)
    created_at = Column(DateTime, default=utcnow, nullable=False)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)
    last_active_at = Column(DateTime, nullable=True)
    total_sessions = Column(Integer, default=0, nullable=False)


class SessionRecord(Base):
    """
    Her öğrenme oturumunun kalıcı kaydı.
    """
    __tablename__ = "sessions"

    session_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)

    topic = Column(String, nullable=True)
    subtopic = Column(String, nullable=True)
    study_mode = Column(String, nullable=True)

    started_at = Column(DateTime, default=utcnow, nullable=False)
    ended_at = Column(DateTime, nullable=True)

    camera_used = Column(Boolean, default=False, nullable=False)

    current_state = Column(String, nullable=True)
    retry_count = Column(Integer, default=0, nullable=False)
    intervention_count = Column(Integer, default=0, nullable=False)
    average_focus_score = Column(Float, nullable=True)


class MessageRecord(Base):
    """
    Chat mesajlarının kalıcı kaydı.
    """
    __tablename__ = "messages"

    id = Column(String, primary_key=True, default=generate_uuid)
    session_id = Column(String, ForeignKey("sessions.session_id"), nullable=False)

    role = Column(String, nullable=False)  # user | assistant | system
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=utcnow, nullable=False)

    user_state = Column(String, nullable=True)
    detected_topic = Column(String, nullable=True)
    message_type = Column(String, nullable=True)
    llm_confidence = Column(Float, nullable=True)


class InterventionRecord(Base):
    """
    Mentor müdahalelerinin kaydı.
    """
    __tablename__ = "interventions"

    id = Column(String, primary_key=True, default=generate_uuid)
    session_id = Column(String, ForeignKey("sessions.session_id"), nullable=False)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)

    intervention_type = Column(String, nullable=False)
    message = Column(Text, nullable=False)

    triggered_by = Column(String, nullable=True)
    reason = Column(Text, nullable=True)
    confidence = Column(Float, nullable=True)
    was_successful = Column(Boolean, nullable=True)

    timestamp = Column(DateTime, default=utcnow, nullable=False)


class SessionReportRecord(Base):
    """
    Oturum sonu raporu için özet kayıt.

    Not:
        Zengin analiz alanları JSON string olarak tutulur.
        Böylece migration karmaşıklığı olmadan hızlı ilerlenebilir.
    """
    __tablename__ = "session_reports"

    id = Column(String, primary_key=True, default=generate_uuid)
    session_id = Column(String, ForeignKey("sessions.session_id"), nullable=False)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)

    topic = Column(String, nullable=True)

    message_count = Column(Integer, default=0, nullable=False)
    intervention_count = Column(Integer, default=0, nullable=False)
    retry_count = Column(Integer, default=0, nullable=False)

    topics_covered = Column(Text, default="[]", nullable=False)
    focus_score = Column(Float, nullable=True)
    summary_text = Column(Text, nullable=True)

    # Yeni: zengin rapor alanları
    behavior_summary = Column(Text, default="{}", nullable=False)
    strengths = Column(Text, default="[]", nullable=False)
    weaknesses = Column(Text, default="[]", nullable=False)
    recommendations = Column(Text, default="[]", nullable=False)
    next_session_plan = Column(Text, default="{}", nullable=False)

    created_at = Column(DateTime, default=utcnow, nullable=False)


class UserProfileRecord(Base):
    """
    Kullanıcının uzun dönem öğrenme profili.
    """
    __tablename__ = "user_profiles"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.user_id"), unique=True, nullable=False)

    preferred_explanation_style = Column(String, nullable=True)
    weak_topics = Column(Text, default="[]", nullable=False)
    strong_topics = Column(Text, default="[]", nullable=False)
    recurring_misconceptions = Column(Text, default="[]", nullable=False)

    avg_session_duration_minutes = Column(Float, default=0.0, nullable=False)
    adaptive_threshold = Column(Float, default=0.75, nullable=False)

    created_at = Column(DateTime, default=utcnow, nullable=False)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)


class BehaviorEventRecord(Base):
    """
    Oturum sırasında tespit edilen davranış olaylarının kaydı.
    """
    __tablename__ = "behavior_events"

    id = Column(String, primary_key=True, default=generate_uuid)
    session_id = Column(String, ForeignKey("sessions.session_id"), nullable=False)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)

    event_type = Column(String, nullable=False)
    state_before = Column(String, nullable=True)
    state_after = Column(String, nullable=True)

    topic = Column(String, nullable=True)
    severity = Column(Float, nullable=True)
    metadata_json = Column(Text, default="{}", nullable=False)

    created_at = Column(DateTime, default=utcnow, nullable=False)


class FocusEventRecord(Base):
    """
    Focus score zaman serisi.
    """
    __tablename__ = "focus_events"

    id = Column(String, primary_key=True, default=generate_uuid)
    session_id = Column(String, ForeignKey("sessions.session_id"), nullable=False)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)

    focus_score = Column(Float, nullable=False)
    source = Column(String, default="text", nullable=False)
    state_label = Column(String, nullable=True)

    created_at = Column(DateTime, default=utcnow, nullable=False)


class UploadedDocumentRecord(Base):
    """
    Kullanıcının yüklediği dokümanların metadata kaydı.
    """
    __tablename__ = "uploaded_documents"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)

    file_name = Column(String, nullable=False)
    file_type = Column(String, nullable=True)
    file_size_bytes = Column(Integer, nullable=True)

    chunk_count = Column(Integer, default=0, nullable=False)
    indexed = Column(Boolean, default=False, nullable=False)
    source_path = Column(String, nullable=True)

    uploaded_at = Column(DateTime, default=utcnow, nullable=False)


# ─────────────────────────────────────────────────────────────────
# INDEXLER
# ─────────────────────────────────────────────────────────────────

Index("ix_sessions_user_id", SessionRecord.user_id)
Index("ix_messages_session_id", MessageRecord.session_id)
Index("ix_messages_timestamp", MessageRecord.timestamp)
Index("ix_interventions_session_id", InterventionRecord.session_id)
Index("ix_session_reports_session_id", SessionReportRecord.session_id)
Index("ix_user_profiles_user_id", UserProfileRecord.user_id)
Index("ix_behavior_events_session_id", BehaviorEventRecord.session_id)
Index("ix_behavior_events_event_type", BehaviorEventRecord.event_type)
Index("ix_focus_events_session_id", FocusEventRecord.session_id)
Index("ix_uploaded_documents_user_id", UploadedDocumentRecord.user_id)


# ─────────────────────────────────────────────────────────────────
# INIT / DEPENDENCY
# ─────────────────────────────────────────────────────────────────

def init_db() -> None:
    """
    Veritabanı tablolarını oluşturur.
    """
    if is_sqlite:
        os.makedirs(settings.data_dir, exist_ok=True)

    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI endpoint'leri için veritabanı oturumu sağlar.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()