# backend/core/database.py
#
# SQLite veritabanı bağlantısı ve tablo tanımları.
# Config'deki database_url kullanılır: sqlite:///./data/mentor.db
#
# Tablolar ve hangi modülün kullandığı:
#   users            → session_agent.py (start_session)
#   sessions         → session_agent.py (start/end_session)
#   messages         → chat.py (update_context)
#   session_reports  → dashboard.py (GET /dashboard/{session_id})
#   interventions    → mentor_agent.py (enrich_intervention)
#
# Sahip: K2
# Bağımlılıklar: config.py, sqlalchemy

import uuid
from datetime import datetime

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
)
from sqlalchemy.orm import declarative_base, sessionmaker

from backend.core.config import settings

# ─────────────────────────────────────────────────────────────────
# BAĞLANTI
# ─────────────────────────────────────────────────────────────────

# check_same_thread=False → FastAPI birden fazla thread kullandığı için gerekli.
# SQLite varsayılan olarak tek thread'e izin verir, bunu kapatıyoruz.
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},
)

# Her endpoint isteği için ayrı DB oturumu açar ve kapatır.
# autocommit=False → her işlemi manuel commit etmemiz gerekir.
# autoflush=False  → commit öncesi DB'ye otomatik yazma yapma.
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

# Tüm tablo sınıflarının miras aldığı temel sınıf.
Base = declarative_base()


# ─────────────────────────────────────────────────────────────────
# TABLOLAR
# ─────────────────────────────────────────────────────────────────

class UserRecord(Base):
    """
    Sisteme giren her kullanıcının kaydı.

    user_id: frontend'den veya SessionStartRequest'ten gelir.
    İlk start_session() çağrısında otomatik oluşturulur.
    Tekrar start_session() çağrılırsa güncellenmez, sadece okunur.

    total_sessions: UserProfile.total_sessions ile senkronize tutulur.
    Mem0 erişilemediğinde bu alan fallback olarak kullanılır.
    """
    __tablename__ = "users"

    user_id        = Column(String, primary_key=True)
    created_at     = Column(DateTime, default=datetime.utcnow)
    total_sessions = Column(Integer, default=0)


class SessionRecord(Base):
    """
    Her öğrenme oturumunun kaydı.

    session_agent.start_session() → oluşturur
    session_agent.end_session()   → ended_at'i doldurur

    camera_used: SessionStartRequest.camera_enabled'dan gelir.
    Hafta 2'de CV Engine bağlandığında bu alan anlamlı hale gelir.
    """
    __tablename__ = "sessions"

    session_id  = Column(String, primary_key=True)
    user_id     = Column(String, ForeignKey("users.user_id"))
    topic       = Column(String, nullable=True)    # SessionStartRequest.topic
    started_at  = Column(DateTime, default=datetime.utcnow)
    ended_at    = Column(DateTime, nullable=True)  # end_session() dolduracak
    camera_used = Column(Boolean, default=False)   # Kamera açık mıydı?


class MessageRecord(Base):
    """
    Her chat mesajının kalıcı kaydı.

    chat.py → update_context() her user/assistant mesajında bunu çağırır.
    short_term.py RAM'de tutar, bu tablo diske yazar.

    user_state: state_model'den gelen anlık durum.
    Dashboard'da "bu oturumda kullanıcı ne zaman takıldı?" sorusunu yanıtlar.

    role: "user" | "assistant"
    """
    __tablename__ = "messages"

    id         = Column(String, primary_key=True,
                        default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("sessions.session_id"))
    role       = Column(String)   # "user" veya "assistant"
    content    = Column(Text)
    timestamp  = Column(DateTime, default=datetime.utcnow)
    # StateEstimate.state değeri — "focused" | "stuck" | "fatigued" | "distracted"
    user_state = Column(String, nullable=True)


class InterventionRecord(Base):
    """
    Mentor Agent'ın ürettiği her müdahalenin kaydı.

    mentor_agent.enrich_intervention() sonrası buraya yazılır.
    Dashboard'daki intervention_count bu tablodan hesaplanır.

    intervention_type: InterventionType enum değeri
                       "hint" | "strategy" | "break" | "question" | "mode_switch"
    triggered_by: MentorIntervention.triggered_by — hangi UserState tetikledi?
    """
    __tablename__ = "interventions"

    id                = Column(String, primary_key=True,
                               default=lambda: str(uuid.uuid4()))
    session_id        = Column(String, ForeignKey("sessions.session_id"))
    user_id           = Column(String, ForeignKey("users.user_id"))
    intervention_type = Column(String)             # InterventionType.value
    message           = Column(Text)               # Kullanıcıya gösterilen metin
    triggered_by      = Column(String)             # UserState.value
    confidence        = Column(Float, nullable=True)
    timestamp         = Column(DateTime, default=datetime.utcnow)


class SessionReportRecord(Base):
    """
    Oturum sonu dashboard için özet veri.

    session_agent.end_session() çağrıldığında dolar.
    dashboard.py → GET /dashboard/{session_id} bu tablodan okur.

    topics_covered: JSON string olarak saklanır.
                    Örnek: '["türev", "integral", "limit"]'
                    dashboard.py'de json.loads() ile listeye çevrilir.

    focus_score: 0.0–1.0 arası.
                 scorer modülü hesaplar, şimdilik None kalabilir.
    """
    __tablename__ = "session_reports"

    id                 = Column(String, primary_key=True,
                                default=lambda: str(uuid.uuid4()))
    session_id         = Column(String, ForeignKey("sessions.session_id"))
    user_id            = Column(String, ForeignKey("users.user_id"))
    topic              = Column(String, nullable=True)
    message_count      = Column(Integer, default=0)
    intervention_count = Column(Integer, default=0)
    retry_count        = Column(Integer, default=0)
    # JSON string: '["türev", "integral"]'
    topics_covered     = Column(Text, default="[]")
    # 0.0–1.0 arası ortalama odak skoru — scorer modülünden gelecek
    focus_score        = Column(Float, nullable=True)
    created_at         = Column(DateTime, default=datetime.utcnow)


# ─────────────────────────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR
# ─────────────────────────────────────────────────────────────────

def init_db() -> None:
    """
    Tabloları oluşturur. Zaten varsa dokunmaz (CREATE TABLE IF NOT EXISTS).
    main.py'nin başında bir kez çağrılır.

    data/ klasörü yoksa SQLite hata verir.
    Bu yüzden klasörü de burada açıyoruz.

    Kullanım (main.py):
        from backend.core.database import init_db
        init_db()
    """
    import os
    os.makedirs(settings.data_dir, exist_ok=True)
    Base.metadata.create_all(bind=engine)


def get_db():
    """
    FastAPI dependency injection için DB oturumu sağlar.
    Her endpoint isteğinde yeni oturum açar, istek bitince kapatır.

    Kullanım (endpoint'lerde):
        from backend.core.database import get_db
        from sqlalchemy.orm import Session
        from fastapi import Depends

        @router.post("/ornek")
        def ornek_endpoint(db: Session = Depends(get_db)):
            users = db.query(UserRecord).all()
            return users

    finally bloğu: hata olsa bile oturum kapanır, bağlantı sızıntısı olmaz.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
