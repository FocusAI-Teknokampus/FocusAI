# backend/core/schemas.py
#
# Bu dosya projenin ortak veri sözleşmesidir.
# Tüm modüller (agents, rag, memory, api) bu dosyadan import yapar.
# Değişiklik yapmadan önce tüm ekiple konuşun.
#
# Sahip: K1 + K2 (Gün 1'de birlikte yazıldı)

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
from enum import Enum
import uuid


# ─────────────────────────────────────────
# ENUM'LAR — sabit değer kümeleri
# ─────────────────────────────────────────

class UserState(str, Enum):
    """
    Kullanıcının anlık çalışma durumu.
    State Modeling modülü (K1) bu değerleri üretir.
    Mentor Agent ve Intervention Agent bu değerleri tüketir.
    """
    FOCUSED     = "focused"      # Normal, verimli çalışma
    DISTRACTED  = "distracted"   # Dikkat dağınıklığı
    FATIGUED    = "fatigued"     # Yorgunluk belirtisi
    STUCK       = "stuck"        # Konuda takılı kalmış
    UNKNOWN     = "unknown"      # Henüz yeterli veri yok


class LearningPattern(str, Enum):
    """
    Öğrenme stratejisi tespiti.
    Intervention Agent (K1) bu değerleri üretir.
    Long-term memory'ye (K2) yazılır.
    """
    SHALLOW_LEARNING = "shallow_learning"  # Hızlı geçiyor, yüzeysel
    DEEP_ATTEMPT     = "deep_attempt"      # Uzun süre deniyor
    MISCONCEPTION    = "misconception"     # Aynı hata tekrar ediyor
    NORMAL           = "normal"            # Sorun yok


class InterventionType(str, Enum):
    """
    Mentor müdahale türleri.
    Mentor Agent (K1) bu değerlerden birini seçer.
    K4 UI'ı bu değere göre farklı stil uygular.
    """
    HINT             = "hint"             # İpucu ver
    STRATEGY         = "strategy"         # Çalışma stratejisi öner
    BREAK            = "break"            # Mola öner
    MODE_SWITCH      = "mode_switch"      # Farklı mod öner (örn: Explain Thinking)
    QUESTION         = "question"         # Soru sor (belirsizlik yüksekse)
    NONE             = "none"             # Müdahale gerekmez


class ResponsePolicyMode(str, Enum):
    """
    State tahmininden sonra cevap stilini secen ara policy katmani.
    """
    DIRECT_HELP = "direct_help"
    GUIDED_HINT = "guided_hint"
    CHALLENGE = "challenge"
    RECOVERY = "recovery"
    CLARIFY = "clarify"


class InputChannel(str, Enum):
    """
    Hangi kanaldan veri geldiği.
    Feature Extractor (K3) her olaya kanal damgası basar.
    """
    TEXT    = "text"    # Chat mesajı
    IMAGE   = "image"   # Soru fotoğrafı
    CAMERA  = "camera"  # MediaPipe kamera sinyali
    VOICE   = "voice"   # Ses girişi (Sprint 3)
    SYSTEM  = "system"  # Sistem olayı (oturum başlat/kapat)


# ─────────────────────────────────────────
# OTURUM MODELLERİ
# Session Agent (K1) ve FastAPI (K3) kullanır
# ─────────────────────────────────────────

class SessionStartRequest(BaseModel):
    """POST /session/start — K4 UI bu modeli gönderir."""
    user_id: str
    topic: Optional[str] = None          # "Bugün ne çalışıyorum?" girişi
    camera_enabled: bool = False          # Kullanıcı kamerayı açtı mı?


class SessionStartResponse(BaseModel):
    """POST /session/start — K3 API bu modeli döner."""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    topic: Optional[str]
    camera_enabled: bool
    started_at: datetime = Field(default_factory=datetime.now)


class SessionEndRequest(BaseModel):
    """POST /session/end — K4 UI oturumu kapatırken gönderir."""
    session_id: str
    user_id: str


class SessionEndResponse(BaseModel):
    """POST /session/end — özet rapor döner."""
    session_id: str
    duration_seconds: int
    ended_at: datetime = Field(default_factory=datetime.now)
    summary: "SessionSummary"            # aşağıda tanımlı


# ─────────────────────────────────────────
# MESAJ MODELLERİ
# Chat endpoint (K3) ve Chat UI (K4) kullanır
# ─────────────────────────────────────────

class ChatMessage(BaseModel):
    """
    POST /chat — kullanıcıdan gelen mesaj.
    image_base64: fotoğraf varsa base64 string.
    """
    session_id: str
    user_id: str
    content: str
    channel: InputChannel = InputChannel.TEXT
    image_base64: Optional[str] = None   # Vision için
    timestamp: datetime = Field(default_factory=datetime.now)


class ChatResponse(BaseModel):
    """
    POST /chat — sistemden dönen yanıt.
    mentor_intervention: varsa proaktif müdahale ayrıca gelir.
    rag_source: RAG kullandıysa "senin notuna göre" ifadesi için.
    """
    session_id: str
    content: str                          # Ana LLM yanıtı
    rag_source: Optional[str] = None      # Hangi nottan alındı?
    mentor_intervention: Optional["MentorIntervention"] = None
    response_policy: Optional[ResponsePolicyMode] = None
    response_reasons: list[str] = Field(default_factory=list)
    dominant_signals: list[str] = Field(default_factory=list)
    policy_path: list[str] = Field(default_factory=list)
    current_state: UserState = UserState.UNKNOWN
    timestamp: datetime = Field(default_factory=datetime.now)


# ─────────────────────────────────────────
# SİNYAL VE ÖZELLİK MODELLERİ
# Feature Extractor (K3) üretir, State Model (K1) tüketir
# ─────────────────────────────────────────

class CameraSignal(BaseModel):
    """
    CV Engine (K3) — MediaPipe çıktısı.
    Eski projedeki FrameData'nın yeni karşılığı.
    Kamera kapalıysa bu model hiç oluşturulmaz.
    """
    timestamp: datetime = Field(default_factory=datetime.now)
    ear_score: float                      # Göz kapağı açıklık oranı (0.0–1.0)
    gaze_on_screen: bool                  # Ekrana bakıyor mu?
    hand_on_chin: bool                    # El çenede mi?
    head_tilt_angle: Optional[float] = None  # Baş eğimi (derece)


class BehaviorSignal(BaseModel):
    """
    Passive Tracking (K3) — kamerasız sinyal.
    Her mesaj olayında Feature Extractor bu modeli üretir.
    """
    timestamp: datetime = Field(default_factory=datetime.now)
    idle_time_seconds: float              # Son mesajdan bu yana geçen süre
    response_time_seconds: float          # Kullanıcı cevabı ne kadar sürede yazdı?
    message_length: int                   # Mesaj uzunluğu (karakter)
    retry_count: int                      # Aynı konuda kaçıncı soru?
    topic: Optional[str] = None           # Tespit edilen konu etiketi
    question_density: float = 0.0         # Mesajdaki soru yoğunluğu
    confusion_score: float = 0.0          # Yardım/karışıklık sinyali
    topic_stability: float = 1.0          # Son mesajlarla konu sürekliliği
    semantic_retry_score: float = 0.0     # Önceki mesajlara anlamsal tekrar benzerliği
    help_seeking_score: float = 0.0       # Dogrudan yardim/cevap talebi yogunlugu
    answer_commitment_score: float = 0.0  # Kendi deneme ve dusunme eforu sinyali


class FeatureVector(BaseModel):
    """
    Feature Extractor (K3) çıktısı → State Model (K1) girdisi.
    Hem kamera hem davranış sinyallerini birleştirir.
    Kamera kapalıysa camera_* alanları None kalır.
    """
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.now)

    # Davranışsal sinyaller (her zaman var)
    idle_time_seconds: float
    retry_count: int
    response_time_seconds: float
    message_length: int
    topic: Optional[str] = None
    question_density: float = 0.0
    confusion_score: float = 0.0
    topic_stability: float = 1.0
    semantic_retry_score: float = 0.0
    help_seeking_score: float = 0.0
    answer_commitment_score: float = 0.0

    # Kamera sinyalleri (opsiyonel)
    ear_score: Optional[float] = None
    gaze_on_screen: Optional[bool] = None
    hand_on_chin: Optional[bool] = None
    head_tilt_angle: Optional[float] = None


# ─────────────────────────────────────────
# STATE VE UNCERTAINTY MODELLERİ
# State Model + Uncertainty Engine (K1) üretir
# ─────────────────────────────────────────

class StateEstimate(BaseModel):
    """
    State Modeling (K1) çıktısı.
    confidence: 0.0–1.0 arası. Düşükse UE soru sorar, müdahale etmez.
    threshold: bu kullanıcı için adaptive eşik (başta sabit 0.75).
    """
    session_id: str
    state: UserState
    predicted_state: UserState = UserState.UNKNOWN
    confidence: float = Field(ge=0.0, le=1.0)
    decision_margin: float = 0.0
    uncertainty_signal: float = 1.0
    learning_pattern: LearningPattern = LearningPattern.NORMAL
    threshold: float = 0.75              # Adaptive threshold — başta sabit
    response_policy: ResponsePolicyMode = ResponsePolicyMode.DIRECT_HELP
    dominant_signals: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    policy_path: list[str] = Field(default_factory=list)
    deviation_features: dict = Field(default_factory=dict)
    state_scores: dict = Field(default_factory=dict)
    state_probabilities: dict = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

    @property
    def should_intervene(self) -> bool:
        """Confidence eşiği aşıldıysa ve durum sorunluysa müdahale et."""
        problematic = self.state in [
            UserState.DISTRACTED,
            UserState.FATIGUED,
            UserState.STUCK
        ]
        return problematic and self.confidence >= self.threshold


# ─────────────────────────────────────────
# MÜDAHALE MODELLERİ
# Mentor Agent (K1) üretir, K4 UI tüketir
# ─────────────────────────────────────────

class MentorIntervention(BaseModel):
    """
    Mentor Agent müdahale çıktısı.
    K4 UI bu modeli alır: tip'e göre farklı renk/stil uygular.
    ChatResponse içine gömülü gelir, WebSocket'ten de push edilebilir.
    """
    intervention_type: InterventionType
    message: str                          # Kullanıcıya gösterilecek metin
    triggered_by: UserState               # Hangi durumdan tetiklendi?
    learning_pattern: Optional[LearningPattern] = None
    confidence: float = Field(ge=0.0, le=1.0)
    decision_reason: Optional[str] = None
    policy_snapshot: dict = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


# ─────────────────────────────────────────
# HAFIZA MODELLERİ
# Short-term (K2) ve Long-term Memory (K2) kullanır
# ─────────────────────────────────────────

class MemoryEntry(BaseModel):
    """
    Long-term memory'ye yazılan olay kaydı.
    K2'nin Mem0 modülü bu formatı kullanır.
    Session kapanınca önemli olaylar buraya yazılır.
    """
    user_id: str
    session_id: str
    entry_type: str                       # "misconception", "topic_completed", "stuck_resolved"
    content: str                          # Ne oldu?
    topic: Optional[str] = None           # Hangi konuda?
    timestamp: datetime = Field(default_factory=datetime.now)


class UserProfile(BaseModel):
    """
    Long-term memory'den okunan kullanıcı profili.
    Session başında yüklenir, Dynamic Persona Prompt (K1) bunu kullanır.
    """
    user_id: str
    preferred_explanation_style: str = "detailed"  # "brief" | "detailed" | "example_heavy"
    weak_topics: list[str] = Field(default_factory=list)          # Zorlandığı konular
    strong_topics: list[str] = Field(default_factory=list)         # İyi olduğu konular
    recurring_misconceptions: list[str] = Field(default_factory=list)  # Tekrarlayan yanlış anlamalar
    avg_session_duration_minutes: float = 0.0
    adaptive_threshold: float = 0.75     # Kişiye özel UE eşiği
    total_sessions: int = 0
    last_session_at: Optional[datetime] = None
    normal_message_length: float = 0.0
    normal_response_delay_seconds: float = 0.0
    typical_retry_level: float = 0.0
    frequent_struggle_topics: list[str] = Field(default_factory=list)
    best_intervention_type: Optional[str] = None
    prefers_hint_first: bool = False
    prefers_direct_explanation: bool = False
    challenge_tolerance: float = 0.5
    intervention_sensitivity: float = 0.5


class ShortTermContext(BaseModel):
    """
    Aktif oturum bağlamı — session boyunca RAM'de tutulur.
    Her agent çağrısında bu bağlam LLM'e eklenir.
    Oturum kapanınca önemli kısımlar MemoryEntry'ye yazılır.
    """
    session_id: str
    user_id: str
    topic: Optional[str] = None
    messages: list[dict] = Field(default_factory=list)            # {"role": "user"/"assistant", "content": "..."}
    current_state: UserState = UserState.UNKNOWN
    retry_count: int = 0
    last_intervention_at: Optional[datetime] = None
    topics_covered: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────
# NOT ve RAG MODELLERİ
# K2 RAG pipeline kullanır
# ─────────────────────────────────────────

class NoteUploadRequest(BaseModel):
    """
    POST /upload — K4 UI bu modeli gönderir.
    Dosya içeriği base64 veya düz metin olarak gelir.
    """
    user_id: str
    filename: str
    content: str                          # Dosya içeriği (metin)
    file_type: str = "txt"               # "txt" | "md" | "pdf"


class NoteUploadResponse(BaseModel):
    """POST /upload — indexleme sonucu."""
    user_id: str
    filename: str
    chunk_count: int                      # Kaç parçaya bölündü?
    indexed: bool
    message: str


class UploadedDocumentSummary(BaseModel):
    """
    GET /upload/documents/{user_id} endpoint'inin dondurdugu dokuman ozeti.
    Bu model, UI'in veritabani satirlarina bagimli kalmamasini saglar.
    """
    filename: str
    file_type: Optional[str] = None
    file_size_bytes: Optional[int] = None
    chunk_count: int = 0
    indexed: bool = False
    uploaded_at: datetime


class RAGResult(BaseModel):
    """
    RAG Agent'ın not araması sonucu.
    ChatResponse içinde kullanılır.
    source_chunk: hangi nottan alındığını UI'a bildirir.
    """
    found: bool
    source_chunk: Optional[str] = None   # İlgili not parçası
    filename: Optional[str] = None       # Hangi dosyadan?
    relevance_score: Optional[float] = None


# ─────────────────────────────────────────
# DASHBOARD / RAPOR MODELLERİ
# K3 endpoint'i üretir, K4 dashboard tüketir
# ─────────────────────────────────────────

class FocusDataPoint(BaseModel):
    """Odak skoru grafiği için tek bir veri noktası."""
    timestamp: datetime
    score: float = Field(ge=0.0, le=1.0)
    state: UserState


class SessionSummary(BaseModel):
    """
    GET /dashboard/{session_id} — oturum sonu rapor.
    K4 dashboard bu modeli alır, grafik ve kartlar üretir.
    """
    session_id: str
    user_id: str
    duration_seconds: int
    topic: Optional[str]

    # Odak analizi
    focus_timeline: list[FocusDataPoint] = Field(default_factory=list)
    avg_focus_score: float = 0.0

    # Öğrenme analizi
    topics_covered: list[str] = Field(default_factory=list)
    detected_patterns: list[LearningPattern] = Field(default_factory=list)
    intervention_count: int = 0
    misconception_count: int = 0

    # Yarın için öneri
    recommended_topics: list[str] = Field(default_factory=list)
    mentor_note: Optional[str] = None    # Genel değerlendirme cümlesi


# ─────────────────────────────────────────
# HISTORY / CONTINUITY / FEEDBACK
# Yeni faz 1 modülleri bu modelleri kullanır
# ─────────────────────────────────────────

class FeedbackRequest(BaseModel):
    """POST /feedback isteği."""
    user_id: str
    session_id: Optional[str] = None
    message_id: Optional[str] = None
    feedback_type: str
    target_type: Optional[str] = None
    target_id: Optional[str] = None
    intervention_type: Optional[str] = None
    value: Optional[str] = None
    notes: Optional[str] = None


class FeedbackResponse(BaseModel):
    """POST /feedback sonucu."""
    status: str = "recorded"
    feedback_id: str
    adaptive_threshold: Optional[float] = None
    intervention_type: Optional[str] = None
    intervention_success_rate: Optional[float] = None
    behavior_change: Optional[dict] = None


# ─────────────────────────────────────────
# WebSocket MODELLERİ
# K3 WebSocket (K4 client) kullanır
# ─────────────────────────────────────────

class WebSocketMessage(BaseModel):
    """
    Server → Client push mesajı.
    event_type'a göre K4 UI farklı bileşen gösterir.
    """
    event_type: str                       # "intervention" | "state_update" | "session_end"
    session_id: str
    payload: dict                         # MentorIntervention veya StateEstimate dict'i
    timestamp: datetime = Field(default_factory=datetime.now)


# Forward reference'ları çöz
SessionEndResponse.model_rebuild()
ChatResponse.model_rebuild()
