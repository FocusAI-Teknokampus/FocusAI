# backend/agents/session_agent.py
#
# Oturum yaşam döngüsünü yönetir.
# Hafta 2 güncellemesi: Mem0 ve ShortTermMemory entegre edildi.
#
# Değişenler:
#   load_profile()     → artık Mem0'dan gerçek profil okuyor
#   _write_to_memory() → artık Mem0'a gerçekten yazıyor
#   RAM store          → ShortTermMemory sınıfına taşındı
#
# Sahip: K1
# Bağımlılıklar: schemas.py, memory/short_term.py, memory/long_term.py

from datetime import datetime
from typing import Optional

from backend.core.schemas import (
    MemoryEntry,
    SessionEndRequest,
    SessionStartRequest,
    SessionStartResponse,
    ShortTermContext,
    UserProfile,
    UserState,
)
from backend.core.config import settings
from backend.memory.short_term import ShortTermMemory
from backend.memory.long_term import LongTermMemory


class SessionAgent:
    """
    Oturum yaşam döngüsünü yönetir.

    start_session()  → oturum açar, context oluşturur
    load_context()   → aktif context'i döner
    load_profile()   → Mem0'dan gerçek profil okur (artık stub değil)
    update_context() → her mesajda context günceller
    end_session()    → özetler, Mem0'a yazar, temizler
    """

    def __init__(self):
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory()

    # ─────────────────────────────────────────────────────────────────
    # OTURUM BAŞLAT
    # ─────────────────────────────────────────────────────────────────

    def start_session(
        self, request: SessionStartRequest
    ) -> SessionStartResponse:
        """
        POST /session/start endpoint'i bu metodu çağırır.
        Yeni ShortTermContext oluşturur, RAM'e yazar.
        """
        from uuid import uuid4
        session_id = str(uuid4())

        context = ShortTermContext(
            session_id=session_id,
            user_id=request.user_id,
            topic=request.topic,
            messages=[],
            current_state=UserState.UNKNOWN,
            retry_count=0,
            topics_covered=[request.topic] if request.topic else [],
        )

        self.short_term.save(context)

        return SessionStartResponse(
            session_id=session_id,
            user_id=request.user_id,
            topic=request.topic,
            camera_enabled=request.camera_enabled,
        )

    # ─────────────────────────────────────────────────────────────────
    # CONTEXT VE PROFİL OKU
    # ─────────────────────────────────────────────────────────────────

    def load_context(self, session_id: str) -> Optional[ShortTermContext]:
        """
        Graph'ın session_node'u her mesajda bunu çağırır.
        Session yoksa None döner.
        """
        return self.short_term.get(session_id)

    def load_profile(self, user_id: str) -> UserProfile:
        """
        Mem0'dan kullanıcı profilini yükler.

        Hafta 1'de: her zaman varsayılan profil dönerdi.
        Hafta 2'de: Mem0'da kayıt varsa gerçek profil, yoksa varsayılan.

        Mem0 erişilemez durumdaysa sistem çökmez —
        LongTermMemory._default_profile() varsayılan döner.
        """
        return self.long_term.get_profile(user_id)

    # ─────────────────────────────────────────────────────────────────
    # CONTEXT GÜNCELLE
    # ─────────────────────────────────────────────────────────────────

    def update_context(
        self,
        session_id: str,
        role: str,
        content: str,
        new_state: Optional[UserState] = None,
        topic: Optional[str] = None,
    ) -> None:
        """
        Her mesaj alışverişinde çağrılır.
        Mesaj ekler, state günceller, konu ekler.
        """
        self.short_term.add_message(session_id, role, content)

        if new_state:
            self.short_term.update_state(session_id, new_state)

        if topic:
            self.short_term.add_topic(session_id, topic)

    # ─────────────────────────────────────────────────────────────────
    # OTURUM KAPAT
    # ─────────────────────────────────────────────────────────────────

    def end_session(self, request: SessionEndRequest) -> dict:
        """
        POST /session/end endpoint'i bunu çağırır.
        1. Önemli olayları çıkar
        2. Mem0'a yaz
        3. RAM'den sil
        """
        context = self.short_term.get(request.session_id)
        if not context:
            return {"status": "not_found"}

        # Oturum tamamlandı kaydı ekle
        session_entry = MemoryEntry(
            user_id=request.user_id,
            session_id=request.session_id,
            entry_type="session_completed",
            content=(
                f"Oturum tamamlandı. "
                f"Konu: {context.topic or 'genel'}. "
                f"Toplam mesaj: {len(context.messages)}."
            ),
            topic=context.topic,
        )

        # Diğer önemli olayları çıkar
        entries = self._extract_memory_entries(context)
        entries.append(session_entry)

        # Mem0'a toplu yaz
        written = self._write_to_memory(
            user_id=request.user_id,
            session_id=request.session_id,
            entries=entries,
        )

        # RAM'den temizle
        self.short_term.delete(request.session_id)

        return {
            "status": "ended",
            "memory_entries_written": written,
            "topics_covered": context.topics_covered,
        }

    def get_recent_messages_as_text(
        self, session_id: str, last_n: int = 6
    ) -> str:
        """
        Son N mesajı LLM prompt'u için düz metin formatına çevirir.
        """
        messages = self.short_term.get_recent_messages(session_id, last_n)
        lines = []
        for msg in messages:
            prefix = "Kullanıcı" if msg["role"] == "user" else "Mentor"
            lines.append(f"{prefix}: {msg['content']}")
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────
    # PRIVATE METODLAR
    # ─────────────────────────────────────────────────────────────────

    def _extract_memory_entries(
        self, context: ShortTermContext
    ) -> list[MemoryEntry]:
        """
        Context'teki önemli olayları MemoryEntry listesine çevirir.
        """
        entries = []

        # Yüksek retry → takılma kaydı
        if context.retry_count >= 3:
            entries.append(MemoryEntry(
                user_id=context.user_id,
                session_id=context.session_id,
                entry_type="high_retry",
                content=(
                    f"{context.retry_count} kez aynı konuda soru soruldu. "
                    f"Konu: {context.topic or 'bilinmiyor'}"
                ),
                topic=context.topic,
            ))

        # İşlenen konuları kaydet
        for topic in context.topics_covered:
            entries.append(MemoryEntry(
                user_id=context.user_id,
                session_id=context.session_id,
                entry_type="topic_studied",
                content=f"{topic} konusu çalışıldı.",
                topic=topic,
            ))

        return entries

    def _write_to_memory(
        self,
        user_id: str,
        session_id: str,
        entries: list[MemoryEntry],
    ) -> int:
        """
        MemoryEntry listesini Mem0'a yazar.

        Hafta 1'de: pass (hiçbir şey yazmıyordu).
        Hafta 2'de: LongTermMemory.write_batch() ile Mem0'a yazıyor.

        Döner: yazılan kayıt sayısı.
        """
        return self.long_term.write_batch(entries)


# ── Geriye dönük uyumluluk ────────────────────────────────────────────────────
# graph.py ve diğer modüller _active_sessions'ı doğrudan import ediyordu.
# ShortTermMemory'nin iç store'una işaret eden alias.
from backend.memory.short_term import _sessions as _active_sessions