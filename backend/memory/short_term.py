# backend/memory/short_term.py
#
# Aktif oturum boyunca RAM'de tutulan kısa süreli hafıza.
# session_agent.py'deki _active_sessions dict'ini formalize eder.
#
# Neden ayrı bir dosya?
# session_agent.py hem oturum mantığını hem RAM store'unu yönetiyordu.
# Bu dosya sadece store sorumluluğunu alıyor.
# session_agent.py iş mantığına odaklanıyor.
#
# Sahip: K2
# Bağımlılıklar: schemas.py

import logging
from datetime import datetime
from typing import Optional

from backend.core.schemas import ShortTermContext, UserState
from backend.core.config import settings

logger = logging.getLogger(__name__)

# ── Global store ──────────────────────────────────────────────────────────────
# Tüm aktif oturumlar burada tutuluyor.
# Key: session_id, Value: ShortTermContext
#
# Production'da bu Redis'e taşınır:
#   - Sunucu restart'ta kaybolmaz
#   - Birden fazla sunucu instance'ı aynı store'u paylaşır
#   - TTL ile otomatik expire edilir
#
# Şimdilik RAM yeterli — tek sunucu, geliştirme ortamı.

_sessions: dict[str, ShortTermContext] = {}


class ShortTermMemory:
    """
    Aktif oturum verilerini RAM'de yönetir.

    Session Agent bu sınıfı kullanır:
        store = ShortTermMemory()
        store.save(context)
        context = store.get(session_id)
        store.delete(session_id)
    """

    def save(self, context: ShortTermContext) -> None:
        """Yeni oturum context'ini kaydeder veya mevcut olanı günceller."""
        _sessions[context.session_id] = context
        logger.debug("Session kaydedildi: %s", context.session_id)

    def get(self, session_id: str) -> Optional[ShortTermContext]:
        """
        Session_id ile context'i getirir.
        Yoksa None döner — API katmanı 404 döndürmeli.
        """
        return _sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        """
        Oturumu RAM'den siler.
        Döner: True → silindi, False → zaten yoktu.
        """
        if session_id in _sessions:
            del _sessions[session_id]
            logger.debug("Session silindi: %s", session_id)
            return True
        return False

    def add_message(
        self,
        session_id: str,
        role: str,        # "user" veya "assistant"
        content: str,
    ) -> bool:
        """
        Session context'ine yeni mesaj ekler.
        Max mesaj limitini uygular (config: short_term_max_messages = 20).

        Limit mantığı:
        İlk mesajı koru (oturum başlangıcı bağlamı önemli),
        ortadan kırp, en son mesajları tut.

        Döner: True → eklendi, False → session bulunamadı.
        """
        context = _sessions.get(session_id)
        if not context:
            return False

        context.messages.append({"role": role, "content": content})

        if len(context.messages) > settings.short_term_max_messages:
            context.messages = (
                context.messages[:1]
                + context.messages[-(settings.short_term_max_messages - 1):]
            )

        return True

    def update_state(
        self,
        session_id: str,
        new_state: UserState,
    ) -> bool:
        """
        Kullanıcının anlık durumunu günceller.
        State Model'in her tahminden sonra çağrılır.
        """
        context = _sessions.get(session_id)
        if not context:
            return False
        context.current_state = new_state
        return True

    def add_topic(self, session_id: str, topic: str) -> bool:
        """
        Oturumda yeni bir konu keşfedilince listeye ekler.
        Tekrar eklemeyi önler.
        """
        context = _sessions.get(session_id)
        if not context:
            return False
        if topic and topic not in context.topics_covered:
            context.topics_covered.append(topic)
        return True

    def increment_retry(self, session_id: str) -> int:
        """
        Retry sayacını bir artırır.
        Döner: güncel retry sayısı. Session yoksa -1.
        """
        context = _sessions.get(session_id)
        if not context:
            return -1
        context.retry_count += 1
        return context.retry_count

    def reset_retry(self, session_id: str) -> None:
        """Uzun/açıklamalı mesaj gelince retry sayacını sıfırla."""
        context = _sessions.get(session_id)
        if context:
            context.retry_count = 0

    def mark_intervention(self, session_id: str) -> None:
        """Son müdahale zamanını günceller (cooldown için)."""
        context = _sessions.get(session_id)
        if context:
            context.last_intervention_at = datetime.now()

    def get_recent_messages(
        self, session_id: str, last_n: int = 6
    ) -> list[dict]:
        """
        Son N mesajı döner.
        MentorAgent'ın LLM prompt'u için kullanılır.
        """
        context = _sessions.get(session_id)
        if not context:
            return []
        return context.messages[-last_n:]

    def list_active_sessions(self) -> list[str]:
        """
        Tüm aktif session_id'leri döner.
        Debug ve monitoring için.
        """
        return list(_sessions.keys())

    def session_count(self) -> int:
        """Kaç aktif oturum var?"""
        return len(_sessions)