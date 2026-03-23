# backend/memory/long_term.py
#
# Kullanıcının oturumlar arası hafızasını yönetir.
# Mem0 local mod kullanır — veriler diske kaydedilir, cloud'a gitmez.
#
# İki görevi var:
#   1. write()       — oturum kapanınca önemli olayları Mem0'a yazar
#   2. get_profile() — oturum açılınca kullanıcı profilini Mem0'dan oluşturur
#
# Sahip: K2
# Bağımlılıklar: mem0ai, schemas.py, config.py

import json
import logging
from typing import Optional

from mem0 import Memory

from backend.core.config import settings
from backend.core.schemas import MemoryEntry, UserProfile

logger = logging.getLogger(__name__)


# ── Mem0 config ───────────────────────────────────────────────────────────────
#
# Local mod: vektörler ChromaDB ile diske kaydediliyor.
# LLM: hafızayı anlamlandırmak için gpt-4o-mini kullanıyoruz.
# Neden mini? Bu görev için tam gpt-4o gerekmez, maliyet düşsün.
# Embedding: OpenAI text-embedding-3-small (config'den geliyor).

_MEM0_CONFIG = {
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "focusai_memories",
            "path": "data/mem0",          # diske kayıt yolu
        },
    },
    "llm": {
        "provider": "openai",
        "config": {
            "api_key": settings.openai_api_key,
            "model": "gpt-4o-mini",       # hafıza işleme için mini yeterli
        },
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "api_key": settings.openai_api_key,
            "model": settings.openai_embedding_model,  # text-embedding-3-small
        },
    },
}


class LongTermMemory:
    """
    Oturumlar arası kalıcı hafıza.

    Nasıl kullanılır:
        memory = LongTermMemory()

        # Oturum kapanınca yaz:
        memory.write(entry)

        # Oturum açılınca oku:
        profile = memory.get_profile(user_id)
    """

    def __init__(self):
        # Mem0 client'ı başlat
        # İlk çalıştırmada ChromaDB collection'ı oluşturur
        # Sonraki çalıştırmalarda mevcut collection'ı yükler
        try:
            self.client = Memory.from_config(_MEM0_CONFIG)
            self._available = True
        except Exception as e:
            # Mem0 başlatılamazsa (paket yok, config hatalı vs.)
            # sistemi çökertme, sadece logla ve devre dışı bırak
            logger.warning(
                "Mem0 başlatılamadı, hafıza devre dışı: %s", e
            )
            self.client = None
            self._available = False

    # ─────────────────────────────────────────────────────────────────
    # YAZMA
    # ─────────────────────────────────────────────────────────────────

    def write(self, entry: MemoryEntry) -> bool:
        """
        Bir olay kaydını Mem0'a yazar.

        entry.entry_type değerlerine göre farklı mesajlar oluşturur:
            "high_retry"     → "X konusunda N kez takıldı"
            "topic_studied"  → "X konusu çalışıldı"

        Parametreler:
            entry : MemoryEntry — session_agent'ın ürettiği olay kaydı

        Döner:
            True  → başarıyla yazıldı
            False → hata oluştu (sistem çalışmaya devam eder)
        """
        if not self._available:
            return False

        # entry_type'a göre Mem0'a gönderilecek mesajı oluştur
        message = self._build_memory_message(entry)
        if not message:
            return False

        try:
            self.client.add(
                message,
                user_id=entry.user_id,
                # metadata: Mem0 bunu saklar, sonra filtreleme için kullanılabilir
                metadata={
                    "session_id": entry.session_id,
                    "entry_type": entry.entry_type,
                    "topic": entry.topic or "",
                },
            )
            logger.debug(
                "Mem0'a yazıldı | user=%s | type=%s | topic=%s",
                entry.user_id, entry.entry_type, entry.topic,
            )
            return True

        except Exception as e:
            logger.error("Mem0 yazma hatası: %s", e)
            return False

    def write_batch(self, entries: list[MemoryEntry]) -> int:
        """
        Birden fazla kaydı sırayla yazar.
        Döner: başarıyla yazılan kayıt sayısı.
        """
        success_count = 0
        for entry in entries:
            if self.write(entry):
                success_count += 1
        return success_count

    # ─────────────────────────────────────────────────────────────────
    # OKUMA
    # ─────────────────────────────────────────────────────────────────

    def get_profile(self, user_id: str) -> UserProfile:
        """
        Mem0'daki geçmiş hafızadan UserProfile oluşturur.

        Nasıl çalışır:
            1. Mem0'dan kullanıcının tüm hafızasını çek
            2. "Zayıf konular neler?" diye ara
            3. "Tekrarlayan hatalar neler?" diye ara
            4. Bulguları UserProfile'a doldur

        Mem0 yoksa veya kayıt yoksa → varsayılan profil döner.
        Sistem hiçbir zaman çökmez.
        """
        if not self._available:
            return self._default_profile(user_id)

        try:
            # Kullanıcının tüm hafıza kayıtlarını çek
            all_memories = self.client.get_all(user_id=user_id)

            if not all_memories or not all_memories.get("results"):
                # Yeni kullanıcı — hiç kaydı yok
                return self._default_profile(user_id)

            # Hafıza kayıtlarından profil bilgilerini çıkar
            return self._build_profile_from_memories(
                user_id=user_id,
                memories=all_memories["results"],
            )

        except Exception as e:
            logger.error("Mem0 okuma hatası: %s", e)
            return self._default_profile(user_id)

    # ─────────────────────────────────────────────────────────────────
    # PRIVATE METODLAR
    # ─────────────────────────────────────────────────────────────────

    def _build_memory_message(self, entry: MemoryEntry) -> Optional[str]:
        """
        MemoryEntry'yi Mem0'un anlayacağı doğal dil cümlesine çevirir.

        Neden doğal dil?
        Mem0, LLM ile bu cümleyi işleyip özümseyecek.
        JSON veya dict değil, cümle göndermek daha iyi sonuç verir.
        """
        if entry.entry_type == "high_retry":
            topic_str = f"'{entry.topic}' konusunda" if entry.topic else "bir konuda"
            return (
                f"Kullanıcı {topic_str} aynı soruyu defalarca sordu ve takıldı. "
                f"Detay: {entry.content}"
            )

        elif entry.entry_type == "topic_studied":
            return (
                f"Kullanıcı bu oturumda {entry.topic or 'bir konu'} çalıştı. "
                f"Detay: {entry.content}"
            )

        elif entry.entry_type == "misconception":
            return (
                f"Kullanıcı şu konuda yanlış anlama yaşadı: {entry.content}"
            )

        elif entry.entry_type == "session_completed":
            return (
                f"Kullanıcı bir oturumu tamamladı. {entry.content}"
            )

        # Bilinmeyen tip — yine de kaydet
        return entry.content

    def _build_profile_from_memories(
        self,
        user_id: str,
        memories: list[dict],
    ) -> UserProfile:
        """
        Mem0'dan gelen ham hafıza listesini UserProfile'a dönüştürür.

        Mem0'un döndürdüğü format:
            [
                {"memory": "Kullanıcı türevde zorlandı", "metadata": {...}},
                {"memory": "Matematik çalışıldı", "metadata": {...}},
                ...
            ]

        Strateji:
        - "takıldı", "zorlandı" içerenleri → weak_topics
        - "yanlış anlama" içerenleri → recurring_misconceptions
        - "çalıştı", "tamamladı" içerenleri → strong_topics (potansiyel)
        - Toplam kayıt sayısından total_sessions tahmini
        """
        weak_topics = []
        strong_topics = []
        misconceptions = []
        session_count = 0

        for mem in memories:
            text = mem.get("memory", "").lower()
            metadata = mem.get("metadata", {})
            topic = metadata.get("topic", "")

            # Zayıflık sinyalleri
            if any(kw in text for kw in ["takıldı", "zorlandı", "defalarca", "aynı soruyu"]):
                if topic and topic not in weak_topics:
                    weak_topics.append(topic)

            # Yanlış anlama sinyalleri
            elif any(kw in text for kw in ["yanlış anlama", "hata", "misconception"]):
                if mem.get("memory") and mem["memory"] not in misconceptions:
                    misconceptions.append(mem["memory"][:100])  # max 100 karakter

            # Güçlü konu sinyalleri
            elif any(kw in text for kw in ["tamamladı", "başarıyla", "iyi gidiyor"]):
                if topic and topic not in strong_topics:
                    strong_topics.append(topic)

            # Oturum sayısını tahmin et
            entry_type = metadata.get("entry_type", "")
            if entry_type == "session_completed":
                session_count += 1

        # Oturum sayısı kaydedilmediyse hafıza kayıt sayısından tahmin et
        if session_count == 0:
            session_count = max(1, len(memories) // 3)

        return UserProfile(
            user_id=user_id,
            preferred_explanation_style="detailed",  # Hafta 3'te kullanıcıdan öğrenilecek
            weak_topics=weak_topics,
            strong_topics=strong_topics,
            recurring_misconceptions=misconceptions,
            adaptive_threshold=settings.default_uncertainty_threshold,
            total_sessions=session_count,
        )

    def _default_profile(self, user_id: str) -> UserProfile:
        """
        Yeni kullanıcı veya Mem0 erişilemez durumda varsayılan profil.
        """
        return UserProfile(
            user_id=user_id,
            preferred_explanation_style="detailed",
            weak_topics=[],
            strong_topics=[],
            recurring_misconceptions=[],
            adaptive_threshold=settings.default_uncertainty_threshold,
            total_sessions=0,
        )