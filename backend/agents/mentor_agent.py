# backend/agents/mentor_agent.py
#
# İki görevi var:
# 1. enrich_intervention() — müdahale mesajını LLM ile kişiselleştirir
# 2. generate_response()   — kullanıcının sorusuna cevap üretir
#
# Dynamic Persona Prompt burada oluşturulur:
# Her LLM çağrısından önce kullanıcı profilinden özel bir system prompt üretilir.
#
# Sahip: K1
# Bağımlılıklar: schemas.py, prompts/*.j2, openai

from __future__ import annotations

from typing import Optional

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - paket eksik oldugunda fallback
    OpenAI = None

from backend.core.config import settings
from backend.core.schemas import (
    ChatMessage,
    InterventionType,
    MentorIntervention,
    ShortTermContext,
    StateEstimate,
    UserProfile,
    UserState,
)


class MentorAgent:
    """
    Proaktif müdahale ve soru cevaplama.

    Kullanım:
        agent = MentorAgent()

        # Müdahaleyi kişiselleştir (graph'ın mentor_node'u)
        enriched = agent.enrich_intervention(intervention, profile, context)

        # Kullanıcı sorusuna cevap ver (graph'ın response_node'u)
        text = agent.generate_response(message, context, profile, ...)
    """

    def __init__(self):
        self.client = None
        self.model = settings.openai_model

    def _get_client(self):
        if self.client is not None:
            return self.client
        if not settings.openai_api_key or OpenAI is None:
            return None
        self.client = OpenAI(api_key=settings.openai_api_key)
        return self.client

    # ─────────────────────────────────────────────────────────────────
    # 1. MÜDAHALEYİ KİŞİSELLEŞTİR
    # ─────────────────────────────────────────────────────────────────

    def enrich_intervention(
        self,
        intervention: MentorIntervention,
        user_profile: Optional[UserProfile] = None,
        session_context: Optional[ShortTermContext] = None,
    ) -> MentorIntervention:
        """
        Uncertainty Engine'in ürettiği şablon müdahaleyi LLM ile zenginleştirir.

        Şablon: "Çok soru soruyorsun, strateji değiştir."
        Kişiselleştirilmiş: "Ahmet, türev konusunda geçen hafta da benzer bir
                             pattern vardı. Bu sefer şöyle dene: ..."
        """
        if intervention.intervention_type == InterventionType.NONE:
            return intervention

        topic = session_context.topic if session_context else None
        system_prompt = self._build_persona_prompt(user_profile, mode="intervention")

        user_prompt = self._build_intervention_prompt(
            intervention=intervention,
            topic=topic,
            user_profile=user_profile,
        )

        client = self._get_client()
        if client is None:
            enriched_message = intervention.message
        else:
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=150,
                    temperature=0.7,
                )
                enriched_message = response.choices[0].message.content.strip()
            except Exception:
                # LLM hatası durumunda şablon mesajla devam et
                enriched_message = intervention.message

        return MentorIntervention(
            intervention_type=intervention.intervention_type,
            message=enriched_message,
            triggered_by=intervention.triggered_by,
            learning_pattern=intervention.learning_pattern,
            confidence=intervention.confidence,
            decision_reason=intervention.decision_reason,
            policy_snapshot=intervention.policy_snapshot,
        )

    # ─────────────────────────────────────────────────────────────────
    # 2. SORUYA CEVAP VER
    # ─────────────────────────────────────────────────────────────────

    def generate_response(
        self,
        message: ChatMessage,
        session_context: Optional[ShortTermContext] = None,
        user_profile: Optional[UserProfile] = None,
        rag_context: Optional[str] = None,
        intervention: Optional[MentorIntervention] = None,
        state_estimate: Optional[StateEstimate] = None,
    ) -> str:
        """
        Kullanıcının mesajına cevap üretir.
        Dynamic Persona Prompt + tüm bağlam LLM'e gönderilir.
        """
        system_prompt = self._build_persona_prompt(user_profile, mode="chat")
        messages = self._build_message_history(
            current_message=message.content,
            session_context=session_context,
            rag_context=rag_context,
            state_estimate=state_estimate,
        )

        client = self._get_client()
        if client is None:
            return self._fallback_response(message, rag_context, intervention, state_estimate)

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    *messages,
                ],
                max_tokens=600,
                temperature=0.5,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            fallback = self._fallback_response(
                message,
                rag_context,
                intervention,
                state_estimate,
            )
            return f"{fallback} (LLM fallback: {str(e)})"

    # ─────────────────────────────────────────────────────────────────
    # DYNAMIC PERSONA PROMPT
    # Her LLM çağrısından önce bu oluşturulur.
    # Kullanıcı profilindeki bilgiler prompt'a enjekte edilir.
    # ─────────────────────────────────────────────────────────────────

    def _build_persona_prompt(
        self,
        profile: Optional[UserProfile],
        mode: str = "chat",             # "chat" | "intervention"
    ) -> str:
        """
        Dynamic Persona Prompt — projenin kişiselleştirme motoru.

        Profil yoksa (yeni kullanıcı) genel bir mentor prompt kullanılır.
        Profil varsa kullanıcıya özel bilgiler eklenir.

        mode="chat"         → soru cevaplama için
        mode="intervention" → müdahale mesajı için (daha kısa, daha odaklı)
        """
        # ── Temel kimlik ──────────────────────────────────────────────
        base = (
            "Sen bir kişisel öğrenme mentörüsün. "
            "Görevin öğrencinin hem sorusunu cevaplamak hem de "
            "nasıl öğrendiğini anlamak. "
            "Samimi, teşvik edici ve net konuş. "
            "Gereksiz uzun cevaplar verme."
        )

        if not profile:
            return base

        # ── Kişisel bağlam ────────────────────────────────────────────
        persona_lines = [base, "\n--- Kullanıcı Profili ---"]

        if profile.preferred_explanation_style == "brief":
            persona_lines.append("Kısa ve öz açıklamalar tercih eder.")
        elif profile.preferred_explanation_style == "example_heavy":
            persona_lines.append("Somut örneklerle daha iyi anlıyor.")
        else:
            persona_lines.append("Detaylı açıklamalardan fayda görüyor.")

        if profile.weak_topics:
            topics = ", ".join(profile.weak_topics)
            persona_lines.append(f"Zorlandığı konular: {topics}.")

        if profile.recurring_misconceptions:
            miscs = " | ".join(profile.recurring_misconceptions)
            persona_lines.append(
                f"Dikkat: bu kullanıcı şu yanlış anlamalara düşüyor: {miscs}. "
                "Cevap verirken bu noktalara özellikle dikkat et."
            )

        if profile.total_sessions > 0:
            persona_lines.append(
                f"Bu kullanıcıyla {profile.total_sessions} oturum geçti, "
                "onu tanıyorsun."
            )

        if mode == "intervention":
            persona_lines.append(
                "\nŞu an bir müdahale mesajı yazıyorsun. "
                "Maksimum 2 cümle, yargılayıcı değil destekleyici ol. "
                "Kullanıcıyı bunaltma."
            )

        return "\n".join(persona_lines)

    # ─────────────────────────────────────────────────────────────────
    # YARDIMCI METODLAR
    # ─────────────────────────────────────────────────────────────────

    def _build_intervention_prompt(
        self,
        intervention: MentorIntervention,
        topic: Optional[str],
        user_profile: Optional[UserProfile],
    ) -> str:
        """
        Müdahale için LLM'e gönderilecek user prompt.
        """
        type_descriptions = {
            InterventionType.HINT:     "ipucu ver",
            InterventionType.STRATEGY: "çalışma stratejisi öner",
            InterventionType.BREAK:    "mola önermesini sağla",
            InterventionType.QUESTION: "nazikçe soru sor",
            InterventionType.MODE_SWITCH: "farklı bir yöntem öner",
        }
        action = type_descriptions.get(
            intervention.intervention_type, "yardımcı ol"
        )

        past_misconceptions = ""
        if user_profile and user_profile.recurring_misconceptions:
            past_misconceptions = (
                f"Geçmiş yanlış anlamalar: "
                f"{', '.join(user_profile.recurring_misconceptions[:2])}"
            )

        return (
            f"Kullanıcı şu durumda: {intervention.triggered_by.value}. "
            f"Öğrenme paterni: {intervention.learning_pattern.value if intervention.learning_pattern else 'bilinmiyor'}. "
            f"{'Çalışılan konu: ' + topic + '.' if topic else ''} "
            f"{past_misconceptions} "
            f"Görеvin: kısa, samimi bir şekilde {action}."
        )

    def _build_message_history(
        self,
        current_message: str,
        session_context: Optional[ShortTermContext],
        rag_context: Optional[str],
        state_estimate: Optional[StateEstimate],
    ) -> list[dict]:
        """
        LLM'e gönderilecek mesaj geçmişini hazırlar.
        Sıra: geçmiş mesajlar → (varsa) not bağlamı → kullanıcının son mesajı
        """
        messages = []

        # Geçmiş mesajlar (son 6)
        if session_context and session_context.messages:
            recent = session_context.messages[-6:]
            for msg in recent:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

        # Not bağlamı varsa sisteme ekle
        if rag_context:
            messages.append({
                "role": "system",
                "content": (
                    f"Kullanıcının kendi notlarından ilgili bölüm:\n"
                    f"---\n{rag_context}\n---\n"
                    f"Cevap verirken bu notları referans al ve "
                    f"'senin notuna göre...' diye başla."
                ),
            })

        # Kullanıcının anlık durumunu sisteme bildir
        if state_estimate and state_estimate.state != UserState.FOCUSED:
            state_hint = {
                UserState.STUCK:       "Kullanıcı bu konuda takılı görünüyor.",
                UserState.FATIGUED:    "Kullanıcı yorgun görünüyor, kısa tut.",
                UserState.DISTRACTED:  "Kullanıcı dağınık, odaklanmasına yardım et.",
            }.get(state_estimate.state, "")
            if state_hint:
                messages.append({"role": "system", "content": state_hint})

        # Son kullanıcı mesajı
        messages.append({"role": "user", "content": current_message})

        return messages

    def _fallback_response(
        self,
        message: ChatMessage,
        rag_context: Optional[str],
        intervention: Optional[MentorIntervention],
        state_estimate: Optional[StateEstimate],
    ) -> str:
        """
        LLM hazir degilse minimum faydali cevap dondur.
        """
        if intervention and intervention.message:
            return intervention.message

        if rag_context:
            snippet = rag_context.strip().splitlines()[0][:180]
            return f"Senin notuna gore ilgili kisim su: {snippet}"

        if state_estimate and state_estimate.state != UserState.FOCUSED:
            return (
                f"Su an {state_estimate.state.value} sinyali goruyorum. "
                "Sorunu bir adim daha detaylandirirsan birlikte ilerleyelim."
            )

        return (
            "Sorunu aldim. LLM baglantisi hazir degil ama istersen problemi "
            "biraz daha acip adim adim ilerleyebiliriz."
        )
