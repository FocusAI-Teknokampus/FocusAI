# backend/agents/session_agent.py

from __future__ import annotations

import json
from typing import Optional
from uuid import uuid4

from sqlalchemy.orm import Session

from backend.core.database import SessionLocal, UserProfileRecord
from backend.core.schemas import (
    MemoryEntry,
    SessionEndRequest,
    SessionStartRequest,
    SessionStartResponse,
    ShortTermContext,
    UserProfile,
    UserState,
)
from backend.memory.long_term import LongTermMemory
from backend.memory.short_term import ShortTermMemory, _sessions as _active_sessions
from backend.services.analytics_service import AnalyticsService
from backend.services.session_service import SessionService


class SessionAgent:
    """
    Session lifecycle coordinator.
    """

    def __init__(self):
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory()

    def start_session(
        self,
        request: SessionStartRequest,
        db: Session,
    ) -> SessionStartResponse:
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

        service = SessionService(db)
        service.create_session(
            session_id=session_id,
            user_id=request.user_id,
            topic=request.topic,
            camera_used=request.camera_enabled,
        )

        return SessionStartResponse(
            session_id=session_id,
            user_id=request.user_id,
            topic=request.topic,
            camera_enabled=request.camera_enabled,
        )

    def load_context(self, session_id: str) -> Optional[ShortTermContext]:
        return self.short_term.get(session_id)

    def load_profile(self, user_id: str, db: Optional[Session] = None) -> UserProfile:
        """
        Load profile by merging DB state with optional long-term memory signals.

        DB remains the source of truth for adaptive threshold and editable profile
        fields. Long-term memory only enriches the returned shape.
        """
        owns_session = db is None
        active_db = db or SessionLocal()
        memory_profile: Optional[UserProfile] = None

        try:
            profile = self.long_term.get_profile(user_id)
            if profile:
                memory_profile = profile
        except Exception:
            pass

        try:
            row = (
                active_db.query(UserProfileRecord)
                .filter(UserProfileRecord.user_id == user_id)
                .first()
            )
            if row:
                return UserProfile(
                    user_id=user_id,
                    preferred_explanation_style=(
                        row.preferred_explanation_style
                        or (memory_profile.preferred_explanation_style if memory_profile else "detailed")
                    ),
                    weak_topics=self._merge_profile_lists(
                        json.loads(row.weak_topics or "[]"),
                        memory_profile.weak_topics if memory_profile else [],
                    ),
                    strong_topics=self._merge_profile_lists(
                        json.loads(row.strong_topics or "[]"),
                        memory_profile.strong_topics if memory_profile else [],
                    ),
                    recurring_misconceptions=self._merge_profile_lists(
                        json.loads(row.recurring_misconceptions or "[]"),
                        memory_profile.recurring_misconceptions if memory_profile else [],
                    ),
                    avg_session_duration_minutes=row.avg_session_duration_minutes or 0.0,
                    adaptive_threshold=(
                        row.adaptive_threshold
                        if row.adaptive_threshold is not None
                        else (memory_profile.adaptive_threshold if memory_profile else 0.75)
                    ),
                    total_sessions=memory_profile.total_sessions if memory_profile else 0,
                    last_session_at=memory_profile.last_session_at if memory_profile else None,
                )
        finally:
            if owns_session:
                active_db.close()

        if memory_profile:
            return memory_profile

        return UserProfile(user_id=user_id)

    def update_context(
        self,
        session_id: str,
        role: str,
        content: str,
        db: Optional[Session] = None,
        new_state: Optional[UserState] = None,
        topic: Optional[str] = None,
        message_type: Optional[str] = None,
        llm_confidence: Optional[float] = None,
    ) -> None:
        self.short_term.add_message(session_id, role, content)

        if new_state is not None:
            self.short_term.update_state(session_id, new_state)

        if topic:
            self.short_term.add_topic(session_id, topic)

        if db is not None:
            context = self.short_term.get(session_id)
            current_state_value = context.current_state.value if context else None

            service = SessionService(db)
            service.save_message(
                session_id=session_id,
                role=role,
                content=content,
                user_state=current_state_value,
                detected_topic=topic,
                message_type=message_type,
                llm_confidence=llm_confidence,
            )

            if context is not None and new_state is not None:
                service.update_session_state(
                    session_id=session_id,
                    state=new_state.value,
                    retry_count=context.retry_count,
                )

    def end_session(
        self,
        request: SessionEndRequest,
        db: Session,
    ) -> dict:
        context = self.short_term.get(request.session_id)
        if not context:
            return {"status": "not_found"}

        session_entry = MemoryEntry(
            user_id=request.user_id,
            session_id=request.session_id,
            entry_type="session_completed",
            content=(
                f"Oturum tamamlandi. "
                f"Konu: {context.topic or 'genel'}. "
                f"Toplam mesaj: {len(context.messages)}."
            ),
            topic=context.topic,
        )

        entries = self._extract_memory_entries(context)
        entries.append(session_entry)

        written = self._write_to_memory(
            user_id=request.user_id,
            session_id=request.session_id,
            entries=entries,
        )

        final_state = context.current_state.value if context.current_state else None

        session_service = SessionService(db)
        analytics_service = AnalyticsService(db)

        session_service.end_session(
            session_id=request.session_id,
            final_state=final_state,
        )
        report = analytics_service.finalize_session_report(request.session_id)

        try:
            from backend.agents.graph import _feature_extractor

            _feature_extractor.reset_session(request.session_id)
        except Exception:
            pass

        self.short_term.delete(request.session_id)

        return {
            "status": "ended",
            "memory_entries_written": written,
            "topics_covered": context.topics_covered,
            "summary_text": report.summary_text if report else None,
        }

    def get_recent_messages_as_text(self, session_id: str, last_n: int = 6) -> str:
        messages = self.short_term.get_recent_messages(session_id, last_n)

        lines = []
        for msg in messages:
            prefix = "Kullanici" if msg["role"] == "user" else "Mentor"
            lines.append(f"{prefix}: {msg['content']}")

        return "\n".join(lines)

    def _extract_memory_entries(self, context: ShortTermContext) -> list[MemoryEntry]:
        entries: list[MemoryEntry] = []

        if context.retry_count >= 3:
            entries.append(
                MemoryEntry(
                    user_id=context.user_id,
                    session_id=context.session_id,
                    entry_type="high_retry",
                    content=(
                        f"{context.retry_count} kez ayni konuda soru soruldu. "
                        f"Konu: {context.topic or 'bilinmiyor'}"
                    ),
                    topic=context.topic,
                )
            )

        for topic in context.topics_covered:
            entries.append(
                MemoryEntry(
                    user_id=context.user_id,
                    session_id=context.session_id,
                    entry_type="topic_studied",
                    content=f"{topic} konusu calisildi.",
                    topic=topic,
                )
            )

        return entries

    def _write_to_memory(
        self,
        user_id: str,
        session_id: str,
        entries: list[MemoryEntry],
    ) -> int:
        try:
            return self.long_term.write_batch(entries)
        except Exception:
            return 0

    def _merge_profile_lists(self, primary: list[str], secondary: list[str]) -> list[str]:
        merged: list[str] = []
        for item in [*primary, *secondary]:
            if item and item not in merged:
                merged.append(item)
        return merged


_active_sessions = _active_sessions
