from __future__ import annotations

from statistics import mean, pstdev
from typing import Any

from sqlalchemy.orm import Session

from backend.core.database import (
    MessageRecord,
    SessionRecord,
    SessionReportRecord,
    UserBaselineRecord,
    UserProfileRecord,
)


class BaselineService:
    """
    Kullanıcı için kişisel baseline metriklerini üretir.
    """

    def __init__(self, db: Session):
        self.db = db

    def refresh_user_baseline(self, user_id: str, sample_size: int = 5) -> dict[str, Any]:
        baseline_data = self._collect_baseline_data(user_id=user_id, sample_size=sample_size)
        if baseline_data["sample_session_count"] == 0:
            existing = (
                self.db.query(UserBaselineRecord)
                .filter(UserBaselineRecord.user_id == user_id)
                .first()
            )
            if existing:
                return self._serialize(existing, baseline_data["metrics"])
            return self._empty_baseline(user_id)

        aggregate = baseline_data["aggregate"]
        baseline = (
            self.db.query(UserBaselineRecord)
            .filter(UserBaselineRecord.user_id == user_id)
            .first()
        )
        if baseline is None:
            baseline = UserBaselineRecord(user_id=user_id)
            self.db.add(baseline)

        baseline.sample_session_count = baseline_data["sample_session_count"]
        baseline.avg_message_length = aggregate["avg_message_length"]
        baseline.avg_response_time_seconds = aggregate["avg_response_time_seconds"]
        baseline.avg_idle_gap_seconds = aggregate["avg_idle_gap_seconds"]
        baseline.avg_messages_per_session = aggregate["avg_messages_per_session"]
        baseline.avg_session_duration_seconds = aggregate["avg_session_duration_seconds"]
        baseline.avg_focus_score = aggregate["avg_focus_score"]
        baseline.question_style = aggregate["question_style"]
        baseline.personalized_threshold = aggregate["personalized_threshold"]

        self._sync_profile_threshold(
            user_id=user_id,
            baseline_threshold=aggregate["personalized_threshold"],
            sample_count=baseline_data["sample_session_count"],
        )

        self.db.commit()
        self.db.refresh(baseline)
        return self._serialize(baseline, baseline_data["metrics"])

    def get_user_baseline(self, user_id: str, sample_size: int = 5) -> dict[str, Any]:
        baseline = (
            self.db.query(UserBaselineRecord)
            .filter(UserBaselineRecord.user_id == user_id)
            .first()
        )
        baseline_data = self._collect_baseline_data(user_id=user_id, sample_size=sample_size)
        metrics = baseline_data["metrics"]

        if baseline is None:
            if baseline_data["sample_session_count"] == 0:
                return self._empty_baseline(user_id)
            return {
                "user_id": user_id,
                "sample_session_count": baseline_data["sample_session_count"],
                "enough_data": baseline_data["sample_session_count"] >= 3,
                **baseline_data["aggregate"],
                "metrics": metrics,
                "updated_at": None,
            }

        return self._serialize(baseline, metrics)

    def get_state_model_baseline(self, user_id: str, sample_size: int = 8) -> dict[str, Any]:
        baseline = self._collect_baseline_data(user_id=user_id, sample_size=sample_size)
        if baseline["sample_session_count"] == 0:
            return {
                "user_id": user_id,
                "sample_session_count": 0,
                "enough_data": False,
                "metrics": {},
                "question_style": None,
                "personalized_threshold": 0.75,
            }

        return {
            "user_id": user_id,
            "sample_session_count": baseline["sample_session_count"],
            "enough_data": baseline["sample_session_count"] >= 3,
            "metrics": baseline["metrics"],
            "question_style": baseline["aggregate"]["question_style"],
            "personalized_threshold": baseline["aggregate"]["personalized_threshold"],
        }

    def _collect_baseline_data(self, user_id: str, sample_size: int) -> dict[str, Any]:
        sessions = (
            self.db.query(SessionRecord)
            .filter(SessionRecord.user_id == user_id, SessionRecord.ended_at.isnot(None))
            .order_by(SessionRecord.ended_at.desc())
            .limit(max(1, sample_size))
            .all()
        )

        if not sessions:
            return {
                "sample_session_count": 0,
                "aggregate": {},
                "metrics": {},
            }

        session_ids = [row.session_id for row in sessions]
        messages = (
            self.db.query(MessageRecord)
            .filter(MessageRecord.session_id.in_(session_ids))
            .order_by(MessageRecord.session_id.asc(), MessageRecord.timestamp.asc())
            .all()
        )
        reports = (
            self.db.query(SessionReportRecord)
            .filter(SessionReportRecord.session_id.in_(session_ids))
            .all()
        )
        report_map = {row.session_id: row for row in reports}

        messages_by_session: dict[str, list[MessageRecord]] = {}
        for message in messages:
            messages_by_session.setdefault(message.session_id, []).append(message)

        user_lengths: list[int] = []
        user_response_times: list[float] = []
        idle_gaps: list[float] = []
        message_counts: list[int] = []
        durations: list[float] = []
        focus_scores: list[float] = []
        retry_counts: list[int] = []
        question_marks = 0
        user_message_count = 0

        for session in sessions:
            session_messages = messages_by_session.get(session.session_id, [])
            session_user_messages = [msg for msg in session_messages if msg.role == "user"]
            message_counts.append(len(session_user_messages))
            retry_counts.append(session.retry_count or 0)

            if session.started_at and session.ended_at:
                durations.append((session.ended_at - session.started_at).total_seconds())

            report = report_map.get(session.session_id)
            if report and report.focus_score is not None:
                focus_scores.append(report.focus_score)
            elif session.average_focus_score is not None:
                focus_scores.append(session.average_focus_score)

            previous_timestamp = None
            previous_role = None

            for message in session_messages:
                if message.role != "user":
                    previous_timestamp = message.timestamp
                    previous_role = message.role
                    continue

                user_message_count += 1
                user_lengths.append(len(message.content))
                if "?" in message.content:
                    question_marks += 1

                if previous_timestamp and message.timestamp:
                    delta = (message.timestamp - previous_timestamp).total_seconds()
                    idle_gaps.append(delta)
                    if previous_role == "assistant":
                        user_response_times.append(delta)

                previous_timestamp = message.timestamp
                previous_role = message.role

        avg_message_length = round(mean(user_lengths), 2) if user_lengths else 0.0
        avg_response_time = round(mean(user_response_times), 2) if user_response_times else 0.0
        avg_idle_gap = round(mean(idle_gaps), 2) if idle_gaps else 0.0
        avg_messages_per_session = round(mean(message_counts), 2) if message_counts else 0.0
        avg_duration = round(mean(durations), 2) if durations else 0.0
        avg_focus = round(mean(focus_scores), 3) if focus_scores else None
        question_style = self._infer_question_style(avg_message_length, question_marks, user_message_count)
        personalized_threshold = self._infer_personalized_threshold(avg_focus)

        aggregate = {
            "avg_message_length": avg_message_length,
            "avg_response_time_seconds": avg_response_time,
            "avg_idle_gap_seconds": avg_idle_gap,
            "avg_messages_per_session": avg_messages_per_session,
            "avg_session_duration_seconds": avg_duration,
            "avg_focus_score": avg_focus,
            "question_style": question_style,
            "personalized_threshold": personalized_threshold,
        }
        metrics = {
            "message_length": self._metric_summary(user_lengths),
            "response_time_seconds": self._metric_summary(user_response_times),
            "idle_time_seconds": self._metric_summary(idle_gaps),
            "retry_count": self._metric_summary(retry_counts),
        }

        return {
            "sample_session_count": len(sessions),
            "aggregate": aggregate,
            "metrics": metrics,
        }

    def _metric_summary(self, values: list[float] | list[int]) -> dict[str, float]:
        if not values:
            return {"mean": 0.0, "stddev": 1.0, "min": 0.0, "max": 0.0}
        clean_values = [float(value) for value in values]
        stddev = pstdev(clean_values) if len(clean_values) > 1 else 0.0
        return {
            "mean": round(mean(clean_values), 3),
            "stddev": round(max(stddev, 1.0), 3),
            "min": round(min(clean_values), 3),
            "max": round(max(clean_values), 3),
        }

    def _sync_profile_threshold(self, user_id: str, baseline_threshold: float, sample_count: int) -> None:
        if sample_count < 3:
            return

        profile = (
            self.db.query(UserProfileRecord)
            .filter(UserProfileRecord.user_id == user_id)
            .first()
        )
        if profile is None:
            return

        if 0.73 <= (profile.adaptive_threshold or 0.75) <= 0.77:
            profile.adaptive_threshold = baseline_threshold

    def _infer_question_style(self, avg_length: float, question_marks: int, total_messages: int) -> str:
        question_ratio = (question_marks / total_messages) if total_messages else 0.0
        if avg_length < 30 and question_ratio >= 0.5:
            return "short_questions"
        if avg_length > 120:
            return "detailed_questions"
        return "balanced"

    def _infer_personalized_threshold(self, avg_focus: float | None) -> float:
        if avg_focus is None:
            return 0.75
        if avg_focus >= 0.8:
            return 0.7
        if avg_focus >= 0.65:
            return 0.73
        if avg_focus <= 0.4:
            return 0.82
        if avg_focus <= 0.5:
            return 0.78
        return 0.75

    def _serialize(self, baseline: UserBaselineRecord, metrics: dict[str, Any]) -> dict[str, Any]:
        return {
            "user_id": baseline.user_id,
            "sample_session_count": baseline.sample_session_count,
            "enough_data": baseline.sample_session_count >= 3,
            "avg_message_length": baseline.avg_message_length,
            "avg_response_time_seconds": baseline.avg_response_time_seconds,
            "avg_idle_gap_seconds": baseline.avg_idle_gap_seconds,
            "avg_messages_per_session": baseline.avg_messages_per_session,
            "avg_session_duration_seconds": baseline.avg_session_duration_seconds,
            "avg_focus_score": baseline.avg_focus_score,
            "question_style": baseline.question_style,
            "personalized_threshold": baseline.personalized_threshold,
            "metrics": metrics,
            "updated_at": baseline.updated_at.isoformat() if baseline.updated_at else None,
        }

    def _empty_baseline(self, user_id: str) -> dict[str, Any]:
        return {
            "user_id": user_id,
            "sample_session_count": 0,
            "enough_data": False,
            "avg_message_length": 0.0,
            "avg_response_time_seconds": 0.0,
            "avg_idle_gap_seconds": 0.0,
            "avg_messages_per_session": 0.0,
            "avg_session_duration_seconds": 0.0,
            "avg_focus_score": None,
            "question_style": None,
            "personalized_threshold": 0.75,
            "metrics": {},
            "updated_at": None,
        }
