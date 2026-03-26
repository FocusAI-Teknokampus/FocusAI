from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional

from sqlalchemy.orm import Session

from backend.core.database import (
    InterventionEffectivenessRecord,
    InterventionPolicyRecord,
)


class InterventionPolicyService:
    """
    Kullanıcı bazında hangi müdahalelerin hangi state içinde işe yaradığını özetler.
    """

    def __init__(self, db: Session):
        self.db = db

    def record_feedback(
        self,
        user_id: str,
        intervention_type: str,
        triggered_state: str,
        was_successful: bool,
        feedback_type: str,
    ) -> dict[str, Any]:
        state_label = triggered_state or "unknown"
        record = (
            self.db.query(InterventionEffectivenessRecord)
            .filter(
                InterventionEffectivenessRecord.user_id == user_id,
                InterventionEffectivenessRecord.state_label == state_label,
                InterventionEffectivenessRecord.intervention_type == intervention_type,
            )
            .first()
        )

        if record is None:
            record = InterventionEffectivenessRecord(
                user_id=user_id,
                state_label=state_label,
                intervention_type=intervention_type,
            )
            self.db.add(record)

        record.total_count = (record.total_count or 0) + 1
        record.success_count = record.success_count or 0
        record.failure_count = record.failure_count or 0
        record.last_feedback_type = feedback_type
        record.last_outcome = was_successful
        record.last_used_at = datetime.utcnow()

        recent_outcomes = self._load_recent_outcomes(record.recent_outcomes_json)
        recent_outcomes.append(bool(was_successful))
        recent_outcomes = recent_outcomes[-5:]
        record.recent_outcomes_json = json.dumps(recent_outcomes)

        if was_successful:
            record.success_count += 1
        else:
            record.failure_count += 1

        self._sync_legacy_rollup(
            user_id=user_id,
            intervention_type=intervention_type,
            was_successful=was_successful,
            feedback_type=feedback_type,
        )

        self.db.flush()
        return self._serialize_effectiveness(record)

    def get_policy_summary(self, user_id: str) -> dict[str, Any]:
        rows = (
            self.db.query(InterventionEffectivenessRecord)
            .filter(InterventionEffectivenessRecord.user_id == user_id)
            .all()
        )

        aggregate: dict[str, dict[str, Any]] = {}
        for row in rows:
            item = aggregate.setdefault(
                row.intervention_type,
                {
                    "user_id": user_id,
                    "intervention_type": row.intervention_type,
                    "total_count": 0,
                    "success_count": 0,
                    "failure_count": 0,
                    "recent_outcomes": [],
                    "states": [],
                    "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                },
            )
            item["total_count"] += row.total_count
            item["success_count"] += row.success_count
            item["failure_count"] += row.failure_count
            item["states"].append(row.state_label)
            item["recent_outcomes"].extend(self._load_recent_outcomes(row.recent_outcomes_json))
            item["updated_at"] = row.updated_at.isoformat() if row.updated_at else item["updated_at"]

        items = []
        for item in aggregate.values():
            total = item["total_count"]
            recent = item["recent_outcomes"][-5:]
            success_rate = round(item["success_count"] / total, 3) if total else None
            recent_success_rate = (
                round(sum(1 for value in recent if value) / len(recent), 3) if recent else None
            )
            items.append(
                {
                    "user_id": item["user_id"],
                    "intervention_type": item["intervention_type"],
                    "total_count": total,
                    "success_count": item["success_count"],
                    "failure_count": item["failure_count"],
                    "success_rate": success_rate,
                    "recent_success_rate": recent_success_rate,
                    "states": sorted(set(item["states"])),
                    "updated_at": item["updated_at"],
                }
            )

        items.sort(
            key=lambda item: (
                item["recent_success_rate"] if item["recent_success_rate"] is not None else -1,
                item["success_rate"] if item["success_rate"] is not None else -1,
                item["total_count"],
            ),
            reverse=True,
        )
        best = items[0] if items else None
        return {
            "best_intervention_type": best["intervention_type"] if best else None,
            "items": items,
        }

    def get_state_policy_summary(self, user_id: str, state_label: str) -> dict[str, dict[str, Any]]:
        rows = (
            self.db.query(InterventionEffectivenessRecord)
            .filter(
                InterventionEffectivenessRecord.user_id == user_id,
                InterventionEffectivenessRecord.state_label == state_label,
            )
            .all()
        )
        return {
            row.intervention_type: self._serialize_effectiveness(row)
            for row in rows
        }

    def get_success_rate(
        self,
        user_id: str,
        intervention_type: str,
        triggered_state: Optional[str] = None,
    ) -> Optional[float]:
        query = self.db.query(InterventionEffectivenessRecord).filter(
            InterventionEffectivenessRecord.user_id == user_id,
            InterventionEffectivenessRecord.intervention_type == intervention_type,
        )
        if triggered_state:
            query = query.filter(InterventionEffectivenessRecord.state_label == triggered_state)

        rows = query.all()
        if not rows:
            return None

        total = sum(row.total_count for row in rows)
        success = sum(row.success_count for row in rows)
        if total == 0:
            return None
        return round(success / total, 3)

    def _sync_legacy_rollup(
        self,
        user_id: str,
        intervention_type: str,
        was_successful: bool,
        feedback_type: str,
    ) -> None:
        rollup = (
            self.db.query(InterventionPolicyRecord)
            .filter(
                InterventionPolicyRecord.user_id == user_id,
                InterventionPolicyRecord.intervention_type == intervention_type,
            )
            .first()
        )
        if rollup is None:
            rollup = InterventionPolicyRecord(
                user_id=user_id,
                intervention_type=intervention_type,
            )
            self.db.add(rollup)

        rollup.total_count = (rollup.total_count or 0) + 1
        rollup.success_count = rollup.success_count or 0
        rollup.failure_count = rollup.failure_count or 0
        rollup.last_feedback_type = feedback_type
        rollup.last_outcome = was_successful
        if was_successful:
            rollup.success_count += 1
        else:
            rollup.failure_count += 1

    def _load_recent_outcomes(self, raw: Optional[str]) -> list[bool]:
        if not raw:
            return []
        try:
            value = json.loads(raw)
            return [bool(item) for item in value if isinstance(item, bool)]
        except Exception:
            return []

    def _serialize_effectiveness(self, row: InterventionEffectivenessRecord) -> dict[str, Any]:
        total = row.total_count or 0
        recent = self._load_recent_outcomes(row.recent_outcomes_json)
        success_rate = round((row.success_count / total), 3) if total else None
        recent_success_rate = (
            round(sum(1 for value in recent if value) / len(recent), 3) if recent else None
        )
        return {
            "user_id": row.user_id,
            "state_label": row.state_label,
            "intervention_type": row.intervention_type,
            "total_count": total,
            "success_count": row.success_count,
            "failure_count": row.failure_count,
            "success_rate": success_rate,
            "recent_success_rate": recent_success_rate,
            "recent_window_count": len(recent),
            "last_feedback_type": row.last_feedback_type,
            "last_outcome": row.last_outcome,
            "last_used_at": row.last_used_at.isoformat() if row.last_used_at else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }
