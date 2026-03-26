from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from sqlalchemy.orm import Session

from backend.core.database import BehaviorEventRecord, MessageRecord, SessionLocal, configure_database


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export user message + feature snapshot pairs for feature labeling."
    )
    parser.add_argument(
        "--output",
        default="data/feature_training_export.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--limit-sessions",
        type=int,
        default=0,
        help="Optional max number of sessions to scan. 0 means all.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_database()
    db = SessionLocal()
    try:
        rows = export_training_rows(db=db, limit_sessions=args.limit_sessions)
    finally:
        db.close()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Exported {len(rows)} rows to {output_path}")


def export_training_rows(db: Session, limit_sessions: int = 0) -> list[dict[str, Any]]:
    session_ids_query = db.query(MessageRecord.session_id).distinct().order_by(MessageRecord.session_id.asc())
    if limit_sessions > 0:
        session_ids_query = session_ids_query.limit(limit_sessions)
    session_ids = [row.session_id for row in session_ids_query.all()]
    if not session_ids:
        return []

    messages = (
        db.query(MessageRecord)
        .filter(MessageRecord.session_id.in_(session_ids), MessageRecord.role == "user")
        .order_by(MessageRecord.session_id.asc(), MessageRecord.timestamp.asc())
        .all()
    )
    snapshots = (
        db.query(BehaviorEventRecord)
        .filter(
            BehaviorEventRecord.session_id.in_(session_ids),
            BehaviorEventRecord.event_type == "state_snapshot",
        )
        .order_by(BehaviorEventRecord.session_id.asc(), BehaviorEventRecord.created_at.asc())
        .all()
    )

    snapshots_by_session: dict[str, list[BehaviorEventRecord]] = {}
    for snapshot in snapshots:
        snapshots_by_session.setdefault(snapshot.session_id, []).append(snapshot)

    rows: list[dict[str, Any]] = []
    messages_by_session: dict[str, list[MessageRecord]] = {}
    for message in messages:
        messages_by_session.setdefault(message.session_id, []).append(message)

    for session_id, session_messages in messages_by_session.items():
        session_snapshots = snapshots_by_session.get(session_id, [])
        matched_snapshots = align_messages_to_snapshots(session_messages, session_snapshots)
        for message, snapshot in zip(session_messages, matched_snapshots):
            metadata = parse_json(getattr(snapshot, "metadata_json", None)) if snapshot else {}
            feature_vector = metadata.get("feature_vector", {}) if isinstance(metadata, dict) else {}
            rows.append(
                {
                    "message_id": message.id,
                    "session_id": message.session_id,
                    "timestamp": message.timestamp.isoformat() if message.timestamp else None,
                    "content": message.content,
                    "detected_topic": message.detected_topic,
                    "user_state": message.user_state,
                    "snapshot_created_at": snapshot.created_at.isoformat() if snapshot and snapshot.created_at else None,
                    "snapshot_state_after": snapshot.state_after if snapshot else None,
                    "feature_vector": feature_vector,
                    "current_scores": {
                        "help_seeking_score": feature_vector.get("help_seeking_score"),
                        "help_seeking_semantic_score": feature_vector.get("help_seeking_semantic_score"),
                        "help_seeking_classifier_score": feature_vector.get("help_seeking_classifier_score"),
                        "answer_commitment_score": feature_vector.get("answer_commitment_score"),
                        "answer_commitment_semantic_score": feature_vector.get("answer_commitment_semantic_score"),
                        "answer_commitment_classifier_score": feature_vector.get("answer_commitment_classifier_score"),
                        "topic": feature_vector.get("topic"),
                        "topic_confidence": feature_vector.get("topic_confidence"),
                        "semantic_retry_score": feature_vector.get("semantic_retry_score"),
                    },
                    "label_help_seeking": None,
                    "label_answer_commitment": None,
                    "review_notes": "",
                }
            )

    return rows


def align_messages_to_snapshots(
    messages: list[MessageRecord],
    snapshots: list[BehaviorEventRecord],
) -> list[Optional[BehaviorEventRecord]]:
    aligned: list[Optional[BehaviorEventRecord]] = []
    snapshot_index = 0

    for message_index, message in enumerate(messages):
        next_message_time = (
            messages[message_index + 1].timestamp
            if message_index + 1 < len(messages)
            else None
        )
        matched: Optional[BehaviorEventRecord] = None

        while snapshot_index < len(snapshots):
            candidate = snapshots[snapshot_index]
            if candidate.created_at and message.timestamp and candidate.created_at < message.timestamp:
                snapshot_index += 1
                continue

            if (
                next_message_time is not None
                and candidate.created_at is not None
                and candidate.created_at > next_message_time
            ):
                break

            matched = candidate
            snapshot_index += 1
            break

        aligned.append(matched)

    return aligned


def parse_json(raw: Optional[str]) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


if __name__ == "__main__":
    main()
