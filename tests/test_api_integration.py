from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
import shutil
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import text

from backend.agents import graph
from backend.agents.session_agent import _active_sessions
from backend.api.main import create_app
from backend.core.config import settings
from backend.core import database as database_module
from backend.core.database import SessionLocal, configure_database
from backend.core.database import UserProfileRecord


class ApiIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_database_url = settings.database_url
        self.original_data_dir = settings.data_dir
        self.original_faiss_index_path = settings.faiss_index_path

        workspace_temp_root = Path.cwd() / ".tmp_testdata"
        workspace_temp_root.mkdir(exist_ok=True)
        self.temp_dir_path = Path(tempfile.mkdtemp(dir=workspace_temp_root))
        base_dir = self.temp_dir_path

        settings.database_url = "sqlite://"
        settings.data_dir = str(base_dir)
        settings.faiss_index_path = str(base_dir / "faiss")
        configure_database(settings.database_url, force=True)

        self.generate_response_patcher = patch.object(
            graph._mentor_agent,
            "generate_response",
            side_effect=self._fake_generate_response,
        )
        self.enrich_intervention_patcher = patch.object(
            graph._mentor_agent,
            "enrich_intervention",
            side_effect=lambda intervention, **_: intervention,
        )
        self.generate_response_patcher.start()
        self.enrich_intervention_patcher.start()

        self.client_context = TestClient(create_app())
        self.client = self.client_context.__enter__()

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)
        self.generate_response_patcher.stop()
        self.enrich_intervention_patcher.stop()
        _active_sessions.clear()

        settings.database_url = self.original_database_url
        settings.data_dir = self.original_data_dir
        settings.faiss_index_path = self.original_faiss_index_path
        configure_database(settings.database_url, force=True)
        shutil.rmtree(self.temp_dir_path, ignore_errors=True)

    def test_startup_runs_schema_migrations(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

        self.assertIsNotNone(database_module.engine)
        with database_module.engine.begin() as connection:
            versions = connection.execute(
                text("SELECT version FROM schema_migrations ORDER BY version ASC")
            ).fetchall()
            tables = {
                row[0]
                for row in connection.execute(
                    text(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                ).fetchall()
            }

        self.assertEqual([row[0] for row in versions], ["001_initial_schema"])
        self.assertIn("sessions", tables)
        self.assertIn("messages", tables)
        self.assertIn("schema_migrations", tables)

    def test_session_chat_feedback_end_to_end_flow(self) -> None:
        start_response = self.client.post(
            "/session/start",
            json={
                "user_id": "integration_user",
                "topic": "programlama",
                "camera_enabled": False,
            },
        )
        self.assertEqual(start_response.status_code, 200)
        session_id = start_response.json()["session_id"]

        self._set_user_threshold("integration_user", threshold=0.65)

        base_time = datetime(2026, 3, 26, 10, 0, 0)
        payloads = [
            self._chat_payload(session_id, "integration_user", "api?", base_time),
            self._chat_payload(
                session_id,
                "integration_user",
                "api?",
                base_time + timedelta(seconds=10),
            ),
            self._chat_payload(
                session_id,
                "integration_user",
                "api?",
                base_time + timedelta(minutes=4, seconds=40),
            ),
        ]

        final_chat = None
        for payload in payloads:
            chat_response = self.client.post("/chat", json=payload)
            self.assertEqual(chat_response.status_code, 200)
            final_chat = chat_response.json()

        assert final_chat is not None
        self.assertEqual(final_chat["current_state"], "stuck")
        self.assertIsNotNone(final_chat["mentor_intervention"])

        intervention_type = final_chat["mentor_intervention"]["intervention_type"]
        follow_up_chat = self.client.post(
            "/chat",
            json=self._chat_payload(
                session_id,
                "integration_user",
                "Python API akisini daha net goruyorum; router, endpoint ve request dogrulamayi ayni ornek uzerinden yaziyorum.",
                base_time + timedelta(minutes=5, seconds=5),
            ),
        )
        self.assertEqual(follow_up_chat.status_code, 200)

        feedback_response = self.client.post(
            "/feedback",
            json={
                "user_id": "integration_user",
                "session_id": session_id,
                "feedback_type": "intervention_helpful",
                "target_type": "intervention",
                "intervention_type": intervention_type,
            },
        )
        self.assertEqual(feedback_response.status_code, 200)
        feedback_payload = feedback_response.json()
        self.assertEqual(feedback_payload["status"], "recorded")
        self.assertEqual(feedback_payload["intervention_type"], intervention_type)
        self.assertIsNotNone(feedback_payload["adaptive_threshold"])
        self.assertLess(feedback_payload["adaptive_threshold"], 0.76)
        self.assertIsNotNone(feedback_payload["behavior_change"])
        self.assertIn(
            feedback_payload["behavior_change"]["measurement_status"],
            {"improved", "unchanged", "worsened"},
        )
        self.assertGreater(
            feedback_payload["behavior_change"]["post_signal_count"],
            0,
        )

        end_response = self.client.post(
            "/session/end",
            json={
                "session_id": session_id,
                "user_id": "integration_user",
            },
        )
        self.assertEqual(end_response.status_code, 200)

        welcome_response = self.client.get("/welcome/integration_user")
        self.assertEqual(welcome_response.status_code, 200)
        welcome_payload = welcome_response.json()
        self.assertTrue(welcome_payload["has_history"])
        self.assertEqual(welcome_payload["last_session"]["session_id"], session_id)
        self.assertIsNotNone(welcome_payload["last_report"])
        self.assertEqual(
            welcome_payload["intervention_policy"]["best_intervention_type"],
            intervention_type,
        )
        self.assertIn("today_start_recommendation", welcome_payload)
        self.assertIn("mini_recall_question", welcome_payload)
        self.assertIsNotNone(welcome_payload["latest_feedback_impact"])
        self.assertIsNotNone(welcome_payload["operational_next_session_plan"])
        self.assertIn("first_prompt", welcome_payload["operational_next_session_plan"])
        self.assertIn("session_structure", welcome_payload["operational_next_session_plan"])

        history_response = self.client.get("/history/sessions/integration_user")
        self.assertEqual(history_response.status_code, 200)
        history_payload = history_response.json()
        self.assertEqual(len(history_payload), 1)
        self.assertEqual(history_payload[0]["session_id"], session_id)
        self.assertGreaterEqual(history_payload[0]["message_count"], 8)

        dashboard_response = self.client.get(f"/dashboard/{session_id}")
        self.assertEqual(dashboard_response.status_code, 200)
        dashboard_payload = dashboard_response.json()
        self.assertEqual(dashboard_payload["report"]["message_count"], 8)
        self.assertGreaterEqual(dashboard_payload["report"]["intervention_count"], 1)
        self.assertEqual(
            dashboard_payload["latest_intervention"]["intervention_type"],
            intervention_type,
        )
        self.assertIn("response_policy", dashboard_payload["latest_state_analysis"])
        self.assertIn("reasons", dashboard_payload["latest_state_analysis"])
        self.assertIsNotNone(dashboard_payload["latest_intervention"]["feedback_impact"])
        self.assertIn("mentor_tactic", dashboard_payload["report"]["next_session_plan"])
        self.assertIn("session_structure", dashboard_payload["report"]["next_session_plan"])

        trend_response = self.client.get("/analytics/focus-trend/integration_user?days=7")
        self.assertEqual(trend_response.status_code, 200)
        trend_payload = trend_response.json()
        self.assertEqual(trend_payload["total_sessions"], 1)
        self.assertEqual(len(trend_payload["points"]), 1)

    def _fake_generate_response(self, *args, **kwargs) -> str:
        intervention = kwargs.get("intervention")
        message = kwargs.get("message")

        if intervention is not None:
            return f"intervention:{intervention.intervention_type.value}"
        if message is not None:
            return f"echo:{message.content}"
        return "echo"

    def _chat_payload(
        self,
        session_id: str,
        user_id: str,
        content: str,
        timestamp: datetime,
    ) -> dict[str, str]:
        return {
            "session_id": session_id,
            "user_id": user_id,
            "content": content,
            "channel": "text",
            "timestamp": timestamp.isoformat(),
        }

    def _set_user_threshold(self, user_id: str, threshold: float) -> None:
        db = SessionLocal()
        try:
            row = (
                db.query(UserProfileRecord)
                .filter(UserProfileRecord.user_id == user_id)
                .first()
            )
            self.assertIsNotNone(row)
            row.adaptive_threshold = threshold
            db.commit()
        finally:
            db.close()


if __name__ == "__main__":
    unittest.main()
