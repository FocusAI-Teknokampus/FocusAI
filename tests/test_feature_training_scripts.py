import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace

from scripts.export_feature_training_data import align_messages_to_snapshots


class FeatureTrainingScriptsTests(unittest.TestCase):
    def test_align_messages_to_snapshots_matches_in_order(self) -> None:
        base_time = datetime(2026, 3, 26, 12, 0, 0)
        messages = [
            SimpleNamespace(timestamp=base_time),
            SimpleNamespace(timestamp=base_time + timedelta(seconds=30)),
        ]
        snapshots = [
            SimpleNamespace(created_at=base_time + timedelta(seconds=5), state_after="stuck"),
            SimpleNamespace(created_at=base_time + timedelta(seconds=35), state_after="focused"),
        ]

        aligned = align_messages_to_snapshots(messages, snapshots)

        self.assertEqual(len(aligned), 2)
        self.assertEqual(aligned[0].state_after, "stuck")
        self.assertEqual(aligned[1].state_after, "focused")


if __name__ == "__main__":
    unittest.main()
