from __future__ import annotations


class FeatureDeviationScorer:
    """
    Ham feature'lari kullanici baseline'ina gore normalize eder.
    """

    def build(self, features, baseline_profile: dict | None) -> dict:
        baseline = baseline_profile or {}
        metrics = baseline.get("metrics", {})
        enough_data = bool(baseline.get("enough_data"))

        return {
            "response_time_seconds": self._metric_deviation(
                value=features.response_time_seconds,
                metric=metrics.get("response_time_seconds"),
                risk_direction="high",
                enough_data=enough_data,
            ),
            "idle_time_seconds": self._metric_deviation(
                value=features.idle_time_seconds,
                metric=metrics.get("idle_time_seconds"),
                risk_direction="high",
                enough_data=enough_data,
            ),
            "message_length": self._metric_deviation(
                value=float(features.message_length),
                metric=metrics.get("message_length"),
                risk_direction="low",
                enough_data=enough_data,
            ),
            "retry_count": self._metric_deviation(
                value=float(features.retry_count),
                metric=metrics.get("retry_count"),
                risk_direction="high",
                enough_data=enough_data,
            ),
            "help_seeking_score": self._metric_deviation(
                value=float(getattr(features, "help_seeking_score", 0.0) or 0.0),
                metric=metrics.get("help_seeking_score"),
                risk_direction="high",
                enough_data=enough_data,
            ),
            "answer_commitment_score": self._metric_deviation(
                value=float(getattr(features, "answer_commitment_score", 0.0) or 0.0),
                metric=metrics.get("answer_commitment_score"),
                risk_direction="low",
                enough_data=enough_data,
            ),
            "baseline_ready": enough_data,
        }

    def _metric_deviation(
        self,
        value: float,
        metric: dict | None,
        risk_direction: str,
        enough_data: bool,
    ) -> dict:
        mean = float((metric or {}).get("mean", 0.0))
        stddev = max(float((metric or {}).get("stddev", 1.0)), 1.0)
        delta = value - mean
        zscore = delta / stddev if stddev else 0.0
        ratio = (delta / mean) if mean else 0.0

        if not enough_data:
            severity = 0.0
        elif risk_direction == "high":
            severity = max(0.0, min(1.0, max(zscore / 2.5, ratio)))
        else:
            severity = max(0.0, min(1.0, max((-zscore) / 2.5, -ratio)))

        return {
            "value": round(value, 3),
            "baseline_mean": round(mean, 3),
            "baseline_stddev": round(stddev, 3),
            "delta": round(delta, 3),
            "ratio": round(ratio, 3),
            "zscore": round(zscore, 3),
            "severity": round(severity, 3),
        }
