from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.state.feature_classifier import (
    ANSWER_COMMITMENT_DATASET,
    HELP_SEEKING_DATASET,
    FeatureIntentClassifier,
    LabeledExample,
)
from backend.state.semantic_features import SemanticFeatureProvider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and save a refreshed feature classifier artifact from labeled export data."
    )
    parser.add_argument(
        "--input",
        default="data/feature_training_export.jsonl",
        help="Path to labeled JSONL export.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output model artifact path. Defaults to config path.",
    )
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="Do not include built-in seed examples during refresh.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.input))
    help_dataset, commitment_dataset = build_datasets(rows=rows, include_seed=not args.no_seed)

    semantic_provider = SemanticFeatureProvider.from_settings()
    classifier = FeatureIntentClassifier.from_datasets(
        semantic_provider=semantic_provider,
        help_dataset=help_dataset,
        commitment_dataset=commitment_dataset,
    )
    artifact_path = classifier.save_artifact(
        artifact_path=args.output,
        metadata={
            "source_file": args.input,
            "include_seed": not args.no_seed,
            "help_examples": len(help_dataset),
            "answer_commitment_examples": len(commitment_dataset),
        },
    )
    print(f"Saved classifier artifact to {artifact_path}")


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_datasets(
    rows: list[dict],
    include_seed: bool,
) -> tuple[tuple[LabeledExample, ...], tuple[LabeledExample, ...]]:
    help_examples = list(HELP_SEEKING_DATASET if include_seed else ())
    commitment_examples = list(ANSWER_COMMITMENT_DATASET if include_seed else ())

    for row in rows:
        content = str(row.get("content", "")).strip()
        if not content:
            continue

        help_label = normalize_label(row.get("label_help_seeking"))
        if help_label is not None:
            help_examples.append(LabeledExample(text=content, label=help_label))

        commitment_label = normalize_label(row.get("label_answer_commitment"))
        if commitment_label is not None:
            commitment_examples.append(LabeledExample(text=content, label=commitment_label))

    if not has_both_classes(help_examples):
        raise ValueError("Help-seeking dataset needs at least one positive and one negative label.")
    if not has_both_classes(commitment_examples):
        raise ValueError("Answer-commitment dataset needs at least one positive and one negative label.")

    return tuple(help_examples), tuple(commitment_examples)


def normalize_label(value: object) -> int | None:
    if value in {None, ""}:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return 1 if float(value) >= 0.5 else 0
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return 1
    if text in {"0", "false", "no", "n"}:
        return 0
    return None


def has_both_classes(examples: list[LabeledExample]) -> bool:
    labels = {example.label for example in examples}
    return labels == {0, 1}


if __name__ == "__main__":
    main()
