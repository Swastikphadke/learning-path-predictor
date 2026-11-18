
"""Held-out evaluation for next-course recommendations."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from app.language_model import (
    train_course_language_model,
    continue_pretraining_on_baskets,
    load_course_language_model,
)
from app.recommendations import predict_next_courses


@dataclass
class RecommendationMetrics:
    model_name: str
    top1_accuracy: float
    hit_at_5: float
    mrr: float
    evaluated_examples: int
    skipped_examples: int


@dataclass
class RecommendationComparison:
    original: RecommendationMetrics
    basket_finetuned: RecommendationMetrics
    train_students: int
    test_students: int


def _evaluate_model(
    checkpoint_path: str | Path,
    test_sequences: Sequence[Sequence[str]],
    model_name: str,
) -> RecommendationMetrics:
    model, tokenizer, _ = load_course_language_model(checkpoint_path)

    top1_hits = 0
    hit_at_5 = 0
    reciprocal_rank_sum = 0.0
    evaluated = 0
    skipped = 0

    for sequence in test_sequences:
        # Each prefix predicts the next course in the original sequence.
        for index in range(1, len(sequence)):
            history = sequence[:index]
            actual_next = sequence[index]

            # Skip targets absent from the training vocabulary.
            if actual_next not in tokenizer.token_to_id:
                skipped += 1
                continue

            recommendations = predict_next_courses(
                model,
                tokenizer,
                history,
                top_n=5,
            )

            ranked_ids = [
                item["course_id"] for item in recommendations
            ]

            evaluated += 1

            if ranked_ids and ranked_ids[0] == actual_next:
                top1_hits += 1

            if actual_next in ranked_ids:
                hit_at_5 += 1
                rank = ranked_ids.index(actual_next) + 1
                reciprocal_rank_sum += 1.0 / rank

    denominator = max(evaluated, 1)

    return RecommendationMetrics(
        model_name=model_name,
        top1_accuracy=top1_hits / denominator,
        hit_at_5=hit_at_5 / denominator,
        mrr=reciprocal_rank_sum / denominator,
        evaluated_examples=evaluated,
        skipped_examples=skipped,
    )


def evaluate_recommendation_models(
    sequences: Sequence[Sequence[str]],
    *,
    seed: int = 42,
    test_fraction: float = 0.20,
    pretrain_epochs: int = 8,
    finetune_epochs: int = 3,
    batch_size: int = 32,
    output_dir: str | Path = "artifacts/evaluation",
) -> RecommendationComparison:
    """Train evaluation-only models on a split and compare held-out predictions.

    This avoids evaluating the existing full-data checkpoints on histories
    they have already seen during training.
    """
    if len(sequences) < 5:
        raise ValueError("At least 5 student histories are required")

    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")

    rows = [list(sequence) for sequence in sequences]

    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)

    split_index = int(len(shuffled) * (1 - test_fraction))
    split_index = min(max(split_index, 1), len(shuffled) - 1)

    train_sequences = shuffled[:split_index]
    test_sequences = shuffled[split_index:]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_path = output_dir / "evaluation_original.pt"
    basket_path = output_dir / "evaluation_basket.pt"

    # Train an original model using only the training partition.
    train_course_language_model(
        train_sequences,
        epochs=pretrain_epochs,
        batch_size=batch_size,
        seed=seed,
        checkpoint_path=original_path,
    )

    # Continue pretraining a separate copy on shuffled baskets.
    continue_pretraining_on_baskets(
        train_sequences,
        checkpoint_path=original_path,
        output_path=basket_path,
        epochs=finetune_epochs,
        batch_size=batch_size,
        seed=seed,
    )

    original_metrics = _evaluate_model(
        original_path,
        test_sequences,
        "Original Transformer",
    )

    basket_metrics = _evaluate_model(
        basket_path,
        test_sequences,
        "Basket-Fine-Tuned Transformer",
    )

    return RecommendationComparison(
        original=original_metrics,
        basket_finetuned=basket_metrics,
        train_students=len(train_sequences),
        test_students=len(test_sequences),
    )
