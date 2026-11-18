
"""Next-course prediction using the trained course language model."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from app.core import COURSE_CATALOG
from app.language_model import load_course_language_model


def predict_next_courses(
    model,
    tokenizer,
    completed_courses: Sequence[str],
    *,
    top_n: int = 5,
) -> list[dict]:
    """Predict the most likely next courses from a student's history."""

    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    completed = list(completed_courses)

    if len(completed) != len(set(completed)):
        raise ValueError("Completed courses must not contain duplicates")

    unknown_courses = [
        course for course in completed
        if course not in COURSE_CATALOG
    ]
    if unknown_courses:
        raise ValueError(f"Unknown course IDs: {unknown_courses}")

    # Validate that the checkpoint knows every supplied course.
    encoded = tokenizer.encode(completed)

    # Keep the most recent history if it exceeds the model context window.
    encoded = encoded[-(model.max_length - 1):]

    input_ids = torch.tensor(
        [[tokenizer.bos_id] + encoded],
        dtype=torch.long,
    )

    model.eval()

    with torch.no_grad():
        logits = model(input_ids)[0, -1]

    # Only recommend catalog courses present in the model vocabulary.
    eligible_ids = [
        token_id
        for token_id, token in tokenizer.id_to_token.items()
        if token in COURSE_CATALOG and token not in completed
    ]

    if not eligible_ids:
        return []

    # Exclude special tokens and completed courses.
    eligible_logits = logits[eligible_ids]
    probabilities = torch.softmax(eligible_logits, dim=-1)

    count = min(top_n, len(eligible_ids))
    top_probabilities, top_indices = torch.topk(probabilities, count)

    recommendations = []

    for probability, index in zip(top_probabilities, top_indices):
        token_id = eligible_ids[index.item()]
        course_id = tokenizer.id_to_token[token_id]

        recommendations.append({
            "course_id": course_id,
            "course_name": COURSE_CATALOG[course_id],
            "probability": round(probability.item(), 4),
        })

    return recommendations


def recommend_from_checkpoint(
    completed_courses: Sequence[str],
    checkpoint_path: str | Path = "artifacts/course_lm.pt",
    *,
    top_n: int = 5,
) -> list[dict]:
    """Load a trained checkpoint and predict the next courses."""

    model, tokenizer, _ = load_course_language_model(checkpoint_path)

    return predict_next_courses(
        model,
        tokenizer,
        completed_courses,
        top_n=top_n,
    )
