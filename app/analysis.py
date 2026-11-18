
"""Milestone 5: Analyze learned course patterns and representations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from thilimarketbasket.mine import frequent_itemsets

from app.core import COURSE_CATALOG
from app.language_model import load_course_language_model


@dataclass
class CourseAnalysis:
    frequent_itemsets: list[dict]
    similar_courses: list[dict]
    checkpoint_path: str


@dataclass
class AttentionAnalysis:
    course_labels: list[str]
    attention_by_head: list[list[list[float]]]


def analyze_course_patterns(
    sequences: Sequence[Sequence[str]],
    checkpoint_path: str | Path = "artifacts/course_lm.pt",
    *,
    min_support: float = 0.05,
    max_itemset_size: int = 3,
    top_k: int = 20,
) -> CourseAnalysis:
    """Analyze frequent itemsets and learned course embedding similarities."""

    if not sequences:
        raise ValueError("At least one student sequence is required")

    if not 0 < min_support <= 1:
        raise ValueError("min_support must be between 0 and 1")

    if max_itemset_size < 1 or top_k < 1:
        raise ValueError("max_itemset_size and top_k must be positive")

    model, tokenizer, _ = load_course_language_model(checkpoint_path)

    # Validate input courses against the catalog and model vocabulary.
    for sequence in sequences:
        unknown = [course for course in sequence if course not in COURSE_CATALOG]
        if unknown:
            raise ValueError(f"Unknown course IDs: {unknown}")
        tokenizer.encode(sequence)

    # 1. Exact frequent-itemset analysis.
    itemsets = frequent_itemsets(
        [list(set(sequence)) for sequence in sequences],
        min_support=min_support,
        max_size=max_itemset_size,
    )

    ranked_itemsets = sorted(
        itemsets.items(),
        key=lambda item: (item[1], len(item[0])),
        reverse=True,
    )[:top_k]

    itemset_results = [
        {
            "courses": sorted(itemset),
            "course_names": [
                COURSE_CATALOG[course] for course in sorted(itemset)
            ],
            "support": round(float(support), 4),
        }
        for itemset, support in ranked_itemsets
    ]

    # 2. Learned embedding similarity.
    course_ids = [
        course for course in COURSE_CATALOG
        if course in tokenizer.token_to_id
    ]

    course_token_ids = [
        tokenizer.token_to_id[course] for course in course_ids
    ]

    embeddings = model.token_embedding.weight.detach()[course_token_ids]
    normalized = F.normalize(embeddings, p=2, dim=1)
    similarities = normalized @ normalized.T

    similar_courses = []

    for i in range(len(course_ids)):
        for j in range(i + 1, len(course_ids)):
            similar_courses.append({
                "course_a": course_ids[i],
                "course_a_name": COURSE_CATALOG[course_ids[i]],
                "course_b": course_ids[j],
                "course_b_name": COURSE_CATALOG[course_ids[j]],
                "similarity": round(
                    similarities[i, j].item(), 4
                ),
            })

    similar_courses.sort(
        key=lambda item: item["similarity"],
        reverse=True,
    )

    return CourseAnalysis(
        frequent_itemsets=itemset_results,
        similar_courses=similar_courses[:top_k],
        checkpoint_path=str(checkpoint_path),
    )


def analyze_attention(
    completed_courses: Sequence[str],
    checkpoint_path: str | Path = "artifacts/course_lm.pt",
) -> AttentionAnalysis:
    """Extract first-layer, per-head causal self-attention weights."""

    if not completed_courses:
        raise ValueError("At least one course is required")

    model, tokenizer, _ = load_course_language_model(checkpoint_path)

    unknown = [
        course for course in completed_courses
        if course not in COURSE_CATALOG
    ]
    if unknown:
        raise ValueError(f"Unknown course IDs: {unknown}")

    encoded = tokenizer.encode(completed_courses)
    encoded = encoded[-(model.max_length - 1):]

    input_ids = torch.tensor(
        [[tokenizer.bos_id] + encoded],
        dtype=torch.long,
    )

    model.eval()

    with torch.no_grad():
        batch_size, sequence_length = input_ids.shape

        positions = torch.arange(sequence_length).unsqueeze(0)

        hidden = (
            model.token_embedding(input_ids)
            * (model.token_embedding.embedding_dim ** 0.5)
        )
        hidden = hidden + model.position_embedding(positions)

        layer = model.transformer.layers[0]

        normalized = layer.norm1(hidden)

        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        padding_mask = input_ids.eq(model.pad_id)

        _, attention_weights = layer.self_attn(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )

    labels = ["BOS"] + tokenizer.decode(encoded)

    return AttentionAnalysis(
        course_labels=labels,
        attention_by_head=attention_weights[0].tolist(),
    )
