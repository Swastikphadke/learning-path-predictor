"""Evaluate frequent-itemset recovery from a basket-fine-tuned course LM."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from thilimarketbasket.mine import frequent_itemsets

from app.language_model import load_course_language_model


@dataclass
class BasketRecoveryResult:
    generated_baskets: list[list[str]]
    reference_itemsets: dict
    generated_itemsets: dict
    precision: float
    recall: float
    f1: float
    recovered_itemsets: set[frozenset[str]]
    checkpoint_path: str


def _itemset_metrics(reference: set, predicted: set) -> tuple[float, float, float]:
    """Return set-level precision, recall and F1."""
    true_positives = len(reference & predicted)
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(reference) if reference else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def _generate_basket(
    model,
    tokenizer,
    target_length: int,
    *,
    temperature: float,
    top_k: int,
    generator: torch.Generator,
) -> list[str]:
    """Sample a basket without duplicate courses or special tokens."""
    course_tokens = [
        token_id
        for token_id, token in tokenizer.id_to_token.items()
        if token not in {"<PAD>", "<BOS>", "<EOS>"}
    ]
    target_length = min(target_length, len(course_tokens), model.max_length - 1)
    if target_length <= 0:
        return []

    chosen: list[int] = []
    model.eval()

    with torch.no_grad():
        for _ in range(target_length):
            input_ids = torch.tensor(
                [[tokenizer.bos_id] + chosen], dtype=torch.long
            )
            logits = model(input_ids)[0, -1].clone()

            # Only real course IDs are eligible; never emit PAD/BOS/EOS.
            allowed = torch.zeros_like(logits, dtype=torch.bool)
            allowed[course_tokens] = True
            if chosen:
                allowed[chosen] = False
            logits[~allowed] = float("-inf")

            logits = logits / max(temperature, 1e-5)
            if top_k > 0 and top_k < int(allowed.sum().item()):
                candidate_logits = logits.clone()
                threshold = torch.topk(candidate_logits, top_k).values[-1]
                logits[candidate_logits < threshold] = float("-inf")

            probabilities = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(
                probabilities, num_samples=1, generator=generator
            ).item()
            chosen.append(next_token)

    return tokenizer.decode(chosen)


def evaluate_basket_itemset_recovery(
    sequences: Sequence[Sequence[str]],
    checkpoint_path: str | Path = "artifacts/course_lm_basket.pt",
    *,
    n_baskets: int | None = None,
    min_support: float = 0.05,
    max_itemset_size: int = 3,
    temperature: float = 0.8,
    top_k: int = 5,
    seed: int = 42,
) -> BasketRecoveryResult:
    """Compare model-generated itemsets with exact itemsets in training histories.

    Uses the same seeded 90/10 split as continue_pretraining_on_baskets.
    The reference itemsets are mined from the training portion; generated
    baskets are sampled to match its basket-length distribution.
    """
    if not sequences:
        raise ValueError("At least one student sequence is required")
    if not 0 < min_support <= 1:
        raise ValueError("min_support must be between 0 and 1")
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")
    if top_k < 0:
        raise ValueError("top_k cannot be negative")

    model, tokenizer, checkpoint = load_course_language_model(checkpoint_path)
    if "basket_finetune" not in checkpoint:
        raise ValueError("Checkpoint has no basket_finetune metadata")

    rows = [list(row) for row in sequences]
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) >= 5:
        split_index = max(1, int(len(shuffled) * 0.9))
        train_sequences = shuffled[:split_index]
    else:
        train_sequences = shuffled

    # Validate that the evaluation histories use this checkpoint's vocabulary.
    for row in train_sequences:
        tokenizer.encode(row)

    reference_itemsets = frequent_itemsets(
        [list(set(row)) for row in train_sequences],
        min_support=min_support,
        max_size=max_itemset_size,
    )

    count = len(train_sequences) if n_baskets is None else n_baskets
    if count < 1:
        raise ValueError("n_baskets must be at least 1")

    rng = random.Random(seed)
    lengths = [max(1, len(row)) for row in train_sequences]
    torch_generator = torch.Generator(device="cpu")
    torch_generator.manual_seed(seed)

    generated_baskets = []
    for _ in range(count):
        target_length = rng.choice(lengths)
        basket = _generate_basket(
            model,
            tokenizer,
            target_length,
            temperature=temperature,
            top_k=top_k,
            generator=torch_generator,
        )
        generated_baskets.append(basket)

    generated_itemsets = frequent_itemsets(
        generated_baskets,
        min_support=min_support,
        max_size=max_itemset_size,
    )

    reference_set = set(reference_itemsets)
    generated_set = set(generated_itemsets)
    precision, recall, f1 = _itemset_metrics(reference_set, generated_set)

    return BasketRecoveryResult(
        generated_baskets=generated_baskets,
        reference_itemsets=reference_itemsets,
        generated_itemsets=generated_itemsets,
        precision=precision,
        recall=recall,
        f1=f1,
        recovered_itemsets=reference_set & generated_set,
        checkpoint_path=str(checkpoint_path),
    )
