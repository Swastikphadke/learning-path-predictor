"""A compact causal Transformer language model for course-ID sequences."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"


class CourseTokenizer:
    """Maps stable course IDs to integer token IDs."""

    def __init__(self, course_ids: Sequence[str]):
        ordered = sorted(set(course_ids))
        self.token_to_id = {PAD_TOKEN: 0, BOS_TOKEN: 1, EOS_TOKEN: 2}
        self.token_to_id.update({course_id: i + 3 for i, course_id in enumerate(ordered)})
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS_TOKEN]

    def encode(self, sequence: Sequence[str]) -> list[int]:
        unknown = [course for course in sequence if course not in self.token_to_id]
        if unknown:
            raise ValueError(f"Unknown course IDs: {unknown}")
        return [self.token_to_id[course] for course in sequence]

    def decode(self, token_ids: Sequence[int]) -> list[str]:
        return [
            self.id_to_token[int(token_id)]
            for token_id in token_ids
            if self.id_to_token[int(token_id)] not in {PAD_TOKEN, BOS_TOKEN, EOS_TOKEN}
        ]

    def to_dict(self) -> dict[str, int]:
        return dict(self.token_to_id)

    @classmethod
    def from_dict(cls, mapping: dict[str, int]) -> "CourseTokenizer":
        tokenizer = cls.__new__(cls)
        tokenizer.token_to_id = dict(mapping)
        tokenizer.id_to_token = {index: token for token, index in mapping.items()}
        return tokenizer


class CourseSequenceDataset(Dataset):
    """Next-token examples: BOS + history predicts history + EOS."""

    def __init__(self, sequences: Sequence[Sequence[str]], tokenizer: CourseTokenizer):
        self.examples = []
        for sequence in sequences:
            encoded = tokenizer.encode(sequence)
            self.examples.append((
                [tokenizer.bos_id] + encoded,
                encoded + [tokenizer.eos_id],
            ))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        return self.examples[index]


def _collate_batch(batch, pad_id: int):
    max_length = max(len(inputs) for inputs, _ in batch)
    inputs = torch.full((len(batch), max_length), pad_id, dtype=torch.long)
    targets = torch.full((len(batch), max_length), -100, dtype=torch.long)
    for row, (input_ids, target_ids) in enumerate(batch):
        inputs[row, :len(input_ids)] = torch.tensor(input_ids, dtype=torch.long)
        targets[row, :len(target_ids)] = torch.tensor(target_ids, dtype=torch.long)
    return inputs, targets


class CourseLanguageModel(nn.Module):
    """Small decoder-style Transformer trained with a causal attention mask."""

    def __init__(
        self,
        vocab_size: int,
        pad_id: int = 0,
        d_model: int = 48,
        nhead: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 96,
        max_length: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.pad_id = pad_id
        self.max_length = max_length
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position_embedding = nn.Embedding(max_length, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.max_length:
            raise ValueError(f"Sequence length {sequence_length} exceeds max_length={self.max_length}")
        positions = torch.arange(sequence_length, device=input_ids.device).unsqueeze(0)
        hidden = self.token_embedding(input_ids) * math.sqrt(self.token_embedding.embedding_dim)
        hidden = hidden + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        padding_mask = input_ids.eq(self.pad_id)
        hidden = self.transformer(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        return self.output(self.final_norm(hidden))


@dataclass
class TrainingResult:
    model: CourseLanguageModel
    tokenizer: CourseTokenizer
    train_loss: list[float]
    validation_loss: float
    checkpoint_path: str


def _evaluate(model, loader, device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            count = targets.ne(-100).sum().item()
            total_loss += loss.item()
            total_tokens += count
    return total_loss / max(total_tokens, 1)


def train_course_language_model(
    sequences: Sequence[Sequence[str]],
    *,
    epochs: int = 8,
    batch_size: int = 32,
    learning_rate: float = 3e-3,
    seed: int = 42,
    checkpoint_path: str | Path = "artifacts/course_lm.pt",
    d_model: int = 48,
) -> TrainingResult:
    """Train and save a compact CPU-friendly next-course language model."""
    if not sequences:
        raise ValueError("At least one student sequence is required")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be at least 1")

    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = torch.device("cpu")

    all_courses = sorted({course for sequence in sequences for course in sequence})
    tokenizer = CourseTokenizer(all_courses)

    shuffled = list(sequences)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) >= 5:
        split_index = max(1, int(len(shuffled) * 0.9))
        train_sequences, validation_sequences = shuffled[:split_index], shuffled[split_index:]
    else:
        train_sequences, validation_sequences = shuffled, shuffled

    train_dataset = CourseSequenceDataset(train_sequences, tokenizer)
    validation_dataset = CourseSequenceDataset(validation_sequences, tokenizer)
    collate = lambda batch: _collate_batch(batch, tokenizer.pad_id)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    model = CourseLanguageModel(tokenizer.vocab_size, pad_id=tokenizer.pad_id, d_model=d_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    history = []

    for _ in range(epochs):
        model.train()
        total_loss = 0.0
        batches = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        history.append(total_loss / max(batches, 1))

    validation_loss = _evaluate(model, validation_loader, device)
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "tokenizer": tokenizer.to_dict(),
            "model_config": {
                "vocab_size": tokenizer.vocab_size,
                "pad_id": tokenizer.pad_id,
                "d_model": d_model,
                "nhead": 4,
                "num_layers": 1,
                "dim_feedforward": 96,
                "max_length": 32,
                "dropout": 0.1,
            },
            "train_loss": history,
            "validation_loss": validation_loss,
        },
        checkpoint,
    )

    return TrainingResult(
        model=model,
        tokenizer=tokenizer,
        train_loss=history,
        validation_loss=validation_loss,
        checkpoint_path=str(checkpoint),
    )


def load_course_language_model(checkpoint_path: str | Path):
    """Load a saved model and tokenizer from a Milestone 2 checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    tokenizer = CourseTokenizer.from_dict(checkpoint["tokenizer"])
    model = CourseLanguageModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer, checkpoint
