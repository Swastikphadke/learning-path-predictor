"""Tests for the Learning Path Predictor milestones."""

import pytest

from app.core import COURSE_CATALOG, generate_student_sequences, milestone_1, milestone_2


def test_milestone_1():
    sequences = milestone_1(n_students=100, seed=7)
    assert len(sequences) == 100
    assert all(isinstance(sequence, list) for sequence in sequences)
    assert all(4 <= len(sequence) <= 14 for sequence in sequences)
    assert all(course_id in COURSE_CATALOG for sequence in sequences for course_id in sequence)
    assert all(len(sequence) == len(set(sequence)) for sequence in sequences)


def test_milestone_1_is_reproducible():
    assert generate_student_sequences(20, seed=123) == generate_student_sequences(20, seed=123)
    assert generate_student_sequences(20, seed=123) != generate_student_sequences(20, seed=456)


def test_milestone_1_rejects_invalid_student_count():
    with pytest.raises(ValueError):
        milestone_1(n_students=0)


def test_milestone_2_trains_and_saves_checkpoint(tmp_path):
    result = milestone_2(
        n_students=40,
        seed=7,
        epochs=1,
        batch_size=8,
        checkpoint_path=str(tmp_path / "course_lm.pt"),
    )
    assert result.model is not None
    assert 3 < result.tokenizer.vocab_size <= len(COURSE_CATALOG) + 3
    assert len(result.train_loss) == 1
    assert result.train_loss[0] > 0
    assert result.validation_loss > 0
    assert (tmp_path / "course_lm.pt").exists()


@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")
