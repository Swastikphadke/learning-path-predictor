"""Tests for the project milestones. Activate each milestone test as you implement it."""

import pytest

from app.core import COURSE_CATALOG, generate_student_sequences, milestone_1


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


@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")
