"""Tests for the Learning Path Predictor milestones."""

import pytest

from app.core import (
    COURSE_CATALOG,
    generate_student_sequences,
    milestone_1,
    milestone_2,
    milestone_2_finetune,
)


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


def test_milestone_2_basket_finetuning(tmp_path):
    initial_checkpoint = tmp_path / "initial.pt"
    output_checkpoint = tmp_path / "basket_finetuned.pt"

    milestone_2(
        n_students=40,
        seed=11,
        epochs=1,
        batch_size=8,
        checkpoint_path=str(initial_checkpoint),
    )
    result = milestone_2_finetune(
        n_students=40,
        seed=11,
        epochs=1,
        batch_size=8,
        checkpoint_path=str(initial_checkpoint),
        output_path=str(output_checkpoint),
        min_support=0.05,
    )

    assert result.model is not None
    assert 3 < result.tokenizer.vocab_size <= len(COURSE_CATALOG) + 3
    assert len(result.train_loss) == 1
    assert result.frequent_itemsets
    assert output_checkpoint.exists()



from app.language_model import CourseLanguageModel, CourseTokenizer
from app.recommendations import predict_next_courses


def test_milestone_3_returns_top_recommendations():
    tokenizer = CourseTokenizer(
        ["CS101", "CS102", "CS103", "CS104", "MA101"]
    )

    model = CourseLanguageModel(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
    )

    recommendations = predict_next_courses(
        model,
        tokenizer,
        ["CS101", "CS102"],
        top_n=3,
    )

    assert len(recommendations) == 3
    assert all("course_id" in item for item in recommendations)
    assert all("course_name" in item for item in recommendations)
    assert all("probability" in item for item in recommendations)


def test_milestone_3_excludes_completed_courses():
    tokenizer = CourseTokenizer(
        ["CS101", "CS102", "CS103", "CS104", "MA101"]
    )

    model = CourseLanguageModel(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
    )

    completed = ["CS101", "CS102"]

    recommendations = predict_next_courses(
        model,
        tokenizer,
        completed,
        top_n=5,
    )

    recommended_ids = {
        item["course_id"] for item in recommendations
    }

    assert not recommended_ids.intersection(completed)


def test_milestone_3_rejects_unknown_courses():
    tokenizer = CourseTokenizer(["CS101", "CS102"])

    model = CourseLanguageModel(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
    )

    with pytest.raises(ValueError, match="Unknown course IDs"):
        predict_next_courses(
            model,
            tokenizer,
            ["INVALID101"],
        )


def test_milestone_3_rejects_duplicate_courses():
    tokenizer = CourseTokenizer(["CS101", "CS102"])

    model = CourseLanguageModel(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
    )

    with pytest.raises(ValueError, match="duplicates"):
        predict_next_courses(
            model,
            tokenizer,
            ["CS101", "CS101"],
        )


def test_milestone_3_rejects_invalid_top_n():
    tokenizer = CourseTokenizer(["CS101", "CS102"])

    model = CourseLanguageModel(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
    )

    with pytest.raises(ValueError, match="top_n"):
        predict_next_courses(
            model,
            tokenizer,
            ["CS101"],
            top_n=0,
        )



@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")
