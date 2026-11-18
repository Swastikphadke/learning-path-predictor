
import pytest

from app.recommendation_evaluation import evaluate_recommendation_models


def test_recommendation_evaluation_rejects_too_few_students():
    with pytest.raises(ValueError, match="At least 5"):
        evaluate_recommendation_models(
            [["CS101", "CS102"]] * 4
        )


def test_recommendation_evaluation_rejects_invalid_split():
    with pytest.raises(ValueError, match="test_fraction"):
        evaluate_recommendation_models(
            [["CS101", "CS102"]] * 10,
            test_fraction=1.0,
        )
