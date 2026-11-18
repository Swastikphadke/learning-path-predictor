
import pytest

from app.analysis import analyze_course_patterns, analyze_attention


def test_analysis_rejects_empty_sequences():
    with pytest.raises(ValueError, match="At least one"):
        analyze_course_patterns([])


def test_attention_rejects_empty_history():
    with pytest.raises(ValueError, match="At least one"):
        analyze_attention([])


def test_analysis_rejects_unknown_course():
    with pytest.raises(ValueError, match="Unknown course IDs"):
        analyze_course_patterns(
            [["INVALID101"]]
        )


def test_attention_rejects_unknown_course():
    with pytest.raises(ValueError, match="Unknown course IDs"):
        analyze_attention(["INVALID101"])
