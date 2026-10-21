"""LLM-as-judge score parser (pure function, no model call)."""
from gitcontext.judge import _parse_scores


def test_parses_clean_json():
    assert _parse_scores('{"correctness": 4, "faithfulness": 5}') == {
        "correctness": 4, "faithfulness": 5}


def test_clips_out_of_range():
    assert _parse_scores('{"correctness": 9, "faithfulness": 0}') == {
        "correctness": 5, "faithfulness": 1}


def test_json_embedded_in_prose():
    text = 'Here is my rating: {"correctness": 3, "faithfulness": 4}. Done.'
    assert _parse_scores(text) == {"correctness": 3, "faithfulness": 4}


def test_fallback_to_first_two_digits():
    assert _parse_scores("correctness: 3 and faithfulness: 2") == {
        "correctness": 3, "faithfulness": 2}


def test_returns_none_when_unparseable():
    assert _parse_scores("no scores at all") is None
