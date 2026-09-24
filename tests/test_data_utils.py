import numpy as np
import pytest

import data_utils


def test_numeric_field_with_blanks_is_parsed_and_imputed():
    """#8: one blank or missing value no longer turns a field into text."""
    metadata = [
        {"ratio": 0.2, "name": "a"},
        {"ratio": "", "name": "b"},
        {"name": "c"},
        {"ratio": 0.6, "name": "d"},
        {"ratio": None, "name": "e"},
    ]
    arrays, stats = data_utils.separate_metadata(metadata)

    ratio = arrays["ratio"]
    assert np.issubdtype(ratio.dtype, np.floating)
    assert not np.isnan(ratio).any()
    # Blanks are imputed with the mean, i.e. 0 after normalisation
    assert ratio[1] == ratio[2] == ratio[4] == 0
    assert ratio[0] == pytest.approx(-ratio[3])
    assert stats["keys"] == ["ratio"]
    assert stats["mean"]["ratio"] == pytest.approx(0.4)


def test_numeric_strings_are_numeric():
    arrays, stats = data_utils.separate_metadata([{"n": "3"}, {"n": 5}, {"n": ""}])
    assert stats["keys"] == ["n"]
    assert stats["mean"]["n"] == pytest.approx(4.0)


def test_text_field_stays_text():
    arrays, stats = data_utils.separate_metadata(
        [{"sym": "4-fold"}, {"sym": 8}, {"sym": ""}])
    assert arrays["sym"].dtype.kind == "U"
    assert list(arrays["sym"]) == ["4-fold", "8", ""]
    assert stats["keys"] == []


def test_constant_field_normalises_to_zero():
    arrays, stats = data_utils.separate_metadata([{"x": 1.0}, {"x": 1.0}])
    assert list(arrays["x"]) == [0.0, 0.0]
    assert stats["std"]["x"] == 0
    assert data_utils.normalize_value(7.0, 1.0, 0.0) == 0.0


def test_normalize_value_matches_training_normalisation():
    arrays, stats = data_utils.separate_metadata([{"x": 1.0}, {"x": 2.0}, {"x": 6.0}])
    mean, std = stats["mean"]["x"], stats["std"]["x"]
    assert data_utils.normalize_value(6.0, mean, std) == pytest.approx(arrays["x"][2])
