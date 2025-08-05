import numpy as np
import pytest
from pynuTS.decomposition import NaiveSAX, sax_distance, analyze_sax_patterns, create_sax_vocabulary

def test_sax_basic_transform():
    ts = np.array([1, 2, 3, 4, 5, 6])
    sax = NaiveSAX(windows=2)
    result = sax.fit_transform(ts)
    assert isinstance(result, str)
    assert len(result) > 0

def test_sax_empty_input():
    sax = NaiveSAX()
    with pytest.raises(ValueError):
        sax.transform([])

def test_sax_invalid_input_type():
    sax = NaiveSAX()
    with pytest.raises(TypeError):
        sax.transform("not a valid input")

def test_sax_distance_simple():
    d = sax_distance("ABC", "ABC")
    assert d == 0.0

def test_sax_distance_diff():
    d = sax_distance("ABC", "ACB")
    assert 0 < d < 1

def test_transform_batch():
    sax = NaiveSAX(windows=2)
    batch = np.array([[1, 2, 3, 4], [4, 3, 2, 1]])
    result = sax.transform_batch(batch)
    assert isinstance(result, list)
    assert all(isinstance(s, str) for s in result)

def test_analyze_sax_patterns():
    patterns = ["ABC", "ABC", "ABD", "ACD", ""]
    result = analyze_sax_patterns(patterns)
    assert 'total_strings' in result
    assert result['total_strings'] == 4
    assert 'most_common_patterns' in result

def test_create_sax_vocabulary():
    patterns = ["ABC", "ABC", "ABD", "XYZ"]
    vocab = create_sax_vocabulary(patterns, min_frequency=2)
    assert "ABC" in vocab
    assert "ABD" not in vocab
