# unit test suite for the textual description of a symbolic encoding

import numpy as np
import pandas as pd
import pytest

from pynuTS.report import describe, segment_summary
from pynuTS.sax import SAX


def segment_lines(text):
    """The rows of the Segments block, and nothing else.

    Filtering on ' mean ' alone would also catch the 'Raw values' header, which
    reports a mean of its own.
    """
    if "Segments:" not in text:
        return []
    block = text.split("Segments:", 1)[1]
    return [line for line in block.splitlines()
            if line.strip() and line.startswith("  ") and line.strip()[0].isdigit()]


@pytest.fixture
def series():
    rng = np.random.default_rng(7)
    hours = np.arange(30 * 24)
    values = (50 + 15 * np.sin(2 * np.pi * hours / 24)
              + 8 * np.sin(2 * np.pi * hours / (24 * 7))
              + rng.normal(0, 2, hours.size))
    values[14 * 24:16 * 24] *= 0.25          # a two day outage
    index = pd.date_range("2024-01-01", periods=hours.size, freq="h")
    return pd.Series(values, index=index, name="consumo_kwh")


@pytest.fixture
def sax():
    return SAX(alphabet=5, windows=24).fit()


class TestSegmentSummary:
    def test_one_row_per_symbol(self, series, sax):
        summary = segment_summary(series, sax)
        assert len(summary) == len(sax.transform(series))
        assert len(summary) == 30

    def test_columns(self, series, sax):
        assert list(segment_summary(series, sax).columns) == \
               ["segment", "start", "end", "n_points", "symbol", "mean", "zscore"]

    def test_symbols_match_the_encoding(self, series, sax):
        summary = segment_summary(series, sax)
        assert "".join(summary.symbol) == sax.transform(series)

    def test_points_add_up_to_the_series_length(self, series, sax):
        assert segment_summary(series, sax).n_points.sum() == len(series)

    def test_the_outage_is_the_lowest_zscore(self, series, sax):
        summary = segment_summary(series, sax)
        assert set(summary.zscore.nsmallest(2).index) == {14, 15}

    def test_index_is_taken_from_the_series(self, series, sax):
        summary = segment_summary(series, sax)
        assert summary.start.iloc[0] == "2024-01-01"

    def test_explicit_index_overrides(self, sax):
        values = np.arange(48.0)
        summary = segment_summary(values, sax, index=pd.date_range("2020-06-01", periods=48, freq="h"))
        assert summary.start.iloc[0] == "2020-06-01"

    def test_works_without_any_index(self, sax):
        summary = segment_summary(np.arange(48.0), sax)
        assert summary.start.iloc[0] == "0" and summary.end.iloc[0] == "23"

    def test_short_last_segment(self, sax):
        summary = segment_summary(np.arange(50.0), sax)
        assert list(summary.n_points) == [24, 24, 2]

    def test_unfitted_sax_raises(self, series):
        with pytest.raises(ValueError):
            segment_summary(series, SAX())

    def test_two_dimensional_input_raises(self, sax):
        with pytest.raises(TypeError):
            segment_summary(np.zeros((3, 24)), sax)


class TestDescribe:
    def test_mentions_the_name_and_the_size(self, series, sax):
        text = describe(series, sax, name="consumo_kwh")
        assert "consumo_kwh" in text
        assert "720 points" in text

    def test_contains_the_encoding(self, series, sax):
        text = describe(series, sax)
        assert sax.transform(series) in text

    def test_legend_covers_every_symbol(self, series, sax):
        text = describe(series, sax)
        legend = text.split("Encoding:")[0]
        for symbol in sax._symbols:
            assert "\n  %s: " % symbol in legend

    def test_legend_reports_the_actual_breakpoints(self, series, sax):
        text = describe(series, sax)
        assert "%+.2f" % sax.breakpoints_[0] in text
        assert "%+.2f" % sax.breakpoints_[-1] in text

    def test_anomalies_are_flagged(self, series, sax):
        text = describe(series, sax, anomalies=[14, 15])
        flagged = [line for line in text.splitlines() if "ANOMALY" in line]
        assert len(flagged) == 2
        assert "Flagged segments: 14, 15." in text

    def test_no_anomalies_means_no_flags(self, series, sax):
        assert "ANOMALY" not in describe(series, sax)

    def test_out_of_range_anomaly_raises(self, series, sax):
        with pytest.raises(ValueError, match="out of range"):
            describe(series, sax, anomalies=[999])

    def test_long_series_are_truncated_but_keep_the_anomalies(self, sax):
        rng = np.random.default_rng(1)
        values = rng.normal(size=200 * 24)
        values[7 * 24:8 * 24] += 40           # one unmistakable spike
        text = describe(values, sax, anomalies=[7], max_segments=10)

        assert "ANOMALY" in text
        assert "further segments omitted" in text
        assert len(segment_lines(text)) <= 10

    def test_short_series_are_listed_in_full(self, series, sax):
        text = describe(series, sax, max_segments=40)
        assert len(segment_lines(text)) == 30
        assert "omitted" not in text

    def test_missing_values_are_reported(self, sax):
        values = np.arange(48.0)
        values[3] = np.nan
        assert "Missing values: 1." in describe(values, sax)

    def test_datetime_segments_collapse_to_a_single_day(self, series, sax):
        text = describe(series, sax)
        assert "  2024-01-01  " in text
        assert "2024-01-01..2024-01-01" not in text

    def test_znormalize_off_changes_the_legend(self, series):
        raw = SAX(alphabet=3, windows=24, znormalize=False).fit()
        text = describe(series, raw)
        assert "value <" in text
        assert "standardized first" not in text

    def test_empty_series(self, sax):
        text = describe(np.empty(0), sax)
        assert "0 points" in text

    def test_output_is_plain_text(self, series, sax):
        text = describe(series, sax, anomalies=[14])
        assert isinstance(text, str)
        assert text == text.strip()
