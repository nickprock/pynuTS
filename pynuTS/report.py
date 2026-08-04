"""
Created on Tue Aug 04 2026

@project: pynuTS
@author: nicola procopio
@description: turn a symbolic encoding into text a language model can read

A language model cannot look at an array. Handing it 720 floats wastes context
and gives it nothing to reason with, while an embedding, however good, is not
something it can read at all.

A symbolic encoding is the one representation that is both compact and legible:
thirty letters instead of seven hundred numbers, and every letter has a meaning
that can be spelled out in one line. That is the property this module exploits.
It builds the description, and stops there: no API client, no key, no network.
What you ask the model is your business.

    from pynuTS.sax import SAX
    from pynuTS.report import describe

    sax = SAX(alphabet=5, windows=24).fit()
    text = describe(series, sax, index=series.index, anomalies=[14, 15])
    # ... then put `text` in whatever prompt you like
"""

import numpy as np
import pandas as pd

from .sax import paa, znorm

__all__ = ["describe", "segment_summary"]


def _format_label(index, position):
    if index is None:
        return str(position)
    label = index[position]
    if isinstance(label, (pd.Timestamp, np.datetime64)):
        stamp = pd.Timestamp(label)
        if stamp.hour or stamp.minute or stamp.second:
            return stamp.strftime("%Y-%m-%d %H:%M")
        return stamp.strftime("%Y-%m-%d")
    return str(label)


def _format_span(start, end):
    """Collapse a segment that begins and ends on the same day to just that day."""
    if start == end:
        return start
    if len(start) == 10 and end.startswith(start):
        return start
    return "%s..%s" % (start, end)


def _as_values(X):
    if isinstance(X, pd.Series):
        return np.asarray(X.values, dtype=float)
    return np.asarray(X, dtype=float)


def segment_summary(X, sax, index=None) -> pd.DataFrame:
    """
    One row per SAX segment, with the symbol and the statistics behind it.

    Useful on its own: flagging anomalies is then a one-liner, for instance
    ``summary[summary.zscore.abs() > 2]``.

    Parameters
    -----------------------
    X : 1D array-like or pandas Series
    sax : a fitted pynuTS.sax.SAX
    index : array-like or None
        default None. Labels for the time axis. Taken from X when it is a
        pandas Series and index is not given.

    Returns
    -----------------------
    summary : pandas DataFrame
        columns: segment, start, end, n_points, symbol, mean, zscore
    """
    if not hasattr(sax, "breakpoints_"):
        raise ValueError("the SAX instance is not fitted yet, call 'fit' first")

    if index is None and isinstance(X, pd.Series):
        index = X.index
    values = _as_values(X)
    if values.ndim != 1:
        raise TypeError("X must be a 1-D series")

    word = sax.transform(values)
    raw_means, lengths = paa(values, sax.windows)
    z_means, _ = paa(znorm(values), sax.windows)

    if lengths.shape[0] == 0:
        return pd.DataFrame({name: [] for name in
                             ("segment", "start", "end", "n_points", "symbol", "mean", "zscore")})

    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(int)
    ends = (starts + lengths - 1).astype(int)

    return pd.DataFrame({
        "segment": np.arange(len(word)),
        "start": [_format_label(index, s) for s in starts],
        "end": [_format_label(index, e) for e in ends],
        "n_points": lengths,
        "symbol": list(word),
        "mean": raw_means,
        "zscore": z_means,
    })


def _legend(sax):
    symbols = sax._symbols
    breakpoints = sax.breakpoints_
    unit = "z" if sax.znormalize else "value"
    lines = []
    for position, symbol in enumerate(symbols):
        if position == 0:
            lines.append("  %s: %s < %+.2f" % (symbol, unit, breakpoints[0]))
        elif position == len(symbols) - 1:
            lines.append("  %s: %s >= %+.2f" % (symbol, unit, breakpoints[-1]))
        else:
            lines.append("  %s: %+.2f <= %s < %+.2f"
                         % (symbol, breakpoints[position - 1], unit, breakpoints[position]))
    return lines


def describe(X, sax, index=None, anomalies=None, name=None, max_segments: int = 40) -> str:
    """
    A compact, self-explaining description of a series and its SAX encoding.

    The text spells out what each symbol stands for, so a model reading it does
    not have to be told the encoding separately.

    Parameters
    -----------------------
    X : 1D array-like or pandas Series
    sax : a fitted pynuTS.sax.SAX
    index : array-like or None
        default None. Labels for the time axis.
    anomalies : array-like of int or None
        default None. Indexes of the *segments* worth pointing out.
    name : str or None
        default None. What the series measures, used in the opening line.
    max_segments : int
        default 40. Above this many segments only the flagged ones and the most
        extreme ones are listed, and the text says how many were left out.

    Returns
    -----------------------
    description : str
    """
    summary = segment_summary(X, sax, index=index)
    values = _as_values(X)
    finite = values[~np.isnan(values)]
    anomalies = set(int(a) for a in (anomalies if anomalies is not None else []))

    unknown = anomalies - set(summary.segment)
    if unknown:
        raise ValueError("anomaly indexes out of range: %s" % sorted(unknown))

    label = 'Time series "%s"' % name if name else "Time series"
    lines = []
    if len(summary):
        span = " from %s to %s" % (summary.start.iloc[0], summary.end.iloc[-1])
    else:
        span = ""
    lines.append("%s: %d points%s." % (label, values.shape[0], span))
    if finite.size:
        lines.append("Raw values: min %.4g, max %.4g, mean %.4g, std %.4g."
                     % (finite.min(), finite.max(), finite.mean(), finite.std()))
    if finite.size < values.shape[0]:
        lines.append("Missing values: %d." % (values.shape[0] - finite.size))

    lines.append("")
    lines.append("SAX encoding: %d symbols, %d segments of up to %d points."
                 % (len(sax._symbols), len(summary), sax.windows))
    if sax.znormalize:
        lines.append("The series is standardized first, so the symbols describe its shape, "
                     "not its level: z is the number of standard deviations from the mean.")
    lines.extend(_legend(sax))
    lines.append("")
    lines.append("Encoding: %s" % "".join(summary.symbol))

    if len(summary):
        lines.append("")
        shown = summary
        omitted = 0
        if len(summary) > max_segments:
            interesting = summary.index.isin(sorted(anomalies))
            ranking = summary.zscore.abs().rank(ascending=False, method="first")
            keep = interesting | (ranking <= max(max_segments - len(anomalies), 0))
            shown = summary[keep]
            omitted = len(summary) - len(shown)

        spans = [_format_span(row.start, row.end) for row in shown.itertuples(index=False)]
        width = max(len(s) for s in spans)

        lines.append("Segments:")
        for row, when in zip(shown.itertuples(index=False), spans):
            flag = "   <- ANOMALY" if row.segment in anomalies else ""
            lines.append("  %4d  %-*s  %s  mean %10.4g  z %+.2f%s"
                         % (row.segment, width, when, row.symbol, row.mean, row.zscore, flag))
        if omitted:
            lines.append("  (%d further segments omitted, closest to the average)" % omitted)

    if anomalies:
        lines.append("")
        lines.append("Flagged segments: %s." % ", ".join(str(a) for a in sorted(anomalies)))

    return "\n".join(lines)
