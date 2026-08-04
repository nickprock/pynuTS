"""SAX and Chronos-style tokenization, side by side on the same series.

Two symbolic representations of time series, twenty years apart, built on the
same move: replace real numbers with symbols from a finite alphabet. They were
designed for opposite purposes, and every design choice follows from that.

Run it with:

    python demos/symbolic_representations.py
"""

import numpy as np
import pandas as pd

from pynuTS.quantize import MeanScaleQuantizer
from pynuTS.report import describe, segment_summary
from pynuTS.sax import SAX, znorm


def hourly_demand(days=30, outage=(14, 16), seed=7):
    """Hourly demand with a daily and a weekly cycle, plus a two day outage."""
    rng = np.random.default_rng(seed)
    hours = np.arange(days * 24)
    values = (50
              + 15 * np.sin(2 * np.pi * hours / 24)
              + 8 * np.sin(2 * np.pi * hours / (24 * 7))
              + rng.normal(0, 2, hours.size))
    values[outage[0] * 24:outage[1] * 24] *= 0.25
    index = pd.date_range("2024-01-01", periods=hours.size, freq="h")
    return pd.Series(values, index=index, name="demand_kwh")


def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    series = hourly_demand()
    values = series.to_numpy()

    rule("The series")
    print("%d hourly points, from %s to %s"
          % (len(series), series.index[0].date(), series.index[-1].date()))
    print("min %.1f  max %.1f  mean %.1f  std %.1f"
          % (values.min(), values.max(), values.mean(), values.std()))

    # ------------------------------------------------------------------ SAX
    rule("SAX: built to be compared and indexed")
    sax = SAX(alphabet=5, windows=24).fit()
    word = sax.transform(series)
    print("one symbol per day, alphabet of 5:")
    print("  %s" % word)
    print("\n%d floats -> %d characters, a %.0fx reduction"
          % (len(values), len(word), len(values) / len(word)))

    print("\nThe breakpoints do not come from this series, they are the quantiles")
    print("of a standard normal, the same for everybody:")
    print("  %s" % np.round(sax.breakpoints_, 4))

    print("\nThat, plus the z-normalization, is what makes two series comparable.")
    doubled = series * 50 + 1000
    print("  same series, x50 and +1000 :  %s" % sax.transform(doubled))
    print("  identical                  :  %s" % (sax.transform(doubled) == word))

    print("\nAnd it is what MINDIST rests on. On the symbols alone it gives a")
    print("distance that never overstates the true one:")
    other = hourly_demand(seed=99, outage=(3, 4))
    lower_bound = sax.mindist(word, sax.transform(other), n=len(values))
    true_distance = np.linalg.norm(znorm(values) - znorm(other.to_numpy()))
    print("  MINDIST on 30 characters   :  %.3f" % lower_bound)
    print("  euclidean on 720 numbers   :  %.3f" % true_distance)
    print("  lower bound holds          :  %s" % (lower_bound <= true_distance))
    print("\n  A candidate whose MINDIST already exceeds the best distance so far")
    print("  can be dropped without opening it. That is the whole point.")

    # ------------------------------------------------- Chronos-style tokens
    rule("Chronos-style tokenization: built to be fed to a language model")
    quantizer = MeanScaleQuantizer(n_bins=4096).fit()
    tokens = quantizer.transform(series)
    print("one token per point, vocabulary of 4096:")
    print("  %s ..." % " ".join(str(t) for t in tokens[:16]))

    reconstructed = quantizer.inverse_transform(tokens)
    error = np.abs(reconstructed - values)
    print("\nNo temporal aggregation, so the length is unchanged: %d -> %d tokens"
          % (len(values), len(tokens)))
    print("but it is invertible, which SAX is not:")
    print("  max reconstruction error   :  %.4f  (%.3f%% of the range)"
          % (error.max(), 100 * error.max() / np.ptp(values)))
    print("  half a bin, times the scale:  %.4f"
          % (quantizer.bin_width() / 2 * quantizer.scales_[0]))

    print("\nScaling by the mean absolute value keeps the sign and the zero,")
    print("so a positive series stays positive and the model can be asked to")
    print("continue it. z-normalization would throw that away.")

    print("\nResolution depends on how the dynamic range compares to the mean:")
    for n_bins in (64, 256, 1024, 4096):
        q = MeanScaleQuantizer(n_bins=n_bins).fit()
        e = np.abs(q.inverse_transform(q.transform(series)) - values).max()
        print("  n_bins %5d  ->  max error %8.4f  (%.3f%% of the range)"
              % (n_bins, e, 100 * e / np.ptp(values)))

    # --------------------------------------------------------- the contrast
    rule("The same move, opposite purposes")
    rows = [
        ("scaling", "z-normalization", "divide by mean |x|"),
        ("keeps the level", "no", "no"),
        ("keeps the sign", "no", "yes"),
        ("time axis", "PAA, 24 points per symbol", "one token per point"),
        ("cut points", "equiprobable gaussian", "uniform over [-15, 15]"),
        ("alphabet", "5", "4096"),
        ("output length", "%d" % len(word), "%d" % len(tokens)),
        ("invertible", "no", "yes, up to half a bin"),
        ("distance with a bound", "yes, MINDIST", "no"),
        ("readable by a human", "yes", "no"),
    ]
    width = max(len(r[0]) for r in rows)
    print("  %-*s  %-26s  %s" % (width, "", "SAX (2003)", "Chronos-style (2024)"))
    print("  %s  %s  %s" % ("-" * width, "-" * 26, "-" * 22))
    for label, left, right in rows:
        print("  %-*s  %-26s  %s" % (width, label, left, right))

    # ------------------------------------------------------ the LLM bridge
    rule("Why the small alphabet is worth something again")
    summary = segment_summary(series, sax)
    anomalies = summary.index[summary.zscore.abs() > 1.5].tolist()
    text = describe(series, sax, anomalies=anomalies, name="demand_kwh", max_segments=8)
    print("A SAX encoding can be written down in a way a language model reads")
    print("directly. An embedding cannot. Neither can 4096-way token ids.\n")
    print(text)
    print("\nThat text is what you put in a prompt. pynuTS stops here on purpose:")
    print("no API client, no key, no network.")


if __name__ == "__main__":
    main()
