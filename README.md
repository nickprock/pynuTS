# pynuTS

## A little Python library for Time Series

<br>

![peanuts](http://www.pngall.com/wp-content/uploads/2016/06/Peanut-Free-PNG-Image.png)

<br>

**pynuTS** is a little python library based on my articles pubblished in 2020 on [**IAML**](https://www.linkedin.com/company/iaml/) blog.

The articles are written in italian, you can read them at the follow links:

* [Breve introduzione al Dynamic Time Warping](https://nicoprocopio.blogspot.com/2020/06/breve-introduzione-al-dynamic-time.html)

* [Individuare pattern con il SAX encoding](https://nicoprocopio.blogspot.com/2020/04/individuare-pattern-col-sax-encoding.html)

* [Trattare i valori mancanti nelle serie storiche](https://nicoprocopio.blogspot.com/2021/06/trattare-i-valori-mancanti-nelle-serie.html)

<br>

![iaml](http://lcsl.mit.edu/courses/mlcc/mlcc2019/img/logos/iaml.png)

<br>

## What pynuTS is (and what it is not)

pynuTS is a small, deliberately readable library about **symbolic
representations of time series**: turning real numbers into symbols from a
finite alphabet, and what you can do once you have.

That idea is older than it looks and younger than it sounds. SAX did it in 2003
so that series could be indexed and compared cheaply. Chronos does it in 2024 so
that a transformer built for language can read a time series at all. Same move,
opposite purposes, and every design choice follows from the purpose:

|  | SAX (2003) | Chronos-style (2024) |
| --- | --- | --- |
| scaling | z-normalization | divide by mean \|x\| |
| keeps the sign | no | yes |
| time axis | PAA, w points per symbol | one token per point |
| cut points | equiprobable gaussian | uniform over a fixed range |
| alphabet | 3 to 10 symbols | thousands of tokens |
| invertible | no | yes, up to half a bin |
| distance with a proven bound | yes, MINDIST | no |
| readable by a human, or an LLM | yes | no |

`python demos/symbolic_representations.py` prints that comparison computed on a
real series, side by side.

Around this core sit the classics the library started from: **DTW clustering**
with DBA barycenters, **rolling mean imputation**, and a compact
**ARIMA/SARIMA generator** for synthetic data.

It is meant for **learning, teaching and prototyping**. Every algorithm is
implemented in plain numpy in a few dozen readable lines, which is exactly what
the mature libraries cannot offer. For production work on real volumes you
probably want one of these instead, and that is a recommendation, not a
disclaimer:

| Need | Use |
| --- | --- |
| Fast DTW, DBA barycenters, soft-DTW | [tslearn](https://github.com/tslearn-team/tslearn), [aeon](https://github.com/aeon-toolkit/aeon), [dtaidistance](https://github.com/wannesm/dtaidistance) |
| SAX / SFA / BOSS / WEASEL | [pyts](https://github.com/johannfaouzi/pyts), [aeon](https://github.com/aeon-toolkit/aeon) |
| Missing value imputation | [PyPOTS](https://github.com/WenjieDu/PyPOTS), [sktime](https://github.com/sktime/sktime) |
| Forecasting, foundation models | [Nixtla](https://github.com/Nixtla/statsforecast), [darts](https://github.com/unit8co/darts), [Chronos](https://github.com/amazon-science/chronos-forecasting) |

### Known approximations

* **`pynuTS.quantize` implements the tokenization scheme, not the model.** There
  are no weights here, no vocabulary layout and no special tokens: Chronos
  reserves a couple of ids for padding and end-of-sequence, which is beside the
  point being made.
* **`NaiveSAX` is the older, non-canonical variant**, kept for backward
  compatibility. It uses empirical quantiles instead of gaussian breakpoints and
  does not z-normalize, so it gives no lower-bounding guarantee. Use
  `pynuTS.sax.SAX` for anything new.
* **`DTWKmeans` is plain numpy**, so it is fine for hundreds of short series and
  not for hundreds of thousands. There is no JIT, no pruning and no GPU.

## What's New?

New features in *version 0.5.0*:

* **`pynuTS.sax.SAX`, canonical SAX**: z-normalization, equiprobable gaussian
  breakpoints and **`MINDIST` with the lower-bounding guarantee**, verified on
  24000 random pairs across six families of series. The breakpoints do not come
  from the data, which is what makes encodings comparable across series
* **`pynuTS.quantize.MeanScaleQuantizer`**: mean scaling plus uniform binning,
  the scheme that turns a series into tokens for a language model. Invertible,
  with a round-trip error bounded by half a bin
* **`pynuTS.report.describe`**: a compact, self-explaining textual description of
  a series and its encoding, ready to drop into a prompt. No API client, no key,
  no network
* `demos/symbolic_representations.py`, the two representations side by side on
  the same series, with a smoke test so it cannot rot
* `norm_ppf` and `gaussian_breakpoints`, an inverse normal CDF accurate to
  machine precision without pulling in scipy

New features in *version 0.4.0*:

* **`DTWKmeans` now averages its centroids with DBA** (DTW Barycenter Averaging,
  Petitjean et al. 2011) instead of the element-wise mean, which was never a
  valid centroid under DTW. `averaging='mean'` keeps the old behaviour
* new module `pynuTS.barycenter` exposing `dba` and `medoid_index`
* new `pynuTS.dtw.dtw_path`, the optimal alignment between two series
* new `'sqeuclidean'` criterion, the one for which the DBA descent is provable,
  and now the default for `DTWKmeans`

New features in *version 0.3.0*:

* **the library installs and imports again**: the dead `sklearn` dependency was
  replaced by `scikit-learn`, `setup.py` gave way to `pyproject.toml`, and two
  unused imports of private NumPy namespaces that broke `decomposition` on
  NumPy 2 are gone
* **new built-in DTW engine** (`pynuTS.dtw`) with a real Sakoe-Chiba band, so the
  unmaintained `dtw` package - no longer installable on Python 3.12+ - is not
  needed anymore. Verified identical to `dtw-python` on 500 random pairs
* `pynuTS.generator` and `pynuTS.datasets` were promoted out of `demos/` and are
  now part of the installed package
* `NaiveSAX` and `TsImputer` gained real `fit`/`transform` methods and work in a
  scikit-learn `Pipeline`
* continuous integration on Python 3.10 to 3.13
* several silent bugs fixed, see the list below

### Bug fixes in 0.3.0

* `TsImputer` silently appended spurious rows instead of imputing when given a
  `Series` with a `DatetimeIndex`
* `TsImputer` fed each imputed value into the next one, so long runs of missing
  values drifted towards a constant. Every value is now computed from the
  original series
* `maximum_distance_recommended` raised `IndexError` on consecutive missing
  values, the most common real world case
* `NaiveSAX` returned an empty string for the *whole* series if a single `NaN`
  was present anywhere
* `naive_dtw` raised `NameError` on every call (numpy was never imported) and
  did not initialize the borders of the cost matrix to infinity, so the distance
  it computed was wrong
* `DTWKmeans` could silently drop a series from every cluster, produced `NaN`
  centroids on series of different lengths, and reseeded the global `random`
  module as a side effect of the constructor

### Breaking changes in 0.4.0

* `DTWKmeans` defaults changed from `averaging='mean'` to `averaging='dba'` and
  from `criterion='euclidean'` to `criterion='sqeuclidean'`. Clusterings and
  `inertia_` values will differ from 0.3.0. Pass
  `DTWKmeans(..., averaging='mean', criterion='euclidean')` for the old
  behaviour. The two defaults move together on purpose: DBA only provably
  descends when the warping path is optimal for the same squared error its
  update step minimizes
* DBA costs roughly 1.5x to 2x the time of the plain mean, and reaches a lower
  inertia in exchange

### Breaking changes in 0.3.0

* `pynuTS.naive_dtw.naive_dtw` is deprecated, use `pynuTS.dtw.dtw_distance`
* `NaiveSAX` now raises `ValueError` on a time window made only of missing
  values, where it used to silently return an empty string
* `demos/generator.py` and `demos/ts_gen.py` are now thin shims over
  `pynuTS.generator` and `pynuTS.datasets`
* Python 3.10 or newer is required (the code has always used the walrus
  operator, so 3.7 never actually worked)

## Getting Started

### Prerequisites

Python 3.10 or higher. The only dependencies are `numpy`, `pandas`,
`scikit-learn` and `tqdm`.

### Installation

```bash
git clone https://github.com/nickprock/pynuTS.git
cd pynuTS
pip install .
```

or, for development:

```bash
pip install -e ".[test,demos]"
pytest
```

## Usage

### Dynamic Time Warping

```python
import numpy as np
from pynuTS.dtw import dtw_distance

# the two series may have different lengths
a = np.array([1, 2, 3, 5, 5, 5, 6])
b = np.array([1, 1, 2, 2, 3, 5])

print(dtw_distance(a, b))          # unconstrained
print(dtw_distance(a, b, w=2))     # Sakoe-Chiba band of half-width 2
```

### Time Series Clustering

```python
import numpy as np
import pandas as pd
from pynuTS.clustering import DTWKmeans

list_of_series = [
    pd.Series(2.5 * np.random.randn(100) + 3),
    pd.Series(2 * np.random.randn(100) + 5),
    pd.Series(-2.5 * np.random.randn(100) + 3),
]

clts = DTWKmeans(num_clust=2, num_iter=5, seed=42)
clts.fit(list_of_series)
print(clts.inertia_)

list_new = [pd.Series(3.5 * np.random.randn(100) + 2)]
print(clts.predict(list_new))
```

### DTW Barycenter Averaging

Averaging time series with the arithmetic mean assumes that points sharing an
index correspond to each other. Under DTW they do not, and that assumption is
what smears a shifted peak into a shape none of the inputs ever had:

```python
import numpy as np
from pynuTS.barycenter import dba

peak = np.exp(-np.linspace(-3, 3, 60) ** 2)
shifted = [np.roll(peak, k) for k in (-9, -4, 0, 4, 9)]

print(peak.max())                      # 0.997  the real amplitude
print(np.mean(shifted, axis=0).max())  # 0.711  the mean flattens it
print(dba(shifted).max())              # 0.997  DBA keeps it
```

`DTWKmeans` uses `dba` for its centroids by default.

### SAX, canonical

```python
import numpy as np
from pynuTS.sax import SAX, znorm

sax = SAX(alphabet=5, windows=24).fit()      # fit looks at no data at all

a = np.sin(np.linspace(0, 20, 720))
b = a * 50 + 1000                            # same shape, different level

print(sax.transform(a))                      # 'deeecbaaaceeedbaaabd...'
print(sax.transform(a) == sax.transform(b))  # True: z-normalization removed the level

# MINDIST works on the 30 characters and never overstates the true distance
c = np.sin(np.linspace(0, 20, 720) + 1.0)
lower_bound = sax.mindist(sax.transform(a), sax.transform(c), n=720)
assert lower_bound <= np.linalg.norm(znorm(a) - znorm(c))
```

That inequality is the point: a candidate whose `mindist` already exceeds the
best distance found so far can be discarded without ever touching the raw
series, and no true match is lost.

### Tokenization for a language model

```python
import numpy as np
from pynuTS.quantize import MeanScaleQuantizer

x = np.sin(np.linspace(0, 6, 100)) * 20 + 100

q = MeanScaleQuantizer(n_bins=4096).fit()
tokens = q.transform(x)                      # array([2184, 2186, 2187, ...])
back = q.inverse_transform(tokens)

print(np.abs(back - x).max())                # <= half a bin, times the scale
print(q.bin_width() / 2 * q.scales_[0])
```

### Describing a series to an LLM

A language model cannot read an array, and it cannot read an embedding either.
It can read this:

```python
from pynuTS.report import describe, segment_summary

summary = segment_summary(series, sax)
anomalies = summary.index[summary.zscore.abs() > 1.5].tolist()

print(describe(series, sax, anomalies=anomalies, name="demand_kwh"))
```

```
Time series "demand_kwh": 720 points from 2024-01-01 to 2024-01-30 23:00.
Raw values: min 9.963, max 74.85, mean 47.39, std 14.77.

SAX encoding: 5 symbols, 30 segments of up to 24 points.
The series is standardized first, so the symbols describe its shape, not its
level: z is the number of standard deviations from the mean.
  a: z < -0.84
  b: -0.84 <= z < -0.25
  c: -0.25 <= z < +0.25
  d: +0.25 <= z < +0.84
  e: z >= +0.84

Encoding: dddcbbcdddccbcaadcbbcdddccbcdd

Segments:
     0  2024-01-01  d  mean      52.44  z +0.34
    ...
    14  2024-01-15  a  mean      13.39  z -2.30   <- ANOMALY
    15  2024-01-16  a  mean      14.29  z -2.24   <- ANOMALY

Flagged segments: 14, 15.
```

pynuTS stops there on purpose. What you ask the model, and which model you ask,
is your business.

### SAX Encoding, the old NaiveSAX

```python
import numpy as np
from pynuTS.decomposition import NaiveSAX

ts1 = 2.5 * np.random.randn(100) + 3
ts2 = 4.5 * np.random.randn(100) + 13

sax = NaiveSAX(windows=10, bounds=[0.33, 0.66], levels=['a', 'b', 'c'])

# per series breakpoints: the encoding is scale invariant, so two series on
# very different scales can produce the same string
print(sax.fit_transform(ts1))

# shared breakpoints: fit once, transform many, and the strings become
# comparable across series
sax.fit(np.concatenate([ts1, ts2]))
print(sax.transform(ts1), sax.transform(ts2))
```

### Imputation

```python
import numpy as np
import pandas as pd
from pynuTS.impute import TsImputer, maximum_distance_recommended

X = pd.Series([1, 2, np.nan, 3, 5, np.nan],
              index=pd.date_range('2024-01-01', periods=6, freq='D'))

dist = maximum_distance_recommended(X.values)
print(TsImputer(m_avg=dist).fit_transform(X))
```

### Synthetic datasets

```python
from pynuTS.generator import AR, ARIMA, SARIMA
from pynuTS.datasets import make_flat_dataset, make_slopes_dataset

print(AR(c=1.0, pcoeff=[0.5], sigma=0.1).generate(100)[:5])
print(SARIMA(pcoeff=[0.5], d=1, Pcoeff=[0.3], m=12, sigma=0.5).generate(100)[:5])

# labelled toy datasets for the clustering demos
series = make_flat_dataset([-1.0, 0.0, 1.0], samples=10, lengths=[50], random_seed=0)
```

## Directory Structure

```
├── pynuTS/                # Main library directory
│   ├── __init__.py       # Public API
│   ├── barycenter.py     # DTW Barycenter Averaging (DBA)
│   ├── clustering.py     # Time series clustering using DTW
│   ├── datasets.py       # Labelled toy datasets
│   ├── decomposition.py  # NaiveSAX, the old non-canonical variant
│   ├── dtw.py            # Dynamic Time Warping engine (distance and path)
│   ├── generator.py      # AR, MA, ARMA, ARIMA, SARIMA generators
│   ├── impute.py         # Time series imputation
│   ├── naive_dtw.py      # Deprecated wrapper over pynuTS.dtw
│   ├── quantize.py       # Mean-scaled uniform tokenization (Chronos-style)
│   ├── report.py         # Textual description of an encoding, for an LLM
│   ├── sax.py            # Canonical SAX with MINDIST lower bounding
│   └── version.py        # Stores the library version
├── demos/                 # Notebooks, plotting helpers, compatibility shims
├── test/                  # Unit tests
├── pyproject.toml         # Packaging configuration
└── README.md              # This file
```

## Contributing

To learn more about making a contribution to pynuTS, please see our [Contribution Guide](https://github.com/nickprock/pynuTS/blob/main/CONTRIBUTING.md).

## Citation

If you use pynuTS in a scientific publication, please cite:

```
@misc{pynuTS,
  author =       {Nicola Procopio and Marcello Morchio},
  title =        {pynuTS},
  version = 	 {0.5.0},
  howpublished = {\url{https://github.com/nickprock/pynuTS/}},
  year =         {2021}
}
```

License
---

The code present in this project is licensed under the MIT LICENSE.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Licenza Creative Commons" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International</a>.
