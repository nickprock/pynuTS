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

pynuTS is a small, deliberately readable implementation of three classic time
series techniques: **DTW based clustering**, **SAX encoding** and **rolling mean
imputation**, plus a compact **ARIMA/SARIMA generator** for building synthetic
datasets.

It is meant for **learning, teaching and prototyping**. Every algorithm is
implemented in plain numpy in a few dozen readable lines, which is exactly what
the mature libraries cannot offer.

For production work on real volumes you probably want one of these instead, and
that is a recommendation, not a disclaimer:

| Need | Use |
| --- | --- |
| Fast DTW, DBA barycenters, soft-DTW | [tslearn](https://github.com/tslearn-team/tslearn), [aeon](https://github.com/aeon-toolkit/aeon), [dtaidistance](https://github.com/wannesm/dtaidistance) |
| SAX / SFA / BOSS / WEASEL | [pyts](https://github.com/johannfaouzi/pyts), [aeon](https://github.com/aeon-toolkit/aeon) |
| Missing value imputation | [PyPOTS](https://github.com/WenjieDu/PyPOTS), [sktime](https://github.com/sktime/sktime) |
| Forecasting | [Nixtla](https://github.com/Nixtla/statsforecast), [darts](https://github.com/unit8co/darts) |

### Known approximations

* **`NaiveSAX` uses empirical quantiles**, not the equiprobable gaussian
  breakpoints of the original paper, and does not z-normalize. It therefore does
  **not** give the MINDIST lower-bounding guarantee, so it cannot be used for
  indexing. Encodings are comparable across series only if you `fit` once and
  `transform` many, see below.
* **`DTWKmeans` is plain numpy**, so it is fine for hundreds of short series and
  not for hundreds of thousands. There is no JIT, no pruning and no GPU.

## What's New?

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

### SAX Encoding

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
│   ├── decomposition.py  # Time series decomposition using SAX
│   ├── dtw.py            # Dynamic Time Warping engine (distance and path)
│   ├── generator.py      # AR, MA, ARMA, ARIMA, SARIMA generators
│   ├── impute.py         # Time series imputation
│   ├── naive_dtw.py      # Deprecated wrapper over pynuTS.dtw
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
  version = 	 {0.4.0},
  howpublished = {\url{https://github.com/nickprock/pynuTS/}},
  year =         {2021}
}
```

License
---

The code present in this project is licensed under the MIT LICENSE.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Licenza Creative Commons" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International</a>.
