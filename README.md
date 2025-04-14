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

## Overview

The `pynuTS` library provides a comprehensive set of tools for time series analysis. It is designed to facilitate the generation, manipulation, and analysis of time series data, offering a range of functionalities suitable for both research and practical applications. The library's core purpose is to empower users with the ability to create, understand, and model time series data effectively.

This repository contains the source code for the `pynuTS` library. The library's architecture is modular, with distinct components for time series generation, clustering, decomposition, imputation, and visualization. Each module within the library is designed to fulfill a specific role, enabling users to perform various time series operations.

The target use cases include:

*   **Research and Development:** Exploring time series models, algorithms, and techniques.
*   **Data Analysis and Exploration:**  Analyzing and visualizing time series datasets.
*   **Prototyping and Experimentation:** Quickly building and testing time series-based solutions.
*   **Educational Purposes:** Providing a learning resource for time series analysis concepts.

The library aims to be a versatile tool for anyone working with time series data, from researchers to data scientists.


#### Work in progress

> The project is work in progress. It is mantained by some voluntiers and me.

### What's New?

New features in *version 0.2.2*:

* changing the names of some hyperparameters in DTWKMeans
* bug fixing
* demos update

New features in *version 0.2.1*:

* SAX Encoding refactoring: new module decompose
* Time series generator *(experimental)*
* New demo notebooks

## Technology Stack

*   **Language:** Python
*   **Frameworks:**
    *   `sklearn (scikit-learn)`: Used for compatibility with the scikit-learn ecosystem (e.g., `BaseEstimator`, `TransformerMixin`).
*   **Libraries:**
    *   `pandas`: Data manipulation and analysis (Series, DataFrame, time series indexing).
    *   `numpy`: Numerical computation (array operations, random number generation, polynomial calculations).
    *   `tqdm`: Progress bar display.
    *   `dtw`: Dynamic Time Warping.
    *   `matplotlib`: Data visualization.
    *   `dataclasses`: Data class definitions.
    *   `collections`: `deque` for efficient buffer management.
    *   `pathlib`: File system interaction.
    *   `random`: Random number generation.
    *   `numpy.polynomial.polynomial`: Polynomial manipulation.
    *   `typing`: Type hinting.
    *   `setuptools`: Packaging and distribution.
*   **Tools:**
    *   `pytest`: Testing framework.

## Directory Structure

```
├── pynuTS/                # Main library directory
│   ├── __init__.py       # Makes pynuTS a package, imports version
│   ├── clustering.py     # Time series clustering using DTW
│   ├── decomposition.py  # Time series decomposition using SAX
│   ├── impute.py         # Time series imputation
│   ├── naive_dtw.py      # Naive implementation of Dynamic Time Warping
│   └── version.py        # Stores the library version
├── demos/                 # Example and demonstration code
│   ├── __init__.py
│   ├── generator.py      # Time series generators (AR, MA, ARMA, ARIMA, SARIMA)
│   ├── ts_gen.py         # Time series dataset generation
│   ├── ts_utils.py       # Utility functions for demos
│   └── ts_viz.py         # Time series visualization functions
├── test/                  # Unit tests
│   ├── __init__.py
│   ├── test_clustering.py # Tests for clustering
│   ├── test_decomposition.py # Tests for decomposition
│   ├── test_generator.py  # Tests for time series generators
│   └── test_ts_gen.py     # Tests for time series dataset generation
├── setup.py               # Packaging and distribution configuration
└── README.md              # This file
```

*   `pynuTS/`: Contains the core library modules.
    *   `__init__.py`: Initializes the `pynuTS` package.
    *   `clustering.py`: Implements the DTW k-means clustering algorithm.
    *   `decomposition.py`: Implements SAX encoding.
    *   `impute.py`: Implements time series imputation.
    *   `naive_dtw.py`: Provides a naive implementation of Dynamic Time Warping.
    *   `version.py`: Stores the library's version number.
*   `demos/`: Contains example code and demonstrations.
    *   `generator.py`: Implements various time series generators (AR, MA, ARMA, ARIMA, SARIMA).
    *   `ts_gen.py`: Provides functions to generate sample time series datasets.
    *   `ts_utils.py`: Contains utility functions for the demos.
    *   `ts_viz.py`: Provides functions for visualizing time series.
*   `test/`: Contains unit tests for the library.
*   `setup.py`:  Defines the package metadata and dependencies.
*   `README.md`: This documentation file.



## Getting Started

1.  **Prerequisites:**

    *   Python 3.7 or higher.
    *   Install the required packages by running `pip install -r requirements.txt` (if a `requirements.txt` file exists). Otherwise, install dependencies manually:
        ```bash
        pip install pandas numpy tqdm dtw scikit-learn matplotlib
        ```

2.  **Installation:**

    *   **Clone the repository:**

        ```bash
        git clone https://github.com/nickprock/pynuTS.git
        cd pynuTS
        ```

    *   **Install the library:**

        ```bash
        python setup.py install
        ```
        or, for development:

        ```bash
        python setup.py develop
        ```

3.  **Module Usage:**

    *   **Time Series Generation (`demos/generator.py`):** The `demos/generator.py` module provides classes for generating various time series based on AR, MA, ARMA, ARIMA and SARIMA models.  These generators can be used to create synthetic datasets for testing or experimentation.

        ```python
        from demos.generator import AR
        import pandas as pd

        # Create an AR(1) time series generator
        ar_generator = AR(c=1.0, pcoeff=[0.5], sigma=0.1)

        # Generate 100 data points
        time_series_data = ar_generator.generate(100)

        # Convert to a pandas Series for easier manipulation
        time_series = pd.Series(time_series_data)

        print(time_series.head())
        ```

    *   **Time Series Clustering (`pynuTS/clustering.py`):** The `pynuTS/clustering.py` module provides the `DTWKmeans` class, which implements the k-means clustering algorithm using Dynamic Time Warping (DTW) for measuring the similarity between time series.

        ```python
        from pynuTS.clustering import DTWKmeans
        import pandas as pd
        import numpy as np

        # Generate some sample time series data
        ts1 = pd.Series(2.5 * np.random.randn(100) + 3)
        ts2 = pd.Series(2 * np.random.randn(100) + 5)
        ts3 = pd.Series(-2.5 * np.random.randn(100) + 3)
        list_of_series = [ts1, ts2, ts3]

        # Initialize and fit the DTWKmeans model
        kmeans = DTWKmeans(num_clust=2, num_iter=5)
        kmeans.fit(list_of_series)

        # Predict the cluster assignments for new time series
        ts4 = pd.Series(3.5 * np.random.randn(100) + 2)
        ts5 = pd.Series(-3.5 * np.random.randn(100) + 2)
        list_new = [ts4, ts5]
        cluster_assignments = kmeans.predict(list_new)

        print(cluster_assignments)
        ```

    *   **Time Series Decomposition (`pynuTS/decomposition.py`):** The `pynuTS/decomposition.py` module provides the `NaiveSAX` class, which implements the SAX (Symbolic Aggregate approXimation) encoding algorithm for dimensionality reduction.

        ```python
        from pynuTS.decomposition import NaiveSAX
        import numpy as np

        # Generate sample time series data
        ts_data = np.random.rand(100)

        # Initialize and transform the time series using SAX
        sax = NaiveSAX(windows=10, bounds=[0.33, 0.66], levels=['a', 'b', 'c'])
        sax_encoding = sax.fit_transform(ts_data)

        print(sax_encoding)
        ```

    *   **Time Series Imputation (`pynuTS/impute.py`):** The `pynuTS/impute.py` module provides the `TsImputer` class to handle missing values in time series using a rolling mean.

        ```python
        from pynuTS.impute import TsImputer
        import numpy as np

        # Sample time series with missing values
        ts_data = np.array([1, 2, np.nan, 4, 5, np.nan, 7])

        # Initialize and apply the imputer
        imputer = TsImputer(m_avg=1) # Use a moving average window of 1
        imputed_ts = imputer.fit_transform(ts_data)

        print(imputed_ts)
        ```

    *   **Time Series Visualization (`demos/ts_viz.py`):** The `demos/ts_viz.py` module provides functions for visualizing time series data.

        ```python
        from demos.ts_viz import plot_list_of_ts
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt

        # Generate some sample time series data
        ts1 = pd.Series(2.5 * np.random.randn(100) + 3, name = 0)
        ts2 = pd.Series(2 * np.random.randn(100) + 5, name = 0)
        list_of_series = [ts1, ts2]

        # Plot the time series
        plot_list_of_ts(list_of_series)
        plt.show()
        ```

## Functional Analysis

### 1. Main Responsibilities of the System

The `pynuTS` library is responsible for providing tools and functionalities for various aspects of time series analysis. Its core responsibilities include:

*   **Time Series Generation:** Generating synthetic time series data based on different models (AR, MA, ARMA, ARIMA, SARIMA). This functionality allows users to create datasets for testing and experimentation with time series algorithms. The `demos/generator.py` module handles this responsibility.
*   **Time Series Clustering:** Clustering time series data using the DTW k-means algorithm. This allows users to group similar time series together, enabling tasks like anomaly detection and pattern recognition. The `pynuTS/clustering.py` module provides this functionality.
*   **Time Series Decomposition:** Performing dimensionality reduction on time series data using SAX encoding. This allows users to simplify and analyze complex time series data. The `pynuTS/decomposition.py` module handles this task.
*   **Time Series Imputation:** Imputing missing values in time series data using a rolling mean. This ensures data integrity and allows users to work with incomplete time series datasets. The `pynuTS/impute.py` module provides this functionality.
*   **Time Series Visualization:** Providing functions for visualizing time series data, enabling users to gain insights and communicate findings effectively. The `demos/ts_viz.py` module handles this responsibility.
*   **Utilities and Demos:** Providing utility functions and example code for demonstration and testing. The `demos` directory contains example code and helper functions to facilitate the use of the library.

The foundational services provided by the system include the time series generation models, the DTW-based clustering, the SAX encoding, and the imputation functionality. These services are designed to be modular and reusable, forming the building blocks for more complex time series analysis tasks.

### 2. Problems the System Solves

The `pynuTS` library is designed to solve several problems related to time series analysis:

*   **Data Generation:** It addresses the need for synthetic time series data generation for model development, testing, and experimentation. The ability to generate time series based on different models (AR, MA, ARMA, ARIMA, SARIMA) allows users to simulate real-world scenarios and evaluate the performance of their algorithms.
*   **Clustering and Pattern Recognition:** It provides a solution for grouping similar time series together using DTW. This is crucial for tasks like anomaly detection, pattern recognition, and time series classification.
*   **Dimensionality Reduction:** The SAX encoding functionality addresses the problem of high-dimensional time series data by reducing its complexity while preserving its essential characteristics. This is beneficial for tasks like indexing, similarity search, and visualization.
*   **Handling Missing Data:** The imputation functionality addresses the common problem of missing values in time series datasets. This allows users to work with incomplete data and prevent errors in subsequent analysis.
*   **Data Exploration and Visualization:** It provides tools for exploring and visualizing time series data, enabling users to gain insights, identify patterns, and communicate their findings effectively.

These functionalities address common challenges faced by researchers, data scientists, and analysts working with time series data.

### 3. Interaction of Modules and Components

The modules within the `pynuTS` library interact with each other to provide a cohesive time series analysis experience. The interactions and dependency flows are as follows:

*   `setup.py` is the entry point for packaging and distribution and uses `pynuTS/version.py` to get the library version.
*   The time series generation modules (`demos/generator.py`) are self-contained and do not depend on other modules directly, except for the use of `params_to_poly` for polynomial representation.
*   The `pynuTS/clustering.py` module depends on the `dtw` library for calculating the DTW distance.
*   The `pynuTS/decomposition.py` module is self-contained.
*   The `pynuTS/impute.py` module is self-contained.
*   The `demos/ts_gen.py` module provides functions for generating time series datasets. These functions are independent but are designed to be used with the other modules, particularly `pynuTS/clustering.py` and `pynuTS/decomposition.py`.
*   The `demos/ts_viz.py` module depends on `matplotlib` for visualization and is designed to visualize time series data generated or processed by other modules.
*   The `demos/ts_utils.py` module provides utility functions for the demos, used to support the other modules.
*   The test suite in the `test` directory imports and utilizes the modules from the `pynuTS` library and the `demos` directory to verify their functionality.

The design emphasizes loose coupling between modules. The `sklearn` base classes (`BaseEstimator`, `TransformerMixin`) promote interoperability with the scikit-learn framework.

### 4. User-Facing vs. System-Facing Functionalities

The `pynuTS` library offers functionalities for both end-users and internal system processes:

*   **User-Facing Functionalities:**
    *   **Time Series Generation:** The time series generators (AR, MA, ARMA, ARIMA, SARIMA) in `demos/generator.py` provide users with the ability to create synthetic time series datasets.
    *   **Time Series Clustering:** The `DTWKmeans` class in `pynuTS/clustering.py` allows users to cluster time series data.
    *   **Time Series Decomposition:** The `NaiveSAX` class in `pynuTS/decomposition.py` enables users to perform SAX encoding for dimensionality reduction.
    *   **Time Series Imputation:** The `TsImputer` class in `pynuTS/impute.py` allows users to handle missing values.
    *   **Time Series Visualization:** The visualization functions in `demos/ts_viz.py` enable users to visualize time series data.
*   **System-Facing Functionalities:**
    *   **Internal Algorithms:** The core algorithms for DTW, SAX, and rolling mean imputation are implemented internally within the respective modules.
    *   **Utility Functions:** The utility functions in `demos/ts_utils.py` support the demos, but are not directly exposed to users.
    *   **Test Suite:** The test suite is used internally to ensure the correctness and reliability of the library.

The user-facing functionalities are designed to be easy to use and integrate into user workflows. The system-facing functionalities are designed to provide the underlying infrastructure and algorithms for the user-facing features.

**Interface or Abstract Class with Common Annotations or Behaviors:**

*   `BaseEstimator` and `TransformerMixin` from `sklearn.base` are used to define a common interface for the clustering, decomposition, and imputation classes. This ensures consistency and allows these classes to integrate seamlessly with the scikit-learn ecosystem.

## Architectural Patterns and Design Principles Applied

*   **Object-Oriented Programming (OOP):** The library makes extensive use of OOP principles, including classes, inheritance, and polymorphism, to structure the code and promote modularity and reusability.
*   **Strategy Pattern:** The different time series generation models (AR, MA, ARMA, ARIMA, SARIMA) can be seen as strategies. The `GeneratorBase` and the `SARIMA` class provide a consistent interface (`generate` method) for these strategies.
*   **Composition:** The `SARIMA` class composes `BaseARMAGenerator`, using it internally for generating time series.
*   **Factory Pattern:** The `GeneratorBase` class and its subclasses can be considered a form of the factory pattern, providing a common interface for creating different time series generators.
*   **Use of `sklearn`'s `BaseEstimator` and `TransformerMixin`:**  The `DTWKmeans`, `NaiveSAX`, and `TsImputer` classes inherit from `BaseEstimator` and `TransformerMixin`. This is a crucial design choice as it makes these classes compatible with the scikit-learn framework. They can be used in pipelines, cross-validation

## Contributing

To learn more about making a contribution to pynuTS, please see our [Contribution Guide](https://github.com/nickprock/pynuTS/blob/main/CONTRIBUTING.md).


## Citation

If you use pynuTS in a scientific publication, please cite:

```
@misc{pynuTS,
  author =       {Nicola Procopio and Marcello Morchio},
  title =        {pynuTS},
  version = 	 {0.2.2}
  howpublished = {\url{https://github.com/nickprock/pynuTS/}},
  year =         {2021}
}
```

License
---

The code present in this project is licensed under the MIT LICENSE.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Licenza Creative Commons" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International</a>.
