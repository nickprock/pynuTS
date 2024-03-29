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

## Installation
------------

### Dependencies
~~~~~~~~~~~~
* Python (>= 3.8.5)
* NumPy (>= 1.19.2)
* Pandas (>= 1.1.3)
* Scikit-learn (>= 0.23.2)
* tqdm (>= 4.50.2)
* dtw (>= 1.4.0)
~~~~~~~~~~~~

### User Installation

The easiest way to install pynuTS is using:

```
sudo apt install git

pip install git+https://github.com/nickprock/pynuTS.git@main
```

Or clone the repo and:

```
pip install pynuTS-master.zip
```

## Demos

After installation, you can try the demo notebooks.

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
