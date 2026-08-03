"""Backward compatible shim.

The sample dataset builders moved into the installed package as
``pynuTS.datasets``. This module is kept so that the notebooks in this folder,
which do ``from ts_gen import make_flat_dataset, ...``, keep working.
"""

from pynuTS.datasets import (  # noqa: F401
    make_binary_code_dataset,
    make_flat_dataset,
    make_slopes_dataset,
)
