"""Backward compatible shim.

The time series generators moved into the installed package as
``pynuTS.generator``. This module is kept so that the notebooks in this folder,
which do ``from generator import AR, MA, ARMA, ARIMA, SARIMA``, keep working.
"""

from pynuTS.generator import *  # noqa: F401,F403
from pynuTS.generator import (  # noqa: F401
    ARIMA,
    ARMA,
    AR,
    MA,
    SARIMA,
    BaseARMAGenerator,
    GeneratorBase,
    get_free_random_path,
    params_to_poly,
    save_new_csv,
    with_ts_index,
)
