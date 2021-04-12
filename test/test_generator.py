# embryo of unit test suite for pynuTS clustering

import pytest
from demos.generator import AR,MA,ARMA
import numpy as np
import pandas as pd
import random


class TestAR(object):
    def test_default_creation(self):
        m = AR()
        assert m

    def test_constant_generation(self):
        m = AR(c=1,sigma=0)
        x = m.generate(10)
        assert x == [1]*10

    def test_linear(self):
        m = AR(c=1,sigma=0,coeff=[1])
        x = m.generate(10)
        assert x == [float(x) for x in range(1,11)]

    def test_linear_multiple_calls(self):
        m = AR(c=1,sigma=0,coeff=[1])
        x = m.generate(10)
        assert x == [float(x) for x in range(1,11)]
        x = m.generate(10)
        assert x == [float(x) for x in range(11,21)]

    def test_white_noise(self):
        m = AR(c=0,sigma=1,coeff=[])
        s = pd.Series(m.generate(1000))
        assert s.std() == pytest.approx(1.0,abs=1e2)
        assert s.mean() == pytest.approx(0.0,abs=1e2)

    def test_periodic(self):
        m = AR(c=1,sigma=0,coeff=[1,-1])
        x = m.generate(10)
        assert x == [1.0, 2.0, 2.0, 1.0, 0.0, 0.0, 1.0, 2.0, 2.0, 1.0]

    def test_seeded_run(self):
        random.seed(1)
        m = AR(c=1,sigma=1,coeff=[1,-1,0.5])
        x = m.generate(10)
        expected = [ 2.28818475, 4.73763036,
                     3.51578142, 0.15769978,
                    -1.08143967, 1.55008577,
                     2.68827216, 0.16063711, 
                    -0.55328019, 1.76359339]
        assert x == pytest.approx(expected)

class TestMA(object):
    def test_default_creation(self):
        m = MA()
        assert m

    def test_constant_generation(self):
        m = MA(mu=1,sigma=0)
        x = m.generate(10)
        assert x == [1]*10

    def test_white_noise(self):
        m = MA(mu=0,sigma=1,coeff=[])
        s = pd.Series(m.generate(1000))
        assert s.std() == pytest.approx(1.0,abs=1e2)
        assert s.mean() == pytest.approx(0.0,abs=1e2)

    def test_seeded_run(self):
        random.seed(1)
        m = MA(mu=0,sigma=1,coeff=[1,2,3])
        x = m.generate(10)
        expected = [ 1.28818475, 2.73763036, 
                     4.09215092, 6.06523763, 
                     2.62429158,-2.39091857, 
                    -5.46874604,-5.67278323,
                    -3.18772026,-5.60728182]
        assert x == pytest.approx(expected)


class TestARMA(object):
    def test_default_creation(self):
        m = ARMA()
        assert m

    def test_constant_generation(self):
        m = ARMA(c=1,sigma=0)
        x = m.generate(10)
        assert x == [1]*10

    def test_linear(self):
        m = ARMA(c=1,sigma=0,pcoeff=[1])
        x = m.generate(10)
        assert x == [float(x) for x in range(1,11)]

    def test_linear_multiple_calls(self):
        m = ARMA(c=1,sigma=0,pcoeff=[1])
        x = m.generate(10)
        assert x == [float(x) for x in range(1,11)]
        x = m.generate(10)
        assert x == [float(x) for x in range(11,21)]

    def test_white_noise(self):
        m = ARMA(c=0,sigma=1,pcoeff=[])
        s = pd.Series(m.generate(1000))
        assert s.std() == pytest.approx(1.0,abs=1e2)
        assert s.mean() == pytest.approx(0.0,abs=1e2)

    def test_periodic(self):
        m = ARMA(c=1,sigma=0,pcoeff=[1,-1])
        x = m.generate(10)
        assert x == [1.0, 2.0, 2.0, 1.0, 0.0, 0.0, 1.0, 2.0, 2.0, 1.0]

    def test_seeded_run_ar(self):
        random.seed(1)
        m = ARMA(c=1,sigma=1,pcoeff=[1,-1,0.5])
        x = m.generate(10)
        expected = [ 2.28818475, 4.73763036,
                     3.51578142, 0.15769978,
                    -1.08143967, 1.55008577,
                     2.68827216, 0.16063711, 
                    -0.55328019, 1.76359339]
        assert x == pytest.approx(expected)

    def test_seeded_run_ma(self):
        random.seed(1)
        m = ARMA(c=0,sigma=1,qcoeff=[1,2,3])
        x = m.generate(10)
        expected = [ 1.28818475, 2.73763036, 
                     4.09215092, 6.06523763, 
                     2.62429158,-2.39091857, 
                    -5.46874604,-5.67278323,
                    -3.18772026,-5.60728182]
        assert x == pytest.approx(expected)