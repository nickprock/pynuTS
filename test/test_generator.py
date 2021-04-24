# embryo of unit test suite for pynuTS clustering

import pytest
from demos.generator import AR,MA,ARMA,ARIMA
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

    def test_linear_using_buffer(self):
        m = AR(c=1,sigma=0,coeff=[1],x_buff=[4])
        x = m.generate(10)
        assert x == [float(x) for x in range(5,15)]

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

    def test_seeded_run_using_buffer(self):
        random.seed(1)
        m = MA(mu=0,sigma=1,coeff=[1,2,3],e_buff=[10,9,8])
        x = m.generate(10)
        expected = [57.28818475315546,  45.73763036185523,
                    28.09215092394896,   6.0652376348325605,
                     2.6242915779000637,-2.390918573400903,
                    -5.468746036302335, -5.672783226762393,
                    -3.1877202581453714,-5.60728181909532]
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

    def test_linear_using_buffer(self):
        m = ARMA(c=1,sigma=0,pcoeff=[1],x_buff=[4])
        x = m.generate(10)
        assert x == [float(x) for x in range(5,15)]

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

    def test_seeded_run_ma_using_buffer(self):
        random.seed(1)
        m = ARMA(c=0,sigma=1,qcoeff=[1,2,3],e_buff=[10,9,8])
        x = m.generate(10)
        expected = [57.28818475315546,  45.73763036185523,
                    28.09215092394896,   6.0652376348325605,
                     2.6242915779000637,-2.390918573400903,
                    -5.468746036302335, -5.672783226762393,
                    -3.1877202581453714,-5.60728181909532]
        assert x == pytest.approx(expected)

    def test_seeded_run_arma_using_buffers(self):
        random.seed(1)
        m = ARMA(c=1,sigma=1,pcoeff=[1,-1,0.5],qcoeff=[1,2,-1],
                             x_buff=[-4,-6,8],e_buff=[2,1,-1])
        x = m.generate(10)
        expected = [13.288184753155463,  3.0258151150106967,
                    -0.1702187141958067, 5.360557169581937,
                     4.870192584384071, -2.2317357514497194,
                    -5.832221183458628, -1.469479506162701,
                     0.9338154760988293,-1.031684748519612]
        assert x == pytest.approx(expected)


class TestARIMA(object):

    def test_default_creation(self):
        m = ARIMA()
        assert m

    def test_constant_generation(self):
        m = ARIMA(c=1,sigma=0)
        x = m.generate(10)
        assert x == [1]*10

    def test_linear(self):
        m = ARIMA(c=1,sigma=0,pcoeff=[1])
        x = m.generate(10)
        assert x == [float(x) for x in range(1,11)]

    def test_linear_multiple_calls(self):
        m = ARIMA(c=1,sigma=0,pcoeff=[1])
        x = m.generate(10)
        assert x == [float(x) for x in range(1,11)]
        x = m.generate(10)
        assert x == [float(x) for x in range(11,21)]

    def test_linear_using_buffer(self):
        m = ARIMA(c=1,sigma=0,pcoeff=[1],x_buff=[4])
        x = m.generate(10)
        assert x == [float(x) for x in range(5,15)]

    def test_white_noise(self):
        m = ARIMA(c=0,sigma=1,pcoeff=[])
        s = pd.Series(m.generate(1000))
        assert s.std() == pytest.approx(1.0,abs=1e2)
        assert s.mean() == pytest.approx(0.0,abs=1e2)

    def test_periodic(self):
        m = ARIMA(c=1,sigma=0,pcoeff=[1,-1])
        x = m.generate(10)
        assert x == [1.0, 2.0, 2.0, 1.0, 0.0, 0.0, 1.0, 2.0, 2.0, 1.0]

    def test_seeded_run_ar(self):
        random.seed(1)
        m = ARIMA(c=1,sigma=1,pcoeff=[1,-1,0.5])
        x = m.generate(10)
        expected = [ 2.28818475, 4.73763036,
                     3.51578142, 0.15769978,
                    -1.08143967, 1.55008577,
                     2.68827216, 0.16063711, 
                    -0.55328019, 1.76359339]
        assert x == pytest.approx(expected)

    def test_seeded_run_ma(self):
        random.seed(1)
        m = ARIMA(c=0,sigma=1,qcoeff=[1,2,3])
        x = m.generate(10)
        expected = [ 1.28818475, 2.73763036, 
                     4.09215092, 6.06523763, 
                     2.62429158,-2.39091857, 
                    -5.46874604,-5.67278323,
                    -3.18772026,-5.60728182]
        assert x == pytest.approx(expected)

    def test_seeded_run_ma_using_buffer(self):
        random.seed(1)
        m = ARIMA(c=0,sigma=1,qcoeff=[1,2,3],e_buff=[10,9,8])
        x = m.generate(10)
        expected = [57.28818475315546,  45.73763036185523,
                    28.09215092394896,   6.0652376348325605,
                     2.6242915779000637,-2.390918573400903,
                    -5.468746036302335, -5.672783226762393,
                    -3.1877202581453714,-5.60728181909532]
        assert x == pytest.approx(expected)

    def test_seeded_run_arma_using_buffers(self):
        random.seed(1)
        m = ARIMA(c=1,sigma=1,pcoeff=[1,-1,0.5],qcoeff=[1,2,-1],
                             x_buff=[-4,-6,8],e_buff=[2,1,-1])
        x = m.generate(10)
        expected = [13.288184753155463,  3.0258151150106967,
                    -0.1702187141958067, 5.360557169581937,
                     4.870192584384071, -2.2317357514497194,
                    -5.832221183458628, -1.469479506162701,
                     0.9338154760988293,-1.031684748519612]
        assert x == pytest.approx(expected)

    def test_seeded_random_walk_as_ar(self):
        random.seed(1)
        m = ARIMA(c=0,sigma=1,pcoeff=[1])
        x = m.generate(10)
        expected = [1.2881847531554629,  2.737630361855234, 
                    2.8039661707934957,  2.039422519821864, 
                    0.9472493047177224,  0.9785838215494392, 
                   -0.04351934846143368,-1.4803487935639636, 
                   -1.2810368170802098, -1.147662212421605]
        assert x == pytest.approx(expected)

    def test_seeded_random_walk_as_arima(self):
        random.seed(1)
        m = ARIMA(c=0,sigma=1,d=1)
        x = m.generate(10)
        expected = [1.2881847531554629,  2.737630361855234, 
                    2.8039661707934957,  2.039422519821864, 
                    0.9472493047177224,  0.9785838215494392, 
                   -0.04351934846143368,-1.4803487935639636, 
                   -1.2810368170802098, -1.147662212421605]
        assert x == pytest.approx(expected)

    def test_full_arima(self):
        random.seed(1)
        m = ARIMA(pcoeff=[1,-1],d=2,qcoeff=[1,1,1,1,1])
        x = m.generate(10)
        expected = [ 1.2881847531554629, 6.602184621321622,
                    17.457781022136512, 31.868581360411298,
                    45.24023840821498,  54.99613204963708,
                    60.84370147353184,  62.18073093086161,
                    58.83054165128004,  52.11658886910866]
        assert x == pytest.approx(expected)

    def test_full_arima_10_equivalent_to_5_and_5(self):
        random.seed(1)
        m1 = ARIMA(pcoeff=[1,-1],d=2,qcoeff=[1,1,1,1,1])
        x1 = m1.generate(10)
        random.seed(1)
        m2 = ARIMA(pcoeff=[1,-1],d=2,qcoeff=[1,1,1,1,1])
        x2 = m2.generate(5) + m2.generate(5)
        assert x1 == pytest.approx(x2)
