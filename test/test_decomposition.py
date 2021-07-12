# embryo of unit test suite for pynuTS clustering

import pytest
import numpy as np
import pandas as pd
import random

from pynuTS.decomposition import NaiveSAX

class TestBasicObject:
    def test_default_object_contruction(self):
        sax = NaiveSAX()
        assert sax

    @pytest.mark.skip(reason="Input validation with exceptions not implemented")
    @pytest.mark.parametrize("init_parmas,expected_exception", 
                [({'levels':1},TypeError),({'bounds':1},TypeError),({'windows':[0.5,0.6]},TypeError),
                 ({'quantile':[0.5,0.6]},TypeError),                     # type of quantile is not validated
                 ({'levels':['A','B'],'bounds':[0.25,0.75]},TypeError)    # mismatch levels vs bounds does not throw exception :-(
                 ])
    def test_bad_parameters_type(self,init_parmas,expected_exception):
        #  levels: list = ["A", "B", "C"], bounds: list = [0.25, 0.75], windows: int = 2, quantile: bool = True
        with pytest.raises(expected_exception):
            sax = NaiveSAX(**init_parmas)

class TestFitTransform:
    @pytest.mark.parametrize("input_series,expected_encoding",
                        [([],''),  # Series of len zero, encoded is a len zero string
                         ([np.nan],'')])  # Series of NaN, encoded is a len zero string
    def test_corner_cases(self,input_series,expected_encoding):
        sax = NaiveSAX()
        assert sax.fit_transform(input_series) == expected_encoding

    @pytest.mark.parametrize("input_series_len,window,expected_encoding_len",
                        [(10,1,10),(10,2,5),(10,3,4),(10,5,2),(10,8,2),(10,12,1)])
    def test_encoded_len(self,input_series_len,window,expected_encoding_len):
        X = np.zeros(input_series_len)
        sax = NaiveSAX(windows=window)
        assert len(sax.fit_transform(X)) == expected_encoding_len

    @pytest.mark.parametrize("window,expected_encoding",
                        [(1,'AAABBBBCCC'),(2,'ABBCC'),(3,'ABBC'),(5,'AC'),(8,'AC'),(12,'C')])
    def test_encoding_quantile(self,window,expected_encoding):
        X = np.arange(0.0,10.0) 
        sax = NaiveSAX(windows=window,bounds=[0.25,0.75],levels=['A','B','C'])
        assert sax.fit_transform(X) == expected_encoding

    @pytest.mark.parametrize("window,expected_encoding",
                        [(1,'AAABBBCCCC'),(2,'AABCC'),(3,'ABCC'),(5,'AC'),(8,'BC'),(12,'B')])
    def test_encoding_absolute(self,window,expected_encoding):
        X = np.arange(0.0,10.0) 
        sax = NaiveSAX(windows=window,quantile=False,bounds=[3,6],levels=['A','B','C'])
        assert sax.fit_transform(X) == expected_encoding

