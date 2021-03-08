# embryo of unit test suite for pynuTS clustering

import pytest
from pynuTS.clustering import DTWKmeans
import numpy as np
import pandas as pd

from demos.ts_gen import make_slopes_dataset,make_flat_dataset,lists_of_series_are_equal


class TestFlat(object):
    def test_flat_equal_clusters(self):
        list_levels = [-1.0,0,0.5]
        scalar_size = 10
        list_lenghts = [5]
        list_of_series = make_flat_dataset(list_levels,scalar_size,additive_noise_factor=0.0,level_noise_factor=0.0,lengths=list_lenghts)
        assert len(list_of_series) == scalar_size * len(list_levels)

    def test_flat_unbalanced_clusters(self):
        list_levels = [-1.0,0,0.5]
        list_size = [10,5,20]
        list_lenghts = [5]
        list_of_series = make_flat_dataset(list_levels,list_size,additive_noise_factor=0.3,level_noise_factor=0.3,lengths=list_lenghts)
        assert len(list_of_series) == sum(list_size)

    def test_random_seed(self):
        list_levels = [-1.0,0,0.5]
        list_size = [10,5,20]
        list_lenghts = [5]
        random_seed =  101
        list_of_series_1 = make_flat_dataset(list_levels,list_size,additive_noise_factor=0.3,level_noise_factor=0.3,lengths=list_lenghts,random_seed = random_seed)
        list_of_series_2 = make_flat_dataset(list_levels,list_size,additive_noise_factor=0.3,level_noise_factor=0.3,lengths=list_lenghts,random_seed = random_seed)
        assert lists_of_series_are_equal(list_of_series_1,list_of_series_2)
