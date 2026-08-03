# ### Sample time series datasets for pynuTS
#
# - Binary channel symbols
# - Slopes
# - Flat levels
#

import random

import numpy as np
import pandas as pd


def _make_rngs(random_seed):
    """Return local (stdlib, numpy) generators, leaving the global state alone."""
    return random.Random(random_seed), np.random.default_rng(random_seed)


def _pick_length(lengths, rng):
    """Draw one length uniformly from the given list.

    The old implementation used ``lengths[randint(0, len(lengths)) - 1]``,
    which mapped two of the draws onto the last element and so oversampled it.
    """
    return rng.choice(lengths)


def make_binary_code_dataset(codes,samples,additive_noise_factor=0.01,lengths=None,random_seed=None):
    """Generate a list of binary time series representing a given list of numeric codes. 
    For each code generate a given number of time series, adding some gaussian noise attenuated by a given factor.
    Lenght of each time series is 100 by default, can be changed with the lengths parameter.
    Number of bits transmitted is fixed = 10
    Number of samples per bit is fixed = 10
    
    Arguments:
    -----------------------       
    codes : list of integers to be binary encoded in the series, e.g. [12,334,654]

    samples : number of series produced for each code

    additive_noise_factor : white noise multiplier added to each series

    lenghts : None or list of integers. If none all series will be of lenght 100. Otherwise lengths will be taken
        at random from the elements of the lengths list. Lenghts shall be no less than 10

    random_seed : None or int, seeds both random generators for reproduceability

    Returns:
    -----------------------
    list of Pandas Series. the names of the series are int, and are unique per each code.
        Series names have the same meaning as the id of the cluster.
    """
    list_of_series = []
    if lengths is None:
        lengths = [100]
    if min(lengths) < 10:
        raise ValueError("lengths shall be no less than 10")
    rng, np_rng = _make_rngs(random_seed)
    for i,code in enumerate(codes):
        for _ in range(samples):
            length = _pick_length(lengths, rng)
            bit_len = round(length / 10)
            series = []
            for bit in f'{code:010b}':
                series.extend(list(additive_noise_factor*np_rng.standard_normal(bit_len) + np.ones(bit_len) * int(bit)))
            sample = pd.Series(series,name=i)
            list_of_series.append(sample)
    return(list_of_series)



def make_slopes_dataset(slopes,samples,additive_noise_factor=0.01,intercept_noise_factor=0.1,lengths=None,random_seed=None):
    """Generate a list of time series of various slopes with some random in intercept + white noise on the samples
    Lenght of each time series is 100 by default, can be changed with the lengths parameter.
    
    Arguments:
    -----------------------       
    slopes : list of slopes as angular coefficients, e.g. [1.0,0,-1.0] means 
    
    samples : number of series produced for each slope
    
    additive_noise_factor : white noise multiplier added to each series

    intercept_noise_factor : random offset multiplier for the intercept
    
    lenghts : None or list of integers. If none all series will be of lenght 100. Otherwise lengths will be taken
        at random from the elements of the lengths list. Lenghts shall be no less than 10

    random_seed : None or int, seeds both random generators for reproduceability

    Returns:
    -----------------------
    list of Pandas Series. the names of the series are int, and are unique per each code.
        Series names have the same meaning as the id of the cluster.
    """
    list_of_series = []
    if lengths is None:
        lengths = [100]
    rng, np_rng = _make_rngs(random_seed)
    for i,slope in enumerate(slopes):
        for _ in range(samples):
            length = _pick_length(lengths, rng)
            series = intercept_noise_factor * np.ones(length)*(rng.random()*length-length/2)+\
                     [x * slope for x in range(length)] + \
                     np_rng.standard_normal(length) * additive_noise_factor
            sample = pd.Series(series,name=i)
            list_of_series.append(sample)
    return(list_of_series)


def make_flat_dataset(levels,samples,level_noise_factor=0.01,additive_noise_factor=0.01,lengths=None,random_seed=None):
    """Generate a list of flat time series of various levels with some white noise on the samples
    Lenght of each time series is 100 by default, can be changed with the lengths parameter.
    
    Arguments:
    -----------------------
    slopes : list of levels, e.g. [2.0,1.0,-1.0] means lines mean values will be 2.0, 1.0, -1.0

    samples : int or list of int, number of series produced for each slope
        if int, all clusters have the same size
        if list of int, size of each cluster. Shall have same length as slopes

    level_noise_factor : multiplier to a random factor added to the luevel

    additive_noise_factor : white noise multiplier added to each series

    lenghts : None or list of integers. If none all series will be of lenght 100. Otherwise lengths will be taken
        at random from the elements of the lengths list.

    random_seed : None or int, seeds both random generators for reproduceability

    Returns:
    -----------------------
    list of Pandas Series. the names of the series are int, and are unique per each code.
        Series names have the same meaning as the id of the cluster.
    """
    samples = samples if isinstance(samples,list) else [samples]*len(levels)
    if lengths is None:
        lengths = [100]
    rng, np_rng = _make_rngs(random_seed)
    list_of_series = []
    for i,level,n_samples in zip(range(len(levels)),levels,samples):
        for _ in range(n_samples):
            level_adder = (np_rng.random()-0.5) * level_noise_factor
            length = _pick_length(lengths, rng)
            series = [level+level_adder for _ in range(length)] + \
                     np_rng.standard_normal(length) * additive_noise_factor
            sample = pd.Series(series,name=i)
            list_of_series.append(sample)
    return(list_of_series)
