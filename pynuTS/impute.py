"""
Created on Wed Apr 01 2020

@project: pynuTS
@author: nicola procopio
@last_update: 11/02/2021
@decription: impute missing value with rolling mean
@reference: https://iaml.it/blog/serie-storiche-1-dati-mancanti
"""
import numpy as np
from tqdm import tqdm

class TsImputer:
    """
    Impute missing values in the time series with rolling mean

    Parameters
    -----------------------
    m_avg : int
        default 1. The range of the moving average.
    copy : bool
        default True. If true create a copy of X the input, else overwrite.
    
    Returns
    ----------------------
    a 1D numpy array

    Examples
    ----------------------
    >> import numpy as np
    >> from pynuTS.impute import TsImputer, maximum_distance_recommended
    >> X = np.array([1, 2, np.nan, 3, 5, np.nan])
    >> dist = maximum_distance_recommended(X)
    >> imputer = TsImputer(m_avg = dist)
    >> X_new = imputer.fit_transform(X)
    """
    def __init__(self, m_avg: int = 1, copy : bool = True):
        if (m_avg is None) | (m_avg<1):
            raise ValueError ("m_avg must be a positive integer")

        self.m_avg = m_avg
        self.copy = copy
    
    def fit_transform(self, X):
        na_list = np.where(np.isnan(X))[0].tolist()
        if self.copy:
            temp=X.copy()
        else:
            temp=X
        
        def moving_average(data, position, m_avg):
            low_bound = position - m_avg
            if low_bound<0:
                low_bound=0
            upp_bound = position + m_avg + 1
            if upp_bound>len(data):
                upp_bound=len(data)
            test = data[low_bound:upp_bound]
            new_val = sum(test[~np.isnan(test)])/(len(test[~np.isnan(test)]))
            return new_val
        
        for i in tqdm(na_list):
            temp[i]=moving_average(data=temp, position=i, m_avg=self.m_avg)
        
        return temp


def maximum_distance_recommended(X):
    """
    Recommend the maximum range without missing value

    Parameters
    -----------------------
    X : 1D numpy array

    Returns
    -----------------------
    max_range : int.
        the maximum distance recommended
    
    Examples
    ----------------------
    >> import numpy as np
    >> from pynuTS.impute import maximum_distance_recommended
    >> X = np.array([1,2,np.nan,3, 5, np.nan])
    >> dist = maximum_distance_recommended(X)
        the maximum range recommended for the 'm_avg' parameter is 1
    """
    na_list = np.where(np.isnan(X))[0].tolist()
    diff=[j-i for i, j in zip(na_list[:-1], na_list[1:])]
    diff.sort()
    if len(diff)>0:
        max_range = diff[0] - 1
        if max_range==0: max_range=list(set(diff))[1]-1
    else:
        max_range = int((X.shape[0]-1)/2)-1
    print("the maximum range recommended for the 'm_avg' parameter is {0} ".format(max_range))
    return max_range