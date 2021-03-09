# ### Utilities for the TS demos

import pandas as pd

def get_clustered_list(list_of_series,clusters_dict):
    """Return a list of pandas series with series names adjusted according to the given clusters dictionary
    
    Arguments:
 
    list_of_series : list of Pandas Series. the name of the series is irrelevent

    clusters_dict : dictiorary associating clusters and series indexes in the input list
        example:  {0: [6, 7, 8], 1: [0, 1, 2], 2: [3, 4, 5]}
        means: cluster 0 is composed by series 6,7,8
               cluster 1 is composed by series 0,1,2
               cluster 2 is composed by series 3,4,5

    Returns:  copy of the list of series with names matching the clusters_dict
    """
    new_list = list_of_series.copy()
    for k in clusters_dict.keys():
        for i in clusters_dict[k]:
            new_list[i].name = k
    return new_list

def lists_of_series_are_equal(list_of_series_1,list_of_series_2):
    """Equality test for list of series"""
    return all([(s1==s2).all() for s1,s2 in zip(list_of_series_1,list_of_series_2)])
