import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ### Helper functions for simple TS visualization

def plot_list_of_ts_over_subplots(list_of_series,figsize=(14,3)):
    """Plot a clustered list of timeseries, each cluster on a subplot with lines of same color.
    Cluster name is retrieved from the name of each series.
    
    Arguments:
    -----------------------    
        list_of_series : list of Pandas Series. the name of the series is the id of the cluster and shall be int
            
        figsize : tuple to set matplotlib figure size
    """
    df = pd.DataFrame(list_of_series)
    clusters = list(set(df.index))
    if len(clusters) > 1:
        fig, axs = plt.subplots(len(clusters), sharex=True, sharey=False,figsize=figsize)
        for i,c in enumerate(clusters):
            df_cluster = df.loc[[c]]
            for _,l in df_cluster.iterrows():
                axs[i].plot(l,label=c,color=index_to_color(c))
    else:
        fig, axs = plt.subplots(len(clusters), sharex=True, sharey=True,figsize=figsize)
        for _,l in df.iterrows():
            axs.plot(l,label=clusters[0],color=index_to_color(clusters[0]))
        

def plot_list_of_ts(list_of_series,figsize=(14,3),figure=None,linewidth=1,marker=None):
    """Plot a clustered list of timeseries, each cluster with line of the same color
    Cluster name is retrieved from the name of each series.
    
    Arguments:
    -----------------------    
        list_of_series : list of Pandas Series. the name of the series is the id of the cluster and shall be int
            
        figsize : tuple to set matplotlib figure size

        figure : existing matplotlib figure to allow adding series to an existing figure, otherwise None (defualt)

        linewidth : integer as in matplotlib 2DLines

        marker : string e.g. 'o' as in matplotlib 2DLines

    Returns :
    -----------------------
        figure : matplotlib figure of the plot

    """
    if figure is None:
        figure = plt.figure(figsize=figsize)
    for l in list_of_series:
        plt.plot(l,label=l.name,color=index_to_color(l.name),linewidth=1,marker=marker)
    return figure

def index_to_color(i,cmap_name='tab10'):
    """Get a color from an integer.

    Arguments :
     -----------------------   
    i : int, to be converted to a color usable in matplotlib functions
    
    cmap_name : string to identify a matplotlib palette to pick colors from.
        see https://matplotlib.org/stable/tutorials/colors/colormaps.html

    Returns:
    tuple : color tuple
    """
    cmap = plt.get_cmap(cmap_name)
    index = i % cmap.N
    return cmap.colors[index]


