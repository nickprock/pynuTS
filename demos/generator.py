#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import random


from dataclasses import dataclass,field
from collections import deque
from itertools import zip_longest
from typing import List

#
# ### [Auto-Regressive Integrated Moving-Average model](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average)
#
@dataclass
class ARIMA:
    """Auto Regressive Moving Average time series generator
    ARMA time series of orders (p,d,q) is computed as follows
    Xt = c + error(t) +        X(t-1) +        X(t-2) + ... +             X(t-d) + 
                      + phi1 * X(t-1) + phi2 * X(t-2) + ... + phip(t-p) * X(t-p) +
                        theta1 * error(t-1) + theta2 * error(t-2) + ... + thetaq * error(t-q)
    
    Params:
    ---------
    pcoeff : list of p floats, [phi1 , phi2, ... , phip]  
    d      : order of integration
    qcoeff : list of q floats, [theta1 , theta2, ... , thetaq]  
    c      : float, c is the drift constant
    sigma  : float, standard deviation of the gaussian random error, with mean = 0
    e_buff: list of floats, keeps memory of the last q values of the error variable. 
            e_buff[0] is the oldest, e_buff[q-1] is the more recent. 
            Empty list by default
    x_buff: list of floats, keeps memory of the last p elements of the time series. 
            x_buff[0] is the oldest, x_buff[p-1] is the more recent. 
            Empty list by default
            
    """
    pcoeff : List[float] = field(default_factory=list)
    d      : int = 0 
    qcoeff : List[float] = field(default_factory=list)
    c      : float = 0.0
    sigma  : float = 1.0
    e_buff : List[float] = field(default_factory=list)
    x_buff : List[float] = field(default_factory=list)

    def __post_init__(self):
        dcoeff = [1] * self.d
        pdcoeff = [p+d for p,d in zip_longest(self.pcoeff,dcoeff,fillvalue=0.0)]
        self._model = ARMA(c=self.c, pcoeff = pdcoeff, qcoeff = self.qcoeff, sigma=self.sigma, 
                                     x_buff = self.x_buff, e_buff = self.e_buff)
        self.x_buff = self._model.x_buff
        self.e_buff = self._model.e_buff

    def generate(self, n = 100):
        return self._model.generate(n)

# 
# ### [Auto-Regressive Moving-Average model](https://en.wikipedia.org/wiki/Autoregressive%E2%80%93moving-average_model)
# 

@dataclass
class ARMA:
    """Auto Regressive Moving Average time series generator
    ARMA time series of orders (p,q) is defined as follows
    Xt = c + error(t) + phi1 * X(t-1) + phi2 * X(t-2) + ... + phip(t-p) * X(t-p) + 
                        theta1 * error(t-1) + theta2 * error(t-2) + ... + thetaq * error(t-q)
    
    Params:
    ---------
    pcoeff : list of p floats, [phi1 , phi2, ... , phip]  
    qcoeff : list of q floats, [theta1 , theta2, ... , thetaq]  
    c      : float, c is the drift constant
    sigma  : float, standard deviation of the gaussian random error, with mean = 0
    e_buff: list of floats, keeps memory of the last q values of the error variable. 
            e_buff[0] is the oldest, e_buff[q-1] is the more recent. 
            Empty list by default
    x_buff: list of floats, keeps memory of the last p elements of the time series. 
            x_buff[0] is the oldest, x_buff[p-1] is the more recent. 
            Empty list by default
            
    """
    pcoeff : List[float] = field(default_factory=list)
    qcoeff : List[float] = field(default_factory=list)
    c      : float = 0.0
    sigma  : float = 1.0
    e_buff : List[float] = field(default_factory=list)
    x_buff : List[float] = field(default_factory=list)

    def __post_init__(self):
        self.x_buff = deque([0]*len(self.pcoeff) + self.x_buff,maxlen = len(self.pcoeff)) 
        self.e_buff = deque([0]*len(self.qcoeff) + self.e_buff,maxlen = len(self.qcoeff)) 

    def generate(self, n = 100):
        x = []
        for _ in range(n):
            x.append(self._next_x())
        return x

    def _next_x(self):
        next_e = random.gauss(0,self.sigma)
        zx = zip(self.pcoeff[::-1],self.x_buff)
        ze = zip(self.qcoeff[::-1],self.e_buff)
        next_x = self.c + sum([p*xt for p,xt in zx]) 
        next_x += next_e + sum([q*et for q,et in ze])
        self.x_buff.append(next_x)
        self.e_buff.append(next_e)
        return next_x

# 
# ### [Autoregressive model](https://en.wikipedia.org/wiki/Autoregressive_model)
# 

@dataclass
class AR:
    """Auto Regressive time series generator
    AR time series of order p is defined as follows
    Xt = c + phi1 * X(t-1) + phi2 * X(t-2) + ... + phip(t-p) * X(t-p) + error(t)
    
    Params:
    ---------
    coeff : list of p floats, [phi1 , phi2, ... , phip]  
    c     : float, c is the drift constant
    sigma : float, standard deviation of the gaussian random error, with mean = 0
    x_buff: list of floats, keeps memory of the last p elements of the time series. 
            x_buff[0] is the oldest, x_buff[p-1] is the more recent.
            Empty list by default
    """
    coeff : List[float] = field(default_factory=list)
    c     : float = 0.0
    sigma : float = 1.0
    x_buff: List[float] = field(default_factory=list)

    def __post_init__(self):
        self._model = ARMA(c=self.c, pcoeff=self.coeff, sigma=self.sigma, x_buff = self.x_buff)
        self.x_buff = self._model.x_buff

    def generate(self, n = 100):
        return self._model.generate(n)

# 
# ### [Moving-Average model](https://en.wikipedia.org/wiki/Moving-average_model)
# 

@dataclass
class MA:
    """Moving Average time series generator
    MA time series of order p is defined as follows
    Xt = mu + error(t) + theta1 * error(t-1) + theta2 * error(t-2) + ... + thetaq * error(t-q)
    
    Params:
    ---------
    coeff : list of q floats, [theta1 , theta2, ... , thetaq]  
    mu    : float, mu is the mean value of the series
    sigma : float, standard deviation of the gaussian random error, with mean = 0
    e_buff: list of floats, keeps memory of the last q values of the error variable. 
            e_buff[0] is the oldest, e_buff[q-1] is the more recent. 
            Empty list by default
    """
    coeff : List[float] = field(default_factory=list)
    mu    : float = 0.0
    sigma : float = 1.0
    e_buff: List[float] = field(default_factory=list)


    def __post_init__(self):
        self._model = ARMA(c=self.mu, qcoeff=self.coeff, sigma=self.sigma, e_buff = self.e_buff)
        self.e_buff = self._model.e_buff

    def generate(self, n = 100):
        return self._model.generate(n)
        