#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import random


from dataclasses import dataclass,field
from typing import List


# 
# 
# ### [Autoregressive model](https://en.wikipedia.org/wiki/Autoregressive_model)
# 
# the notation AR(p) refers to the autoregressive model of order p. The AR(p) model is written
# 
# ![AR equation](img/ar.svg)
# 
# where $\varphi _{1},\ldots ,\varphi _{p}$ are parameters, c is a constant, and the random variable $\varepsilon _{t}$ is white noise. The value of p is called the order of the AR model.
# 
# Some constraints are necessary on the values of the parameters so that the model remains stationary. For example, processes in the AR(1) model with ${\displaystyle |\varphi _{1}|\geq 1}$ are not stationary.


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

    def generate(self, n = 100):
        x = []
        for _ in range(n):
            x.append(self._next_x())
        return x

    def _next_x(self):
        next_e = random.gauss(0,self.sigma)
        next_x = self.c + sum([p*xt for p,xt in zip(self.coeff,self.x_buff[::-1])]) + next_e
        if len(self.coeff) > 0 :
            self.x_buff.append(next_x)
            self.x_buff = self.x_buff[-len(self.coeff):]
        return next_x  

# 
# 
# ### [Moving-Average model](https://en.wikipedia.org/wiki/Moving-average_model)
# 
# The notation MA(q) refers to the moving average model of order q:
# 
# ![MA equation](img/ma.svg)
# 
# where μ is the mean of the series, the $\theta _{1},\ldots ,\theta _{q}$ are the parameters of the model and the  $\varepsilon _{t}, \varepsilon _{t-1},\ldots ,\varepsilon _{t-q}$ are white noise error terms. The value of q is called the order of the MA model


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

    def generate(self, n = 100):
        x = []
        for _ in range(n):
            x.append(self._next_x())
        return x

    def _next_x(self):
        next_e = random.gauss(0,self.sigma)
        next_x = self.mu + next_e + sum([q*et for q,et in zip(self.coeff,self.e_buff[::-1])])
        if len(self.coeff) > 0 :
            self.e_buff.append(next_e)
            self.e_buff = self.e_buff[-len(self.coeff):]
        return next_x
        

# 
# 
# ### [Auto-Regressive Moving-Average model](https://en.wikipedia.org/wiki/Autoregressive%E2%80%93moving-average_model)
# 
# The notation ARMA(p, q) refers to the model with p autoregressive terms and q moving-average terms. This model contains the AR(p) and MA(q) models,
# 
# ![ARMA equation](img/arma.svg)
# 
# where $\varphi _{1},\ldots ,\varphi _{p}$ are parameters of the AR component, c is a constant, the $\theta _{1},\ldots ,\theta _{q}$ are the parameters of the MA model and the $\varepsilon _{t}, \varepsilon _{t-1},\ldots ,\varepsilon _{t-q}$  are white noise error terms. The vaues (p,q) are called the AR and MA orders of the ARMA model.
# 
# The ARMA model is essentially an infinite impulse response filter applied to white noise, with some additional interpretation placed on it.
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

    def generate(self, n = 100):
        x = []
        for _ in range(n):
            x.append(self._next_x())
        return x

    def _next_x(self):
        next_e = random.gauss(0,self.sigma)
        next_x = self.c + sum([p*xt for p,xt in zip(self.pcoeff,self.x_buff[::-1])]) 
        next_x += next_e + sum([q*et for q,et in zip(self.qcoeff,self.e_buff[::-1])])
        if len(self.pcoeff) > 0 :
            self.x_buff.append(next_x)
            self.x_buff = self.x_buff[-len(self.pcoeff):]
        if len(self.qcoeff) > 0 :
            self.e_buff.append(next_e)
            self.e_buff = self.e_buff[-len(self.qcoeff):]
        return next_x
