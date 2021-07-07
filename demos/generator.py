#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import Polynomial 
import random,sys


from dataclasses import dataclass,field
from collections import deque
from collections.abc import Generator
from itertools import zip_longest
from typing import List


@dataclass
class BaseARMAGenerator(Generator):
    """Auto Regressive Moving Average time series generator
    ARMA time series of orders (p,q) is defined as follows
    
    Xt = c + error(t) + phi1 * X(t-1) + phi2 * X(t-2) + ... + phip(t-p) * X(t-p) + 
                        theta1 * error(t-1) + theta2 * error(t-2) + ... + thetaq * error(t-q)
    
    All other types of time series
    AR, MA, ARMA, ARIMA, SARIMA 
    can be referred to this generator
    
    Params:
    ---------
    phi_coeff   : list of p floats, [phi1 , phi2, ... , phip]  
    theta_coeff : list of q floats, [theta1 , theta2, ... , thetaq]  
    c      : float, c is the drift constant  
    mu     : float, mu is the mean value for MA series
    sigma  : float, standard deviation of the gaussian random error, with mean = 0
    x_buff: list of floats,  known elements of the time series. 
            x_buff[0] is the oldest, x_buff[-1] is the more recent. 
            Only the more recent len(phi_coeff) values are used. 
            Empty list by default
            Padded with 0 (in the past) if less than len(phi_coeff) are elements are provided
    e_buff: list of floats, known values of the error variable. 
            e_buff[0] is the oldest, e_buff[-1] is the more recent. 
            Only the more recent len(theta_coeff) values are used. 
            Empty list by default.
            Padded with 0 (in the past) if less than len(theta_coeff) are elements are provided
    """
    phi_coeff   : List[float] = field(default_factory=list)
    theta_coeff : List[float] = field(default_factory=list)
    c      : float = 0.0
    mu     : float = 0.0
    sigma  : float = 1.0
    e_buff : List[float] = field(default_factory=list)
    x_buff : List[float] = field(default_factory=list)

    def __post_init__(self):
        self._x_buffer = deque([0]*len(self.phi_coeff) + self.x_buff,maxlen = len(self.phi_coeff)) 
        self._e_buffer = deque([0]*len(self.theta_coeff) + self.e_buff,maxlen = len(self.theta_coeff)) 

        
    def send(self, ignored_arg):
        # return the next elenement of the generator
        next_e = random.gauss(self.mu,self.sigma)
        #print(f'{self.phi_coeff[::-1]=},{self._x_buffer=}')
        #print(f'{self.theta_coeff[::-1]=},{self._e_buffer=}')
        zx = list(zip(self.phi_coeff[::-1],self._x_buffer))
        ze = list(zip(self.theta_coeff[::-1],self._e_buffer))
        #print(f'{zx=}')
        #print(f'{ze=}')
        next_x = self.c + sum([p*xt for p,xt in zx]) 
        next_x += next_e + sum([q*et for q,et in ze])
        self._x_buffer.append(next_x)
        self._e_buffer.append(next_e)
        #print(f'{next_x=}')
        #print(f'{next_e=}')
        #print(next_x,file=sys.stderr)
        return next_x
    
    def throw(self, type=None, value=None, traceback=None):
        raise StopIteration
        
def params_to_poly(params):
    """Return the numpy polynomial with coefficients derived by regression parameters.
    The parametrs are all negated because the polynomials in the time series models are
    in the form (1 - sum_i(phy_i * B^i))
    
    Arguments:
    ----------
    params : list(float), list of regression parameters (phi_i in the example above)
    
    Returns:
    ----------
    numpy.polynomial.polynomial.Polynomial
    """
    return Polynomial([1]+[-phy for phy in params])


#
# ### [Auto-Regressive Integrated Moving-Average model](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average)
#
@dataclass
class SARIMA:
    """Auto Regressive, Integrated, Moving Average time series generator with Seasonality

    Params:
    ---------
    c      : float, c is the drift constant 
    mu     : float, mu is the mean value of the gaussian noise for MA series
    pcoeff : auto-regressive coefficents, list of p floats, [phi1 , phi2, ... , phip]  
    d      : order of integration
    qcoeff : moving-average coefficents, list of q floats, [theta1 , theta2, ... , thetaq]  
    s_pcoeff : seasonal auto-regressive coefficents, list of p floats, [phi1 , phi2, ... , phip]  
    s_d      : seasonal order of integration
    s_qcoeff : seasonal moving-average coefficents, list of q floats, [theta1 , theta2, ... , thetaq]  
    sigma  : float, standard deviation of the gaussian random error, with mean = 0
    e_buff: list of floats, keeps memory of the last q values of the error variable. 
            e_buff[0] is the oldest, e_buff[q-1] is the more recent. 
            Empty list by default
    x_buff: list of floats, keeps memory of the last p elements of the time series. 
            x_buff[0] is the oldest, x_buff[p-1] is the more recent. 
            Empty list by default
            
    """
    c      : float = 0.0
    mu     : float = 0.0
    pcoeff : List[float] = field(default_factory=list)
    d      : int = 0 
    qcoeff : List[float] = field(default_factory=list)
    Pcoeff : List[float] = field(default_factory=list)
    D      : int = 0 
    Qcoeff : List[float] = field(default_factory=list)
    m      : int = 0
    sigma  : float = 1.0
    e_buff : List[float] = field(default_factory=list)
    x_buff : List[float] = field(default_factory=list)

    def __post_init__(self):
        # non seasonal components
        ar_p = params_to_poly(self.pcoeff)          # polynimal represntation of AR parameters
        d_p  = params_to_poly([1]) ** self.d        # polynomial representation of integration term
        ma_p = params_to_poly(self.qcoeff)          # polynomial representation of MA parameters

        # seasonal components - polynomial coefficient are mutuiplied by seasonality
        # seasonal AR
        sar_p = np.zeros((self.m+1) * len(self.Pcoeff))
        for i,p in enumerate(self.Pcoeff) :
            sar_p[(i+1)*self.m] = p
        sar_p = params_to_poly(sar_p[1:])
        # seasonal MA
        sma_p = np.zeros((self.m+1) * len(self.Qcoeff))
        for i,q in enumerate(self.Qcoeff) :
            sma_p[(i+1)*self.m] = q
        sma_p = params_to_poly(sma_p[1:])
        # seasonal Integration
        sd_p = np.zeros(self.m+1)
        sd_p[self.m] = 1
        sd_p = params_to_poly(sd_p[1:]) ** self.D

        ar_d_sar_sd_p = ar_p * d_p * sar_p * sd_p   # polynomial multiplication of the left side of SARIMA equation
        ma_sma_p = ma_p * sma_p                     # polynomial multiplication of the right side of SARIMA equation

        self._generator = BaseARMAGenerator(phi_coeff=list(-ar_d_sar_sd_p)[1:],  # AR and D polyn. coefficients, excluded term 0, negated  
                                            theta_coeff=list(ma_sma_p)[1:],      # MA polyn. coefficients, excluded term 0 
                                            sigma=self.sigma,c=self.c,mu=self.mu,
                                            x_buff=self.x_buff,e_buff=self.e_buff)
                                            
        self.x_buff = self._generator.x_buff
        self.e_buff = self._generator.e_buff

    def __str__(self):
        str = f'SARIMA({len(self.pcoeff)},{self.d},{len(self.qcoeff)})({len(self.Pcoeff)},{self.D},{len(self.Qcoeff)}){self.m}, sigma={self.sigma}, drift={self.c}'
        return str

    def generate(self, n = 100):
        return [next(self._generator) for _ in range(n)]





@dataclass
class GeneratorBase:
    """Basic generator wrapper

    Params:
    ---------
    sigma  : float, standard deviation of the gaussian random error, with mean = 0 (significant in all models)            
    """
    sigma  : float = 1.0
       
    def __post_init__(self):
        self.wrapped_sarima = SARIMA(sigma=self.sigma)

    def __str__(self):
        str = f'GeneratorBase: sigma={self.sigma}'
        return str

    def generate(self, n = 100):
        return self.wrapped_sarima.generate(n)
    
@dataclass
class AR(GeneratorBase):
    """Auto Regressive time series generator 

    Params: same of GeneratorBase plus
    ---------
    c      : float, c is the drift constant 
    pcoeff : auto-regressive coefficents, list of p floats, [phi1 , phi2, ... , phip]  
    x_buff: list of floats, keeps memory of the last p elements of the time series. 
            x_buff[0] is the oldest, x_buff[p-1] is the more recent. 
            Empty list by default
            
    """
    c      : float = 0.0
    pcoeff : List[float] = field(default_factory=list)
    x_buff : List[float] = field(default_factory=list)
       
    def __post_init__(self):
        self.wrapped_sarima = SARIMA(c=self.c,pcoeff=self.pcoeff,
                            sigma=self.sigma,
                            x_buff=self.x_buff)
    def __str__(self):
        str = f'AR({len(self.pcoeff)}), sigma={self.sigma}, drift={self.c}'
        return str

    def generate(self, n = 100):
        return self.wrapped_sarima.generate(n)
    
@dataclass
class MA(GeneratorBase):
    """Moving Average time series generator 

    Params: same of GeneratorBase plus
    ---------
    mu     : float, mu is the mean value of the gaussian noise for MA series
    qcoeff : moving-average coefficents, list of q floats, [theta1 , theta2, ... , thetaq]  
    e_buff: list of floats, keeps memory of the last q values of the error variable. 
            e_buff[0] is the oldest, e_buff[q-1] is the more recent. 
            Empty list by default
    """
    mu     : float = 0.0
    qcoeff : List[float] = field(default_factory=list)
    e_buff : List[float] = field(default_factory=list)
       
    def __post_init__(self):
        self.wrapped_sarima = SARIMA(qcoeff=self.qcoeff,mu=self.mu,
                            sigma=self.sigma,
                            e_buff=self.e_buff)
    def __str__(self):
        str = f'MA({len(self.qcoeff)}), sigma={self.sigma}, mu={self.mu}'
        return str

@dataclass
class ARMA(AR,MA):
    """Auto Regressive Moving Average time series generator 

    Params: same of AR plus MA
    ---------
    """
       
    def __post_init__(self):
        self.wrapped_sarima = SARIMA(c=self.c,pcoeff=self.pcoeff,qcoeff=self.qcoeff,mu=self.mu,
                            sigma=self.sigma,
                            e_buff=self.e_buff,x_buff=self.x_buff)
    def __str__(self):
        str = f'ARMA({len(self.pcoeff)},{len(self.qcoeff)}), sigma={self.sigma}, drift={self.c}, mu={self.mu}'
        return str

@dataclass
class ARIMA(ARMA):
    """Auto Regressive, Integrated,  Moving Average time series generator 

    Params: same of ARMA plus
    ---------
    d      : order of integration
    """
    d      : int = 0   
    
    def __post_init__(self):
        self.wrapped_sarima = SARIMA(c=self.c,pcoeff=self.pcoeff,qcoeff=self.qcoeff,mu=self.mu,d=self.d,
                            sigma=self.sigma,
                            e_buff=self.e_buff,x_buff=self.x_buff)
    def __str__(self):
        str = f'ARIMA({len(self.pcoeff)},{self.d},{len(self.qcoeff)}), sigma={self.sigma}, drift={self.c}, mu={self.mu}'
        return str
        

################## utilities for files

import datetime
from pathlib import Path

def save_new_csv(s,path='.',root_name='test_ts',freq='H',start_time=None,columns=['Timestamp','Value']):    
    csv_path = get_free_random_path(path,root_name,'csv')
    s_csv = with_ts_index(s,freq,start_time)
    s_csv.index.name,s_csv.name=columns
    s_csv.to_csv(csv_path)
    return(csv_path)
    
def with_ts_index(s,freq='H',start_time=None):
    if start_time is None :
        now = datetime.datetime.now()
        start_time = datetime.datetime(now.year-1,now.month,1)
    index = pd.date_range(start=start_time,freq=freq,periods=len(s))
    s_ts = s.copy()
    s_ts.index = index
    return s_ts

def get_free_random_path(path,root_name,extension):
    while (free_path := Path(path)/(f'{root_name}_{random.randint(0,1000):04d}.{extension}')).exists():
        pass
    return free_path
