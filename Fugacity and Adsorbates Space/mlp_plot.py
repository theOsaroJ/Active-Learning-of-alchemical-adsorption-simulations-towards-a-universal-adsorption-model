#!/usr/bin/env python
# coding: utf-8

#importing important libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import warnings
warnings.filterwarnings(action='ignore')

##------------Reading data------------##
data=pd.read_csv('mlpr_sq.csv')
actual= data.iloc[:,0].values
predicted= data.iloc[:,1].values

##------------Calculating R_squared-------------##
coeff= np.corrcoef(actual, predicted)
corr= coeff[0,1]
r_sq= corr**2
r_sq= np.round(r_sq,3)
print(r_sq)
