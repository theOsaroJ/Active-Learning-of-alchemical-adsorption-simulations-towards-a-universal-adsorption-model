#!/usr/bin/env python
# coding: utf-8

##------------------Importing important libraries----------------------##

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data= pd.read_csv('Cut_CompleteData.csv')
fug= data.iloc[:,0].values
ch= data.iloc[:,1].values
bl= data.iloc[:,2].values
es= data.iloc[:,3].values
si= data.iloc[:,4].values

for i in range(len(fug)):
    if fug[i] == 2500:
        fug[i] = 2750
        
    if fug[i] == 5000:
        fug[i] = 5500
        
    if fug[i] == 7500:
        fug[i] = 7750
        
    if fug[i] == 10000:
        fug[i] = 10500
        
    if fug[i] == 25000:
        fug[i] = 27500
        
    if fug[i] == 50000:
        fug[i] = 55000
        
    if fug[i] == 75000:
        fug[i] = 77500
        
    if fug[i] == 100000:
        fug[i] = 105000
        
    if fug[i] == 500000:
        fug[i] = 550000
        
    if fug[i] == 1000000:
        fug[i] = 1050000  
        
    if fug[i] == 5000000:
        fug[i] = 5500000
        
    if fug[i] == 7500000:
        fug[i] = 7750000

for j in range(len(ch)):
    if ch[j] == 0:
        ch[j] = 0.05
    if ch[j] == 0.2:
        ch[j] = 0.3
    if ch[j] == 0.5:
        ch[j] = 0.6

for k in range(len(bl)):
    if bl[k] == 0:
        bl[k] = 0.2
        
    if bl[k] == 1:
        bl[k] = 1.13
        
    if bl[k] == 1.3:
        bl[k] = 1.45
        
for l in range(len(es)):
    if es[l] == 30:
        es[l] = 35
        
    if es[l] == 50:
        es[l] = 58
        
    if es[l] == 60:
        es[l] = 62
    
    if es[l] == 90:
        es[l] = 94
        
    if es[l] == 100:
        es[l] = 102
        
    if es[l] == 120:
        es[l] = 127
        
    if es[l] == 150:
        es[l] = 154
        
    if es[l] == 190:
        es[l] = 197
        
    if es[l] == 200:
        es[l] = 212
    
    
for m in range(len(si)):
    if si[m] == 3:
        si[m] = 3.1
    
    if si[m] == 3.5:
        si[m] = 3.54
        
    if si[m] == 3.65:
        si[m] = 3.78
        
    if si[m] == 3.8:
        si[m] = 3.98
        
    if si[m] == 4:
        si[m] = 4.15
        
    if si[m] == 4.2:
        si[m] = 4.42
        
    if si[m] == 4.35:
        si[m] = 4.64
        
    if si[m] == 4.5:
        si[m] = 4.86
        
    if si[m] == 5:
        si[m] = 5.08
        
    if si[m] == 5.15:
        si[m] = 5.1
data.to_csv('TestData.csv', index=False)
