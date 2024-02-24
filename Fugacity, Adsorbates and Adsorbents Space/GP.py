#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os
import math
import warnings
import tensorflow as tf
import gpflow
from gpflow.utilities import print_summary
warnings.filterwarnings(action='ignore')
import random

# Set the chunk size for reading CSV files
chunk_size = 5000

# Reading the dataset in chunks
df_chunks = pd.read_csv('Prior.csv', delimiter=',', chunksize=chunk_size)

# Process each chunk
df_list = []
for df_chunk in df_chunks:
    df_list.append(df_chunk)

# Concatenate the chunks into a single DataFrame
df = pd.concat(df_list)

# Repeat the same steps for the 'testing_pca.csv' file
df3_chunks = pd.read_csv('mmm.csv', delimiter=',', chunksize=chunk_size)
df3_list = []
for df3_chunk in df3_chunks:
    df3_list.append(df3_chunk)
df3 = pd.concat(df3_list)

# Extract the necessary columns from the DataFrame

fu = df.iloc[:, 0].values / 1e5
ch = df.iloc[:, 1].values
bl = df.iloc[:, 2].values
ep = df.iloc[:, 3].values
si = df.iloc[:, 4].values
y = df.iloc[:, 5].values
pc1_1 = df.iloc[:, 6].values
pc2_1 = df.iloc[:, 7].values

# Extract the necessary columns from the testing DataFrame

fug_test = df3.iloc[:, 0].values / 1e5
ch_test = df3.iloc[:, 1].values
bl_test = df3.iloc[:, 2].values
eps_test = df3.iloc[:, 3].values
sig_test = df3.iloc[:, 4].values
y_true = df3.iloc[:, 5].values
pc1_test = df3.iloc[:, 6].values
pc2_test = df3.iloc[:, 7].values


#Replacing y if some y value in zero
for i in range(len(y)):
        if (y[i] == 0):
                y[i] = 0.000001

#Transforming 1D arrays to 2D
pc1_1 = np.atleast_2d(pc1_1).flatten().reshape(-1,1)
pc2_1 = np.atleast_2d(pc2_1).flatten().reshape(-1,1)
fu = np.atleast_2d(fu).flatten().reshape(-1,1)
ch = np.atleast_2d(ch).flatten().reshape(-1,1)
bl = np.atleast_2d(bl).flatten().reshape(-1,1)
ep = np.atleast_2d(ep).flatten().reshape(-1,1)
si = np.atleast_2d(si).flatten().reshape(-1,1)
y = np.atleast_2d(y).flatten()

pc1_test = np.atleast_2d(pc1_test).flatten().reshape(-1,1)
pc2_test = np.atleast_2d(pc2_test).flatten().reshape(-1,1)
fug_test= np.atleast_2d(fug_test).flatten().reshape(-1,1)
ch_test= np.atleast_2d(ch_test).flatten().reshape(-1,1)
bl_test= np.atleast_2d(bl_test).flatten().reshape(-1,1)
eps_test = np.atleast_2d(eps_test).flatten().reshape(-1,1)
sig_test = np.atleast_2d(sig_test).flatten().reshape(-1,1)

pc1_true = pc1_1
pc2_1 = pc2_1
fu_true = fu
ch_true = ch
bl_true = bl
ep_true = ep
si_true = si
y_actual = y


#Taking logbase 10 of the input vector
fu = np.log10(fu)
# ch = np.log10(ch)
# bl= np.log10(bl)
ep = np.log10(ep)
si = np.log10(si)
y = np.log10(y)

#Normalizing y
y_std= np.std(y, ddof=1)
y_m= np.mean(y)
y_s= (y - y_m)/y_std

#print(len(x),len(y))
#Taking the log of X_test
fug_test = np.log10(fug_test)
# ch_test = np.log10(ch_test)
# bl_test = np.log10(bl_test)
eps_test = np.log10(eps_test)
sig_test= np.log10(sig_test)

#Extracting the mean and std. dev for fu
fu_m = np.mean(fu)
fu_std = np.std(fu,ddof=1)

#Extracting the mean and std. dev for ch
ch_m = np.mean(ch)
ch_std = np.std(ch,ddof=1)

#Extracting the mean and std. dev for bl
bl_m = np.mean(bl)
bl_std = np.std(bl_test,ddof=1)

#Extracting the mean and std. dev for ep
ep_m = np.mean(ep)
ep_std = np.std(ep,ddof=1)

#Extracting the mean and std. dev for sig
si_m= np.mean(si)
si_std= np.std(si,ddof=1)

#Extracting the mean and std. dev for pc1 and pc2
pc1_m= np.mean(pc1_test)
pc1_std= np.std(pc1_test,ddof=1)

pc2_m= np.mean(pc2_test)
pc2_std= np.std(pc2_test,ddof=1)

#Standardising fugacity,charge,bondlen,eps and sig in log-space
fu_s = (fu - fu_m)/fu_std
ch_s= (ch - ch_m)/ch_std
bl_s= (bl - bl_m)/bl_std
ep_s = (ep - ep_m)/ep_std
si_s= (si - si_m)/si_std
pc1_s= (pc1_1 - pc1_m)/pc1_std
pc2_s= (pc2_1 - pc2_m)/pc2_std

#Standardising X_test in log-space
fug_test = (fug_test - fu_m)/fu_std
ch_test= (ch_test - ch_m)/ch_std
bl_test= (bl_test - bl_m)/bl_std
eps_test = (eps_test - ep_m)/ep_std
sig_test= (sig_test - si_m)/si_std
pc1_test= (pc1_test - pc1_m)/pc1_std
pc2_test= (pc2_test - pc2_m)/pc2_std

# #Initializing scaled down training and prediction set
x_s= np.vstack((fu_s.flatten(),ch_s.flatten(), bl_s.flatten(), ep_s.flatten(),si_s.flatten(),pc1_s.flatten(),pc2_s.flatten())).T
X_test = np.vstack((fug_test.flatten(),ch_test.flatten(),bl_test.flatten(),eps_test.flatten(), sig_test.flatten(),pc1_test.flatten(),pc2_test.flatten())).T

kernel = gpflow.kernels.RationalQuadratic() + gpflow.kernels.White() + gpflow.kernels.Matern12() + gpflow.kernels.Matern12() + gpflow.kernels.White() + gpflow.kernels.RationalQuadratic()
model = gpflow.models.GPR(
        data=(x_s,y_s.reshape(-1, 1)),
        kernel=kernel,
        noise_variance=10**-5)


gpflow.utilities.set_trainable(model.likelihood.variance,False)

#Optimize model with scipy
optimizer = gpflow.optimizers.Scipy()
optimizer.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=10000),method="L-BFGS-B")

y_pred, var = model.predict_f(X_test)

#Removing the tensor_flow print
y_pred=y_pred.numpy()

#transforming the y_pred
y_pred= (y_pred*y_std) + y_m

#Replacing y_pred if some y_pred value in zero
for i in range(len(y_pred)):
        if (y_pred[i] == 0):
                y_pred[i] = 0.000001

sigma= np.zeros(len(var))
for i in range(len(var)):
    sigma[i]= np.sqrt(var[i])

rel_error = np.zeros(len(sigma))
#finding the relative error:
for i in range(len(sigma)):
    rel_error[i] = abs(sigma[i])/abs(y_pred[i])

abs_error = np.zeros(len(sigma))
#finding the absolute error:
for i in range(len(sigma)):
    abs_error[i] = abs(sigma[i])

abs_m = np.mean(abs_error)

#define the limit for uncertainty
lim = 0.05
Max = abs_m
index = np.argmax(abs_error)

#transforming the index to original pressure point

X_test[:,0] = (X_test[:,0]*fu_std) + fu_m
X_test[:,0] = 10**(X_test[:,0])
X_test[:,0] = 1e5*(X_test[:,0])
X_test[:,0] = np.round(X_test[:,0],1)

X_test[:,1] = (X_test[:,1]*ch_std) + ch_m
X_test[:,1] = (X_test[:,1])
X_test[:,1] = np.round(X_test[:,1],2)

X_test[:,2] = (X_test[:,2]*bl_std) + bl_m
X_test[:,2] = (X_test[:,2])
X_test[:,2] = np.round(X_test[:,2],3)

X_test[:,3] = (X_test[:,3]*ep_std) + ep_m
X_test[:,3] = 10**(X_test[:,3])
X_test[:,3] = np.round(X_test[:,3],2)

X_test[:,4] = (X_test[:,4]*si_std) + si_m
X_test[:,4] = 10**(X_test[:,4])
X_test[:,4] = np.round(X_test[:,4],3)

X_test[:,5] = (X_test[:,5]*pc1_std) + pc1_m

X_test[:,6] = (X_test[:,6]*pc2_std) + pc2_m

xx= pd.DataFrame(X_test,columns=['fugacity','charge','bond length','epsilon','sigma','PC1', 'PC2'])
xx.to_csv('Xtest.csv', index=False)
#checking the whether the maximum uncertainty is less than out desired limit
if (Max >= lim):
        Data = str(X_test[index])
        Data = Data.replace("[","")
        Data = Data.replace("]","")
        print(X_test[index,0],X_test[index,1], X_test[index,2], X_test[index,3], X_test[index,4], X_test[index,5], X_test[index,6],y_true[index])
        print("NOT_DONE ")
        print(rel_error[index])
        print(Max)

else:
        Data = str(X_test[index])
        Data = Data.replace("[","")
        Data = Data.replace("]","")
        print(X_test[index,0],X_test[index,1], X_test[index,2], X_test[index,3], X_test[index,4], X_test[index,5], X_test[index,6],y_true[index])
        print("DONE")
        print("Final Maximum Error=", rel_error[index])
        print(Max)

y_pred = 10**y_pred
pred= y_pred

rel_error = 100*(rel_error)

#finding the mean of relative error
rel_m = np.mean(rel_error)
# #printing mean of rel error and rrmse for each iteration in a separate mean.csv file
os.system("echo -n "+str(rel_m)+","+str(abs_m)+" >> mean.csv")

## Printing the final predicted data
a=pd.DataFrame(pred,columns=['Predicted'])
a.to_csv('pred.csv',index=False)

import matplotlib.pyplot as plt
actual= df3.iloc[:, 5].values
plt.plot(actual,pred,'o')

from sklearn.metrics import r2_score
# calculate the r2
r2 = r2_score(actual, pred)
print('R-squared: %.2f' %r2)
